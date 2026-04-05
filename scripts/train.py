import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time
from torch.utils.data import DataLoader

# Add scripts folder to path so we can import our modules
sys.path.append(os.path.dirname(__file__))
from dataset import (load_all_data, split_data, compute_norm_stats,
                     normalise, MotorFaultDataset, get_weighted_sampler)
from model import CNNGRUModel, count_parameters


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'window_size'  : 800,
    'batch_size'   : 64,
    'epochs'       : 30,
    'learning_rate': 1e-3,
    'weight_decay' : 1e-4,
    'train_ratio'  : 0.8,
    'random_seed'  : 42,
    'num_classes'  : 6,
    'num_channels' : 9,
}



# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
CHECKPOINT   = os.path.join(RESULTS_DIR, 'best_model.pt')
HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.json')


# ── Compute class weights for loss function ───────────────────────────────────
def get_class_weights(labels, num_classes=6):
    """
    Compute per-class weights for weighted cross entropy loss.
    Weight = total_samples / (num_classes * class_count)
    Rare classes get higher weight.
    """
    class_counts = np.bincount(labels.astype(int), minlength=num_classes)
    total        = len(labels)
    weights      = total / (num_classes * class_counts)
    weights      = torch.tensor(weights, dtype=torch.float32)

    print("Class weights for loss function:")
    class_names = {0:'VREC', 1:'OP', 2:'REVD', 3:'2PSC', 4:'1PSC', 5:'NF'}
    for i, (w, c) in enumerate(zip(weights, class_counts)):
        print(f"  {class_names[i]:<6} count={c:>4}, weight={w:.4f}")

    return weights


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for i, (batch_windows, batch_labels) in enumerate(loader):
        # Stop early if cap reached
        

        batch_windows = batch_windows.to(device)
        batch_labels  = batch_labels.to(device)

        optimiser.zero_grad()
        logits = model(batch_windows)
        loss   = criterion(logits, batch_labels)
        loss.backward()
        optimiser.step()

        total_loss    += loss.item() * len(batch_labels)
        preds          = logits.argmax(dim=1)
        correct       += (preds == batch_labels).sum().item()
        total_samples += len(batch_labels)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for i, (batch_windows, batch_labels) in enumerate(loader):

            batch_windows = batch_windows.to(device)
            batch_labels  = batch_labels.to(device)

            logits = model(batch_windows)
            loss   = criterion(logits, batch_labels)

            total_loss    += loss.item() * len(batch_labels)
            preds          = logits.argmax(dim=1)
            correct       += (preds == batch_labels).sum().item()
            total_samples += len(batch_labels)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ── Validation loop ───────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for batch_windows, batch_labels in loader:
            batch_windows = batch_windows.to(device)
            batch_labels  = batch_labels.to(device)

            logits = model(batch_windows)
            loss   = criterion(logits, batch_labels)

            total_loss    += loss.item() * len(batch_labels)
            preds          = logits.argmax(dim=1)
            correct       += (preds == batch_labels).sum().item()
            total_samples += len(batch_labels)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy

def check_for_leakage(train_dataset, test_dataset):
    """
    Verify no windows from the same time region appear in both splits.
    Checks a sample of windows for exact matches.
    """
    print("\nChecking for data leakage...")
    
    # Sample 100 random test windows
    test_indices  = np.random.choice(len(test_dataset), 
                                      min(100, len(test_dataset)), 
                                      replace=False)
    train_indices = np.random.choice(len(train_dataset),
                                      min(1000, len(train_dataset)),
                                      replace=False)

    # Convert samples to numpy for comparison
    test_samples  = np.array([test_dataset.windows[i] 
                               for i in test_indices])
    train_samples = np.array([train_dataset.windows[i] 
                               for i in train_indices])

    # Check for exact matches
    matches = 0
    for test_w in test_samples:
        for train_w in train_samples:
            if np.allclose(test_w, train_w, atol=1e-6):
                matches += 1

    print(f"Exact matches found: {matches}")
    if matches == 0:
        print("✓ No leakage detected")
    else:
        print(f"✗ WARNING: {matches} identical windows in train and test")

    # Also check label distribution in datasets
    train_label_dist = np.bincount(
        np.array(train_dataset.targets), minlength=6
    )
    test_label_dist  = np.bincount(
        np.array(test_dataset.targets),  minlength=6
    )
    
    print("\nWindow-level class distribution:")
    print(f"{'Class':<8} {'Train':>8} {'Test':>8} {'Ratio':>8}")
    class_names = {0:'VREC', 1:'OP', 2:'REVD', 3:'2PSC', 4:'1PSC', 5:'NF'}
    for i in range(6):
        ratio = (test_label_dist[i] / train_label_dist[i] 
                 if train_label_dist[i] > 0 else 0)
        print(f"{class_names[i]:<8} {train_label_dist[i]:>8} "
              f"{test_label_dist[i]:>8} {ratio:>8.2f}")

# ── Main training function ────────────────────────────────────────────────────
def train(config=CONFIG):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}\n")

    # Reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # ── Data ──────────────────────────────────────────────────────────
    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    data, labels = load_all_data(verbose=True)

    print("\nSPLITTING DATA")
    print("=" * 50)
    train_data, train_labels, test_data, test_labels = split_data(
        data, labels,
        train_ratio = config['train_ratio'],
        random_seed = config['random_seed']
    )

    print("\nNORMALISING")
    print("=" * 50)
    mean, std  = compute_norm_stats(train_data)
    train_data = normalise(train_data, mean, std)
    test_data  = normalise(test_data,  mean, std)

    # Save normalisation stats for inference later
    norm_stats = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(os.path.join(RESULTS_DIR, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)

    print("\nCREATING DATASETS")
    print("=" * 50)
    train_dataset = MotorFaultDataset(
        train_data, train_labels,
        window_size = config['window_size']
    )
    test_dataset = MotorFaultDataset(
        test_data, test_labels,
        window_size = config['window_size']
    )
    
    check_for_leakage(train_dataset, test_dataset)

    # Weighted sampler for training only
    sampler      = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size = config['batch_size'],
        sampler    = sampler
    )
    test_loader  = DataLoader(
        test_dataset,
        batch_size = config['batch_size'],
        shuffle    = False
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────
    print("\nINITIALISING MODEL")
    print("=" * 50)
    model = CNNGRUModel(
        num_channels = config['num_channels'],
        window_size  = config['window_size'],
        num_classes  = config['num_classes']
    ).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # ── Loss and optimiser ────────────────────────────────────────────
    class_weights = get_class_weights(train_labels, config['num_classes'])
    criterion     = nn.CrossEntropyLoss(
        weight = class_weights.to(device)
    )
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr           = config['learning_rate'],
        weight_decay = config['weight_decay']
    )

    # Learning rate scheduler — reduce lr when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5,
        patience=5
    )

    # ── Training loop ─────────────────────────────────────────────────
    print("\nTRAINING")
    print("=" * 50)

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss' : [], 'test_acc' : []
    }
    best_test_acc = 0.0

    for epoch in range(1, config['epochs'] + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, device,
            max_batches = config.get('max_train_batches', None)
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device,
            max_batches = config.get('max_test_batches', None)
        )

        # Step scheduler
        scheduler.step(test_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc' ].append(train_acc)
        history['test_loss' ].append(test_loss)
        history['test_acc'  ].append(test_acc)

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'config'     : config,
                'norm_stats' : norm_stats,
                'test_acc'   : test_acc
            }, CHECKPOINT)
            saved = ' ← best saved'
        else:
            saved = ''

        elapsed = time.time() - start
        print(f"Epoch {epoch:>3}/{config['epochs']} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Test loss: {test_loss:.4f} acc: {test_acc:.4f} | "
              f"{elapsed:.1f}s{saved}")

    # Save training history
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Model saved to:     {CHECKPOINT}")
    print(f"History saved to:   {HISTORY_PATH}")

    return model, history


if __name__ == '__main__':
    train()