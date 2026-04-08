import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score
)

sys.path.append(os.path.dirname(__file__))
from dataset import (load_all_data, split_data, compute_norm_stats,
                     normalise, MotorFaultDataset)
from model import CNNGRUModel

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['VREC', 'OP', 'REVD', '2PSC', '1PSC', 'NF']
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR  = os.path.join(RESULTS_DIR, 'figures')
CHECKPOINT   = os.path.join(RESULTS_DIR, 'best_model.pt')


def load_model_and_data(checkpoint_path, window_size=800, stride=800):
    """Load trained model and prepare test dataset."""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config     = checkpoint['config']
    norm_stats = checkpoint['norm_stats']

    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"with test acc {checkpoint['test_acc']:.4f}")

    # Recreate model
    model = CNNGRUModel(
        num_channels = config['num_channels'],
        window_size  = config['window_size'],
        num_classes  = config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Reload data using saved normalisation stats
    data, labels = load_all_data(verbose=False)
    _, _, test_data, test_labels = split_data(
        data, labels,
        train_ratio = config['train_ratio'],
        random_seed = config['random_seed'],
        verbose     = False
    )

    # Apply saved normalisation stats
    mean = np.array(norm_stats['mean'], dtype=np.float32)
    std  = np.array(norm_stats['std'],  dtype=np.float32)
    test_data = normalise(test_data, mean, std)

    # Create test dataset
    test_dataset = MotorFaultDataset(
        test_data, test_labels,
        window_size = window_size,
        stride      = stride
    )

    return model, test_dataset, config


def get_all_predictions(model, dataset, batch_size=64, device='cpu'):
    """Run inference on entire dataset and return predictions and labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds  = []
    all_labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_windows, batch_labels in loader:
            batch_windows = batch_windows.to(device)
            logits        = model(batch_windows)
            preds         = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    # Also compute normalised version
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalised
    sns.heatmap(
        cm_norm, annot=True, fmt='.3f', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (normalised)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved confusion matrix to {save_path}")


def plot_per_class_metrics(metrics_dict, save_path):
    """Bar chart of precision, recall, F1 per class."""
    classes    = list(metrics_dict.keys())
    precision  = [metrics_dict[c]['precision']  for c in classes]
    recall     = [metrics_dict[c]['recall']     for c in classes]
    f1         = [metrics_dict[c]['f1']         for c in classes]

    x     = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label='Precision', color='#2196F3')
    ax.bar(x,         recall,    width, label='Recall',    color='#4CAF50')
    ax.bar(x + width, f1,        width, label='F1-Score',  color='#FF9800')

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision, Recall and F1-Score')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [ax.patches[:len(classes)],
                 ax.patches[len(classes):2*len(classes)],
                 ax.patches[2*len(classes):]]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=7
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved per-class metrics to {save_path}")


def plot_training_curves(history_path, save_path):
    """Plot loss and accuracy curves from training history."""
    with open(history_path) as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train', color='#2196F3')
    ax1.plot(epochs, history['test_loss'],  label='Test',  color='#FF9800')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train', color='#2196F3')
    ax2.plot(epochs, history['test_acc'],  label='Test',  color='#FF9800')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.9, 1.01)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle('CNN-GRU Training History', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved training curves to {save_path}")


def evaluate(checkpoint_path=None, window_size=800, stride=800):
    """Full evaluation pipeline."""

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT

    os.makedirs(FIGURES_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # ── Load model and data ───────────────────────────────────────────
    print("=" * 55)
    print("LOADING MODEL AND DATA")
    print("=" * 55)
    model, test_dataset, config = load_model_and_data(
        checkpoint_path, window_size, stride
    )

    # ── Get predictions ───────────────────────────────────────────────
    print("\nRunning inference on test set...")
    y_pred, y_true = get_all_predictions(
        model, test_dataset,
        batch_size = 64,
        device     = device
    )
    print(f"Total windows evaluated: {len(y_pred):,}")

    # ── Overall metrics ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("OVERALL METRICS")
    print("=" * 55)
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    # ── Per-class metrics ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PER-CLASS METRICS")
    print("=" * 55)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(6))
    )

    metrics_dict = {}
    print(f"\n{'Class':<8} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Support':>10}")
    print("-" * 52)

    for i, name in enumerate(CLASS_NAMES):
        metrics_dict[name] = {
            'precision': float(precision[i]),
            'recall'   : float(recall[i]),
            'f1'       : float(f1[i]),
            'support'  : int(support[i])
        }
        print(f"{name:<8} {precision[i]:>10.4f} {recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} {support[i]:>10}")

    # Macro averages
    print("-" * 52)
    print(f"{'Macro avg':<8} {precision.mean():>10.4f} "
          f"{recall.mean():>10.4f} {f1.mean():>10.4f} "
          f"{support.sum():>10}")

    # Full sklearn report
    print("\nFull Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names = CLASS_NAMES
    ))

    # ── Save metrics to JSON ──────────────────────────────────────────
    results = {
        'overall_accuracy' : float(overall_acc),
        'window_size'      : window_size,
        'stride'           : stride,
        'per_class'        : metrics_dict,
        'macro_avg'        : {
            'precision': float(precision.mean()),
            'recall'   : float(recall.mean()),
            'f1'       : float(f1.mean())
        }
    }

    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {results_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        save_path = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    )

    plot_per_class_metrics(
        metrics_dict,
        save_path = os.path.join(FIGURES_DIR, 'per_class_metrics.png')
    )

    history_path = os.path.join(RESULTS_DIR, 'training_history.json')
    if os.path.exists(history_path):
        plot_training_curves(
            history_path,
            save_path = os.path.join(FIGURES_DIR, 'training_curves.png')
        )

    return results


if __name__ == '__main__':
    results = evaluate()

