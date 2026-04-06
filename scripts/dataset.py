import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import json
import os

# ── paths ──────────────────────────────────────────────────────────────
def get_data_dir():
    """Returns data directory, checks environment variable first."""
    env_path = os.environ.get('MOTOR_DATA_DIR')
    if env_path:
        return env_path
    # Default local path
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

DATA_DIR = get_data_dir()
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'eda_config.json')

# ── fault file map ─────────────────────────────────────────────────────
FAULT_FILES = {
    'NF'  : 'Preprocessed_No_failed.mat',
    'OP'  : 'Preprocessed_Disconnect_Phase_10_11_21_.mat',
    '2PSC': 'Preprocessed_Short_between_two_phases_.mat',
    '1PSC': 'Preprocessed_Test_Data_Short_phases_Ln_G_.mat',
    'REVD': 'Preprocessed_Rotor_Current_Failed_R_.mat',
    'VREC': 'Preprocessed_Test_Data_Rotor_Current_Faild.mat'
}
def load_all_data(verbose=True):
    all_data   = []
    all_labels = []

    for fault_key, fname in FAULT_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        mat  = scipy.io.loadmat(path)

        # Convert to float32 immediately — halves memory usage
        experiments = mat['train_data'].astype(np.float32)
        labels      = mat['label_data'].flatten()

        if verbose:
            print(f"Loaded {fault_key}: {experiments.shape}, "
                  f"label={labels[0]}")

        all_data.append(experiments)
        all_labels.append(labels)

    data   = np.concatenate(all_data,   axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if verbose:
        # Show memory usage
        mem_gb = data.nbytes / 1e9
        print(f"\nTotal loaded: {data.shape}")
        print(f"Memory usage: {mem_gb:.2f} GB")

    return data, labels

def split_data(data, labels, train_ratio=0.8, random_seed=42, verbose=True):
    """
    Split at experiment level BEFORE windowing.
    Stratified so each class keeps its ratio in both splits.
    
    Returns:
        train_data, train_labels, test_data, test_labels
    """
    from sklearn.model_selection import train_test_split

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels,
        train_size=train_ratio,
        stratify=labels,        # preserve class ratios in both splits
        random_state=random_seed
    )

    print(f"Train: {train_data.shape[0]} experiments")
    print(f"Test:  {test_data.shape[0]} experiments")

    # Verify class distribution in each split
    print("\nClass distribution:")
    print(f"{'Class':<8} {'Train':>8} {'Test':>8}")
    for cls in np.unique(labels):
        tr = np.sum(train_labels == cls)
        te = np.sum(test_labels  == cls)
        print(f"{cls:<8} {tr:>8} {te:>8}")

    return train_data, train_labels, test_data, test_labels

def compute_norm_stats(train_data, verbose=True):
    """
    Compute mean and std per channel from TRAINING data only.
    Shape: train_data is (N, 10000, 9)
    Returns mean and std of shape (9,)
    """
    # Reshape to (N * 10000, 9) to compute stats across all time points
    reshaped = train_data.reshape(-1, train_data.shape[2])
    mean = reshaped.mean(axis=0)  # (9,)
    std  = reshaped.std(axis=0)   # (9,)

    # Avoid division by zero for constant channels
    std = np.where(std == 0, 1.0, std)

    print("Normalisation statistics (per channel):")
    print(f"{'Channel':<10} {'Mean':>10} {'Std':>10}")
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f"Ch {i:<7} {m:>10.4f} {s:>10.4f}")

    return mean, std


def normalise(data, mean, std):
    """
    Apply z-score normalisation using training statistics.
    Works for any split — train, test.
    """
    return (data - mean) / std

class MotorFaultDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for motor fault classification.
    Windows are computed on the fly instead of stored in memory.
    """

    def __init__(self, data, labels, window_size, stride=None):
        """
        Args:
            data:        np.ndarray (N, 10000, 9) — normalised experiments
            labels:      np.ndarray (N,)
            window_size: int
            stride:      int (default: window_size // 2)
        """
        self.data        = data
        self.labels      = labels
        self.window_size = window_size
        self.stride      = stride if stride is not None else window_size // 2
        self.signal_len  = data.shape[1]

        # Build index map — (experiment_idx, start_time) for each window
        # This stores integers only, not the actual data
        self.index_map = []
        for exp_idx in range(len(data)):
            start = 0
            while start + self.window_size <= self.signal_len:
                self.index_map.append((exp_idx, start))
                start += self.stride

        # Still need targets list for weighted sampler
        self.targets = [
            int(labels[exp_idx])
            for exp_idx, _ in self.index_map
        ]

        print(f"Created {len(self.index_map):,} windows "
              f"(window={self.window_size}, stride={self.stride})")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        exp_idx, start = self.index_map[idx]

        # Slice window on the fly — only loads what's needed
        window = self.data[exp_idx, start:start + self.window_size, :]

        # Transpose to (channels, time) for CNN
        window = torch.tensor(window.T, dtype=torch.float32)
        label  = torch.tensor(self.targets[idx], dtype=torch.long)

        return window, label

def get_weighted_sampler(dataset, verbose=True):
    """
    Creates a WeightedRandomSampler so each class is sampled equally
    during training, handling the 2.38x class imbalance.
    """
    labels  = np.array(dataset.targets)
    classes = np.unique(labels)

    # Weight per class = 1 / count
    class_counts  = np.array([np.sum(labels == c) for c in classes])
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = np.array([class_weights[c] for c in labels])
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    print("Class weights for sampler:")
    class_names = {0:'VREC', 1:'OP', 2:'REVD', 3:'2PSC', 4:'1PSC', 5:'NF'}
    for c, w, count in zip(classes, class_weights, class_counts):
        print(f"  {class_names[c]:<6} count={count:>4}, weight={w:.6f}")

    return sampler
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # 1. Load
    data, labels = load_all_data()

    # 2. Split BEFORE windowing
    train_data, train_labels, test_data, test_labels = split_data(data, labels)

    # 3. Normalise using TRAINING stats only
    mean, std = compute_norm_stats(train_data)
    train_data = normalise(train_data, mean, std)
    test_data  = normalise(test_data,  mean, std)

    # 4. Create datasets with window_size=800 for quick test
    train_dataset = MotorFaultDataset(train_data, train_labels, window_size=800)
    test_dataset  = MotorFaultDataset(test_data,  test_labels,  window_size=800)

    # 5. Create sampler for training
    sampler = get_weighted_sampler(train_dataset)

    # 6. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # 7. Verify a batch
    batch_windows, batch_labels = next(iter(train_loader))
    print(f"\nBatch verification:")
    print(f"Window shape: {batch_windows.shape}")   # expect (32, 9, 800)
    print(f"Labels shape: {batch_labels.shape}")    # expect (32,)
    print(f"Label values: {batch_labels.unique()}")
    