import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """Single CNN block: Conv → BatchNorm → ReLU → MaxPool → Dropout"""

    def __init__(self, in_channels, out_channels, kernel_size=7,
                 dropout=0.2):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                padding      = kernel_size // 2  # same padding — keeps time dim
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class CNNGRUModel(nn.Module):
    """
    CNN-GRU fusion model for motor fault classification.
    
    CNN extracts spatial features across channels.
    GRU learns temporal patterns in the extracted features.

    Args:
        num_channels:  number of input signal channels (default: 9)
        window_size:   number of time points per window (default: 800)
        num_classes:   number of fault classes (default: 6)
        cnn_channels:  list of output channels per CNN block
        kernel_size:   conv kernel size
        cnn_dropout:   dropout rate in CNN blocks
        gru_hidden:    GRU hidden state size
        gru_layers:    number of GRU layers
        gru_dropout:   dropout between GRU layers
        fc_dropout:    dropout before final classifier
    """

    def __init__(
        self,
        num_channels = 9,
        window_size  = 800,
        num_classes  = 6,
        cnn_channels = [32, 64, 128],
        kernel_size  = 7,
        cnn_dropout  = 0.2,
        gru_hidden   = 256,
        gru_layers   = 2,
        gru_dropout  = 0.3,
        fc_dropout   = 0.4
    ):
        super(CNNGRUModel, self).__init__()

        self.num_channels = num_channels
        self.window_size  = window_size

        # ── CNN Blocks ────────────────────────────────────────────────
        cnn_in_channels = [num_channels] + cnn_channels[:-1]

        self.cnn_blocks = nn.ModuleList([
            CNNBlock(
                in_channels  = in_ch,
                out_channels = out_ch,
                kernel_size  = kernel_size,
                dropout      = cnn_dropout
            )
            for in_ch, out_ch in zip(cnn_in_channels, cnn_channels)
        ])

        # After 3 maxpool(2) layers, time dimension = window_size // 8
        self.cnn_out_time = window_size // (2 ** len(cnn_channels))
        self.cnn_out_ch   = cnn_channels[-1]  # 128

        # ── GRU ───────────────────────────────────────────────────────
        # GRU input: (batch, time, features)
        # CNN output: (batch, 128, reduced_time) → need to transpose
        self.gru = nn.GRU(
            input_size  = self.cnn_out_ch,   # 128 features per time step
            hidden_size = gru_hidden,         # 256
            num_layers  = gru_layers,         # 2
            dropout     = gru_dropout if gru_layers > 1 else 0,
            batch_first = True                # (batch, time, features)
        )

        # ── Classifier ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, num_channels, window_size)
        Returns:
            logits: tensor of shape (batch, num_classes)
        """
        # ── CNN forward ───────────────────────────────────────────────
        # x: (batch, 9, 800)
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        # x: (batch, 128, 100)  for window=800

        # ── Reshape for GRU ───────────────────────────────────────────
        # GRU expects (batch, time, features)
        # CNN output is (batch, features, time) → transpose
        x = x.permute(0, 2, 1)
        # x: (batch, 100, 128)

        # ── GRU forward ───────────────────────────────────────────────
        gru_out, _ = self.gru(x)
        # gru_out: (batch, 100, 256)

        # Take only the last time step output
        x = gru_out[:, -1, :]
        # x: (batch, 256)

        # ── Classifier forward ────────────────────────────────────────
        logits = self.classifier(x)
        # logits: (batch, 6)

        return logits


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # ── Sanity check ──────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test with each window size
    window_sizes = [100, 200, 400, 600, 800]

    for w in window_sizes:
        model = CNNGRUModel(
            num_channels = 9,
            window_size  = w,
            num_classes  = 6
        ).to(device)

        # Create a dummy batch
        dummy_input = torch.randn(32, 9, w).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        print(f"Window={w:>4} | "
              f"Input: {tuple(dummy_input.shape)} | "
              f"Output: {tuple(output.shape)} | "
              f"Params: {count_parameters(model):,}")

    print("\nModel architecture (window=800):")
    print(CNNGRUModel(window_size=800))