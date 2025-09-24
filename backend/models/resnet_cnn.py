"""
📌 MODEL DESIGNATION:
Figure2CNN is validated ONLY for RAMAN spectra input.
Any use for FTIR modeling is invalid and deprecated.
See milestone: @figure2cnn-raman-only-milestone
"""
import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """ 
    Basic 1-D residual block:
    Conv1d -> ReLU -> Conv1d (+ skip connection).
    If channel count changes, a 1x1 Conv aligns the skip path.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size, padding=padding)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

    def describe_model(self):
        """Print architecture and flattened size (for debug). """
        print(r"\n Model Summary:")
        print(r" - Conv Block: 4 Layers")
        print(f" - Input length: {self.flattened_size} after conv/pool")
        print(f" - Classifier: {self.classifier}\n")


class ResNet1D(nn.Module):
    """ 
    Lightweight 1-D ResNet for Raman spectra (length 500, single channel).
    """

    def __init__(self, input_length: int = 500, num_classes: int = 2):
        super().__init__()

        # Three residual stages
        self.stage1 = ResidualBlock1D(1, 16)
        self.stage2 = ResidualBlock1D(16, 32)
        self.stage3 = ResidualBlock1D(32, 64)

        # Global aggregation + classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # -> [B, 64, 1]
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).squeeze(-1)     # -> [B, 64]
        return self.fc(x)
