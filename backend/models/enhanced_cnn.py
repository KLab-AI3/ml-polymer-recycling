"""
All neural network blocks and architectures in models/enhanced_cnn.py are custom implementations, developed to expand the model registry for advanced polymer spectral classification. While inspired by established deep learning concepts (such as residual connections, attention mechanisms, and multi-scale convolutions), they are are unique to this project and tailored for 1D spectral data.

Registry expansion: The purpose is to enrich the available models.
Literature inspiration: SE-Net, ResNet, Inception.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock1D(nn.Module):
    """1D attention mechanism for spectral data."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: [batch, channels, length]
        b, c, _ = x.size()

        # Global average pooling
        y = self.global_pool(x).view(b, c)

        # Fully connected layers
        y = self.fc(y).view(b, c, 1)

        # Apply attention weights
        return x * y.expand_as(x)


class EnhancedResidualBlock1D(nn.Module):
    """Enhanced residual block with attention and improved normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_attention: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout1d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Skip connection
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        )

        # Attention mechanism
        self.attention = (
            AttentionBlock1D(out_channels) if use_attention else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply attention
        out = self.attention(out)

        out = out + identity
        return self.relu(out)


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolution block for capturing features at different scales."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Different kernel sizes for multi-scale feature extraction
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=9, padding=4)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Parallel convolutions with different kernel sizes
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)

        # Concatenate along channel dimension
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        return self.relu(out)


class EnhancedCNN(nn.Module):
    """Enhanced CNN with attention, multi-scale features, and improved architecture."""

    def __init__(
        self,
        input_length: int = 500,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_length = input_length
        self.num_classes = num_classes

        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Multi-scale feature extraction
        self.multiscale_block = MultiScaleConvBlock(32, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Enhanced residual blocks
        self.res_block1 = EnhancedResidualBlock1D(64, 96, use_attention=use_attention)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.res_block2 = EnhancedResidualBlock1D(96, 128, use_attention=use_attention)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.res_block3 = EnhancedResidualBlock1D(128, 160, use_attention=use_attention)

        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Calculate feature size after convolutions
        self.feature_size = 160

        # Enhanced classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure input is 3D: [batch, channels, length]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Feature extraction
        x = self.initial_conv(x)
        x = self.multiscale_block(x)
        x = self.pool1(x)

        x = self.res_block1(x)
        x = self.pool2(x)

        x = self.res_block2(x)
        x = self.pool3(x)

        x = self.res_block3(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x

    def get_feature_maps(self, x):
        """Extract intermediate feature maps for visualization."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        features = {}

        x = self.initial_conv(x)
        features["initial"] = x

        x = self.multiscale_block(x)
        features["multiscale"] = x
        x = self.pool1(x)

        x = self.res_block1(x)
        features["res1"] = x
        x = self.pool2(x)

        x = self.res_block2(x)
        features["res2"] = x
        x = self.pool3(x)

        x = self.res_block3(x)
        features["res3"] = x

        return features


class EfficientSpectralCNN(nn.Module):
    """Efficient CNN designed for real-time inference with good performance."""

    def __init__(self, input_length: int = 500, num_classes: int = 2):
        super().__init__()

        # Efficient feature extraction with depthwise separable convolutions
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Depthwise separable convolutions
            self._make_depthwise_sep_conv(32, 64),
            nn.MaxPool1d(2),
            self._make_depthwise_sep_conv(64, 96),
            nn.MaxPool1d(2),
            self._make_depthwise_sep_conv(96, 128),
            nn.MaxPool1d(2),
            # Final feature extraction
            nn.Conv1d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

        self._initialize_weights()

    def _make_depthwise_sep_conv(self, in_channels, out_channels):
        """Create depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv1d(
                in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class HybridSpectralNet(nn.Module):
    """Hybrid network combining CNN and attention mechanisms."""

    def __init__(self, input_length: int = 500, num_classes: int = 2):
        super().__init__()

        # CNN backbone
        self.cnn_backbone = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )

        # Final pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # CNN feature extraction
        x = self.cnn_backbone(x)

        # Prepare for attention: [batch, length, channels]
        x = x.transpose(1, 2)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)

        # Back to [batch, channels, length]
        x = attn_out.transpose(1, 2)

        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def create_enhanced_model(model_type: str = "enhanced", **kwargs):
    """Factory function to create enhanced models."""
    models = {
        "enhanced": EnhancedCNN,
        "efficient": EfficientSpectralCNN,
        "hybrid": HybridSpectralNet,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(models.keys())}"
        )

    return models[model_type](**kwargs)
