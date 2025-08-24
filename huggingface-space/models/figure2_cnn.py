# 📌 MODEL DESIGNATION:
# Figure2CNN is validated ONLY for RAMAN spectra input.
# Any use for FTIR modeling is invalid and deprecated.
# See milestone: @figure2cnn-raman-only-milestone

import torch
import torch.nn as nn


class Figure2CNN(nn.Module):
    """ 
    CNN architecture based on Figure 2 of the referenced research paper.
    Designed for 1D spectral data input of length 500
    """

    def __init__(self, input_length=500, input_channels=1):
        super(Figure2CNN, self).__init__()

        self.input_channels = input_channels


        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Dynamically calculate flattened size after conv + pooling
        self.flattened_size = self._get_flattened_size(input_channels, input_length)

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary output
        )

    def _get_flattened_size(self,input_channels, input_length):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_length)
            out = self.conv_block(dummy_input)
            return out.view(1, -1).shape[1]

    def forward(self, x):
        """ 
        Defines the forward pass of the Figure2CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, input_length).

        Returns:
            torch.Tensor: Output tensor containing class scores.
        """
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)

    def describe_model(self):
        """Print architecture and flattened size (for debug). """
        print(r"\n Model Summary:")
        print(r" - Conv Block: 4 Layers")
        print(f" - Input length: {self.flattened_size} after conv/pool")
        print(f" - Classifier: {self.classifier}\n")
        