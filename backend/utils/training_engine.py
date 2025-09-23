"""
Core Training Engine for the POLYMEROS project.

This module contains the primary logic for model training and validation,
encapsulated in a reusable `TrainingEngine` class. It is designed to be
called by different interfaces, such as the command-line script
(train_model.py) and the web UI's TrainingManager.

This approach ensures that the core training process is consistent,
maintainable, and follows the DRY (Don't Repeat Yourself) principle.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

from .training_types import (
    TrainingConfig,
    TrainingProgress,
    get_cv_splitter,
    augment_spectral_data,
)
from backend.models.registry import build as build_model


class TrainingEngine:
    """Encapsulates the core model training and validation logic."""

    def __init__(self, config: TrainingConfig):
        """
        Initializes the TrainingEngine with a given configuration.

        Args:
            config (TrainingConfig): The configuration object for the training run.
        """
        self.config = config
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """Selects the appropriate compute device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def run(
        self, X: np.ndarray, y: np.ndarray, progress_callback: callable = None
    ) -> dict:
        """
        Executes the full cross-validation training and evaluation loop.
        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Label data.
            progress_callback (callable, optional):
                A function to call with
                progress updates. Defaults to None.

        Returns:
                dict: A dictionary containing the final
                results and metrics.
        """
        cv_splitter = get_cv_splitter(self.config.cv_strategy, self.config.num_folds)

        fold_accuracies = []
        all_conf_matrices = []
        final_model_state = None

        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y), 1):
            if progress_callback:
                progress_callback(
                    {
                        "type": "fold_start",
                        "fold": fold,
                        "total_folds": self.config.num_folds,
                    }
                )

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Apply data augmentation if enabled
            if self.config.enable_augmentation:
                X_train, y_train = augment_spectral_data(
                    X_train, y_train, noise_level=self.config.noise_level
                )

            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.long),
                ),
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                )
            )

            model = build_model(self.config.model_name, self.config.target_len).to(
                self.device
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config.learning_rate
            )
            criterion = nn.CrossEntropyLoss()

            for epoch in range(self.config.epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs = inputs.unsqueeze(1).to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                if progress_callback:
                    progress_callback(
                        {
                            "type": "epoch_end",
                            "fold": fold,
                            "epoch": epoch + 1,
                            "total_epochs": self.config.epochs,
                            "loss": running_loss / len(train_loader),
                        }
                    )

            # Validation
            model.eval()
            all_true, all_pred = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.unsqueeze(1).to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    all_true.extend(labels.cpu().numpy())
                    all_pred.extend(predicted.cpu().numpy())

            acc = accuracy_score(all_true, all_pred)
            fold_accuracies.append(acc)
            all_conf_matrices.append(confusion_matrix(all_true, all_pred).tolist())
            final_model_state = model.state_dict()

            if progress_callback:
                progress_callback({"type": "fold_end", "fold": fold, "accuracy": acc})

        return {
            "fold_accuracies": fold_accuracies,
            "confusion_matrices": all_conf_matrices,
            "mean_accuracy": np.mean(fold_accuracies),
            "std_accuracy": np.std(fold_accuracies),
            "model_state_dict": final_model_state,
        }
