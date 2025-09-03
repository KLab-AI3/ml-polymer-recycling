"""
This script performs 10-fold cross-validation on FTIR data using a CNN model.
It includes optional preprocessing steps such as baseline correction, smoothing, and normalization.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
from datetime import datetime
import warnings

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from models.figure2_cnn import Figure2CNN
from depracated_scripts.preprocess_ftir import preprocess_ftir

# Argument parser
parser = argparse.ArgumentParser(
    description="Run 10-fold CV on FTIR data with optional preprocessing.")
parser.add_argument(
    "--target-len", type=int, default=500,
    help="Number of points to resample spectra to"
)
parser.add_argument(
    "--baseline", action="store_true",
    help="Apply baseline correction"
)
parser.add_argument(
    "--smooth", action="store_true",
    help="Apply Savitzky-Golay smoothing"
)
parser.add_argument(
    "--normalize", action="store_true",
    help="Apply min-max normalization"
)
parser.add_argument(
    "--batch-size", type=int, default=16,
    help="Batch size for training"
)
parser.add_argument(
    "--epochs", type=int, default=10,
    help="Number of training epochs."
)
parser.add_argument(
    "--learning-rate", type=float,
    default=1e-3, help="Learning rate for optimizer."
)

args = parser.parse_args()


# Print configuration
print("Preprocessing Configuration:")
print(f"    Reseample to    : {args.target_len} points")
print(f"    Baseline Correct: {'✅' if args.baseline else '❌'}")
print(f"    Smoothing       : {'✅' if args.smooth else '❌'}")
print(f"    Normalization   : {'✅' if args.normalize else '❌'}")

# Constants
DATASET_PATH = 'datasets/ftir'
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
NUM_FOLDS = 10
LEARNING_RATE = args.learning_rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Load and preprocess dataset
print("🔄 Loading and preprocessing FTIR data ...")
X, y = preprocess_ftir(
    DATASET_PATH,
    target_len=args.target_len,
    baseline_correction=args.baseline,
    apply_smoothing=args.smooth,
    normalize=args.normalize
)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
print(f"✅ Data Loaded: {X.shape[0]} samples, {X.shape[1]} features each.")
input_channels = 4

# Cross-validation setup
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_accuracies = []
all_conf_matrices = []

# Cross-validation loop
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/{NUM_FOLDS} Training...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = Figure2CNN(input_length=args.target_len, input_channels=input_channels).to(DEVICE)
        model.describe_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()


        # Training
        for epoch in range(EPOCHS):
            model.train()
            RUNNING_LOSS = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.size(0), input_channels, args.target_len).to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                RUNNING_LOSS += loss.item()

        # After fold loop (outside the epoch loop), print 1 line:
        print(f"✅ Fold {fold} completed. Final training loss: {RUNNING_LOSS:.4f}")

        torch.save(model.state_dict(), "outputs/ftir_model.pth")

        # Evaluation
        model.eval()
        all_true = []
        all_pred = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.view(inputs.size(0), input_channels, args.target_len).to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_true.extend(labels.cpu().numpy())
                all_pred.extend(predicted.cpu().numpy())

        accuracy = 100 * np.mean(np.array(all_true) == np.array(all_pred))
        fold_accuracies.append(accuracy)
        conf_mat = confusion_matrix(all_true, all_pred)
        all_conf_matrices.append(conf_mat)

        print(f"✅ Fold {fold} Accuracy: {accuracy:.2f}%")
        print(f"Confusion Matrix Fold {fold}):\n{conf_mat}")

# Final summary
print("\n📊 Final Cross-Validation Results:")
for i, acc in enumerate(fold_accuracies, 1):
    print(f"Fold {i}: {acc:.2f}%")

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print(f"\n✅ Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
print("✅ Model saved to outputs/ftir_model.pth")


# Diagnostics log saving
def save_diagnostics_log(fold_accs, conf_matrices, config_args, output_path="logs/ftir_cv_diagnostics.json"):
    fold_metrics = [
        {
            "fold": i + 1,
            "accuracy": acc,
            "confusion_matrix": cm.tolist()
        }
        for i, (acc, cm) in enumerate(zip(fold_accs, conf_matrices))
    ]
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessing": {
            "target_len": config_args.target_len,
            "baseline_correction": config_args.baseline,
            "smoothin": config_args.smooth,
            "normalization": config_args.normalize,
        },
        "fold_metrics": fold_metrics,
        "overall": {
            "mean_accuracy": float(np.mean(fold_accs)),
            "std_accuracy": float(np.std(fold_accs)),
            "num_folds": len(fold_accs),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "device": str(DEVICE)
        }
    }
    os.makedirs("logs", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"🧠 Diagnostics written to {output_path}")


# Run diagnostics save
save_diagnostics_log(fold_accuracies, all_conf_matrices, args)
