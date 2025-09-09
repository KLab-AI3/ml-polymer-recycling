import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
import argparse, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import random
import json

from utils.training_engine import TrainingEngine
from utils.training_manager import TrainingConfig

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add project-specific imports
from scripts.preprocess_dataset import preprocess_dataset
from models.registry import choices as model_choices, build as build_model


# Argument parser for CLI usage
parser = argparse.ArgumentParser(
    description="Run 10-fold CV on Raman data with optional preprocessing.")
parser.add_argument("--target-len", type=int, default=500)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--model", type=str, default="figure2", choices=model_choices())
def parse_args():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Run 10-fold CV on Raman data with optional preprocessing."
    )
    parser.add_argument("--target-len", type=int, default=500)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="figure2", choices=model_choices())
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-path", type=str, default="datasets/rdwp")
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--cv-strategy", type=str, default="stratified_kfold", choices=["stratified_kfold", "kfold"])

args = parser.parse_args()
    return parser.parse_args()

# Constants
# Raman-only dataset (RDWP)
DATASET_PATH = 'datasets/rdwp'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
NUM_FOLDS = 10

# Ensure output dirs exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)
def cli_progress_callback(progress_data: dict):
    """A simple callback to print progress to the console."""
    if progress_data["type"] == "fold_start":
        print(f"\n🔁 Fold {progress_data['fold']}/{progress_data['total_folds']}")
    elif progress_data["type"] == "epoch_end":
        # Print progress on the same line
        print(
            f"  Epoch {progress_data['epoch']}/{progress_data['total_epochs']} | Loss: {progress_data['loss']:.4f}",
            end="\r",
        )
    elif progress_data["type"] == "fold_end":
        print(f"\n✅ Fold {progress_data['fold']} Accuracy: {progress_data['accuracy'] * 100:.2f}%")

print("Preprocessing Configuration:")
print(f"    Resample to     : {args.target_len}")

print(f"    Baseline Correct: {'✅' if args.baseline else '❌'}")
print(f"    Smoothing       : {'✅' if args.smooth else '❌'}")
print(f"    Normalization   : {'✅' if args.normalize else '❌'}")
def save_diagnostics_log(results: dict, config: TrainingConfig, output_path: str):
    """Saves a JSON log file with training diagnostics."""
    fold_metrics = [
        {"fold": i + 1, "accuracy": float(acc), "confusion_matrix": cm}
        for i, (acc, cm) in enumerate(
            zip(results["fold_accuracies"], results["confusion_matrices"])
        )
    ]
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": config.model_name,
        "config": config.to_dict(),
        "fold_metrics": fold_metrics,
        "overall": {
            "mean_accuracy": results["mean_accuracy"],
            "std_accuracy": results["std_accuracy"],
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"🧠 Diagnostics written to {output_path}")

# Load + Preprocess data
print("🔄 Loading and preprocessing data ...")
X, y = preprocess_dataset(
    DATASET_PATH,
    target_len=args.target_len,
    baseline_correction=args.baseline,
    apply_smoothing=args.smooth,
    normalize=args.normalize
)
X, y = np.array(X, np.float32), np.array(y, np.int64)
print(f"✅ Data Loaded: {X.shape[0]} samples, {X.shape[1]} features each.")
print(f"🔍 Using model: {args.model}")

# CV
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_accuracies = []
all_conf_matrices = []
def main():
    """Main function to run the training process from the CLI."""
    args = parse_args()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🔁 Fold {fold}/{NUM_FOLDS}")
    # Ensure output dirs exist
    os.makedirs("outputs/weights", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Create TrainingConfig from CLI args
    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset_path,
        target_len=args.target_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_folds=args.num_folds,
        baseline_correction=args.baseline,
        smoothing=args.smooth,
        normalization=args.normalize,
        device=args.device,
        cv_strategy=args.cv_strategy,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.long)))
    print("🔄 Loading and preprocessing data...")
    X, y = preprocess_dataset(config.dataset_path, target_len=config.target_len)
    print(f"✅ Data Loaded: {X.shape[0]} samples, {X.shape[1]} features each.")
    print(f"🔍 Using model: {config.model_name}")

    # Model selection
    model = build_model(args.model, args.target_len).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # Run training
    engine = TrainingEngine(config)
    results = engine.run(X, y, progress_callback=cli_progress_callback)

    for epoch in range(args.epochs):
        model.train()
        RUNNING_LOSS = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1).to(DEVICE)
            labels = labels.to(DEVICE)
    # Save final model and logs
    model_path = f"outputs/weights/{config.model_name}_model.pth"
    torch.save(results["model_state_dict"], model_path)
    print(f"\n✅ Model saved to {model_path}")

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            RUNNING_LOSS += loss.item()
    log_path = f"outputs/logs/{config.model_name}_cli_diagnostics.json"
    save_diagnostics_log(results, config, log_path)

    # After fold loop (outside the epoch loop), print 1 line:
    print(f"✅ Fold {fold} done. Final loss: {RUNNING_LOSS:.4f}")

    # Evaluation
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.unsqueeze(1).to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

    acc = 100 * np.mean(np.array(all_true) == np.array(all_pred))
    fold_accuracies.append(acc)
    all_conf_matrices.append(confusion_matrix(all_true, all_pred))
    print(f"✅ Fold {fold} Accuracy: {acc:.2f}%")

# Save model checkpoint **after** final fold
model_path = f"outputs/{args.model}_model.pth"
torch.save(model.state_dict(), model_path)

# Summary
mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
print("\n📊 Cross-Validation Results:")
for i, a in enumerate(fold_accuracies, 1):
    print(f"Fold {i}: {a:.2f}%")
print(f"\n✅ Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
print(f"✅ Model saved to {model_path}")

# Save diagnostics


def save_diagnostics_log(fold_acc, confs, args_param, output_path):
    fold_metrics = [
    {"fold": i + 1, "accuracy": float(a), "confusion_matrix": c.tolist()}
    for i, (a, c) in enumerate(zip(fold_acc, confs))
]
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessing": {
            "target_len": args_param.target_len,
            "baseline": args_param.baseline,
            "smooth": args_param.smooth,
            "normalize": args_param.normalize,
        },
        "fold_metrics": fold_metrics,
        "overall": {
            "mean_accuracy": float(np.mean(fold_acc)),
            "std_accuracy": float(np.std(fold_acc)),
            "num_folds": len(fold_acc),
            "batch_size": args_param.batch_size,
            "epochs": args_param.epochs,
            "learning_rate": args_param.learning_rate,
            "device": str(DEVICE)
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"🧠 Diagnostics written to {output_path}")

log_path = f"outputs/logs/raman_{args.model}_diagnostics.json"
save_diagnostics_log(fold_accuracies, all_conf_matrices, args, log_path)
if __name__ == "__main__":
    main()