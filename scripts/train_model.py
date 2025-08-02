<<<<<<< HEAD
import os
import sys
=======
import os, sys, json
>>>>>>> e484a46 (Initial migration from original polymer_project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
import argparse, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

<<<<<<< HEAD
import random
import json

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

=======
# Add project-specific imports
from scripts.preprocess_dataset import preprocess_dataset
from models.figure2_cnn import Figure2CNN
from models.resnet_cnn import ResNet1D
>>>>>>> e484a46 (Initial migration from original polymer_project)

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
<<<<<<< HEAD
parser.add_argument("--model", type=str, default="figure2", choices=model_choices())

=======
parser.add_argument("--model", type=str, default="figure2",
                    choices=["figure2", "resnet"])
>>>>>>> e484a46 (Initial migration from original polymer_project)
args = parser.parse_args()

# Constants
# Raman-only dataset (RDWP)
DATASET_PATH = 'datasets/rdwp'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
NUM_FOLDS = 10

# Ensure output dirs exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

print("Preprocessing Configuration:")
<<<<<<< HEAD
print(f"    Resample to     : {args.target_len}")

=======
print(f"    Reseample to    : {args.target_len}")
>>>>>>> e484a46 (Initial migration from original polymer_project)
print(f"    Baseline Correct: {'✅' if args.baseline else '❌'}")
print(f"    Smoothing       : {'✅' if args.smooth else '❌'}")
print(f"    Normalization   : {'✅' if args.normalize else '❌'}")

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

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🔁 Fold {fold}/{NUM_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_loader = DataLoader(
<<<<<<< HEAD
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.long)))

    # Model selection
    model = build_model(args.model, args.target_len).to(DEVICE)
=======
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),batch_size=args.batch_size)

    # Model selection
    model = (Figure2CNN if args.model == "figure2" else ResNet1D)(
        input_length=args.target_len).to(DEVICE)
>>>>>>> e484a46 (Initial migration from original polymer_project)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        RUNNING_LOSS = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1).to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            RUNNING_LOSS += loss.item()

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
<<<<<<< HEAD
    fold_metrics = [
    {"fold": i + 1, "accuracy": float(a), "confusion_matrix": c.tolist()}
    for i, (a, c) in enumerate(zip(fold_acc, confs))
]
=======
    fold_metrics = [{"fold": i+1, "accuracy": acc,
                    "confusion_matrix": c.tolist()}
        for i, (a, c) in enumerate(zip(fold_acc, confs))]
>>>>>>> e484a46 (Initial migration from original polymer_project)
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