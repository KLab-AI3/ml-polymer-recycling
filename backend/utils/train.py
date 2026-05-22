"""
Main Training Script

This script orchestrates the model training process. It is configuration-driven
and uses MLflow for experiment tracking.

Usage:
    python scripts/train.py --config-path configs/base_config.yaml
"""

from pathlib import Path
import sys
import argparse
import yaml
from typing import Dict, Optional, Any

import pandas as pd
import torch
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Ensure the backend is in the path to import registry and preprocessing
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import TARGET_LEN
from backend.utils.preprocessing import preprocess_spectrum
from models.registry import build


def load_data(data_path: Path, target_len: int):
    """Load and preprocess data from a CSV file."""
    df = pd.read_csv(data_path)

    # This is a placeholder for your actual data loading.
    # You need to parse your 'spectra' column into x and y values.
    # For this example, we assume 'y_values' are stored as a string of numbers.
    # A more robust solution would use np.load or similar if data is saved in binary format.

    all_y = []
    # This loop is inefficient and for demonstration only. Vectorize in production.
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {data_path.name}"):
        # Dummy x_values, as preprocess_spectrum primarily uses y_values
        x_values = range(len(row['spectrum'].split()))
        y_values = [float(y) for y in row['spectrum'].split()]
        _, y_processed = preprocess_spectrum(
            x_values, y_values, modality='raman')
        all_y.append(y_processed)

    features = torch.tensor(all_y, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(df['label'].values, dtype=torch.long)

    return TensorDataset(features, labels)


def train(config: dict, jobs_db: Optional[Dict[str, Any]] = None, job_id: Optional[str] = None):
    """Main training and validation loop."""
    try:
        # --- MLflow Setup ---
        mlflow.set_experiment(config['experiment_name'])
        with mlflow.start_run(run_name=config.get('run_name', 'default_run')) as run:
            mlflow.log_params(config)
            if jobs_db and job_id:
                jobs_db[job_id]['mlflow_run_id'] = run.info.run_id
                jobs_db[job_id]['status'] = 'RUNNING'
            print(f"MLflow Run ID: {run.info.run_id}")

            # --- Data Loading ---
            data_dir = Path(config['data_dir'])
            train_dataset = load_data(data_dir / config['train_csv'], TARGET_LEN)
            val_dataset = load_data(data_dir / config['val_csv'], TARGET_LEN)

            train_loader = DataLoader(
                train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

            # --- Model, Optimizer, Loss ---
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            model = build(config['model_name'], TARGET_LEN).to(device)
            optimizer = getattr(torch.optim, config['optimizer'])(
                model.parameters(), lr=config['learning_rate'])
            criterion = getattr(torch.nn, config['loss_function'])()

            # --- Training Loop ---
            best_val_loss = float('inf')
            for epoch in range(config['epochs']):
                model.train()
                train_loss = 0.0
                for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
                    features, labels = features.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

                # --- Validation Loop ---
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                print(
                    f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # --- Progress Update for Web UI ---
                if jobs_db and job_id:
                    progress = (epoch + 1) / config['epochs']
                    jobs_db[job_id]['progress'] = progress
                    jobs_db[job_id]['metrics']['train_loss'].append(avg_train_loss)
                    jobs_db[job_id]['metrics']['val_loss'].append(avg_val_loss)
                    jobs_db[job_id]['current_epoch'] = epoch + 1

                # --- Save Best Model ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    mlflow.pytorch.log_model(
                        model, "model", registered_model_name=f"{config.get('run_name', 'default_run')}_best")
                    print(
                        f"New best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        if jobs_db and job_id:
            jobs_db[job_id]['status'] = 'COMPLETED'
            jobs_db[job_id]['progress'] = 1.0
        print("✅ Training complete.")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        if jobs_db and job_id:
            jobs_db[job_id]['status'] = 'FAILED'
            jobs_db[job_id]['error'] = str(e)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a spectral classification model.")
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Run training from CLI without web-specific job tracking
    train(config=config)
