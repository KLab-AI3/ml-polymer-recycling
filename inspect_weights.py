"""
Diagnostic script to inspect the weights within a PyTorch .pth file.

This utility loads a model's state dictionary and prints summary statistics
(mean, std, min, max) for each parameter tensor. It helps diagnose issues
like corrupted weights from failed or interrupted training runs, which might
result in a model producing constant, incorrect outputs.

Usage:
    python scripts/inspect_weights.py path/to/your/model_weights.pth
"""

import torch
import argparse
import os
from pathlib import Path
import sys

# Add project root to path to allow imports from other modules
sys.path.append(str(Path(__file__).resolve().parent.parent))


def inspect_weights(file_path: str):
    """
    Loads a model state_dict from a .pth file and prints statistics
    for each parameter tensor to help diagnose corrupted weights.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    print(f"🔍 Inspecting weights for: {file_path}\n")

    try:
        # Load the state dictionary
        # Use weights_only=True for security and to supress the warning
        try:
            state_dict = torch.load(
                file_path, map_location=torch.device("cpu"), weights_only=True
            )
        except TypeError:  # Fallback for older torch versions
            state_dict = torch.load(file_path, map_location=torch.device("cpu"))

        # Handle checkpoints that save the model in a sub-dictionary
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        if not state_dict:
            print("⚠️ State dictionary is empty.")
            return

        print(
            f"{'Parameter Name':<40} {'Shape':<20} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}"
        )
        print("-" * 120)
        all_stds = []

        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                # Ensure tensor is float for stats, but don't fail if not
                try:
                    param_float = param.float()
                    mean_val = f"{param_float.mean().item():.4e}"
                    std_val_float = param_float.std().item()
                    std_val = f"{std_val_float:.4e}"
                    min_val = f"{param_float.min().item():.4e}"
                    max_val = f"{param_float.max().item():.4e}"
                    all_stds.append(std_val_float)
                except (RuntimeError, TypeError):
                    mean_val, std_val, min_val, max_val = "N/A", "N/A", "N/A", "N/A"

                shape_str = str(list(param.shape))
                print(
                    f"{name:<40} {shape_str:<20} {mean_val:<15} {std_val:<15} {min_val:<15} {max_val:<15}"
                )
            else:
                print(f"{name:<40} {'Non-Tensor':<20} {str(param):<60}")

        print("\n" + "-" * 120)
        print("✅ Inspection complete.")
        print("\nDiagnosis:")
        print(
            "- If you see all zeros, NaNs, or very small (e.g., e-38) uniform values, the weights file is likely corrupted."
        )
        if all(s < 1e-6 for s in all_stds if s is not None):
            print(
                "- WARNING: All parameter standard deviations are extremely low. The model may be 'dead' and insensitive to input."
            )
        else:
            print(
                "- The weight statistics appear varied, suggesting the file is not corrupted with zeros/NaNs."
            )
            print(
                "- If the model still produces constant output, it is likely poorly trained."
            )

        print("\nRecommendation: Retraining the model is the correct solution.")

    except Exception as e:
        print(f"❌ An error occurred while inspecting the weights file: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch model weights in a .pth file."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the .pth model weights file."
    )
    args = parser.parse_args()
    inspect_weights(args.file_path)
