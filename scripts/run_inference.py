import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

import argparse
import warnings
import logging

import numpy as np
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from scripts.preprocess_dataset import resample_spectrum, label_file
from models.registry import choices as model_choices, build as build_model



# =============================================
# ✅ Raman-Only Inference Script
# This script supports prediction on a single Raman spectrum (.txt file).
# FTIR inference has been deprecated and removed for scientific integrity.
# See: @raman-pipeline-focus-milestone
# =============================================


warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning
)


def load_raman_spectrum(filepath):
    """Load a 2-column Raman spectrum from a .txt file"""
    x_vals, y_vals = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue
    return np.array(x_vals), np.array(y_vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a single Raman spectrum (.txt file)."
    )
    parser.add_argument("--arch", type=str, default="figure2", choices=model_choices(),
                    help="Model architecture (must match the provided weights).")  # NEW
    parser.add_argument(
        "--target-len", type=int, required=True,
        help="Target length to match model input"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to Raman .txt file."
    )
    parser.add_argument(
        "--model", default="random",
        help="Path to .pth model file, or specify 'random' to use untrained weights."
    )
    parser.add_argument(
        "--output", default=None,
        help="Where to write prediction result. If omitted, prints to stdout."
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--quiet", action="store_true",
        help="Show only warnings and errors"
    )
    verbosity.add_argument(
        "--verbose", action="store_true",
        help="Show debug-level logging"
    )

    args = parser.parse_args()

    # configure logging
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    try:
        # Load & preprocess Raman spectrum
        if os.path.isdir(args.input):
            parser.error(f"Input must be a single Raman .txt file, got a directory: {args.input}")

        x_raw, y_raw = load_raman_spectrum(args.input)
        if len(x_raw) < 10:
            parser.error("Spectrum too short for inference.")

        data = resample_spectrum(x_raw, y_raw, target_len=args.target_len)
        # Shape = (1, 1, target_len) — valid input for Raman inference
        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)


        # 2. Load Model (via shared model registry)
        model = build_model(args.arch, args.target_len).to(DEVICE)
        if args.model != "random":
            state = torch.load(args.model, map_location="cpu") # broad compatibility
            model.load_state_dict(state)
        model.eval()
        
        

        # 3. Inference
        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).item()

        # 4. True Label
        try:
            true_label = label_file(args.input)
            label_str = f"True Label: {true_label}"
        except FileNotFoundError:
            label_str = "True Label: Unknown"

        result = f"Predicted Label: {pred} {label_str}\nRaw Logits: {logits.tolist()}"
        logging.info(result)

        # 5. Save or stdout
        if args.output:
            # ensure parent dir exists (e.g., outputs/inference/)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as fout:
                fout.write(result)
            logging.info("Result saved to %s", args.output)

        sys.exit(0)

    except Exception as e:
        logging.error(e)
        sys.exit(1)
