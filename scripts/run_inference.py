# scripts/run_inference.py
"""
CLI inference with preprocessing parity.
Applies: resample → baseline (deg=2) → smooth (w=11,o=2) → normalize
unless explicitly disabled via flags.

Usage (examples):
python scripts/run_inference.py \
    --input datasets/rdwp/sta-1.txt \
    --arch figure2 \
    --weights outputs/figure2_model.pth \
    --target-len 500

# Disable smoothing only:
python scripts/run_inference.py --input ... --arch resnet --weights ... --disable-smooth
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import logging
from pathlib import Path
from typing import cast
from torch import nn

import numpy as np
import torch
import torch.nn.functional as F

from models.registry import build, choices
from utils.preprocessing import preprocess_spectrum, TARGET_LENGTH
from scripts.plot_spectrum import load_spectrum
from scripts.discover_raman_files import label_file


def parse_args():
    p = argparse.ArgumentParser(description="Raman spectrum inference (parity with CLI preprocessing).")
    p.add_argument("--input", required=True, help="Path to a single Raman .txt file (2 columns: x, y).")
    p.add_argument("--arch", required=True, choices=choices(), help="Model architecture key.")
    p.add_argument("--weights", required=True, help="Path to model weights (.pth).")
    p.add_argument("--target-len", type=int, default=TARGET_LENGTH, help="Resample length (default: 500).")

    # Default = ON; use disable- flags to turn steps off explicitly.
    p.add_argument("--disable-baseline", action="store_true", help="Disable baseline correction.")
    p.add_argument("--disable-smooth", action="store_true", help="Disable Savitzky–Golay smoothing.")
    p.add_argument("--disable-normalize", action="store_true", help="Disable min-max normalization.")

    p.add_argument("--output", default=None, help="Optional output JSON path (defaults to outputs/inference/<name>.json).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device (default: cpu).")
    return p.parse_args()


def _load_state_dict_safe(path: str):
    """Load a state dict safely across torch versions & checkpoint formats."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)  # newer torch
    except TypeError:
        obj = torch.load(path, map_location="cpu")  # fallback for older torch

    # Accept either a plain state_dict or a checkpoint dict that contains one
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

    if not isinstance(obj, dict):
        raise ValueError(
            "Loaded object is not a state_dict or checkpoint with a state_dict. "
            f"Type={type(obj)} from file={path}"
        )

    # Strip DataParallel 'module.' prefixes if present
    if any(key.startswith("module.") for key in obj.keys()):
        obj = {key.replace("module.", "", 1): val for key, val in obj.items()}

    return obj


def main():
    logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # --- Load raw spectrum
    x_raw, y_raw = load_spectrum(str(in_path))
    if len(x_raw) < 10:
        raise ValueError("Input spectrum has too few points (<10).")

    # --- Preprocess (single source of truth)
    _, y_proc = preprocess_spectrum(
        np.array(x_raw),
        np.array(y_raw),
        target_len=args.target_len,
        do_baseline=not args.disable_baseline,
        do_smooth=not args.disable_smooth,
        do_normalize=not args.disable_normalize,
        out_dtype="float32",
    )

    # --- Build model & load weights (safe)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = cast(nn.Module, build(args.arch, args.target_len)).to(device)
    state = _load_state_dict_safe(args.weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logging.info("Loaded with non-strict keys. missing=%d unexpected=%d", len(missing), len(unexpected))

    model.eval()

    # Shape: (B, C, L) = (1, 1, target_len)
    x_tensor = torch.from_numpy(y_proc[None, None, :]).to(device)

    with torch.no_grad():
        logits = model(x_tensor).float().cpu()  # shape (1, num_classes)
        probs = F.softmax(logits, dim=1)

    probs_np = probs.numpy().ravel().tolist()
    logits_np = logits.numpy().ravel().tolist()
    pred_label = int(np.argmax(probs_np))

    # Optional ground-truth from filename (if encoded)
    true_label = label_file(str(in_path))

    # --- Prepare output
    out_dir = Path("outputs") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (out_dir / f"{in_path.stem}_{args.arch}.json")

    result = {
        "input_file": str(in_path),
        "arch": args.arch,
        "weights": str(args.weights),
        "target_len": args.target_len,
        "preprocessing": {
            "baseline": not args.disable_baseline,
            "smooth": not args.disable_smooth,
            "normalize": not args.disable_normalize,
        },
        "predicted_label": pred_label,
        "true_label": true_label,
        "probs": probs_np,
        "logits": logits_np,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logging.info("Predicted Label: %d  True Label: %s", pred_label, true_label)
    logging.info("Raw Logits: %s", logits_np)
    logging.info("Result saved to %s", out_path)


if __name__ == "__main__":
    main()
