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
import csv
import logging
from pathlib import Path
from typing import cast, Dict, List, Any
from torch import nn
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.registry import build, choices, build_multiple, validate_model_list
from utils.preprocessing import preprocess_spectrum, TARGET_LENGTH
from utils.multifile import parse_spectrum_data, detect_file_format
from scripts.plot_spectrum import load_spectrum
from scripts.discover_raman_files import label_file


def parse_args():
    p = argparse.ArgumentParser(
        description="Raman/FTIR spectrum inference with multi-model support."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to spectrum file (.txt, .csv, .json) or directory for batch processing.",
    )

    # Model selection - either single or multiple
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--arch", choices=choices(), help="Single model architecture key."
    )
    group.add_argument(
        "--models",
        help="Comma-separated list of models for comparison (e.g., 'figure2,resnet,resnet18vision').",
    )

    p.add_argument(
        "--weights",
        help="Path to model weights (.pth). For multi-model, use pattern with {model} placeholder.",
    )
    p.add_argument(
        "--target-len",
        type=int,
        default=TARGET_LENGTH,
        help="Resample length (default: 500).",
    )

    # Modality support
    p.add_argument(
        "--modality",
        choices=["raman", "ftir"],
        default="raman",
        help="Spectroscopy modality for preprocessing (default: raman).",
    )

    # Default = ON; use disable- flags to turn steps off explicitly.
    p.add_argument(
        "--disable-baseline", action="store_true", help="Disable baseline correction."
    )
    p.add_argument(
        "--disable-smooth",
        action="store_true",
        help="Disable Savitzky–Golay smoothing.",
    )
    p.add_argument(
        "--disable-normalize",
        action="store_true",
        help="Disable min-max normalization.",
    )

    p.add_argument(
        "--output",
        default=None,
        help="Output path - JSON for single file, CSV for multi-model comparison.",
    )
    p.add_argument(
        "--output-format",
        choices=["json", "csv"],
        default="json",
        help="Output format for results.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu).",
    )

    # File format options
    p.add_argument(
        "--file-format",
        choices=["auto", "txt", "csv", "json"],
        default="auto",
        help="Input file format (auto-detect by default).",
    )

    return p.parse_args()


# /////////////////////////////////////////////////////////


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


# /////////////////////////////////////////////////////////


def run_single_model_inference(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    model_name: str,
    weights_path: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Run inference with a single model."""
    start_time = time.time()

    # Preprocess spectrum
    _, y_proc = preprocess_spectrum(
        x_raw,
        y_raw,
        target_len=args.target_len,
        modality=args.modality,
        do_baseline=not args.disable_baseline,
        do_smooth=not args.disable_smooth,
        do_normalize=not args.disable_normalize,
        out_dtype="float32",
    )

    # Build model & load weights
    model = cast(nn.Module, build(model_name, args.target_len)).to(device)
    state = _load_state_dict_safe(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logging.info(
            f"Model {model_name}: Loaded with non-strict keys. missing={len(missing)} unexpected={len(unexpected)}"
        )

    model.eval()

    # Run inference
    x_tensor = torch.from_numpy(y_proc[None, None, :]).to(device)

    with torch.no_grad():
        logits = model(x_tensor).float().cpu()
        probs = F.softmax(logits, dim=1)

    processing_time = time.time() - start_time
    probs_np = probs.numpy().ravel().tolist()
    logits_np = logits.numpy().ravel().tolist()
    pred_label = int(np.argmax(probs_np))

    # Map prediction to class name
    class_names = ["Stable", "Weathered"]
    predicted_class = (
        class_names[pred_label]
        if pred_label < len(class_names)
        else f"Class_{pred_label}"
    )

    return {
        "model": model_name,
        "prediction": pred_label,
        "predicted_class": predicted_class,
        "confidence": max(probs_np),
        "probs": probs_np,
        "logits": logits_np,
        "processing_time": processing_time,
    }


# /////////////////////////////////////////////////////////


def run_multi_model_inference(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    model_names: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Run inference with multiple models for comparison."""
    results = {}

    for model_name in model_names:
        try:
            # Generate weights path - either use pattern or assume same weights for all
            if args.weights and "{model}" in args.weights:
                weights_path = args.weights.format(model=model_name)
            elif args.weights:
                weights_path = args.weights
            else:
                # Default weights path pattern
                weights_path = f"outputs/{model_name}_model.pth"

            if not Path(weights_path).exists():
                logging.warning(f"Weights not found for {model_name}: {weights_path}")
                continue

            result = run_single_model_inference(
                x_raw, y_raw, model_name, weights_path, args, device
            )
            results[model_name] = result

        except Exception as e:
            logging.error(f"Failed to run inference with {model_name}: {str(e)}")
            continue

    return results


# /////////////////////////////////////////////////////////


def save_results(
    results: Dict[str, Any], output_path: Path, format: str = "json"
) -> None:
    """Save results to file in specified format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    elif format == "csv":
        # Convert to tabular format for CSV
        if "models" in results:  # Multi-model results
            rows = []
            for model_name, model_result in results["models"].items():
                row = {
                    "model": model_name,
                    "prediction": model_result["prediction"],
                    "predicted_class": model_result["predicted_class"],
                    "confidence": model_result["confidence"],
                    "processing_time": model_result["processing_time"],
                }
                # Add individual class probabilities
                if "probs" in model_result:
                    for i, prob in enumerate(model_result["probs"]):
                        row[f"prob_class_{i}"] = prob
                rows.append(row)

            # Write CSV
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        else:  # Single model result
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                writer.writeheader()
                writer.writerow(results)


def main():
    logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
    args = parse_args()

    # Input validation
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Determine if this is single or multi-model inference
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        model_names = validate_model_list(model_names)
        if not model_names:
            raise ValueError(f"No valid models found in: {args.models}")
        multi_model = True
    else:
        model_names = [args.arch]
        multi_model = False

    # Load and parse spectrum data
    if args.file_format == "auto":
        file_format = None  # Auto-detect
    else:
        file_format = args.file_format

    try:
        # Read file content
        with open(in_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse spectrum data with format detection
        x_raw, y_raw = parse_spectrum_data(content, str(in_path))
        x_raw = np.array(x_raw, dtype=np.float32)
        y_raw = np.array(y_raw, dtype=np.float32)

    except Exception as e:
        x_raw, y_raw = load_spectrum(str(in_path))
        x_raw = np.array(x_raw, dtype=np.float32)
        y_raw = np.array(y_raw, dtype=np.float32)
        logging.warning(
            f"Failed to parse with new parser, falling back to original: {e}"
        )
        x_raw, y_raw = load_spectrum(str(in_path))

    if len(x_raw) < 10:
        raise ValueError("Input spectrum has too few points (<10).")

    # Setup device
    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    # Run inference
    model_results = {}  # Initialize to avoid unbound variable error
    if multi_model:
        model_results = run_multi_model_inference(
            np.array(x_raw, dtype=np.float32),
            np.array(y_raw, dtype=np.float32),
            model_names,
            args,
            device,
        )

        # Get ground truth if available
        true_label = label_file(str(in_path))

        # Prepare combined results
        results = {
            "input_file": str(in_path),
            "modality": args.modality,
            "models": model_results,
            "true_label": true_label,
            "preprocessing": {
                "baseline": not args.disable_baseline,
                "smooth": not args.disable_smooth,
                "normalize": not args.disable_normalize,
                "target_len": args.target_len,
            },
            "comparison": {
                "total_models": len(model_results),
                "agreements": (
                    sum(
                        1
                        for i, (_, r1) in enumerate(model_results.items())
                        for j, (_, r2) in enumerate(
                            list(model_results.items())[i + 1 :]
                        )
                        if r1["prediction"] == r2["prediction"]
                    )
                    if len(model_results) > 1
                    else 0
                ),
            },
        }

        # Default output path for multi-model
        default_output = (
            Path("outputs")
            / "inference"
            / f"{in_path.stem}_comparison.{args.output_format}"
        )

    else:
        # Single model inference
        model_result = run_single_model_inference(
            x_raw, y_raw, model_names[0], args.weights, args, device
        )
        true_label = label_file(str(in_path))

        results = {
            "input_file": str(in_path),
            "modality": args.modality,
            "arch": model_names[0],
            "weights": str(args.weights),
            "target_len": args.target_len,
            "preprocessing": {
                "baseline": not args.disable_baseline,
                "smooth": not args.disable_smooth,
                "normalize": not args.disable_normalize,
            },
            "predicted_label": model_result["prediction"],
            "predicted_class": model_result["predicted_class"],
            "true_label": true_label,
            "confidence": model_result["confidence"],
            "probs": model_result["probs"],
            "logits": model_result["logits"],
            "processing_time": model_result["processing_time"],
        }

        # Default output path for single model
        default_output = (
            Path("outputs")
            / "inference"
            / f"{in_path.stem}_{model_names[0]}.{args.output_format}"
        )

    # Save results
    output_path = Path(args.output) if args.output else default_output
    save_results(results, output_path, args.output_format)

    # Log summary
    if multi_model:
        logging.info(
            f"Multi-model inference completed with {len(model_results)} models"
        )
        for model_name, result in model_results.items():
            logging.info(
                f"{model_name}: {result['predicted_class']} (confidence: {result['confidence']:.3f})"
            )
        logging.info(f"Results saved to {output_path}")
    else:
        logging.info(
            f"Predicted Label: {results['predicted_label']} ({results['predicted_class']})"
        )
        logging.info(f"Confidence: {results['confidence']:.3f}")
        logging.info(f"True Label: {results['true_label']}")
        logging.info(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
