"""Multi-file processing utilities for batch inference.
Handles multiple file uploads and iterative processing.
Supports TXT, CSV, and JSON file formats with automatic detection."""

from typing import List, Dict, Any, Tuple, Optional
import time
import io
from pathlib import Path
import numpy as np
import json
import csv
import hashlib

from backend.utils.preprocessing import preprocess_spectrum
from backend.utils.errors import ErrorHandler
from backend.utils.confidence import calculate_softmax_confidence
from backend.config import TARGET_LEN


def detect_file_format(filename: str, content: str) -> str:
    """Automatically detect file format based on exstention and content

    Args:
        filename: Name of the file
        content: Content of the file

    Returns:
        File format: .'txt', .'csv', .'json'
    """
    # First try by extension
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        try:
            json.loads(content)
            return "json"
        except json.JSONDecodeError:
            pass
    elif suffix == ".csv":
        return "csv"
    elif suffix == ".txt":
        return "txt"

    # If extension doesn't match or is unclear, try content detection
    content_stripped = content.strip()

    # Try JSON
    if content_stripped.startswith(("{", "[")):
        try:
            json.loads(content)
            return "json"
        except json.JSONDecodeError:
            pass

    # Try CSV (look for commas in first few lines)
    lines = content_stripped.split("\n")[:5]
    comma_count = sum(line.count(",") for line in lines)
    if comma_count > len(lines):  # More commas than lines suggests CSV
        return "csv"

    # Default to TXT
    return "txt"


def parse_json_spectrum(content: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse spectrum data from JSON format.

    Expected formats:
    - {"wavenumbers": [...], "intensities": [...]}
    - {"x": [...], "y": [...]}
    - [{"wavenumber": val, "intensity": val}, ...]
    """

    try:
        data = json.loads(content)

        # Format 1: Object with arrays
        if isinstance(data, dict):
            x_key = None
            y_key = None

            # Try common key names for x-axis
            for key in ["wavenumbers", "wavenumber", "x", "freq", "frequency"]:
                if key in data:
                    x_key = key
                    break

            # Try common key names for y-axis
            for key in ["intensities", "intensity", "y", "counts", "absorbance"]:
                if key in data:
                    y_key = key
                    break

            if x_key and y_key:
                x_vals = np.array(data[x_key], dtype=float)
                y_vals = np.array(data[y_key], dtype=float)
                return x_vals, y_vals

        # Format 2: Array of objects
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            x_vals = []
            y_vals = []

            for item in data:
                # Try to find x and y values
                x_val = None
                y_val = None

                for x_key in ["wavenumber", "wavenumbers", "x", "freq"]:
                    if x_key in item:
                        x_val = float(item[x_key])
                        break

                for y_key in ["intensity", "intensities", "y", "counts"]:
                    if y_key in item:
                        y_val = float(item[y_key])
                        break

                if x_val is not None and y_val is not None:
                    x_vals.append(x_val)
                    y_vals.append(y_val)

            if x_vals and y_vals:
                return np.array(x_vals), np.array(y_vals)

        raise ValueError(
            "JSON format not recognized. Expected wavenumber/intensity pairs."
        )

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Failed to parse JSON spectrum: {str(e)}") from e


def parse_csv_spectrum(
    content: str, filename: str = "unknown"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse spectrum data from CSV format.

    Handles various CSV formats with headers or without.
    """
    try:
        # Use StringIO to treat string as file-like object
        csv_file = io.StringIO(content)

        # Try to detect delimiter
        sample = content[:1024]
        delimiter = ","
        if sample.count(";") > sample.count(","):
            delimiter = ";"
        elif sample.count("\t") > sample.count(","):
            delimiter = "\t"

        # Read CSV
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        rows = list(csv_reader)

        if not rows:
            raise ValueError("Empty CSV file")

        # Check if first row is header
        has_header = False
        try:
            # If first row contains non-numeric data, it's likely a header
            float(rows[0][0])
            float(rows[0][1])
        except (ValueError, IndexError):
            has_header = True

        data_rows = rows[1:] if has_header else rows

        # Extract x and y values
        x_vals = []
        y_vals = []

        for i, row in enumerate(data_rows):
            if len(row) < 2:
                continue

            try:
                x_val = float(row[0])
                y_val = float(row[1])
                x_vals.append(x_val)
                y_vals.append(y_val)
            except ValueError:
                ErrorHandler.log_warning(
                    f"Could not parse CSV row {i+1}: {row}", f"Parsing {filename}"
                )
                continue

        if len(x_vals) < 10:
            raise ValueError(
                f"Insufficient data points ({len(x_vals)}). Need at least 10 points."
            )

        return np.array(x_vals), np.array(y_vals)

    except Exception as e:
        raise ValueError(f"Failed to parse CSV spectrum: {str(e)}") from e


def parse_spectrum_data(
    text_content: str, filename: str = "unknown", file_format: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse spectrum data from text content with automatic format detection.
    Args:
        text_content: Raw text content of the spectrum file
        filename: Name of the file for error reporting
        file_format: Force specific format ('txt', 'csv', 'json') or None for auto-detection
    Returns:
        Tuple of (x_values, y_values) as numpy arrays
    Raises:
        ValueError: If the data cannot be parsed
    """
    try:
        # Detect format if not specified
        if file_format is None:
            file_format = detect_file_format(filename, text_content)

        # Parse based on detected/specified format
        if file_format == "json":
            x, y = parse_json_spectrum(text_content)
        elif file_format == "csv":
            x, y = parse_csv_spectrum(text_content, filename)
        else:  # Default to TXT format
            x, y = parse_txt_spectrum(text_content, filename)

        # Common validation for all formats
        validate_spectrum_data(x, y, filename)

        return x, y

    except Exception as e:
        raise ValueError(f"Failed to parse spectrum data: {str(e)}") from e


def parse_txt_spectrum(
    content: str, filename: str = "unknown"
) -> Tuple[np.ndarray, np.ndarray]:
    """Robustly parse spectrum data from TXT format."""
    lines = content.strip().split("\n")
    x_vals, y_vals = [], []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith(("#", "%")):
            continue

        try:
            # Handle different separators
            parts = line.replace(",", " ").replace(";", " ").replace("\t", " ").split()

            # Find the first two valid numbers in the line
            numbers = []
            for part in parts:
                if part:  # Skip empty strings from multiple spaces
                    try:
                        numbers.append(float(part))
                    except ValueError:
                        continue  # Ignore non-numeric parts

            if len(numbers) >= 2:
                x_vals.append(numbers[0])
                y_vals.append(numbers[1])
            else:
                ErrorHandler.log_warning(
                    f"Could not find two numbers on line {i+1}: '{line}'",
                    f"Parsing {filename}",
                )

        except ValueError as e:
            ErrorHandler.log_warning(
                f"Error parsing line {i+1}: '{line}'. Error: {e}",
                f"Parsing {filename}",
            )
            continue

    if len(x_vals) < 10:
        raise ValueError(
            f"Insufficient data points ({len(x_vals)}). Need at least 10 points."
        )

    return np.array(x_vals), np.array(y_vals)


def validate_spectrum_data(x: np.ndarray, y: np.ndarray, filename: str) -> None:
    """
    Validate parsed spectrum data for common issues.
    """
    # Check for NaNs
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")

    # Check monotonic increasing x (sort if needed)
    if not np.all(np.diff(x) >= 0):
        # Sort by x values if not monotonic
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        ErrorHandler.log_warning(
            "Wavenumbers were not monotonic - data has been sorted",
            f"Parsing {filename}",
        )

    # Check reasonable range for spectroscopy
    if min(x) < 0 or max(x) > 10000 or (max(x) - min(x)) < 100:
        ErrorHandler.log_warning(
            f"Unusual wavenumber range: {min(x):.1f} - {max(x):.1f} cm⁻¹",
            f"Parsing {filename}",
        )


def process_single_file(
    filename: str,
    text_content: str,
    model_choice: str,
    run_inference_func,
    label_file_func,
    modality: str,
    target_len: int,
) -> Optional[Dict[str, Any]]:
    """
    Process a single spectrum file

    Args:
        filename: Name of the file
        text_content: Raw text content
        model_choice: Selected model name
        run_inference_func: Function to run inference
        label_file_func: Function to extract ground truth label

    Returns:
        Dictionary with processing results or None if failed
    """
    start_time = time.time()

    try:
        # 1. Parse spectrum data
        x_raw, y_raw = parse_spectrum_data(text_content, filename)

        # 2. Preprocess spectrum using the full, modality-aware pipeline
        x_resampled, y_resampled = preprocess_spectrum(
            x_raw, y_raw, modality=modality, target_len=target_len
        )

        # 3. Run inference, passing modality
        cache_key = hashlib.md5(
            f"{y_resampled.tobytes()}{model_choice}".encode()
        ).hexdigest()
        prediction, logits_list, probs, logits = run_inference_func(
            y_resampled, model_choice, modality=modality, cache_key=cache_key
        )

        if prediction is None:
            raise ValueError("Inference returned None. Model may have failed to load.")

        # ==Calculate confidence==
        if logits is not None:
            probs_np, max_confidence, confidence_level, confidence_emoji = (
                calculate_softmax_confidence(logits)
            )
        else:
            # Fallback for older models or if logits are not returned
            probs_np = np.array(probs) if probs is not None else np.array([])
            max_confidence = float(np.max(probs_np)) if probs_np.size > 0 else 0.0
            confidence_level = "LOW"
            confidence_emoji = "🔴"

        # ==Get ground truth==
        ground_truth = label_file_func(filename)
        ground_truth = (
            ground_truth if ground_truth is not None and ground_truth >= 0 else None
        )

        # ==Get predicted class==
        label_map = {0: "Stable (Unweathered)", 1: "Weathered (Degraded)"}
        predicted_class = label_map.get(int(prediction), f"Unknown ({prediction})")

        processing_time = time.time() - start_time

        return {
            "filename": filename,
            "success": True,
            "prediction": int(prediction),
            "predicted_class": predicted_class,
            "confidence": max_confidence,
            "confidence_level": confidence_level,
            "confidence_emoji": confidence_emoji,
            "logits": logits_list if logits_list else [],
            "probabilities": probs_np.tolist() if len(probs_np) > 0 else [],
            "ground_truth": ground_truth,
            "processing_time": processing_time,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_resampled": x_resampled,
            "y_resampled": y_resampled,
        }

    except ValueError as e:
        ErrorHandler.log_error(e, f"processing {filename}")
        return {
            "filename": filename,
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time,
        }


def process_multiple_files(
    uploaded_files: List,
    model_choice: str,
    run_inference_func,
    label_file_func,
    modality: str,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """
    Process multiple uploaded files

    Args:
        uploaded_files: List of uploaded file objects
        model_choice: Selected model name
        run_inference_func: Function to run inference
        label_file_func: Function to extract ground truth label
        progress_callback: Optional callback to update progress

    Returns:
        List of processing results
    """
    results = []
    total_files = len(uploaded_files)

    ErrorHandler.log_message(
        f"Starting batch processing of {total_files} files with modality '{modality}'"
    )

    for i, uploaded_file in enumerate(uploaded_files):
        if progress_callback:
            progress_callback(i, total_files, uploaded_file.name)

        try:
            # ==Read file content==
            raw = uploaded_file.read()
            text_content = raw.decode("utf-8") if isinstance(raw, bytes) else raw

            # ==Process the file==
            result = process_single_file(
                filename=uploaded_file.name,
                text_content=text_content,
                model_choice=model_choice,
                run_inference_func=run_inference_func,
                label_file_func=label_file_func,
                modality=modality,
                target_len=TARGET_LEN,
            )

            if result:
                results.append(result)

        except ValueError as e:
            ErrorHandler.log_error(e, f"reading file {uploaded_file.name}")
            results.append(
                {
                    "filename": uploaded_file.name,
                    "success": False,
                    "error": f"Failed to read file: {str(e)}",
                }
            )

    if progress_callback:
        progress_callback(total_files, total_files, "Complete")

    ErrorHandler.log_message(
        f"Completed batch processing: {sum(1 for r in results if r.get('success', False))}/{total_files} successful"
    )

    return results
