"""Multi-file processing utilities for batch inference.
Handles multiple file uploads and iterative processing.
Supports TXT, CSV, and JSON file formats with automatic detection."""

from typing import List, Dict, Any, Tuple, Optional, Union
import time
import streamlit as st
import numpy as np
import pandas as pd
import json
import csv
import io
from pathlib import Path

from .preprocessing import resample_spectrum
from .errors import ErrorHandler, safe_execute
from .results_manager import ResultsManager
from .confidence import calculate_softmax_confidence


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
        except:
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
        except:
            pass

    # Try CSV (look for commas in first few lines)
    lines = content_stripped.split("\n")[:5]
    comma_count = sum(line.count(",") for line in lines)
    if comma_count > len(lines):  # More commas than lines suggests CSV
        return "csv"

    # Default to TXT
    return "txt"


# /////////////////////////////////////////////////////


def parse_json_spectrum(
    content: str, filename: str = "unknown"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse spectrum data from JSON format.

    Expected formats:
    - {"wavenumbers": [...], "intensities": [...]}
    - {"x": [...], "y": [...]}
    - [{"wavenumber": val, "intensity": val}, ...]
    """

    try:
        data = json.load(content)

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
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to parse JSON spectrum: {str(e)}")


# /////////////////////////////////////////////////////


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
        raise ValueError(f"Failed to parse CSV spectrum: {str(e)}")


# /////////////////////////////////////////////////////


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
            x, y = parse_json_spectrum(text_content, filename)
        elif file_format == "csv":
            x, y = parse_csv_spectrum(text_content, filename)
        else:  # Default to TXT format
            x, y = parse_txt_spectrum(text_content, filename)

        # Common validation for all formats
        validate_spectrum_data(x, y, filename)

        return x, y

    except Exception as e:
        raise ValueError(f"Failed to parse spectrum data: {str(e)}")


# /////////////////////////////////////////////////////


def parse_txt_spectrum(
    content: str, filename: str = "unknown"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse spectrum data from TXT format (original implementation).
    """
    lines = content.strip().split("\n")

    # ==Remove empty lines and comments==
    data_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("%"):
            data_lines.append(line)

    if not data_lines:
        raise ValueError("No data lines found in file")

    # ==Try to parse==
    x_vals, y_vals = [], []

    for i, line in enumerate(data_lines):
        try:
            # Handle different separators
            parts = line.replace(",", " ").split()
            numbers = [
                p
                for p in parts
                if p.replace(".", "", 1)
                .replace("-", "", 1)
                .replace("+", "", 1)
                .isdigit()
            ]
            if len(numbers) >= 2:
                x_val = float(numbers[0])
                y_val = float(numbers[1])
                x_vals.append(x_val)
                y_vals.append(y_val)

        except ValueError:
            ErrorHandler.log_warning(
                f"Could not parse line {i+1}: {line}", f"Parsing {filename}"
            )
            continue

    if len(x_vals) < 10:  # ==Need minimum points for interpolation==
        raise ValueError(
            f"Insufficient data points ({len(x_vals)}). Need at least 10 points."
        )

    return np.array(x_vals), np.array(y_vals)


# /////////////////////////////////////////////////////


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


# /////////////////////////////////////////////////////


def process_single_file(
    filename: str,
    text_content: str,
    model_choice: str,
    load_model_func,
    run_inference_func,
    label_file_func,
) -> Optional[Dict[str, Any]]:
    """
    Process a single spectrum file

    Args:
        filename: Name of the file
        text_content: Raw text content
        model_choice: Selected model name
        load_model_func: Function to load the model
        run_inference_func: Function to run inference
        label_file_func: Function to extract ground truth label

    Returns:
        Dictionary with processing results or None if failed
    """
    start_time = time.time()

    try:
        # ==Parse spectrum data==
        result, success = safe_execute(
            parse_spectrum_data,
            text_content,
            filename,
            error_context=f"parsing {filename}",
            show_error=False,
        )

        if not success or result is None:
            return None

        x_raw, y_raw = result

        # ==Resample spectrum==
        result, success = safe_execute(
            resample_spectrum,
            x_raw,
            y_raw,
            500,  # TARGET_LEN
            error_context=f"resampling {filename}",
            show_error=False,
        )

        if not success or result is None:
            return None

        x_resampled, y_resampled = result

        # ==Run inference==
        result, success = safe_execute(
            run_inference_func,
            y_resampled,
            model_choice,
            error_context=f"inference on {filename}",
            show_error=False,
        )

        if not success or result is None:
            ErrorHandler.log_error(
                Exception("Inference failed"), f"processing {filename}"
            )
            return None

        prediction, logits_list, probs, inference_time, logits = result

        # ==Calculate confidence==
        if logits is not None:
            probs_np, max_confidence, confidence_level, confidence_emoji = (
                calculate_softmax_confidence(logits)
            )
        else:
            probs_np = np.array([])
            max_confidence = 0.0
            confidence_level = "LOW"
            confidence_emoji = "🔴"

        # ==Get ground truth==
        try:
            ground_truth = label_file_func(filename)
            ground_truth = ground_truth if ground_truth >= 0 else None
        except Exception:
            ground_truth = None

        # ==Get predicted class==
        label_map = {0: "Stable (Unweathered)", 1: "Weathered (Degraded)"}
        predicted_class = label_map.get(prediction, f"Unknown ({prediction})")

        processing_time = time.time() - start_time

        return {
            "filename": filename,
            "success": True,
            "prediction": prediction,
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

    except Exception as e:
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
    load_model_func,
    run_inference_func,
    label_file_func,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """
    Process multiple uploaded files

    Args:
        uploaded_files: List of uploaded file objects
        model_choice: Selected model name
        load_model_func: Function to load the model
        run_inference_func: Function to run inference
        label_file_func: Function to extract ground truth label
        progress_callback: Optional callback to update progress

    Returns:
        List of processing results
    """
    results = []
    total_files = len(uploaded_files)

    ErrorHandler.log_info(f"Starting batch processing of {total_files} files")

    for i, uploaded_file in enumerate(uploaded_files):
        if progress_callback:
            progress_callback(i, total_files, uploaded_file.name)

        try:
            # ==Read file content==
            raw = uploaded_file.read()
            text_content = raw.decode("utf-8") if isinstance(raw, bytes) else raw

            # ==Process the file==
            result = process_single_file(
                uploaded_file.name,
                text_content,
                model_choice,
                load_model_func,
                run_inference_func,
                label_file_func,
            )

            if result:
                results.append(result)

                # ==Add successful results to the results manager==
                if result.get("success", False):
                    ResultsManager.add_results(
                        filename=result["filename"],
                        model_name=model_choice,
                        prediction=result["prediction"],
                        predicted_class=result["predicted_class"],
                        confidence=result["confidence"],
                        logits=result["logits"],
                        ground_truth=result["ground_truth"],
                        processing_time=result["processing_time"],
                        metadata={
                            "confidence_level": result["confidence_level"],
                            "confidence_emoji": result["confidence_emoji"],
                        },
                    )

        except Exception as e:
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

    ErrorHandler.log_info(
        f"Completed batch processing: {sum(1 for r in results if r.get('success', False))}/{total_files} successful"
    )

    return results


def display_batch_results(batch_results: list):
    """Renders a clean, consolidated summary of batch processing results using metrics and a pandas DataFrame replacing the old expander list"""
    if not batch_results:
        st.info("No batch results to display.")
        return

    successful_runs = [r for r in batch_results if r.get("success", False)]
    failed_runs = [r for r in batch_results if not r.get("success", False)]

    # 1. High Level Metrics
    st.markdown("###### Batch Summary")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Files Processed", f"{len(batch_results)}")
    metric_cols[1].metric("✔️ Successful", f"{len(successful_runs)}")
    metric_cols[2].metric("❌ Failed", f"{len(failed_runs)}")

    # 3 Hidden Failure Details
    if failed_runs:
        with st.expander(
            f"View details for {len(failed_runs)} failed file(s)", expanded=False
        ):
            for r in failed_runs:
                st.error(f"**File:** `{r.get('filename', 'unknown')}`")
                st.caption(
                    f"Reason for failure: {r.get('error', 'No details provided')}"
                )


# Legacy display batch results
# def display_batch_results(results: List[Dict[str, Any]]) -> None:
#     """
#     Display batch processing results in the UI

#     Args:
#         results: List of processing results
#     """
#     if not results:
#         st.warning("No results to display")
#         return

#     successful = [r for r in results if r.get("success", False)]
#     failed = [r for r in results if not r.get("success", False)]

#     # ==Summary==
#     col1, col2, col3 = st.columns(3, border=True)
#     with col1:
#         st.metric("Total Files", len(results))
#     with col2:
#         st.metric("Successful", len(successful),
#                   delta=f"{len(successful)/len(results)*100:.1f}%")
#     with col3:
#         st.metric("Failed", len(
#             failed), delta=f"-{len(failed)/len(results)*100:.1f}%" if failed else "0%")

#     # ==Results tabs==
#     tab1, tab2 = st.tabs(["✅Successful", "❌ Failed"], width="stretch")

#     with tab1:
#         with st.expander("Successful"):
#             if successful:
#                 for result in successful:
#                     with st.expander(f"{result['filename']}", expanded=False):
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.write(
#                                 f"**Prediction:** {result['predicted_class']}")
#                             st.write(
#                                 f"**Confidence:** {result['confidence_emoji']} {result['confidence_level']} ({result['confidence']:.3f})")
#                         with col2:
#                             st.write(
#                                 f"**Processing Time:** {result['processing_time']:.3f}s")
#                             if result['ground_truth'] is not None:
#                                 gt_label = {0: "Stable", 1: "Weathered"}.get(
#                                     result['ground_truth'], "Unknown")
#                                 correct = "✅" if result['prediction'] == result['ground_truth'] else "❌"
#                                 st.write(
#                                     f"**Ground Truth:** {gt_label} {correct}")
#             else:
#                 st.info("No successful results")

#     with tab2:
#         if failed:
#             for result in failed:
#                 with st.expander(f"❌ {result['filename']}", expanded=False):
#                     st.error(f"Error: {result.get('error', 'Unknown error')}")
#         else:
#             st.success("No failed files!")


def create_batch_uploader() -> List:
    """
    Create multi-file uploader widget

    Returns:
        List of uploaded files
    """
    uploaded_files = st.file_uploader(
        "Upload multiple Raman spectrum files (.txt)",
        type="txt",
        accept_multiple_files=True,
        help="Select multiple .txt files with wavenumber and intensity columns",
        key="batch_uploader",
    )

    return uploaded_files if uploaded_files else []
