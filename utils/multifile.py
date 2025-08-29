"""Multi-file processing utiltities for batch inference.
Handles multiple file uploads and iterative processing."""

from typing import List, Dict, Any, Tuple, Optional
import time
import streamlit as st 
import numpy as np 

from .preprocessing import resample_spectrum
from .errors import ErrorHandler, safe_execute
from .results_manager import ResultsManager
from .confidence import calculate_softmax_confidence

def parse_spectrum_data(text_content: str, filename: str = "unknown") -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse spectrum data from text content
    
    Args:
        text_content: Raw text content of the spectrum file
        filename: Name of the file for error reporting
    
    Returns:
        Tuple of (x_values, y_values) as numpy arrays
    
    Raises:
        ValueError: If the data cannot be parsed
    """
    try:
        lines = text_content.strip().split('\n')

        #==Remove empty lines and comments==
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('%'):
                data_lines.append(line)

        if not data_lines:
            raise ValueError("No data lines found in file")

        #==Try to parse==
        x_vals, y_vals = [], []

        for i, line in enumerate(data_lines):
            try:
                #=Try comma separation first, then space=
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split()

                if len(parts) < 2:
                    ErrorHandler.log_warning(f"Line {i+1} has fewer than 2 columns, skipping", f"Parsing {filename}")
                    continue

                x_val = float(parts[0].strip())
                y_val = float(parts[1].split())

                x_vals.append(x_val)
                y_vals.append(y_val)

            except (ValueError, IndexError) as e:
                ErrorHandler.log_warning(f"Could not parse line {i+1}: {line}", f"Parsing {filename}")
                continue

            if len(x_vals) < 10:    #==Need minimum points for interpolation==
                raise ValueError(f"Insufficient data points ({len(x_vals)}). Need at least 10 points.")

        return np.array(x_vals), np.array(y_vals)
                
    except Exception as e:
        raise ValueError(f"Failed to parse spectrum data: {str(e)}")

def process_single_file(
    filename: str,
    text_content: str,
    model_choice: str,
    load_model_func,
    run_inference_func,
    label_file_func
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
        #==Parse spectrum data==
        x_raw, y_raw, success = safe_execute(
            parse_spectrum_data,
            text_content,
            filename,
            error_context=f"parsing {filename}",
            show_error=False
        )

        if not success:
            return None

        #==Resample spectrum==
        x_resampled, y_resampled, success = safe_execute(
            resample_spectrum,
            x_raw,
            y_raw,
            500,    # TARGET_LEN
            error_context=f"resampling {filename}",
            show_error=False
        )

        if not success:
            return None

        #==Run inference==
        prediction, logits_list, probs, inference_time, logits, success = safe_execute(
            run_inference_func,
            y_resampled,
            model_choice,
            error_context=f"inference on {filename}",
            show_error=False
        )

        if not success or prediction is None:
            ErrorHandler.log_error(Exception("Inference failed"), f"processing {filename}")
            return None

        #==Calculate confidence==
        if logits is not None:
            probs_np, max_confidence, confidence_level, confidence_emoji = calculate_softmax_confidence(logits)
        else:
            probs_np = np.array([])
            max_confidence = 0.0
            confidence_level = "LOW"
            confidence_emoji = "🔴"

        #==Get ground truth==
        try:
            ground_truth = label_file_func(filename)
            ground_truth = ground_truth if ground_truth >= 0 else None
        except Exception:
            ground_truth = None

        #==Get predicted class==
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
            "processing_time": time.time() - start_time
        }

def process_multiple_files(
    uploaded_files: List,
    model_choice: str,
    load_model_func,
    run_inference_func,
    label_file_func,
    progress_callback=None
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
            #==Read file content==
            raw = uploaded_file.read()
            text_content = raw.decode('utf-8') if isinstance(raw, bytes) else raw

            #==Process the file==
            result = process_single_file(
                uploaded_file.name,
                text_content,
                model_choice,
                load_model_func,
                run_inference_func,
                label_file_func
            )

            if result:
                results.append(result)

                #==Add successful results to the results manager==
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
                            "confidence_emoji": result["confidence_emoji"]
                        }
                    )

        except Exception as e:
            ErrorHandler.log_error(e, f"reading file {uploaded_file.name}")
            results.append({
                "filename": uploaded_file.name,
                "success": False,
                "error": f"Failed to read file: {str(e)}"
            })

    if progress_callback:
        progress_callback(total_files, total_files, "Complete")

    ErrorHandler.log_info(f"Completed batch processing: {sum(1 for r in results if r.get('success', False))}/{total_files} successful")

    return results

def display_batch_results(results: List[Dict[str, Any]]) -> None:
    """
    Display batch processing results in the UI
    
    Args:
        results: List of processing results
    """
    if not results:
        st.warning("No results to display")
        return

    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    #==Summary==
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", len(results))
    with col2:
        st.metric("Successful", len(successful), delta=f"{len(successful)/len(results)*100:.1f}%")
    with col3:
        st.metric("Failed", len(failed), delta=f"-{len(failed)/len(results)*100:.1f}%" if failed else "0%")

    #==Results tabs==
    tab1, tab2 = st.tabs(["✅Successful", "❌ Failed"])

    with tab1:
        if successful:
            for result in successful:
                with st.expander(f"{result['filename']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Prediction:** {result['predicted_class']}")
                        st.write(f"**Confidence:** {result['confidence_emoji']} {result['confidence_level']} ({result['confidence']:.3f})")
                    with col2:
                        st.write(f"**Processing Time:** {result['processing_time']:.3f}s")
                        if result['ground_truth'] is not None:
                            gt_label = {0: "Stable", 1: "Weathered"}.get(result['ground_truth'], "Unknown")
                            correct = "✅" if result['prediction'] == result['ground_truth'] else "❌"
                            st.write(f"**Ground Truth:** {gt_label} {correct}")
        else:
            st.info("No successful results")

    with tab2:
        if failed:
            for result in failed:
                with st.expander(f"❌ {result['filename']}", expanded=False):
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            st.success("No failed files!")

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
        key="batch_uploader"
    )

    return uploaded_files if uploaded_files else [] 
