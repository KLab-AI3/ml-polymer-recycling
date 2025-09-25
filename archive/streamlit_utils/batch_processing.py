"""This file provides utilities for **batch processing** spectral data files (such as Raman spectra) for polymer classification. Its main goal is to process multiple files efficiently—either synchronously or asynchronously—using one or more machine learning models, and to collect, summarize, and export the results. It is designed for integration with a Streamlit-based UI, supporting file uploads and batch inference."""

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import streamlit as st

from utils.preprocessing import preprocess_spectrum
from utils.multifile import parse_spectrum_data
from utils.async_inference import submit_batch_inference, wait_for_batch_completion
from core_logic import run_inference


@dataclass
class BatchProcessingResult:
    """Result from batch processing operation."""

    filename: str
    model_name: str
    prediction: int
    confidence: float
    logits: List[float]
    inference_time: float
    status: str = "success"
    error: Optional[str] = None
    ground_truth: Optional[int] = None


class BatchProcessor:
    """Handles batch processing of spectral data files."""

    def __init__(self, modality: str = "raman"):
        self.modality = modality
        self.results: List[BatchProcessingResult] = []

    def process_files_sync(
        self,
        file_data: List[Tuple[str, str]],  # (filename, content)
        model_names: List[str],
        target_len: int = 500,
    ) -> List[BatchProcessingResult]:
        """Process files synchronously."""
        results = []

        for filename, content in file_data:
            for model_name in model_names:
                try:
                    # Parse spectrum data
                    x_raw, y_raw = parse_spectrum_data(content)

                    # Preprocess
                    x_proc, y_proc = preprocess_spectrum(
                        x_raw, y_raw, modality=self.modality, target_len=target_len
                    )

                    # Run inference
                    start_time = time.time()
                    prediction, logits_list, probs, inference_time, logits = (
                        run_inference(y_proc, model_name)
                    )

                    if prediction is not None:
                        confidence = max(probs) if probs is not None else 0.0

                        result = BatchProcessingResult(
                            filename=filename,
                            model_name=model_name,
                            prediction=int(prediction),
                            confidence=confidence,
                            logits=logits_list or [],
                            inference_time=inference_time or 0.0,
                            ground_truth=self._extract_ground_truth(filename),
                        )
                    else:
                        result = BatchProcessingResult(
                            filename=filename,
                            model_name=model_name,
                            prediction=-1,
                            confidence=0.0,
                            logits=[],
                            inference_time=0.0,
                            status="failed",
                            error="Inference failed",
                        )

                    results.append(result)

                except Exception as e:
                    result = BatchProcessingResult(
                        filename=filename,
                        model_name=model_name,
                        prediction=-1,
                        confidence=0.0,
                        logits=[],
                        inference_time=0.0,
                        status="failed",
                        error=str(e),
                    )
                    results.append(result)

        self.results.extend(results)
        return results

    def process_files_async(
        self,
        file_data: List[Tuple[str, str]],
        model_names: List[str],
        target_len: int = 500,
        max_concurrent: int = 3,
    ) -> List[BatchProcessingResult]:
        """Process files asynchronously."""
        results = []

        # Process files in chunks to manage concurrency
        chunk_size = max_concurrent
        file_chunks = [
            file_data[i : i + chunk_size] for i in range(0, len(file_data), chunk_size)
        ]

        for chunk in file_chunks:
            chunk_results = self._process_chunk_async(chunk, model_names, target_len)
            results.extend(chunk_results)

        self.results.extend(results)
        return results

    def _process_chunk_async(
        self, file_chunk: List[Tuple[str, str]], model_names: List[str], target_len: int
    ) -> List[BatchProcessingResult]:
        """Process a chunk of files asynchronously."""
        results = []

        for filename, content in file_chunk:
            try:
                # Parse and preprocess
                x_raw, y_raw = parse_spectrum_data(content)
                x_proc, y_proc = preprocess_spectrum(
                    x_raw, y_raw, modality=self.modality, target_len=target_len
                )

                # Submit async inference for all models
                task_ids = submit_batch_inference(
                    model_names=model_names,
                    input_data=y_proc,
                    inference_func=run_inference,
                )

                # Wait for completion
                inference_results = wait_for_batch_completion(task_ids, timeout=60.0)

                # Process results
                for model_name in model_names:
                    if model_name in inference_results:
                        model_result = inference_results[model_name]

                        if "error" not in model_result:
                            prediction, logits_list, probs, inference_time, logits = (
                                model_result
                            )
                            confidence = max(probs) if probs else 0.0

                            result = BatchProcessingResult(
                                filename=filename,
                                model_name=model_name,
                                prediction=prediction or -1,
                                confidence=confidence,
                                logits=logits_list or [],
                                inference_time=inference_time or 0.0,
                                ground_truth=self._extract_ground_truth(filename),
                            )
                        else:
                            result = BatchProcessingResult(
                                filename=filename,
                                model_name=model_name,
                                prediction=-1,
                                confidence=0.0,
                                logits=[],
                                inference_time=0.0,
                                status="failed",
                                error=model_result["error"],
                            )
                    else:
                        result = BatchProcessingResult(
                            filename=filename,
                            model_name=model_name,
                            prediction=-1,
                            confidence=0.0,
                            logits=[],
                            inference_time=0.0,
                            status="failed",
                            error="No result received",
                        )

                    results.append(result)

            except Exception as e:
                # Create error results for all models
                for model_name in model_names:
                    result = BatchProcessingResult(
                        filename=filename,
                        model_name=model_name,
                        prediction=-1,
                        confidence=0.0,
                        logits=[],
                        inference_time=0.0,
                        status="failed",
                        error=str(e),
                    )
                    results.append(result)

        return results

    def _extract_ground_truth(self, filename: str) -> Optional[int]:
        """Extract ground truth label from filename."""
        try:
            from core_logic import label_file

            return label_file(filename)
        except:
            return None

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for batch processing results."""
        if not self.results:
            return {}

        successful_results = [r for r in self.results if r.status == "success"]
        failed_results = [r for r in self.results if r.status == "failed"]

        stats = {
            "total_files": len(set(r.filename for r in self.results)),
            "total_inferences": len(self.results),
            "successful_inferences": len(successful_results),
            "failed_inferences": len(failed_results),
            "success_rate": (
                len(successful_results) / len(self.results) if self.results else 0
            ),
            "models_used": list(set(r.model_name for r in self.results)),
            "average_inference_time": (
                np.mean([r.inference_time for r in successful_results])
                if successful_results
                else 0
            ),
            "total_processing_time": sum(r.inference_time for r in successful_results),
        }

        # Calculate accuracy if ground truth is available
        gt_results = [r for r in successful_results if r.ground_truth is not None]
        if gt_results:
            correct_predictions = sum(
                1 for r in gt_results if r.prediction == r.ground_truth
            )
            stats["accuracy"] = correct_predictions / len(gt_results)
            stats["samples_with_ground_truth"] = len(gt_results)

        return stats

    def export_results(self, format: str = "csv") -> str:
        """Export results to specified format."""
        # Placeholder implementation to ensure a string is always returned
        return "Export functionality not implemented yet."
