"""Performance benchmarking and monitoring for the backend API."""

import logging
import psutil
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import os
import tempfile
import sys
# Configure performance logger
performance_logger = logging.getLogger('performance')
performance_logger.setLevel(logging.INFO)

# Determine writable log directory (env override), fallback to temp dir
tmp_base = Path(os.getenv("PERF_LOG_DIR", tempfile.gettempdir()))
log_dir = tmp_base / "ml_polymer_logs"

# Try to create and use a file handler; if that fails, fallback to stdout StreamHandler
try:
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "performance.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    performance_logger.addHandler(file_handler)
except Exception as e:
    # Fallback to stdout so HF Spaces / container logs capture the output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    performance_logger.addHandler(stream_handler)
    performance_logger.warning("Could not create file handler for performance logs, using stdout: %s", e)

class PerformanceBenchmark:
    """Context manager for benchmarking operations."""

    def __init__(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = 0
        self.start_memory = 0
        self.duration = 0
        self.memory_delta = 0
        self.performance_data = {}

    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - self.start_memory

        # Log performance data
        perf_data = {
            "operation": self.operation_name,
            "duration_seconds": round(duration, 4),
            "memory_start_mb": round(self.start_memory, 2),
            "memory_end_mb": round(end_memory, 2),
            "memory_delta_mb": round(memory_delta, 2),
            "timestamp": datetime.utcnow().isoformat(),
            **self.metadata
        }

        performance_logger.info(f"BENCHMARK: {perf_data}")

        # Store in class for retrieval
        self.duration = duration
        self.memory_delta = memory_delta
        self.performance_data = perf_data

def log_model_performance(model_name: str, inference_time: float,
                         preprocessing_time: float, total_time: float,
                         memory_usage: float, spectrum_length: int):
    """Log model inference performance metrics."""
    perf_data = {
        "operation": "model_inference",
        "model_name": model_name,
        "inference_time": round(inference_time, 4),
        "preprocessing_time": round(preprocessing_time, 4),
        "total_time": round(total_time, 4),
        "memory_usage_mb": round(memory_usage, 2),
        "spectrum_length": spectrum_length,
        "timestamp": datetime.utcnow().isoformat()
    }

    performance_logger.info(f"MODEL_PERF: {perf_data}")

def get_system_performance():
    """Get current system performance metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "timestamp": datetime.utcnow().isoformat()
    }
