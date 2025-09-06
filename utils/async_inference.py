"""
Asynchronous inference utilities for polymer classification.
Supports async processing for improved UI responsiveness.
"""

import concurrent.futures
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import streamlit as st
import numpy as np


class InferenceStatus(Enum):
    """Enumeration of possible statuses for an inference task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InferenceTask:
    """Represents an asynchronous inference task."""

    task_id: str
    model_name: str
    input_data: np.ndarray
    status: InferenceStatus = InferenceStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class AsyncInferenceManager:
    """Manages asynchronous inference tasks for multiple models."""

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, InferenceTask] = {}
        self._task_counter = 0

    def generate_task_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter}_{int(time.time() * 1000)}"

    def submit_inference(
        self,
        model_name: str,
        input_data: np.ndarray,
        inference_func: Callable,
        **kwargs,
    ) -> str:
        """Submit an inference task for asynchronous execution."""
        task_id = self.generate_task_id()
        task = InferenceTask(
            task_id=task_id, model_name=model_name, input_data=input_data
        )

        self.tasks[task_id] = task

        # Submit to thread pool
        self.executor.submit(self._run_inference, task, inference_func, **kwargs)

        return task_id

    def _run_inference(
        self, task: InferenceTask, inference_func: Callable, **kwargs
    ) -> None:
        """Execute inference task."""
        try:
            task.status = InferenceStatus.RUNNING
            task.start_time = time.time()

            # Run inference
            result = inference_func(task.input_data, task.model_name, **kwargs)

            task.result = result
            task.status = InferenceStatus.COMPLETED
            task.end_time = time.time()

        except (
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:  # Replace with specific exceptions
            task.error = str(e)
            task.status = InferenceStatus.FAILED
            task.end_time = time.time()

    def get_task_status(self, task_id: str) -> Optional[InferenceTask]:
        """Get status of a specific task."""
        return self.tasks.get(task_id)

    def get_completed_tasks(self) -> List[InferenceTask]:
        """Get all completed tasks."""
        return [
            task
            for task in self.tasks.values()
            if task.status == InferenceStatus.COMPLETED
        ]

    def get_failed_tasks(self) -> List[InferenceTask]:
        """Get all failed tasks."""
        return [
            task
            for task in self.tasks.values()
            if task.status == InferenceStatus.FAILED
        ]

    def wait_for_completion(self, task_ids: List[str], timeout: float = 30.0) -> bool:
        """Wait for specific tasks to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_done = all(
                self.tasks[tid].status
                in [InferenceStatus.COMPLETED, InferenceStatus.FAILED]
                for tid in task_ids
                if tid in self.tasks
            )
            if all_done:
                return True
            time.sleep(0.1)
        return False

    def cleanup_completed_tasks(self, max_age: float = 300.0) -> None:
        """Clean up old completed tasks."""
        current_time = time.time()
        to_remove = []

        for task_id, task in self.tasks.items():
            if (
                task.end_time
                and current_time - task.end_time > max_age
                and task.status in [InferenceStatus.COMPLETED, InferenceStatus.FAILED]
            ):
                to_remove.append(task_id)

        for task_id in to_remove:
            del self.tasks[task_id]

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class AsyncInferenceManagerSingleton:
    """Singleton wrapper for AsyncInferenceManager."""

    _instance: Optional[AsyncInferenceManager] = None

    @classmethod
    def get_instance(cls) -> AsyncInferenceManager:
        """Get the singleton instance of AsyncInferenceManager."""
        if cls._instance is None:
            cls._instance = AsyncInferenceManager()
        return cls._instance


def get_async_inference_manager() -> AsyncInferenceManager:
    """Get or create the singleton async inference manager."""
    return AsyncInferenceManagerSingleton.get_instance()


@st.cache_resource
def get_cached_async_manager():
    """Get cached async inference manager for Streamlit."""
    return AsyncInferenceManager()


def submit_batch_inference(
    model_names: List[str], input_data: np.ndarray, inference_func: Callable, **kwargs
) -> List[str]:
    """Submit batch inference for multiple models."""
    manager = get_async_inference_manager()
    task_ids = []

    for model_name in model_names:
        task_id = manager.submit_inference(
            model_name=model_name,
            input_data=input_data,
            inference_func=inference_func,
            **kwargs,
        )
        task_ids.append(task_id)

    return task_ids


def check_inference_progress(task_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Check progress of multiple inference tasks."""
    manager = get_async_inference_manager()
    progress = {}

    for task_id in task_ids:
        task = manager.get_task_status(task_id)
        if task:
            progress[task_id] = {
                "model_name": task.model_name,
                "status": task.status.value,
                "duration": task.duration,
                "error": task.error,
            }

    return progress


def wait_for_batch_completion(
    task_ids: List[str],
    timeout: float = 30.0,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Wait for batch inference completion with progress updates."""
    manager = get_async_inference_manager()
    start_time = time.time()

    while time.time() - start_time < timeout:
        progress = check_inference_progress(task_ids)

        if progress_callback:
            progress_callback(progress)

        # Check if all tasks are done
        all_done = all(
            status["status"] in ["completed", "failed"] for status in progress.values()
        )

        if all_done:
            break

        time.sleep(0.2)

    # Collect results
    results = {}
    for task_id in task_ids:
        task = manager.get_task_status(task_id)
        if task:
            if task.status == InferenceStatus.COMPLETED:
                results[task.model_name] = task.result
            elif task.status == InferenceStatus.FAILED:
                results[task.model_name] = {"error": task.error}

    return results
