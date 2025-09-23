# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring, redefined-outer-name, unused-argument, unused-import, singleton-comparison, broad-except
"""
seeds.py

Universal reproducibility controls for the polymer aging ML pipeline.
Provides centralized seed management to ensure consistent results across
all random operations in training, validation, and inference.

* NOTE: This module should be imported and used at the start of any script
*       involving randomness to guarantee reproducible results.
"""

import os
import random
import numpy as np
import torch


def set_global_seeds(seed: int = 42):
    """
    Set random seeds for all major libraries to ensure reproducibility.

    Args:
        seed (int): Random seed value to use across all libraries

    Note:
        This function should be called at the beginning of any script
        that involves random operations (training, data splitting, etc.)
    """
    # Python built-in random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)

    # PyTorch CUDA random (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Additional CUDA reproducibility settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✅ Global seeds set to {seed} for reproducibility")


def get_default_seed():
    """
    Get the default seed value used across the project.

    Returns:
        int: Default seed value (42)
    """
    return 42


def create_fold_seeds(base_seed: int = 42, num_folds: int = 10):
    """
    Create deterministic seeds for cross-validation folds.

    Args:
        base_seed (int): Base seed for generating fold seeds
        num_folds (int): Number of CV folds

    Returns:
        list: List of unique seeds for each fold
    """
    # Use base seed to create deterministic but unique seeds for each fold
    np.random.seed(base_seed)
    fold_seeds = np.random.randint(0, 2**31-1, size=num_folds)
    return fold_seeds.tolist()


def create_augmentation_seed(base_seed: int = 42, fold: int = 0):
    """
    Create a deterministic seed for data augmentation within a specific fold.

    Args:
        base_seed (int): Base seed
        fold (int): Current fold number

    Returns:
        int: Deterministic seed for augmentation in this fold
    """
    return base_seed + 1000 + fold


def verify_reproducibility():
    """
    Verify that random operations are reproducible after setting seeds.

    Returns:
        bool: True if reproducibility check passes
    """
    # Test Python random
    set_global_seeds(42)
    python_rand_1 = random.random()

    set_global_seeds(42)
    python_rand_2 = random.random()

    # Test NumPy random
    set_global_seeds(42)
    numpy_rand_1 = np.random.random()

    set_global_seeds(42)
    numpy_rand_2 = np.random.random()

    # Test PyTorch random
    set_global_seeds(42)
    torch_rand_1 = torch.rand(1).item()

    set_global_seeds(42)
    torch_rand_2 = torch.rand(1).item()

    # Check if all are reproducible
    python_reproducible = python_rand_1 == python_rand_2
    numpy_reproducible = numpy_rand_1 == numpy_rand_2
    torch_reproducible = torch_rand_1 == torch_rand_2

    all_reproducible = python_reproducible and numpy_reproducible and torch_reproducible

    if all_reproducible:
        print("✅ Reproducibility verification passed")
    else:
        print("❌ Reproducibility verification failed")
        print(f"   Python: {python_reproducible}")
        print(f"   NumPy: {numpy_reproducible}")
        print(f"   PyTorch: {torch_reproducible}")

    return all_reproducible


if __name__ == "__main__":
    print("🧪 Testing reproducibility controls...")
    # Test seed setting
    set_global_seeds(42)

    # Test fold seed generation
    fold_seeds = create_fold_seeds(42, 10)
    print(f"📊 Generated fold seeds: {fold_seeds}")

    # Test augmentation seed generation
    aug_seeds = [create_augmentation_seed(42, i) for i in range(5)]
    print(f"📊 Generated augmentation seeds: {aug_seeds}")

    # Verify reproducibility
    verify_reproducibility()

    print("✅ Reproducibility controls test completed!")
