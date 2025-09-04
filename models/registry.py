# models/registry.py
from typing import Callable, Dict, List, Any
from models.figure2_cnn import Figure2CNN
from models.resnet_cnn import ResNet1D
from models.resnet18_vision import ResNet18Vision

# Internal registry of model builders keyed by short name.
_REGISTRY: Dict[str, Callable[[int], object]] = {
    "figure2": lambda L: Figure2CNN(input_length=L),
    "resnet": lambda L: ResNet1D(input_length=L),
    "resnet18vision": lambda L: ResNet18Vision(input_length=L),
}

# Model specifications with metadata for enhanced features
_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "figure2": {
        "input_length": 500,
        "num_classes": 2,
        "description": "Figure 2 baseline custom implemetation",
        "modalities": ["raman", "ftir"],
        "citation": "Neo et al., 2023, Resour. Conserv. Recycl., 188, 106718",
    },
    "resnet": {
        "input_length": 500,
        "num_classes": 2,
        "description": "(Residual Network) uses skip connections to train much deeper networks",
        "modalities": ["raman", "ftir"],
        "citation": "Custom ResNet implementation",
    },
    "resnet18vision": {
        "input_length": 500,
        "num_classes": 2,
        "description": "excels at image recognition tasks by using 'residual blocks' to train more efficiently",
        "modalities": ["raman", "ftir"],
        "citation": "ResNet18 Vision adaptation",
    },
}

# Placeholder for future model expansions
_FUTURE_MODELS = {
    "densenet1d": {
        "description": "DenseNet1D for spectroscopy (placeholder)",
        "status": "planned",
    },
    "ensemble_cnn": {
        "description": "Ensemble of CNN variants (placeholder)",
        "status": "planned",
    },
}


def choices():
    """Return the list of available model keys."""
    return list(_REGISTRY.keys())


def planned_models():
    """Return the list of planned future model keys."""
    return list(_FUTURE_MODELS.keys())


def build(name: str, input_length: int):
    """Instantiate a model by short name with the given input length."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choices: {choices()}")
    return _REGISTRY[name](input_length)


def build_multiple(names: List[str], input_length: int) -> Dict[str, Any]:
    """Nuild multiple models for comparison."""
    models = {}
    for name in names:
        if name in _REGISTRY:
            models[name] = build(name, input_length)
        else:
            raise ValueError(f"Unknown model '{name}'. Available: {choices()}")
    return models


def register_model(
    name: str, builder: Callable[[int], object], spec: Dict[str, Any]
) -> None:
    """Dynamically register a new model."""
    if name in _REGISTRY:
        raise ValueError(f"Model '{name}' already registered.")
    if not callable(builder):
        raise TypeError("Builder must be a callable that accepts an integer argument.")
    _REGISTRY[name] = builder
    _MODEL_SPECS[name] = spec


def spec(name: str):
    """Return expected input length and number of classes for a model key."""
    if name in _MODEL_SPECS:
        return _MODEL_SPECS[name].copy()
    raise KeyError(f"Unknown model '{name}'. Available: {choices()}")


def get_model_info(name: str) -> Dict[str, Any]:
    """Get comprehensive model information including metadata."""
    if name in _MODEL_SPECS:
        return _MODEL_SPECS[name].copy()
    elif name in _FUTURE_MODELS:
        return _FUTURE_MODELS[name].copy()
    else:
        raise KeyError(f"Unknown model '{name}'")


def models_for_modality(modality: str) -> List[str]:
    """Get list of models that support a specific modality."""
    compatible = []
    for name, spec_info in _MODEL_SPECS.items():
        if modality in spec_info.get("modalities", []):
            compatible.append(name)
    return compatible


def validate_model_list(names: List[str]) -> List[str]:
    """Validate and return list of available models from input list."""
    available = choices()
    valid_models = []
    for name in names:
        if name is available:
            valid_models.append(name)
    return valid_models


__all__ = [
    "choices",
    "build",
    "spec",
    "build_multiple",
    "register_model",
    "get_model_info",
    "models_for_modality",
    "validate_model_list",
    "planned_models",
]
