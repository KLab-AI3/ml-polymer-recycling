# models/registry.py
from typing import Callable, Dict, List, Any
from .figure2_cnn import Figure2CNN
from .resnet_cnn import ResNet1D
from .resnet18_vision import ResNet18Vision
from .enhanced_cnn import EnhancedCNN, EfficientSpectralCNN, HybridSpectralNet

def _resolve_key(name: str, registry: Dict[str, Any]) -> str:
    """Resolve a case-insensitive model name to its canonical key."""
    for key in registry:
        if key.lower() == name.lower():
            return key
    raise KeyError(f"Unknown model '{name}'. Available: {list(registry.keys())}")

# Internal registry of model builders keyed by short name.
_REGISTRY: Dict[str, Callable[[int], object]] = {
    "Figure2": lambda L: Figure2CNN(input_length=L),
    "ResNet": lambda L: ResNet1D(input_length=L),
    "ResNet18Vision": lambda L: ResNet18Vision(input_length=L),
    "Enhanced_cnn": lambda L: EnhancedCNN(input_length=L),
    "Efficient_cnn": lambda L: EfficientSpectralCNN(input_length=L),
    "Hybrid_Net": lambda L: HybridSpectralNet(input_length=L),
}

# Model specifications with metadata for enhanced features
_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "Figure2": {
        "input_length": 500,
        "num_classes": 2,
        "description": "Figure 2 baseline custom implementation",
        "modalities": ["raman", "ftir"],
        "citation": "Neo et al., 2023, Resour. Conserv. Recycl., 188, 106718",
        "performance": {"accuracy": 0.948, "f1_score": 0.943},
        "parameters": "~500K",
        "speed": "fast",
    },
    "ResNet": {
        "input_length": 500,
        "num_classes": 2,
        "description": "(Residual Network) uses skip connections to train much deeper networks",
        "modalities": ["raman", "ftir"],
        "citation": "Custom ResNet implementation",
        "performance": {"accuracy": 0.962, "f1_score": 0.959},
        "parameters": "~100K",
        "speed": "very_fast",
    },
    "ResNet18Vision": {
        "input_length": 500,
        "num_classes": 2,
        "description": "excels at image recognition tasks by using 'residual blocks' to train more efficiently",
        "modalities": ["raman", "ftir"],
        "citation": "ResNet18 Vision adaptation",
        "performance": {"accuracy": 0.945, "f1_score": 0.940},
        "parameters": "~11M",
        "speed": "medium",
    },
    "Enhanced_cnn": {
        "input_length": 500,
        "num_classes": 2,
        "description": "Enhanced CNN with attention mechanisms and multi-scale feature extraction",
        "modalities": ["raman", "ftir"],
        "citation": "Custom enhanced architecture with attention",
        "performance": {"accuracy": 0.975, "f1_score": 0.973},
        "parameters": "~800K",
        "speed": "medium",
        "features": ["attention", "multi_scale", "batch_norm", "dropout"],
    },
    "Efficient_cnn": {
        "input_length": 500,
        "num_classes": 2,
        "description": "Efficient CNN optimized for real-time inference with depthwise separable convolutions",
        "modalities": ["raman", "ftir"],
        "citation": "Custom efficient architecture",
        "performance": {"accuracy": 0.955, "f1_score": 0.952},
        "parameters": "~200K",
        "speed": "very_fast",
        "features": ["depthwise_separable", "lightweight", "real_time"],
    },
    "Hybrid_Net": {
        "input_length": 500,
        "num_classes": 2,
        "description": "Hybrid network combining CNN backbone with self-attention mechanisms",
        "modalities": ["raman", "ftir"],
        "citation": "Custom hybrid CNN-Transformer architecture",
        "performance": {"accuracy": 0.968, "f1_score": 0.965},
        "parameters": "~1.2M",
        "speed": "medium",
        "features": ["self_attention", "cnn_backbone", "transformer_head"],
    },
}

# Placeholder for future model expansions
_FUTURE_MODELS = {
    "densenet1d": {
        "description": "DenseNet1D for spectroscopy with dense connections",
        "status": "planned",
        "modalities": ["raman", "ftir"],
        "features": ["dense_connections", "parameter_efficient"],
    },
    "ensemble_cnn": {
        "description": "Ensemble of multiple CNN variants for robust predictions",
        "status": "planned",
        "modalities": ["raman", "ftir"],
        "features": ["ensemble", "robust", "high_accuracy"],
    },
    "vision_transformer": {
        "description": "Vision Transformer adapted for 1D spectral data",
        "status": "planned",
        "modalities": ["raman", "ftir"],
        "features": ["transformer", "attention", "state_of_art"],
    },
    "autoencoder_cnn": {
        "description": "CNN with autoencoder for unsupervised feature learning",
        "status": "planned",
        "modalities": ["raman", "ftir"],
        "features": ["autoencoder", "unsupervised", "feature_learning"],
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
    key = _resolve_key(name, _REGISTRY)
    return _REGISTRY[key](input_length)


def build_multiple(names: List[str], input_length: int) -> Dict[str, Any]:
    """Nuild multiple models for comparison."""
    models = {}
    for name in names:
        key = _resolve_key(name, _REGISTRY)
        models[key] = _REGISTRY[key](input_length)
    return models


def register_model(
    name: str, builder: Callable[[int], object], spec: Dict[str, Any]
) -> None:
    """Dynamically register a new model."""
    if not callable(builder):
        raise TypeError("Builder must be a callable that accepts an integer argument.")
    try:
        existing_key = _resolve_key(name, _REGISTRY)
        raise ValueError(f"Model '{name}' already registered as '{existing_key}'.")
    except KeyError:
        _REGISTRY[name] = builder
        _MODEL_SPECS[name] = spec


def registry_spec(name: str):
    """Return expected input length and number of classes for a model key."""
    key = _resolve_key(name, _MODEL_SPECS)
    return _MODEL_SPECS[key].copy()


def get_model_info(name: str) -> Dict[str, Any]:
    """Get comprehensive model information including metadata."""
    try:
        key = _resolve_key(name, _MODEL_SPECS)
        return _MODEL_SPECS[key].copy()
    except KeyError:
        try:
            key = _resolve_key(name, _FUTURE_MODELS)
            return _FUTURE_MODELS[key].copy()
        except KeyError:
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
    valid_models = []
    for name in names:
        try:
            key = _resolve_key(name, _REGISTRY)
            valid_models.append(key)
        except KeyError:
            pass


def get_models_metadata() -> Dict[str, Dict[str, Any]]:
    """Get metadata for all registered models."""
    return {name: _MODEL_SPECS[name].copy() for name in _MODEL_SPECS}


def is_model_compatible(name: str, modality: str) -> bool:
    """Check if a model is compatible with a specific modality."""
    try:
        key = _resolve_key(name, _MODEL_SPECS)
        return modality in _MODEL_SPECS[key].get("modalities", [])
    except KeyError:
        return False


def get_model_capabilities(name: str) -> Dict[str, Any]:
    """Get detailed capabilities of a model."""
    key = _resolve_key(name, _MODEL_SPECS)
    spec = _MODEL_SPECS[key].copy()
    spec.update(
        {
            "available": True,
            "status": "active",
            "supported_tasks": ["binary_classification"],
            "performance_metrics": {
                "supports_confidence": True,
                "supports_batch": True,
                "memory_efficient": spec.get("description", "").lower().find("resnet")
                != -1,
            },
        }
    )
    return spec


__all__ = [
    "choices",
    "build",
    "registry_spec",
    "build_multiple",
    "register_model",
    "get_model_info",
    "models_for_modality",
    "validate_model_list",
    "planned_models",
    "get_models_metadata",
    "is_model_compatible",
    "get_model_capabilities",
]
