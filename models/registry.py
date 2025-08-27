# models/registry.py
from typing import Callable, Dict
from models.figure2_cnn import Figure2CNN
from models.resnet_cnn import ResNet1D
from models.resnet18_vision import ResNet18Vision 

# Internal registry of model builders keyed by short name.
_REGISTRY: Dict[str, Callable[[int], object]] = {
    "figure2": lambda L: Figure2CNN(input_length=L),
    "resnet": lambda L: ResNet1D(input_length=L),
    "resnet18vision": lambda L: ResNet18Vision(input_length=L)
}

def choices():
    """Return the list of available model keys."""
    return list(_REGISTRY.keys())

def build(name: str, input_length: int):
    """Instantiate a model by short name with the given input length."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choices: {choices()}")
    return _REGISTRY[name](input_length)

def spec(name: str):
    """Return expected input length and number of classes for a model key."""
    if name == "figure2":
        return {"input_length": 500, "num_classes": 2}
    if name == "resnet":
        return {"input_length": 500, "num_classes": 2}
    if name == "resnet18vision":
        return {"input_length": 500, "num_classes": 2}
    raise KeyError(f"Unknown model '{name}'")


__all__ = ["choices", "build"]
