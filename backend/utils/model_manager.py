import torch
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from backend.models.registry import build as build_model, get_model_info as get_registry_model_info, choices
from backend.config import TARGET_LEN
from backend.pydantic_models import ModelInfo

class ModelManager:
    """
    Centralized manager for discovering, loading, and caching ML models and their weights.
    Ensures consistent model loading logic across different services.
    """

    def __init__(self):
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self._weights_cache: Dict[str, torch.nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ ModelManager initialized on {self.device}")

    def _load_state_dict(self, model_path: Path) -> Optional[Dict]:
        """Load state dict with caching."""
        try:
            if not model_path.exists():
                return None

            mtime = model_path.stat().st_mtime
            cache_key = f"{model_path}:{mtime}"

            if cache_key not in self._weights_cache:
                # Simple cap for cache size, could be replaced with LRU
                if len(self._weights_cache) > 10:
                    self._weights_cache.clear()
                self._weights_cache[cache_key] = torch.load(
                    model_path, map_location="cpu", weights_only=True
                )
            return self._weights_cache[cache_key]
        except (FileNotFoundError, RuntimeError, OSError) as e:
            print(f"Error loading state dict from {model_path}: {e}")
            return None

    def load_model(self, model_name: str, target_len: int = TARGET_LEN) -> Tuple[torch.nn.Module, bool, Path]:
        """
        Load a trained model for inference, including its weights.
        Caches the loaded model.

        Args:
            model_name (str): Name of the model architecture (from registry).
            target_len (int): Expected input length for the model.

        Returns:
            Tuple[torch.nn.Module, bool, Path]: The loaded model, a boolean indicating
            if weights were successfully loaded, and the path to the loaded weights.
        """
        if model_name in self._model_cache:
            model_entry = self._model_cache[model_name]
            return model_entry['model'], model_entry['weights_loaded'], model_entry['weights_path']

        if model_name not in choices():
            print(f"⚠️ Model '{model_name}' not found in registry.")
            return None, False, Path("")

        model = build_model(model_name, target_len)
        weights_loaded = False
        loaded_path = Path("")

        # Define standard weight file naming convention
        potential_weight_paths = [
            Path("backend/models/weights") / f"{model_name}_model.pth",
            Path("backend/models/weights") / f"{model_name}.pth",
        ]

        for weight_path in potential_weight_paths:
            if weight_path.exists():
                try:
                    state_dict = self._load_state_dict(weight_path)
                    if state_dict:
                        model.load_state_dict(state_dict, strict=True)
                        model.to(self.device)
                        model.eval()
                        weights_loaded = True
                        loaded_path = weight_path
                        print(f"✅ Loaded weights for {model_name} from {loaded_path}")
                        break
                except (OSError, RuntimeError, KeyError) as e:
                    print(f"❌ Error loading weights for {model_name} from {weight_path}: {e}")
                    continue
            else:
                print(f"🔍 Weights not found for {model_name} at {weight_path}")

        if not weights_loaded:
            print(f"⚠️ No weights loaded for model '{model_name}'. Model will use random initialization.")
            model.to(self.device)
            model.eval() # Ensure model is in eval mode even if no weights loaded

        self._model_cache[model_name] = {
            'model': model,
            'weights_loaded': weights_loaded,
            'weights_path': loaded_path,
            'target_len': target_len,
            'device': self.device
        }
        return model, weights_loaded, loaded_path

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific model."""
        if model_name not in choices():
            return None
        info = get_registry_model_info(model_name)
        # Add runtime info if model is loaded
        if model_name in self._model_cache:
            cached_info = self._model_cache[model_name]
            info['weights_loaded'] = cached_info['weights_loaded']
            info['weights_path'] = str(cached_info['weights_path'])
            info['device'] = str(cached_info['device'])
            info['available'] = True
        else:
            # Check if weights exist even if not loaded yet
            weights_exist = any((Path("backend/models/weights") / f"{model_name}_model.pth").exists() or \
                                (Path("backend/models/weights") / f"{model_name}.pth").exists()
                                for _ in [0]) # Dummy loop to check both paths
            info['weights_loaded'] = False
            info['weights_path'] = None
            info['device'] = str(self.device)
            info['available'] = weights_exist # Mark as available if weights are present

        return info

    def get_available_models(self) -> List[ModelInfo]:
        """Get a list of all models with their availability status."""
        models_list = []
        for model_name in choices():
            info = self.get_model_info(model_name)
            if info:
                models_list.append(ModelInfo(
                    name=model_name,
                    description=info.get("description", ""),
                    input_length=info.get("input_length", TARGET_LEN),
                    num_classes=info.get("num_classes", 2),
                    supported_modalities=info.get("modalities", ["raman", "ftir"]),
                    performance=info.get("performance", {}),
                    parameters=info.get("parameters"),
                    speed=info.get("speed"),
                    citation=info.get("citation"),
                    available=info.get("available", False) # Use the 'available' status from get_model_info
                ))
        return models_list

# Global instance of the ModelManager
model_manager = ModelManager()
