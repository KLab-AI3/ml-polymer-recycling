"""
Model performance optimization utilities.
Includes model quantization, pruning, and optimization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from pathlib import Path


class ModelOptimizer:
    """Utility class for optimizing trained models."""

    def __init__(self):
        self.optimization_history = []

    def quantize_model(
        self, model: nn.Module, dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """Apply dynamic quantization to reduce model size and inference time."""
        # Prepare for quantization
        model.eval()

        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv1d}, dtype=dtype  # Layers to quantize
        )

        return quantized_model

    def prune_model(
        self, model: nn.Module, pruning_ratio: float = 0.2, structured: bool = False
    ) -> nn.Module:
        """Apply magnitude-based pruning to reduce model parameters."""
        model_copy = type(model)(
            model.input_length if hasattr(model, "input_length") else 500
        )
        model_copy.load_state_dict(model.state_dict())

        # Collect modules to prune
        modules_to_prune = []
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                modules_to_prune.append((module, "weight"))

        if structured:
            # Structured pruning (entire channels/filters)
            for module, param_name in modules_to_prune:
                if isinstance(module, nn.Conv1d):
                    prune.ln_structured(
                        module, name=param_name, amount=pruning_ratio, n=2, dim=0
                    )
                else:
                    prune.l1_unstructured(module, name=param_name, amount=pruning_ratio)
        else:
            # Unstructured pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )

        # Make pruning permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)

        return model_copy

    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply multiple optimizations for faster inference."""
        model.eval()

        # Fuse operations where possible
        optimized_model = self._fuse_conv_bn(model)

        # Apply quantization
        optimized_model = self.quantize_model(optimized_model)

        return optimized_model

    def _fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """Fuse convolution and batch normalization layers."""
        model_copy = type(model)(
            model.input_length if hasattr(model, "input_length") else 500
        )
        model_copy.load_state_dict(model.state_dict())

        # Simple fusion for sequential Conv1d + BatchNorm1d patterns
        for name, module in model_copy.named_children():
            if isinstance(module, nn.Sequential):
                self._fuse_sequential_conv_bn(module)

        return model_copy

    def _fuse_sequential_conv_bn(self, sequential: nn.Sequential):
        """Fuse Conv1d + BatchNorm1d in sequential modules."""
        layers = list(sequential.children())
        i = 0
        while i < len(layers) - 1:
            if isinstance(layers[i], nn.Conv1d) and isinstance(
                layers[i + 1], nn.BatchNorm1d
            ):
                # Fuse the layers
                if isinstance(layers[i], nn.Conv1d) and isinstance(
                    layers[i + 1], nn.BatchNorm1d
                ):
                    if isinstance(layers[i + 1], nn.BatchNorm1d):
                        if isinstance(layers[i], nn.Conv1d) and isinstance(
                            layers[i + 1], nn.BatchNorm1d
                        ):
                            fused = self._fuse_conv_bn_layer(layers[i], layers[i + 1])
                        else:
                            fused = None
                    else:
                        fused = None
                else:
                    fused = None
                if fused:
                    # Replace in sequential
                    new_layers = layers[:i] + [fused] + layers[i + 2 :]
                    sequential = nn.Sequential(*new_layers)
                    layers = new_layers
            i += 1

    def _fuse_conv_bn_layer(self, conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
        """Fuse a single Conv1d and BatchNorm1d layer."""
        # Create new conv layer
        fused_conv = nn.Conv1d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size[0],
            conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
            conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding,
            conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation,
            conv.groups,
            bias=True,  # Always add bias after fusion
        )

        # Calculate fused parameters
        w_conv = conv.weight.clone()
        w_bn = bn.weight.clone()
        b_bn = bn.bias.clone()
        mean_bn = (
            bn.running_mean.clone()
            if bn.running_mean is not None
            else torch.zeros_like(bn.weight)
        )
        var_bn = (
            bn.running_var.clone()
            if bn.running_var is not None
            else torch.zeros_like(bn.weight)
        )
        eps = bn.eps

        # Fuse weights
        factor = w_bn / torch.sqrt(var_bn + eps)
        fused_conv.weight.data = w_conv * factor.reshape(-1, 1, 1)

        # Fuse bias
        if conv.bias is not None:
            b_conv = conv.bias.clone()
        else:
            b_conv = torch.zeros_like(b_bn)

        fused_conv.bias.data = (b_conv - mean_bn) * factor + b_bn

        return fused_conv

    def benchmark_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 1, 500),
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark model performance."""
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append(end_time - start_time)

        # Calculate statistics
        times = np.array(times)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate model size (approximate)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)

        return {
            "mean_inference_time": float(np.mean(times)),
            "std_inference_time": float(np.std(times)),
            "min_inference_time": float(np.min(times)),
            "max_inference_time": float(np.max(times)),
            "fps": 1.0 / float(np.mean(times)),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
        }

    def compare_optimizations(
        self,
        original_model: nn.Module,
        optimizations: Optional[List[str]] = None,
        input_shape: Tuple[int, ...] = (1, 1, 500),
    ) -> Dict[str, Dict[str, Any]]:
        if optimizations is None:
            optimizations = ["quantize", "prune", "full_optimize"]
        results = {}

        # Benchmark original model
        results["original"] = self.benchmark_model(original_model, input_shape)

        for opt in optimizations:
            try:
                if opt == "quantize":
                    optimized_model = self.quantize_model(original_model)
                elif opt == "prune":
                    optimized_model = self.prune_model(
                        original_model, pruning_ratio=0.3
                    )
                elif opt == "full_optimize":
                    optimized_model = self.optimize_for_inference(original_model)
                else:
                    continue

                # Benchmark optimized model
                benchmark_results = self.benchmark_model(optimized_model, input_shape)

                # Calculate improvements
                speedup = (
                    results["original"]["mean_inference_time"]
                    / benchmark_results["mean_inference_time"]
                )
                size_reduction = (
                    results["original"]["model_size_mb"]
                    - benchmark_results["model_size_mb"]
                ) / results["original"]["model_size_mb"]
                param_reduction = (
                    results["original"]["total_parameters"]
                    - benchmark_results["total_parameters"]
                ) / results["original"]["total_parameters"]

                benchmark_results.update(
                    {
                        "speedup": speedup,
                        "size_reduction_ratio": size_reduction,
                        "parameter_reduction_ratio": param_reduction,
                    }
                )

                results[opt] = benchmark_results

            except (RuntimeError, ValueError, TypeError) as e:
                results[opt] = {"error": str(e)}

        return results

    def suggest_optimizations(
        self,
        model: nn.Module,
        target_speed: Optional[float] = None,
        target_size: Optional[float] = None,
    ) -> List[str]:
        """Suggest optimization strategies based on requirements."""
        suggestions = []

        # Get baseline metrics
        baseline = self.benchmark_model(model)

        if target_speed and baseline["mean_inference_time"] > target_speed:
            suggestions.append("Apply quantization for 2-4x speedup")
            suggestions.append("Use pruning to reduce model size by 20-50%")
            suggestions.append(
                "Consider using EfficientSpectralCNN for real-time inference"
            )

        if target_size and baseline["model_size_mb"] > target_size:
            suggestions.append("Apply magnitude-based pruning")
            suggestions.append("Use quantization to reduce model size")
            suggestions.append("Consider knowledge distillation to a smaller model")

        # Model-specific suggestions
        if baseline["total_parameters"] > 1000000:
            suggestions.append(
                "Model is large - consider using efficient architectures"
            )

        return suggestions
