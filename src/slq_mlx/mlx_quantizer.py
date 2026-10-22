"""MLX port of block-wise GPTQ quantization orchestrator.

Ported from SLQ src/quant/quantizer.py, adapted for MLX-LM model structure.
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

from .mlx_fast_obq import MLXFastOBQ
from .mlx_model_utils import (
    InputCollector,
    ForwardInterrupt,
    select_layers_mlx,
    get_model_layers,
    get_pre_blocks,
)
from .mlx_weight_store import WeightStore, save_qparams


class HookedLinear(nn.Module):
    """Wrapper to capture inputs to a Linear layer during forward pass."""

    def __init__(self, layer: nn.Linear, hook_fn):
        super().__init__()
        object.__setattr__(self, 'layer', layer)
        object.__setattr__(self, 'hook_fn', hook_fn)
        # Copy weight and bias (bias may not exist)
        self.weight = layer.weight
        self.bias = getattr(layer, 'bias', None)

    def __call__(self, x: mx.array) -> mx.array:
        self.hook_fn(x)
        return self.layer(x)


def quantize_block(
    block: nn.Module,
    layers: Dict[str, nn.Linear],
    handles: Dict[str, MLXFastOBQ],
    input_args: List[Tuple],
    input_kwargs: List[Dict],
):
    """Quantize a block by running forward pass with hooked layers."""
    # Wrap layers with hooks
    original_layers = {}
    for name, layer in layers.items():
        handle = handles[name]
        original_layers[name] = layer
        hooked = HookedLinear(layer, handle.update)
        # Replace layer in block
        parts = name.split(".")
        parent = block
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], hooked)

    # Forward pass to accumulate Hessians
    for inp_args, inp_kwargs in zip(input_args, input_kwargs):
        block(*inp_args, **inp_kwargs)

    # Restore original layers
    for name, layer in original_layers.items():
        parts = name.split(".")
        parent = block
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], layer)

    mx.eval([h.H for h in handles.values()])


class MLXQuantizer:
    """Block-wise GPTQ quantization of a model at several bitwidths."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        quantizable_modules: str,
        pre_block_modules: List[str],
        block_modules: str,
        save_dir: str,
        obq_kwargs: Dict[str, Any],
        device: Optional[mx.Device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
    ):
        self.model = model
        self.data_loader = data_loader
        self.quantizable_modules = quantizable_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.save_dir = save_dir
        self.obq_kwargs = obq_kwargs
        self.device = device or mx.default_device()
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations

    def quantize(self, bitwidth_options: List[int], calibration_bitwidth: int):
        """Run GPTQ quantization for all blocks."""
        # MLX-LM model structure: model['model'] contains the actual model
        inner_model = self.model['model'] if 'model' in self.model else self.model

        # Get layers (blocks)
        blocks = get_model_layers(inner_model, self.block_modules)
        pre_blocks = get_pre_blocks(inner_model, self.pre_block_modules)

        # Disable cache during calibration
        use_cache = getattr(inner_model, "use_cache", None)
        if use_cache is not None:
            inner_model.use_cache = False

        # Stage 1: Collect inputs to first block
        # Wrap first block with InputCollector
        first_block = blocks[0]
        blocks[0] = InputCollector(first_block, cpu_offload=self.cpu_offload_activations)

        for inp_args, inp_kwargs in self.data_loader:
            try:
                inner_model(*inp_args, **inp_kwargs)
            except ForwardInterrupt:
                pass

        input_args, input_kwargs = blocks[0].input_args, blocks[0].input_kwargs
        blocks[0] = first_block  # Restore original block

        mx.eval(input_args, input_kwargs)

        # Stage 2: Process each block sequentially
        for block_id, block in enumerate(blocks):
            print(f"Processing {self.block_modules}[{block_id}]")

            # Select quantizable layers in this block
            layers = select_layers_mlx(block, self.quantizable_modules)

            # Create FastOBQ handles for each layer
            handles = {
                name: MLXFastOBQ(layer, bitwidth_options, **self.obq_kwargs)
                for name, layer in layers.items()
            }

            # Quantize this block
            quantize_block(block, layers, handles, input_args, input_kwargs)

            # Quantize each layer and save
            for name, handle in handles.items():
                full_name = f"{self.block_modules}.{block_id}.{name}"
                self._quantize_layer(full_name, handle, calibration_bitwidth)

            # Recompute inputs for next block using quantized weights
            new_input_args, new_input_kwargs = [], []
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*inp_args, **inp_kwargs)
                out = out[0] if isinstance(out, (tuple, list)) else out

                if len(inp_args) > 0:
                    new_inp_args = list(inp_args)
                    new_inp_args[0] = out
                    new_input_args.append(tuple(new_inp_args))
                elif "hidden_states" in inp_kwargs:
                    new_inp_kwargs = dict(inp_kwargs)
                    new_inp_kwargs["hidden_states"] = out
                    new_input_kwargs.append(new_inp_kwargs)
                else:
                    raise ValueError("Unsupported block input format")

            input_args, input_kwargs = new_input_args, new_input_kwargs

            del handles
            mx.eval(mx.zeros(1))  # Clear cache

        # Restore cache setting
        if use_cache is not None:
            inner_model.use_cache = use_cache

    def _quantize_layer(self, name: str, handle: MLXFastOBQ, calibration_bitwidth: int):
        """Quantize a single layer and save weights + qparams."""
        layer_save_dir = os.path.join(self.save_dir, name)
        os.makedirs(layer_save_dir, exist_ok=True)

        quantized = handle.quantize()

        for bits, (qweight, scales, zeros) in quantized.items():
            # Save qweight as numpy
            mx.save(os.path.join(layer_save_dir, f"{bits}.npy"), qweight)
            save_qparams(layer_save_dir, bits, qweight, scales, zeros, handle)

            # If this is the calibration bitwidth, replace layer in model for next block calibration
            if bits == calibration_bitwidth:
                self._replace_layer_in_model(name, qweight, scales, zeros, getattr(handle.layer, 'bias', None), handle.group_size)

    def _replace_layer_in_model(self, name: str, qweight: mx.array, scales: mx.array, zeros: mx.array, bias: Optional[mx.array], group_size: int):
        """Replace a layer in the model with its quantized version for calibration."""
        # Parse layer path: e.g., "model.layers.0.self_attn.q_proj"
        parts = name.split(".")
        inner_model = self.model['model'] if 'model' in self.model else self.model

        # Navigate to parent module
        parent = inner_model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        child_name = parts[-1]
        original_layer = getattr(parent, child_name)

        # Create quantized linear module
        qlayer = QuantizedLinear.from_qparams(qweight, scales, zeros, getattr(original_layer, 'bias', None), group_size)
        setattr(parent, child_name, qlayer)


class QuantizedLinear(nn.Module):
    """Linear layer with pre-quantized weights (fake-quantized for inference)."""

    def __init__(self, weight: mx.array, bias: Optional[mx.array] = None):
        super().__init__()
        self.weight = weight
        self.bias = bias

    @classmethod
    def from_qparams(cls, qweight: mx.array, scales: mx.array, zeros: mx.array, bias: Optional[mx.array], group_size: int = 128):
        """Reconstruct fake-quantized weight from qparams."""
        d_row, d_col = qweight.shape
        # Expand scales/zeros to full weight shape
        scale_expanded = mx.repeat(scales, group_size, axis=1)[:, :d_col]
        zero_expanded = mx.repeat(zeros, group_size, axis=1)[:, :d_col]
        fake_weight = qweight.astype(mx.float32) * scale_expanded + zero_expanded
        return cls(fake_weight, bias)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T + (self.bias if self.bias is not None else 0)