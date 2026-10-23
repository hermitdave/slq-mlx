"""MLX model utilities for SLQ quantization.

Adapted for MLX-LM model structure.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
import re


class ForwardInterrupt(Exception):
    """Exception to interrupt forward pass after input collection."""
    pass


class InputCollector(nn.Module):
    """Wrapper to collect inputs to a module during forward pass."""

    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        object.__setattr__(self, 'module', module)
        object.__setattr__(self, 'cpu_offload', cpu_offload)
        object.__setattr__(self, 'input_args', [])
        object.__setattr__(self, 'input_kwargs', [])

    def __call__(self, *args, **kwargs):
        # Store inputs
        if self.cpu_offload:
            args = tuple(a.to_device(mx.cpu) if isinstance(a, mx.array) else a for a in args)
            kwargs = {k: v.to_device(mx.cpu) if isinstance(v, mx.array) else v for k, v in kwargs.items()}
        self.input_args.append(args)
        self.input_kwargs.append(kwargs)
        raise ForwardInterrupt()

    def __getattr__(self, name):
        # Delegate to wrapped module
        module = object.__getattribute__(self, 'module')
        return getattr(module, name)


def detect_quantizable_modules(model: nn.Module, block_modules: str = "layers") -> str:
    """Auto-detect quantizable layer regex from model architecture.

    Inspects the first transformer block and builds a regex matching all
    Linear projections under `self_attn` and `mlp`.

    Args:
        model: Loaded MLX-LM model
        block_modules: Dotted path to the layers list, e.g. "layers"

    Returns:
        Regex string for --quantizable-modules
    """
    blocks = get_model_layers(model, block_modules)
    if not blocks:
        raise ValueError(f"No blocks found at '{block_modules}'")

    first_block = blocks[0]
    proj_attn = set()
    proj_mlp = set()
    prefix_attn = None
    prefix_mlp = None

    for name, submodule in first_block.named_modules():
        if isinstance(submodule, nn.Linear):
            parts = name.split(".")
            if len(parts) >= 2:
                parent = parts[-2]
                proj = parts[-1]
                if parent == "self_attn":
                    prefix_attn = "self_attn"
                    proj_attn.add(proj)
                elif parent == "mlp":
                    prefix_mlp = "mlp"
                    proj_mlp.add(proj)

    if not proj_attn and not proj_mlp:
        # Fallback: match any Linear under the block that ends with _proj
        return ".*(_proj)$"

    # Exclude known non-standard projections that should not be quantized
    # by default (e.g. Spark's attention output gate).
    excluded = {"g_proj"}

    patterns = []
    if prefix_attn and proj_attn:
        projs = "|".join(sorted(p for p in proj_attn if p not in excluded))
        if projs:
            patterns.append(f"{prefix_attn}\\.({projs})")
    if prefix_mlp and proj_mlp:
        projs = "|".join(sorted(proj_mlp))
        patterns.append(f"{prefix_mlp}\\.({projs})")

    return "|".join(patterns)


def select_layers_mlx(module: nn.Module, pattern: str) -> Dict[str, nn.Linear]:
    """Select quantizable Linear layers matching pattern under module.

    Args:
        module: Module to search (e.g., a TransformerBlock)
        pattern: Regex pattern for layer names (e.g., "self_attn.(q|k|v|o)_proj|mlp.(gate|up|down)_proj")

    Returns:
        Dict mapping layer name -> Linear module
    """
    regex = re.compile(pattern)
    layers = {}

    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Linear):
            if regex.search(name):
                layers[name] = submodule

    return layers


def select_layers(model: nn.Module, prefix: str, pattern: str) -> Dict[str, nn.Linear]:
    """Select quantizable Linear layers matching pattern under prefix (legacy API)."""
    return select_layers_mlx(model, pattern)


def to_device(data: Any, device: mx.Device) -> Any:
    """Recursively move tensors to device. In MLX, device is global, so this is a no-op."""
    # MLX handles device placement globally via mx.set_default_device()
    # This function is kept for API compatibility
    return data


def get_model_layers(model: nn.Module, block_modules: str) -> List[nn.Module]:
    """Get list of transformer blocks from model."""
    parts = block_modules.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def get_pre_blocks(model: nn.Module, pre_block_modules: List[str]) -> List[nn.Module]:
    """Get pre-block modules (embed_tokens, norm, etc.)."""
    pre_blocks = []
    for name in pre_block_modules:
        parts = name.split(".")
        current = model
        try:
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            pre_blocks.append(current)
        except AttributeError:
            continue
    return pre_blocks


def detect_pre_block_modules(model: nn.Module) -> List[str]:
    """Auto-detect pre-block module names from model architecture."""
    inner_model = model['model'] if 'model' in model else model
    candidates_embed = ["embed_tokens", "embedding", "tok_embeddings", "wte"]
    candidates_norm = ["norm", "final_norm", "ln_f"]

    found = []
    for name in candidates_embed:
        if hasattr(inner_model, name):
            found.append(name)
            break

    for name in candidates_norm:
        if hasattr(inner_model, name):
            found.append(name)
            break

    return found


def get_submodule(model: nn.Module, path: str) -> nn.Module:
    """Get submodule by dotted path."""
    parts = path.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_submodule(model: nn.Module, path: str, module: nn.Module):
    """Set submodule by dotted path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def set_submodule_weight(model: nn.Module, path: str, weight: mx.array):
    """Set weight parameter of a Linear submodule by dotted path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    target = getattr(parent, parts[-1])
    if hasattr(target, 'weight'):
        target.weight = weight
    else:
        raise ValueError(f"Module at {path} has no 'weight' attribute")


def get_layer_weights(model: nn.Module, name: str) -> Optional[mx.array]:
    """Get weight tensor for a layer by name."""
    try:
        module = get_submodule(model, name)
        return module.weight
    except AttributeError:
        return None


def set_layer_weights(model: nn.Module, name: str, weight: mx.array):
    """Set weight tensor for a layer by name."""
    try:
        module = get_submodule(model, name)
        module.weight = weight
    except AttributeError:
        pass