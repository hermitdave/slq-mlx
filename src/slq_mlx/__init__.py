"""SLQ-MLX: Statistically-Lossless Quantization for MLX."""

from .mlx_fast_obq import MLXFastOBQ
from .mlx_quantizer import MLXQuantizer
from .mlx_sensitivity import run_build_database, LinearSensitivityData, ShapleySensitivityData
from .mlx_search import solve_allocation, compute_avg_bitwidth, write_config
from .mlx_pack import pack_checkpoint, pack_row, pack_weight, effective_bits
from .mlx_checkpoint import main as build_checkpoint_dequantized
from .mlx_weight_store import WeightStore, ErrorDatabase
from .mlx_metrics import compute_perplexity, compute_baseline_topk, compute_kl_and_ear
from .mlx_data import prepare_calibration_data, get_tokenizer
from .mlx_model_utils import select_layers, to_device, get_submodule, set_submodule

__version__ = "0.2.0"
__all__ = [
    "MLXFastOBQ",
    "MLXQuantizer",
    "run_build_database",
    "LinearSensitivityData",
    "ShapleySensitivityData",
    "solve_allocation",
    "compute_avg_bitwidth",
    "write_config",
    "pack_checkpoint",
    "pack_row",
    "pack_weight",
    "effective_bits",
    "build_checkpoint_dequantized",
    "WeightStore",
    "ErrorDatabase",
    "compute_perplexity",
    "compute_baseline_topk",
    "compute_kl_and_ear",
    "prepare_calibration_data",
    "get_tokenizer",
    "select_layers",
    "to_device",
    "get_submodule",
    "set_submodule",
]