"""MLX port of FastOBQ: GPTQ quantization of a single linear layer at several bitwidths from one shared Hessian.

Ported from SLQ (IST-DASLab, COLM 2026) src/quant/fast_obq.py
"""

from typing import Dict, List, Tuple
import numpy as np
import mlx.core as mx
import mlx.nn as nn


def find_params(w: mx.array, bits: int, sym: bool, perchannel: bool) -> Tuple[mx.array, mx.array]:
    """Find quantization params (scale, zero) for weight tensor w.

    Args:
        w: Weight tensor [d_row, d_col] or [d_row, group_size] for per-group
        bits: Target bitwidth
        sym: Symmetric quantization (no zero point)
        perchannel: Per-output-channel scales (True) or per-tensor (False)

    Returns:
        scale: [d_row, 1] if perchannel else scalar
        zero: [d_row, 1] if perchannel else scalar (zeros if sym)
    """
    maxq = 2**bits - 1
    w_f32 = w.astype(mx.float32)

    if perchannel:
        # Per output channel: reduce over input dimension (axis=1)
        w_min = w_f32.min(axis=1, keepdims=True)
        w_max = w_f32.max(axis=1, keepdims=True)
    else:
        # Per tensor
        w_min = w_f32.min()
        w_max = w_f32.max()

    if sym:
        w_max = mx.maximum(mx.abs(w_min), mx.abs(w_max))
        w_min = -w_max
        zero = mx.zeros_like(w_max)
    else:
        zero = w_min

    scale = (w_max - w_min) / maxq
    scale = mx.where(scale == 0, mx.ones_like(scale), scale)  # avoid div by zero

    if sym:
        zero = mx.zeros_like(scale)
    else:
        zero = (w_min / scale).round() * scale  # zero point on quantized grid

    return scale, zero


def quantize(w: mx.array, scale: mx.array, zero: mx.array, maxq: int) -> mx.array:
    """Quantize weight tensor to INT grid."""
    q = mx.clip(((w - zero) / scale).round(), 0, maxq)
    return q.astype(mx.uint8)


def dequantize(q: mx.array, scale: mx.array, zero: mx.array) -> mx.array:
    """Dequantize INT tensor back to float."""
    return q.astype(mx.float32) * scale + zero


class MLXFastOBQ:
    """GPTQ (OBQ-style) quantization of a single linear layer at several bitwidths.

    Accumulates Hessian H = 2 * X^T X / n from calibration inputs,
    then quantizes at multiple bitwidths sharing the same Cholesky factor.
    """

    def __init__(
        self,
        layer: nn.Linear,
        bitwidth_options: List[int],
        group_size: int = 128,
        sym: bool = False,
        perchannel: bool = True,
        rel_damp: float = 1e-2,
    ):
        assert isinstance(layer, nn.Linear), "MLXFastOBQ supports only nn.Linear layers"
        self.layer = layer
        self.bitwidth_options = bitwidth_options
        self.d_row, self.d_col = layer.weight.shape  # [out_features, in_features]
        assert self.d_col % group_size == 0, f"in_features ({self.d_col}) must be divisible by group_size ({group_size})"
        self.group_size = group_size
        self.sym = sym
        self.perchannel = perchannel
        self.rel_damp = rel_damp
        self.H = None  # [d_col, d_col] float32 Hessian
        self.num_samples = 0

    def update(self, x: mx.array) -> None:
        """Update running Hessian estimate with batch of layer inputs.

        Args:
            x: Input tensor [..., d_col] - will be reshaped to [-1, d_col]
        """
        x = x.reshape(-1, x.shape[-1]).astype(mx.float32)  # [batch*seq, d_col]
        batch_size = x.shape[0]

        if self.H is None:
            self.H = mx.zeros((self.d_col, self.d_col), dtype=mx.float32)

        # Running average: H_new = (n_old * H_old + 2 * X^T X) / (n_old + n_new)
        n_old = self.num_samples
        n_new = n_old + batch_size
        if n_old > 0:
            self.H = self.H * (n_old / n_new)
        self.H += 2.0 * (x.T @ x) / n_new
        self.num_samples = n_new

    def _prepare(self) -> Tuple[mx.array, mx.array]:
        """Compute dampened Hessian inverse Cholesky factor and float32 weight clone.

        Returns:
            W: float32 weight [d_row, d_col]
            H_inv_cho: upper Cholesky factor of H^-1 [d_col, d_col]
        """
        H = self.H
        W = self.layer.weight.astype(mx.float32)

        # Handle dead input channels (never activated)
        diag_H = mx.diagonal(H)
        dead = diag_H == 0
        if mx.any(dead):
            # Set unit curvature for dead channels
            H = mx.where(
                mx.expand_dims(dead, 1) | mx.expand_dims(dead, 0),
                mx.eye(self.d_col, dtype=mx.float32),
                H
            )
            W = mx.where(mx.expand_dims(dead, 0), mx.zeros_like(W), W)

        # Dampening
        damp = self.rel_damp * mx.mean(diag_H)
        H = H + mx.eye(self.d_col, dtype=mx.float32) * damp

        # All-zero weight columns are decoupled
        zero_cols = (W == 0).all(axis=0)
        if mx.any(zero_cols):
            H = mx.where(
                mx.expand_dims(zero_cols, 1) | mx.expand_dims(zero_cols, 0),
                mx.eye(self.d_col, dtype=mx.float32),
                H
            )

        # Cholesky inverse (run on CPU as GPU Cholesky not yet supported in MLX)
        original_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            H_cpu = H
            L = mx.linalg.cholesky(H_cpu)  # lower triangular
            H_inv_cho = mx.linalg.cholesky_inv(L).T  # upper triangular (L^-T)
        finally:
            mx.set_default_device(original_device)

        return W, H_inv_cho

    def quantize(self) -> Dict[int, Tuple[mx.array, mx.array, mx.array]]:
        """Quantize layer at all bitwidths using shared Hessian.

        Returns:
            Dict mapping bitwidth -> (qweight [d_row, d_col] uint8, scales [d_row, n_groups], zeros [d_row, n_groups])
        """
        if self.H is None:
            # No calibration samples - use simple RTN quantization as fallback
            print(f"Warning: No calibration samples for layer, using RTN fallback")
            return self._quantize_rtn()

        W, H_inv_cho = self._prepare()
        d_row, d_col, gs = self.d_row, self.d_col, self.group_size
        n_groups = d_col // gs

        out = {}
        for bits in self.bitwidth_options:
            maxq = 2**bits - 1

            qweight = mx.zeros((d_row, d_col), dtype=mx.uint8)
            scales = mx.zeros((d_row, n_groups), dtype=mx.float32)
            zeros = mx.zeros((d_row, n_groups), dtype=mx.float32)

            w = mx.array(W)  # Working copy for error feedback

            # One GPTQ block per group
            for c1 in range(0, d_col, gs):
                c2 = c1 + gs
                w_blk = w[:, c1:c2]  # [d_row, gs]

                # Quantization params for this group
                scale, zero = find_params(w_blk, bits, self.sym, self.perchannel)
                scales[:, c1 // gs] = scale.squeeze(-1)
                zeros[:, c1 // gs] = zero.squeeze(-1)

                # Quantize with stored-precision scale
                scale_f32 = scale.astype(mx.float32)
                H_blk = H_inv_cho[c1:c2, c1:c2]  # [gs, gs]

                errs = mx.zeros_like(w_blk)
                for i in range(gs):
                    # Quantize column i
                    q = quantize(w_blk[:, i:i+1], scale_f32, zero.astype(mx.float32), maxq)
                    qweight[:, c1 + i] = q.squeeze(-1)

                    # Error feedback: err = (w - dequant(q)) / H_ii
                    w_q = dequantize(q, scale_f32, zero.astype(mx.float32))
                    err = (w_blk[:, i] - w_q.squeeze(-1)) / H_blk[i, i]
                    errs[:, i] = err

                    # Update remaining columns in group
                    if i + 1 < gs:
                        w_blk[:, i+1:] -= mx.outer(err, H_blk[i, i+1:])

                # Propagate error to subsequent groups
                if c2 < d_col:
                    w[:, c2:] -= errs @ H_inv_cho[c1:c2, c2:]

            out[bits] = (qweight, scales, zeros)

        self.H = None  # Free memory
        return out

    def _quantize_rtn(self) -> Dict[int, Tuple[mx.array, mx.array, mx.array]]:
        """Simple Round-To-Nearest quantization fallback when no calibration data."""
        out = {}
        d_row, d_col, gs = self.d_row, self.d_col, self.group_size
        n_groups = d_col // gs

        for bits in self.bitwidth_options:
            maxq = 2**bits - 1

            qweight = mx.zeros((d_row, d_col), dtype=mx.uint8)
            scales = mx.zeros((d_row, n_groups), dtype=mx.float32)
            zeros = mx.zeros((d_row, n_groups), dtype=mx.float32)

            w = self.layer.weight.astype(mx.float32)

            for c1 in range(0, d_col, gs):
                c2 = c1 + gs
                w_blk = w[:, c1:c2]

                scale, zero = find_params(w_blk, bits, self.sym, self.perchannel)
                scales[:, c1 // gs] = scale.squeeze(-1)
                zeros[:, c1 // gs] = zero.squeeze(-1)

                scale_f32 = scale.astype(mx.float32)
                for i in range(gs):
                    q = quantize(w_blk[:, i:i+1], scale_f32, zero.astype(mx.float32), maxq)
                    qweight[:, c1 + i] = q.squeeze(-1)

            out[bits] = (qweight, scales, zeros)

        return out