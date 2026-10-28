#!/usr/bin/env python3
"""Dequantize official LiquidAI MLX 4-bit quant to BF16 safetensors."""

import torch
import numpy as np
from safetensors.torch import save_file
from safetensors import safe_open
import os
import sys

def dequantize_layer(qweight, scales, zeros, group_size=64):
    """Dequantize one affine-groupwise 4-bit layer.
    
    Args:
        qweight: uint32 tensor of shape [out_features, in_features // 8]
        scales: float tensor of shape [out_features, in_features // group_size]
        zeros: float tensor of shape [out_features, in_features // group_size]
    """
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * 8
    
    # Unpack uint32 to 4-bit values
    qweight_np = qweight.numpy()
    # View as bytes: [out_features, in_features//8, 4]
    qweight_bytes = qweight_np.view(np.uint8).reshape(out_features, qweight_np.shape[1], 4)
    # Extract low and high 4 bits from each byte
    low = qweight_bytes & 0x0F
    high = (qweight_bytes >> 4) & 0x0F
    # Interleave: [out_features, in_features//8, 2] -> [out_features, in_features//4]
    # qweight_4bit = np.stack([low, high], axis=2).reshape(out_features, in_features // 4)
    # Wait, that gives in_features//4 not in_features
    # Actually: each uint32 has 4 bytes = 8 4-bit values
    # So shape [out_features, in_features//8, 4] -> stack -> [out_features, in_features//8, 2, 4] -> reshape to [out_features, in_features]
    
    # Let me redo this more carefully
    qweight_bytes = qweight_np.view(np.uint8).reshape(out_features, -1)
    # qweight_bytes shape: [out_features, in_features//8 * 4]
    # Each byte has 2 4-bit values
    low = qweight_bytes & 0x0F
    high = (qweight_bytes >> 4) & 0x0F
    qweight_4bit = np.empty((out_features, in_features), dtype=np.uint8)
    qweight_4bit[:, 0::2] = low
    qweight_4bit[:, 1::2] = high
    
    # Reshape scales/zeros for broadcasting
    scales_np = scales.float().numpy()
    zeros_np = zeros.float().numpy()
    num_groups = in_features // group_size
    scales_reshaped = scales_np.reshape(out_features, num_groups, 1).repeat(group_size, axis=2).reshape(out_features, in_features)
    zeros_reshaped = zeros_np.reshape(out_features, num_groups, 1).repeat(group_size, axis=2).reshape(out_features, in_features)
    
    # Dequantize: (q - z) * s
    dequant = (qweight_4bit.astype(np.float32) - zeros_reshaped) * scales_reshaped
    return torch.from_numpy(dequant).to(torch.float16)


def main():
    quant_dir = 'quant/LFM3-1.7B-MLX-4bit'
    src_path = os.path.join(quant_dir, 'model.safetensors')
    dst_path = os.path.join(quant_dir, 'model_dequant.safetensors')
    
    # Load all tensors
    tensors = {}
    with safe_open(src_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    # Find all quantized layers
    quantized_layers = set()
    for key, t in tensors.items():
        if t.dtype == torch.uint32 and key.endswith('.weight'):
            layer_path = key[:-7]
            quantized_layers.add(layer_path)
    
    print(f'Quantized layers: {len(quantized_layers)}')
    
    # Dequantize all quantized layers
    dequantized = {}
    for layer_path in quantized_layers:
        qweight = tensors[layer_path + '.weight']
        scales = tensors[layer_path + '.scales']
        zeros = tensors[layer_path + '.biases']
        
        dequant = dequantize_layer(qweight, scales, zeros)
        dequantized[layer_path + '.weight'] = dequant
        print(f'Dequantized {layer_path}')
    
    # Copy non-quantized tensors (norms, etc.)
    # Exclude scales and biases from quantized layers - they're not model params
    for key, t in tensors.items():
        if key.endswith('.scales') or key.endswith('.biases'):
            continue
        if not (key.endswith('.weight') and key[:-7] in quantized_layers):
            if t.dtype == torch.bfloat16:
                dequantized[key] = t.to(torch.float16)
            else:
                dequantized[key] = t
    
    # Fix key names: strip 'language_model.model.' prefix then add 'model.' to match MLX-LM format
    fixed = {}
    old_prefix = 'language_model.model.'
    new_prefix = 'model.'
    for key, t in dequantized.items():
        if key.startswith(old_prefix):
            fixed[new_prefix + key[len(old_prefix):]] = t
        elif not key.startswith(new_prefix):
            fixed[new_prefix + key] = t
        else:
            fixed[key] = t
    dequantized = fixed
    
    print(f'Total tensors: {len(dequantized)}')
    
    # Save
    save_file(dequantized, dst_path)
    print(f'Saved to: {dst_path}')
    print(f'Size: {os.path.getsize(dst_path)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
