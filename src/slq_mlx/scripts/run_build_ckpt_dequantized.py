#!/usr/bin/env python3
"""Write a fake-quantized checkpoint (legacy / debug export).

Every weight is stored dequantized to float16, so the artifact is *always*
BF16-sized regardless of the SLQ bitwidth. That makes it useless as a size
reduction and it is kept only for:

* debugging -- the weights can be inspected with any safetensors reader;
* runtimes without a quantized GEMM, where the packed format cannot execute.

For a real size reduction use ``run_pack_mlx`` (the default for
``run_build_ckpt``), which stores the integer grid packed and runs it through
MLX's fused ``mx.quantized_matmul``.
"""

import argparse
import logging
import sys

from slq_mlx.mlx_checkpoint import main as _main

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    sys.exit(_main())
