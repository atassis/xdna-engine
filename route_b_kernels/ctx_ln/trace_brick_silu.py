#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Per-op device trace harness for the SiLU (post-dwconv activation) brick.

Same pattern as trace_brick.py (ln_affine_cast) / trace_brick_glu.py, adapted to silu's ABI:
out[r,cols] = silu(in[r,cols]) = in * sigmoid(in), a SYMMETRIC elementwise op (in width == out
width == cols, unlike glu's asymmetric 2D-in/D-out).

Usage (from the layernorm example dir, after building a traced xclbin with Makefile.silutr):
  python3 trace_brick_silu.py --xclbin build/final_silutr_1024x400.xclbin \
      --instr build/insts_silutr_1024x400.txt --kernel MLIR_AIE \
      --trace_size 65536 --trace-file trace_silu.txt --rows 1024 --cols 400
"""
import argparse
import sys

import numpy as np
import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.test import create_npu_kernel


def main(opts):
    rows, cols = int(opts.rows), int(opts.cols)
    rng = np.random.default_rng(seed=42)

    in_np = rng.standard_normal(rows * cols, dtype=np.float32)

    a_in = iron.tensor(in_np, dtype=np.float32, device="npu")
    c_out = iron.zeros(rows * cols, dtype=np.float32)

    npu_opts = create_npu_kernel(opts)
    print(f"Running traced brick (silu): rows={rows} cols={cols}\n")
    # No reference: this run exists to produce a trace, not to verify numerics.
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [a_in, c_out],
        {},
        verify=False,
        verbosity=npu_opts.verbosity,
    )
    print("\nTRACE RUN OK\n" if res == 0 else f"\nTRACE RUN rc={res}\n")
    sys.exit(res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_runtime_args(p)
    p.add_argument("--rows", default=1024)
    p.add_argument("--cols", default=400)
    main(p.parse_args(sys.argv[1:]))
