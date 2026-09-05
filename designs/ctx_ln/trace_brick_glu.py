#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Per-op device trace harness for the GLU (Conformer conv-module gate) brick.

Same pattern as trace_brick.py (ln_affine_cast), adapted to glu's ABI: out[t,D] =
a[t,D] * sigmoid(g[t,D]), where the input row is [a(0..D) | g(D..2D)] (pw1's packed output).

Usage (from the layernorm example dir, after building a traced xclbin with Makefile.glutr):
  python3 trace_brick_glu.py --xclbin build/final_glutr_512x1024.xclbin \
      --instr build/insts_glutr_512x1024.txt --kernel MLIR_AIE \
      --trace_size 65536 --trace-file trace_glu.txt --rows 512 --cols 1024
"""
import argparse
import sys

import numpy as np
import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.test import create_npu_kernel


def main(opts):
    rows, cols = int(opts.rows), int(opts.cols)  # cols = D, the OUTPUT width
    rng = np.random.default_rng(seed=42)

    in_np = rng.standard_normal(rows * 2 * cols, dtype=np.float32)  # [a | g] per row

    a_in = iron.tensor(in_np, dtype=np.float32, device="npu")
    c_out = iron.zeros(rows * cols, dtype=np.float32)

    npu_opts = create_npu_kernel(opts)
    print(f"Running traced brick (glu): rows={rows} cols(D)={cols}\n")
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
    p.add_argument("--rows", default=512)
    p.add_argument("--cols", default=1024)
    main(p.parse_args(sys.argv[1:]))
