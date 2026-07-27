#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Per-op device trace harness for OUR resident bricks.

Runs a traced brick xclbin and dumps the raw trace buffer, which the in-tree
`python/utils/trace/{parse,get_trace_summary}.py` then decode to per-op compute / DMA / stall.

Why this exists: every fusion number so far is a wall-clock DELTA between whole-encoder
configurations, and two separate cost models have already been falsified that way (dispatch count,
then hw-context switch count). This measures the device directly instead.

NOTE ON CORE COUNT: trace egress needs a free shim channel. Our production bricks run 8 cores and
saturate the shim, which makes the traced build fail placement with
`'aie.masterset' op targets same destination South: 3`. Build the traced variant with FEWER cores
(cores=1 routes). Every core runs the identical row loop, so one core is representative of the math;
it is NOT representative of whole-brick wall time, which is the 8-way parallel form.

Usage (from the layernorm example dir, after building a traced xclbin):
  python3 trace_brick.py --xclbin build/final_lnaffcasttr_512x1024.xclbin \
      --instr build/insts_lnaffcasttr_512x1024.txt --kernel MLIR_AIE \
      --trace_size 65536 --trace-file trace.txt --rows 512 --cols 1024
"""
import argparse
import sys

import numpy as np
import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.test import create_npu_kernel

try:
    from ml_dtypes import bfloat16
except ImportError:  # pragma: no cover
    bfloat16 = None


def main(opts):
    rows, cols = int(opts.rows), int(opts.cols)
    rng = np.random.default_rng(seed=42)

    # ln_affine_cast ABI: x f32 [rows*cols], gb f32 [2*cols] = [gamma | beta], out bf16 [rows*cols].
    x_np = rng.standard_normal(rows * cols, dtype=np.float32)
    gb_np = np.concatenate([
        np.ones(cols, dtype=np.float32),      # gamma
        np.zeros(cols, dtype=np.float32),     # beta
    ])

    x = iron.tensor(x_np, dtype=np.float32, device="npu")
    gb = iron.tensor(gb_np, dtype=np.float32, device="npu")
    out = iron.zeros(rows * cols, dtype=bfloat16 if bfloat16 else np.float32)

    npu_opts = create_npu_kernel(opts)
    print(f"Running traced brick: rows={rows} cols={cols}\n")
    # No reference: this run exists to produce a trace, not to verify numerics (the brick's own
    # correctness is gated elsewhere by fused_seam_parity / encoder_parity).
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [x, gb, out],
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
