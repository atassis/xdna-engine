#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Per-op device trace harness for the residual_add (resadd) brick.

Same pattern as trace_brick.py (ln_affine_cast), adapted to resadd's ABI: out[t,c] =
a[t,c] + scale*b[t,c], all f32 (scale is BAKED at build time, matching residual_add_iron.py --
the .xclbin already has it compiled in, so the host never passes it).

Usage (from the layernorm example dir, after building a traced xclbin with Makefile.resaddtr):
  python3 trace_brick_resadd.py --xclbin build/final_resaddtr_512x1024_s050.xclbin \
      --instr build/insts_resaddtr_512x1024_s050.txt --kernel MLIR_AIE \
      --trace_size 65536 --trace-file trace_resadd.txt --rows 512 --cols 1024
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

    a_np = rng.standard_normal(rows * cols, dtype=np.float32)
    b_np = rng.standard_normal(rows * cols, dtype=np.float32)

    a = iron.tensor(a_np, dtype=np.float32, device="npu")
    b = iron.tensor(b_np, dtype=np.float32, device="npu")
    out = iron.zeros(rows * cols, dtype=np.float32)

    npu_opts = create_npu_kernel(opts)
    print(f"Running traced brick (resadd): rows={rows} cols={cols}\n")
    # No reference: this run exists to produce a trace, not to verify numerics.
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [a, b, out],
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
