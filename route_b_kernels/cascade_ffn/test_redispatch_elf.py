#!/usr/bin/env python3
"""ELF re-dispatch + correctness gate for the cascade FFN (M=1 and M-tile).

The re-entrancy fix (mlir-air load_pdi cascade reset) is ELF-PATH ONLY -- the
xclbin path has no reset and aborts on dispatch #2. So the re-entrancy gate runs
the ELF via mlir-air's XRTRunner, whose n_perf_iters loop IS a re-dispatch test:
pre-patch it aborts ('qds_device::wait() unexpected command state'); post-patch
all N+warmup dispatches complete. Correctness is gated by output correlation vs
the numpy/torch golden (the device-faithful bf16 reference, gen_golden.py).

Device discipline: single-tenant NPU -- free it first (stop `npu serve`).
Build the matching ELF first, e.g.:
    M_TILE=4 OUTPUT_FORMAT=elf OUT=artifacts/cascade_ffn_mtile \
        bash route_b_kernels/cascade_ffn/build_cascade_ffn.sh

Run (engine venv on PATH for ml_dtypes; air_env sourced for the air dialect):
    python3 route_b_kernels/cascade_ffn/test_redispatch_elf.py \
        --m-tile 4 --buffers artifacts/cascade_ffn_mtile/iron_baseline/buffers
"""
import argparse
import os
import sys

import numpy as np
import ml_dtypes

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "route_b_kernels", "cascade_ffn"))

from ffn_cascade import build_module  # noqa: E402
from air.backend.xrt_runner import XRTRunner  # noqa: E402

D, FF, NCORES, M_INPUT, EPS = 768, 3072, 8, 8, 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-tile", type=int, default=4, dest="M_TILE")
    ap.add_argument("--buffers", default=None,
                    help="dir with x/Wfc1/bfc1/bfc2/Wfc2/out .bin (defaults by M_TILE)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--min-corr", type=float, default=0.99)
    args = ap.parse_args()
    MT = args.M_TILE
    buf = args.buffers or (
        f"{REPO}/artifacts/cascade_ffn/iron_baseline/buffers" if MT == 1
        else f"{REPO}/artifacts/cascade_ffn_mtile/iron_baseline/buffers")

    def bf16(name, shape):
        raw = np.fromfile(f"{buf}/{name}.bin", dtype=np.uint16)
        return raw.view(ml_dtypes.bfloat16).reshape(shape)

    xshape = (D,) if MT == 1 else (MT, D)
    x = bf16("x", xshape)
    w1 = bf16("Wfc1", (FF, D + 32))   # K-aug: cols 0:D weights, col D = bias_fc1 (gen_golden)
    bfc1 = np.fromfile(f"{buf}/bfc1.bin", dtype=np.uint16)
    bfc2 = np.fromfile(f"{buf}/bfc2.bin", dtype=np.uint16)
    biases = np.concatenate([bfc1, bfc2]).view(ml_dtypes.bfloat16)  # [FF+D]
    w2 = bf16("Wfc2", (D, FF))
    golden = bf16("out", xshape)
    print(f"M_TILE={MT} shapes: x{x.shape} w1{w1.shape} biases{biases.shape} "
          f"w2{w2.shape} out{golden.shape}")

    module = build_module(D, FF, NCORES, M_INPUT, EPS, MT)
    runner = XRTRunner(
        output_format="elf",
        instance_name="ffn_cascade",   # ELF kernel id = "main:ffn_cascade"
        use_lock_race_condition_fix=False,
        stack_size=4096,
        n_perf_iters=args.iters,
        n_warmup_iters=5,
        verbose=False,
    )
    print(f"\n=== {args.iters}+5 ELF dispatches (re-entrancy + correctness) ===")
    runner.run_test(
        module,
        inputs=[x, w1, biases, w2],
        expected_outputs=[golden],
        rtol=0.2, atol=0.05, max_mismatch_percentage=100,  # bf16 noise; gate on corr
        min_correlation=args.min_corr,
    )
    print(f"\n*** RE-ENTRANT + CORRECT: {args.iters}+5 ELF dispatches, NO qds abort, "
          f"corr>={args.min_corr}, M_TILE={MT} ***")


if __name__ == "__main__":
    main()
