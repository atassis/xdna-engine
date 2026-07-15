#!/usr/bin/env python3
"""Minimal repro: an n-D (multi-dim TensorAccessPattern) OUTPUT DMA hangs when its kernel runs as
a CO-RESIDENT hw-context alongside a second xclbin -- but works as the sole context.

Isolates the resident-rails Variant B blocker for an upstream bug report. Each --test runs ONE
dispatch; wrap in `timeout` so a hang shows as a timeout (exit 124) vs PASS (exit 0).

Artifacts (built by build_parakeet_modal_kernels.sh, staged in artifacts/parakeet/ln):
  final_deint_512x4096.xclbin  -- cast with a 3D chunk-major output TAP (the n-D DMA)
  final_cast_512x4096.xclbin   -- the SAME cast kernel, PLAIN contiguous output (no n-D TAP)
  final_cast_512x1024.xclbin   -- a third plain kernel (2nd control context)

  test 1 (control):  deint ALONE (1 ctx)                 -> run deint  -> expect PASS
  test 2 (the bug):  deint + cast@4096 co-resident (2)   -> run deint  -> expect HANG
  test 3 (control):  cast@4096 + cast@1024 (2, no n-D)   -> run cast@4096 -> expect PASS

Run:  for t in 1 2 3; do timeout 30 .venv-iron/bin/python scripts/repro_ndma_multicontext.py --test $t; \
        echo "test $t exit=$?"; done
"""
import argparse, sys
import numpy as np
import pyxrt

A = "artifacts/parakeet/ln"
T, DFF, KRES = 512, 4096, 1024


def load(dev, name):
    xb = pyxrt.xclbin(f"{A}/final_{name}.xclbin")
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    k = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(f"{A}/insts_{name}.txt", dtype=np.uint32)
    return xb, ctx, k, instr  # keep ctx alive (context stays resident)


def run_cast_like(dev, k, instr, in_elems, out_bytes):
    """Dispatch a cast-family kernel (in f32 -> out bf16) with the 8-arg matmul ABI."""
    HO, CA = pyxrt.bo.host_only, pyxrt.bo.cacheable
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    x = np.zeros(in_elems, np.float32)
    bi = pyxrt.bo(dev, instr.nbytes, CA, k.group_id(1))
    bx = pyxrt.bo(dev, x.nbytes, HO, k.group_id(3))
    bo = pyxrt.bo(dev, out_bytes, HO, k.group_id(4))
    dt = pyxrt.bo(dev, 8, HO, k.group_id(5))
    dc = pyxrt.bo(dev, 8, HO, k.group_id(6))
    dtr = pyxrt.bo(dev, 1, HO, k.group_id(7))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    bx.write(x.tobytes(), 0); bx.sync(TO)
    k(3, bi, instr.size, bx, bo, dt, dc, dtr).wait()  # blocks; hang -> outer timeout fires


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, required=True)
    a = ap.parse_args()
    dev = pyxrt.device(0)

    if a.test == 1:
        print("[t1] deint ALONE (1 ctx) -> run deint")
        _keep = load(dev, "deint_512x4096")
        run_cast_like(dev, _keep[2], _keep[3], T * DFF, T * DFF * 2)
        print("[t1] deint completed -> PASS (sole context)")

    elif a.test == 2:
        print("[t2] deint + cast@4096 CO-RESIDENT (2 ctx) -> run deint")
        _cast = load(dev, "cast_512x4096")          # 2nd context, kept alive
        _deint = load(dev, "deint_512x4096")
        run_cast_like(dev, _deint[2], _deint[3], T * DFF, T * DFF * 2)
        print("[t2] deint completed -> (unexpected) PASS")

    elif a.test == 3:
        print("[t3] cast@4096 + cast@1024 CO-RESIDENT (2 ctx, NO n-D) -> run cast@4096")
        _c1024 = load(dev, "cast_512x1024")         # 2nd context, kept alive
        _c4096 = load(dev, "cast_512x4096")
        run_cast_like(dev, _c4096[2], _c4096[3], T * DFF, T * DFF * 2)
        print("[t3] cast@4096 completed -> PASS (2 ctx OK without n-D DMA)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
