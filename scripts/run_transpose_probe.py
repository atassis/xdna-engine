#!/usr/bin/env python3
"""Validate the on-chip COMPUTE-tile transpose xclbin (conv-module Task 0.1) on device.

The element transpose runs on the compute core (transpose_tile.cc); the shim DMA does
only contiguous reads + a unit-inner-stride block scatter -- NO transposing n-D DMA (the
path that hangs when co-resident, blocker npu.rs:740). Transpose is pure data movement,
so the result must be BIT-EXACT vs host x.T.

Gate: BIT-EXACT (0 mismatched elements) and NO HANG. We also split mismatched OUTPUT rows
EVEN vs ODD (ping-pong-buffer parity corruption would land on one parity) and report rel-L2.

ABI (mirrors run_dwconv_silu_fused_probe.py / dwconv1d.py): opcode 3, instr on group_id(1),
sequence buffers next. Design sequence = (X, Y) -> X on group_id(3), Y on group_id(4):
  k(3, bo_instr, n_instr, X[gid3], Y[gid4]).

Usage (NPU must be FREE -- stop npu-serve/npu-vox first):
  .venv-iron/bin/python scripts/run_transpose_probe.py \
      -M 1024 -N 400 -mb 8 --cols 8 --dt bf16
Reads build/final_transpose_<M>x<N>_<dt>.xclbin from the mlir-aie dwconv1d sandbox.
"""
import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

EX = "mlir-aie/programming_examples/ml/dwconv1d/build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, required=True, help="input rows (transpose [M,N]->[N,M])")
    ap.add_argument("-N", type=int, required=True, help="input cols")
    ap.add_argument("-mb", type=int, default=8, help="row-block per tile (informational)")
    ap.add_argument("--cols", type=int, default=8, help="cores (informational)")
    ap.add_argument("--dt", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    M, N, dt = args.M, args.N, args.dt
    tag = f"{M}x{N}_{dt}"
    XCLBIN = f"{EX}/final_transpose_{tag}.xclbin"
    INSTS = f"{EX}/insts_transpose_{tag}.txt"
    for p in (XCLBIN, INSTS):
        if not os.path.exists(p):
            sys.exit(
                f"missing {p} -- build: make -f Makefile.transpose NPU2=1 "
                f"M={M} N={N} mb={args.mb} cols={args.cols} dt={dt} "
                f"build/final_transpose_{tag}.xclbin"
            )

    elem = bfloat16 if dt == "bf16" else np.float32
    uview = np.uint16 if dt == "bf16" else np.uint32
    ebytes = 2 if dt == "bf16" else 4

    rng = np.random.RandomState(0)
    # distinct values everywhere so any mis-placement shows up as a mismatch.
    x = rng.standard_normal(size=(M, N)).astype(elem)
    ref = np.ascontiguousarray(x.T)  # [N, M] golden = exact transpose

    X = np.ascontiguousarray(x).reshape(-1).view(uview)
    instr = np.fromfile(INSTS, dtype=np.uint32)

    import pyxrt

    xclbin = pyxrt.xclbin(XCLBIN)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] {tag}  kernel='{kname}'  instr_words={instr.size}")
    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    xbytes = M * N * ebytes
    ybytes = N * M * ebytes
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, xbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, ybytes, pyxrt.bo.host_only, k.group_id(4))
    bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
    bo_x.write(X.tobytes(), 0); bo_x.sync(TO)

    def once():
        r = k(3, bo_instr, instr.size, bo_x, bo_y)
        r.wait()

    once()  # warm + hang-check (r.wait() would block forever on a hang)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        once()
    dt_ms = (time.perf_counter() - t0) / args.iters * 1e3

    bo_y.sync(FROM)
    Y_u = np.frombuffer(bo_y.read(ybytes, 0), dtype=uview).reshape(N, M)
    Y = Y_u.view(elem)

    ref_u = ref.view(uview)
    mismatch = Y_u != ref_u
    n_bad = int(mismatch.sum())
    bad_rows = np.where(mismatch.any(axis=1))[0]
    even_bad = int((bad_rows % 2 == 0).sum())
    odd_bad = int((bad_rows % 2 == 1).sum())
    n_even = (N + 1) // 2
    n_odd = N // 2

    reff = ref.astype(np.float32)
    yf = Y.astype(np.float32)
    denom = np.linalg.norm(reff)
    relL2 = float(np.linalg.norm(yf - reff) / denom) if denom > 0 else 0.0

    print(f"[run] device time/iter: {dt_ms:.3f} ms   ([{M},{N}] -> [{N},{M}], {dt}, in-core transpose)")
    print(f"[run] BIT-EXACT: {n_bad} mismatched elements out of {N*M}")
    print(f"[run] mismatched OUTPUT rows: EVEN {even_bad}/{n_even}  ODD {odd_bad}/{n_odd}  (0/.. => no ping-pong parity corruption)")
    print(f"[run] rel-L2 vs host x.T: {relL2:.4e}")
    print(f"[run] Y[0,:4]={yf[0,:4]}  ref={reff[0,:4]}")
    ok = (n_bad == 0)
    print(f"[run] COMPUTE-tile transpose on NPU: {'PASS (bit-exact, no hang)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
