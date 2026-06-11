#!/usr/bin/env python3
"""Validate a bf16 -> f32 single_core matmul on the XDNA2 NPU via pyxrt.

Build first (no NPU): rm the stale dtype-agnostic kernel object, then
  source scripts/iron_env.sh
  make -C mlir-aie/programming_examples/basic/matrix_multiplication/single_core \
       NPU2=1 M=$M K=$K N=$N dtype_in=bf16 dtype_out=f32
(the kernel object is named mm_${m}x${k}x${n}.o — tile only, NOT dtype — so a
stale i16 build is silently reused unless removed.)

Host ABI (matrix_multiplication/test.cpp): opcode=3;
  kernel(opcode, instr[gid1,cacheable], n, A[gid3], B[gid4], C[gid5], tmp[gid6], trace[gid7])
A=[M,K] bf16, B=[K,N] bf16 (b_col_maj=0 -> row-major), C=[M,N] f32.

Usage:
  .venv-iron/bin/python scripts/run_npu_matmul_bf16.py -M 512 -K 768 -N 768 --dry
  .venv-iron/bin/python scripts/run_npu_matmul_bf16.py -M 512 -K 768 -N 768
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

EXD = "mlir-aie/programming_examples/basic/matrix_multiplication/single_core/build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=512)
    ap.add_argument("-K", type=int, default=768)
    ap.add_argument("-N", type=int, default=768)
    ap.add_argument("--tile", default="32x32x32")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    M, K, N = a.M, a.K, a.N
    suffix = f"{M}x{K}x{N}_{a.tile}"
    xclbin = f"{EXD}/final_{suffix}.xclbin"
    insts = f"{EXD}/insts_{suffix}.txt"

    rng = np.random.RandomState(0)
    A = rng.uniform(-1, 1, size=(M, K)).astype(bfloat16)
    B = rng.uniform(-1, 1, size=(K, N)).astype(bfloat16)
    ref = A.astype(np.float32) @ B.astype(np.float32)   # bf16 in, f32 accumulate (NPU convention)
    print(f"[ref] A[{M},{K}] @ B[{K},{N}] -> C[{M},{N}]  bf16->f32")

    if a.dry:
        print(f"[dry] would load {xclbin}; ref[0,:4]={ref[0,:4]}")
        return 0
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build it (see header)")
    instr = np.fromfile(insts, dtype=np.uint32)

    import pyxrt
    xb = pyxrt.xclbin(xclbin)
    kname = xb.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}")
    d = pyxrt.device(0); d.register_xclbin(xb)
    ctx = pyxrt.hw_context(d, xb.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    Ab = np.ascontiguousarray(A).view(np.uint16)
    Bb = np.ascontiguousarray(B).view(np.uint16)
    bo_i = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_a = pyxrt.bo(d, Ab.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_b = pyxrt.bo(d, Bb.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_c = pyxrt.bo(d, M * N * 4, pyxrt.bo.host_only, k.group_id(5))
    bo_tmp = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(6))
    bo_tr = pyxrt.bo(d, 4, pyxrt.bo.host_only, k.group_id(7))
    bo_i.write(instr.tobytes(), 0); bo_i.sync(TO)
    bo_a.write(Ab.tobytes(), 0); bo_a.sync(TO)
    bo_b.write(Bb.tobytes(), 0); bo_b.sync(TO)

    def once():
        k(3, bo_i, instr.size, bo_a, bo_b, bo_c, bo_tmp, bo_tr).wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters
    bo_c.sync(FROM)
    C = np.frombuffer(bo_c.read(M * N * 4, 0), np.float32).reshape(M, N)

    d_ = np.abs(C - ref)
    rel = d_.max() / (np.abs(ref).max() + 1e-9)
    fro = np.linalg.norm(d_) / (np.linalg.norm(ref) + 1e-9)
    macs = 2.0 * M * K * N
    ok = rel < 0.03 and not np.isnan(C).any()
    print(f"[run] device time/iter: {dt*1e3:.3f} ms -> {macs/dt/1e9:.1f} GFLOP/s")
    print(f"[run] vs f32(bf16 in): max|Δ|={d_.max():.4e}  max_rel={rel:.3e}  frob_rel={fro:.3e}")
    print(f"[run] C[0,:4]={C[0,:4]}  ref={ref[0,:4]}")
    print(f"[run] bf16 matmul {M}x{K}x{N} on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
