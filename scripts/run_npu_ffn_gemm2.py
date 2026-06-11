#!/usr/bin/env python3
"""Validate the FUSED two-matmul chain  C = (A @ W1) @ W2  on XDNA2 via pyxrt.

One xclbin runs BOTH matmuls in a single host dispatch; the intermediate
H = A@W1 is materialized ON-CHIP (kept bf16 in L2/MemTile) and consumed by the
second matmul without a host round-trip.  See
  mlir-aie/.../single_core/ffn_gemm2_iron.py   (design)
  mlir-aie/.../single_core/Makefile.ffn        (build)

Build first (NO NPU):
  source scripts/iron_env.sh
  rm -f mlir-aie/programming_examples/basic/matrix_multiplication/single_core/build/mm_*.o
  make -C mlir-aie/programming_examples/basic/matrix_multiplication/single_core \
       -f Makefile.ffn NPU2=1 M=64 K=128 P=128 N=128 m=32 k=32 p=32 n=32
(the kernel .o is named by tile geometry only -> remove stale single_core
 bf16_f32-ONLY objects first, else the link misses matmul_bf16_bf16.)

Host ABI (same family as run_npu_matmul_bf16.py): opcode=3;
  kernel(opcode, instr[gid1,cacheable], n_instr,
         A[gid3], W1[gid4], W2[gid5], C[gid6])
A=[M,K] bf16, W1=[K,P] bf16, W2=[P,N] bf16 (all row-major), C=[M,N] f32.
The 4 runtime-sequence buffers map to gid 3,4,5,6 in declaration order
(A, W1, W2, C) -- confirmed from aie.runtime_sequence in the generated MLIR.

NOTE on accuracy: H is kept in bf16 on-chip (mm1 runs bf16->bf16), so the
reference rounds H to bf16 before the second matmul to match the device.

Usage:
  .venv-iron/bin/python scripts/run_npu_ffn_gemm2.py --dry
  .venv-iron/bin/python scripts/run_npu_ffn_gemm2.py        # real NPU path
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

EXD = "mlir-aie/programming_examples/basic/matrix_multiplication/single_core/build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=64)
    ap.add_argument("-K", type=int, default=128)
    ap.add_argument("-P", type=int, default=128)
    ap.add_argument("-N", type=int, default=128)
    ap.add_argument("--tile", default="32x32x32x32", help="m x k x p x n")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    M, K, P, N = a.M, a.K, a.P, a.N
    suffix = f"ffn_{M}x{K}x{P}x{N}_{a.tile}"
    xclbin = f"{EXD}/final_{suffix}.xclbin"
    insts = f"{EXD}/insts_{suffix}.txt"

    rng = np.random.RandomState(0)
    A = rng.uniform(-1, 1, size=(M, K)).astype(bfloat16)
    W1 = rng.uniform(-1, 1, size=(K, P)).astype(bfloat16)
    W2 = rng.uniform(-1, 1, size=(P, N)).astype(bfloat16)

    # Reference: bf16 in, f32 accumulate per matmul, with the intermediate H
    # ROUNDED TO bf16 (matches the on-chip bf16 H of the fused kernel).
    H_f32 = A.astype(np.float32) @ W1.astype(np.float32)   # mm1, f32 accumulate
    H_bf16 = H_f32.astype(bfloat16)                        # on-chip H is bf16
    ref = H_bf16.astype(np.float32) @ W2.astype(np.float32)  # mm2, f32 accumulate
    # also keep a "full f32 H" ref to report the precision cost of bf16 H
    ref_f32H = H_f32 @ W2.astype(np.float32)
    print(f"[ref] C = (A[{M},{K}] @ W1[{K},{P}]) @ W2[{P},{N}] -> C[{M},{N}]  bf16->f32, H bf16 on-chip")

    if a.dry:
        print(f"[dry] would load {xclbin}")
        print(f"[dry] insts={insts}")
        print(f"[dry] ABI: k(3, instr@gid1, n, A@gid3, W1@gid4, W2@gid5, C@gid6)")
        print(f"[dry] ref(bf16 H)[0,:4]   = {ref[0,:4]}")
        print(f"[dry] ref(f32  H)[0,:4]   = {ref_f32H[0,:4]}")
        bf16_vs_f32 = np.abs(ref - ref_f32H).max() / (np.abs(ref_f32H).max() + 1e-9)
        print(f"[dry] bf16-H vs f32-H rel diff = {bf16_vs_f32:.3e} (precision cost of on-chip bf16 H)")
        return 0

    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build it (see header)")
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
    W1b = np.ascontiguousarray(W1).view(np.uint16)
    W2b = np.ascontiguousarray(W2).view(np.uint16)
    bo_i = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_a = pyxrt.bo(d, Ab.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_w1 = pyxrt.bo(d, W1b.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_w2 = pyxrt.bo(d, W2b.nbytes, pyxrt.bo.host_only, k.group_id(5))
    bo_c = pyxrt.bo(d, M * N * 4, pyxrt.bo.host_only, k.group_id(6))
    bo_i.write(instr.tobytes(), 0); bo_i.sync(TO)
    bo_a.write(Ab.tobytes(), 0); bo_a.sync(TO)
    bo_w1.write(W1b.tobytes(), 0); bo_w1.sync(TO)
    bo_w2.write(W2b.tobytes(), 0); bo_w2.sync(TO)

    def once():
        k(3, bo_i, instr.size, bo_a, bo_w1, bo_w2, bo_c).wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters
    bo_c.sync(FROM)
    C = np.frombuffer(bo_c.read(M * N * 4, 0), np.float32).reshape(M, N)

    dd = np.abs(C - ref)
    rel = dd.max() / (np.abs(ref).max() + 1e-9)
    fro = np.linalg.norm(dd) / (np.linalg.norm(ref) + 1e-9)
    macs = 2.0 * M * K * P + 2.0 * M * P * N
    ok = rel < 0.03 and not np.isnan(C).any()
    print(f"[run] device time/iter: {dt*1e3:.3f} ms -> {macs/dt/1e9:.1f} GFLOP/s")
    print(f"[run] vs ref(bf16 H): max|d|={dd.max():.4e}  max_rel={rel:.3e}  frob_rel={fro:.3e}")
    print(f"[run] C[0,:4]={C[0,:4]}  ref={ref[0,:4]}")
    print(f"[run] fused FFN gemm2 {M}x{K}x{P}x{N} on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
