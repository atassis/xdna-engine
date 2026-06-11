#!/usr/bin/env python3
"""Run + validate the WHOLE-ARRAY GEMM->GEMM FUSION SLICE (n_cols=1) on XDNA2.

ONE xclbin computes, with the intermediate H = A@W1 kept ON-CHIP (never returned
to the host):
        C = (A @ W1) @ W2          bf16 in, f32 out
mm1 (A[M,K] @ W1[K,P] -> H[M,P]) runs on the col-0 mm1 core; H is narrowed to
bf16 and FORWARDED through the col-0 MemTile to the col-0 mm2 core, which runs
mm2 (H[M,P] @ W2[P,N] -> C[M,N]) in the SAME dispatch.  This is the smallest
verifiable whole_array-idiom on-chip GEMM->GEMM: ONE host dispatch, H never
leaves the array.

WHY n_cols=1 ONLY: the n_cols>=2 (multi-column, all-to-all H) build hits a hard
AIE2 limit at COMPILE time -- the mm2 core would need n_cols H-slab inputs + 1
W2 input = n_cols+1 INPUT DMA channels, but a compute tile has only 2.  aiecc
fails to place with:
    "tile (0,3) requires 3 input/1 output DMA channels, but only 2 available"
So only the single-column chain is buildable; this runner validates THAT.

Design : mlir-aie/.../whole_array/wa_gemm2_iron.py  (--n-cols 1)
Build  : see route_b_kernels/whole_array_fused/Makefile.wa_gemm2 (or the header
         of that file); artifacts land in
         mlir-aie/.../whole_array/build_gemm2/final_wa_gemm2_1col.xclbin
         and insts_wa_gemm2_1col.txt

Host ABI (opcode 3, from the emitted runtime_sequence; 3 inputs + 1 output):
    kernel(3, instr[gid1,cacheable], n_instr,
           A[gid3]  bf16 [M,K] row-major,
           W1[gid4] bf16 [K,P] row-major,
           W2[gid5] bf16 [P,N] row-major,
           C[gid6]  f32  [M,N] row-major)

  *** DO NOT RUN ON A BUSY NPU. *** The NPU is single-tenant; a 2nd hw context
  crashes (CREATE_HWCTX EINVAL).  Run ONLY on a freed NPU (flm-asr stopped).

Usage:
    .venv-iron/bin/python scripts/run_npu_wa_gemm2.py --dry      # no NPU, checks artifacts + host ref
    .venv-iron/bin/python scripts/run_npu_wa_gemm2.py            # dispatch on a FREED NPU

PASS criterion: max_rel(device C vs host (A@W1->bf16->@W2)) < 0.05, no NaN.
KNOWN-CRASH signature (the thing we are watching for): a hang or
    "qds_device::wait() unexpected command state"
would indicate the on-chip-H pipeline deadlocked (an H-reuse stall).  For
n_cols=1 / N_div_n=1 there is no H re-read across an N-loop, so we EXPECT no
deadlock -- a crash here would be a new finding.
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

EXD = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build_gemm2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=32)
    ap.add_argument("-K", type=int, default=32)
    ap.add_argument("-P", type=int, default=32)
    ap.add_argument("-N", type=int, default=32)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    M, K, P, N = a.M, a.K, a.P, a.N

    xclbin = f"{EXD}/final_wa_gemm2_1col.xclbin"
    insts = f"{EXD}/insts_wa_gemm2_1col.txt"

    rng = np.random.RandomState(0)
    A = rng.uniform(-1, 1, (M, K)).astype(bfloat16)
    W1 = rng.uniform(-1, 1, (K, P)).astype(bfloat16)
    W2 = rng.uniform(-1, 1, (P, N)).astype(bfloat16)

    # Host reference mirroring the device numerics:
    #   mm1 accumulates in f32, the narrow epilogue rounds H to bf16,
    #   mm2 accumulates that bf16 H against bf16 W2 in f32.
    H_f32 = A.astype(np.float32) @ W1.astype(np.float32)   # f32 accumulate
    H_bf = H_f32.astype(bfloat16)                           # narrow epilogue -> bf16
    C_ref = H_bf.astype(np.float32) @ W2.astype(np.float32)  # mm2 f32 accumulate
    print(f"[ref] C = (A[{M},{K}] @ W1[{K},{P}]) @ W2[{P},{N}]  (H kept bf16 on-chip)")
    print(f"[ref] C[0,:4]={C_ref[0,:4]}")

    if a.dry:
        print(f"[dry] xclbin={xclbin}")
        print(f"[dry] insts ={insts}")
        for p in (xclbin, insts):
            print(f"[dry]   {'OK ' if os.path.exists(p) else 'MISSING'} {p}")
        print(f"[dry] host A : bf16 [{M},{K}] row-major nbytes={A.nbytes}")
        print(f"[dry] host W1: bf16 [{K},{P}] row-major nbytes={W1.nbytes}")
        print(f"[dry] host W2: bf16 [{P},{N}] row-major nbytes={W2.nbytes}")
        print(f"[dry] host C : f32  [{M},{N}] row-major nbytes={M*N*4}")
        print("[dry] not touching the NPU.")
        return 0

    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build the n_cols=1 slice first (see header)")
    instr = np.fromfile(insts, dtype=np.uint32)

    import pyxrt
    xb = pyxrt.xclbin(xclbin)
    kname = xb.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}")
    d = pyxrt.device(0); d.register_xclbin(xb)
    ctx = pyxrt.hw_context(d, xb.get_uuid())
    kk = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    Ab = np.ascontiguousarray(A).view(np.uint16)
    W1b = np.ascontiguousarray(W1).view(np.uint16)
    W2b = np.ascontiguousarray(W2).view(np.uint16)
    bo_i = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bo_a = pyxrt.bo(d, Ab.nbytes, pyxrt.bo.host_only, kk.group_id(3))
    bo_w1 = pyxrt.bo(d, W1b.nbytes, pyxrt.bo.host_only, kk.group_id(4))
    bo_w2 = pyxrt.bo(d, W2b.nbytes, pyxrt.bo.host_only, kk.group_id(5))
    bo_c = pyxrt.bo(d, M * N * 4, pyxrt.bo.host_only, kk.group_id(6))  # f32 out
    bo_i.write(instr.tobytes(), 0); bo_i.sync(TO)
    bo_a.write(Ab.tobytes(), 0); bo_a.sync(TO)
    bo_w1.write(W1b.tobytes(), 0); bo_w1.sync(TO)
    bo_w2.write(W2b.tobytes(), 0); bo_w2.sync(TO)

    def once():
        kk(3, bo_i, instr.size, bo_a, bo_w1, bo_w2, bo_c).wait()
    once()  # warm
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters
    bo_c.sync(FROM)
    C = np.frombuffer(bo_c.read(M * N * 4, 0), np.float32).reshape(M, N)

    rel = np.abs(C - C_ref).max() / (np.abs(C_ref).max() + 1e-9)
    ok = rel < 0.05 and not np.isnan(C).any()
    macs = 2.0 * M * K * P + 2.0 * M * P * N
    print(f"[run] device time/iter (BOTH matmuls, 1 dispatch): {dt*1e3:.3f} ms -> {macs/dt/1e9:.1f} GFLOP/s")
    print(f"[run] C[0,:4]={C[0,:4]}  ref={C_ref[0,:4]}")
    print(f"[run] max_rel vs host (A@W1->bf16->@W2) = {rel:.3e}")
    print(f"[run] whole_array GEMM->GEMM (n_cols=1, on-chip H) on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
