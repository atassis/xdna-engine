#!/usr/bin/env python3
"""Rung 5b — run the mlir-aie single_core GEMM on the XDNA2 NPU via pyxrt.

The design (single_core_iron.py) was built with its defaults: dtype_in=i16, dtype_out=i32,
M=K=N=512, tiled 32x32x32. So this is an EXACT integer matmul — no bf16 packing, no tolerance.
We keep inputs small (|x|<=8) so the i32 accumulation can't overflow, giving a deterministic
bit-exact check, plus a rough TOPS number from the device wall time.

Host C++ harness is unbuildable on Arch (broken XRT cmake), so we drive xclbin+insts via pyxrt.
Kernel ABI (from matrix_multiplication/test.cpp): opcode=3;
  kernel(opcode, instr[gid1,cacheable], n, A[gid3], B[gid4], C[gid5], tmp1[gid6,1B], trace[gid7,4B])

Usage:
  .venv-iron/bin/python scripts/run_npu_matmul.py --dry   # validate, no NPU
  .venv-iron/bin/python scripts/run_npu_matmul.py         # REAL run on the NPU (must be free)
"""
import argparse, os, sys, time
import numpy as np

EX = "mlir-aie/programming_examples/basic/matrix_multiplication/single_core/build"
M = K = N = 512

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final_512x512x512_32x32x32.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts_512x512x512_32x32x32.txt")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build: (cd .../single_core && make NPU2=1 use_iron=1)")

    instr = np.fromfile(a.insts, dtype=np.uint32)
    # small ints so i32 accumulation over K=512 cannot overflow -> bit-exact reference
    rng = np.random.RandomState(0)
    A = rng.randint(-8, 9, size=(M, K), dtype=np.int16)
    B = rng.randint(-8, 9, size=(K, N), dtype=np.int16)
    C_ref = (A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32)

    import pyxrt
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}'  instr_words={instr.size}  GEMM {M}x{K}x{N} i16->i32")
    if a.dry:
        print("[dry] artifacts valid; pyxrt imported; kernel read. NOT touching the NPU.")
        return 0

    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())   # CREATE_HWCTX (needs NPU free)
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_a = pyxrt.bo(d, A.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_b = pyxrt.bo(d, B.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_c = pyxrt.bo(d, C_ref.nbytes, pyxrt.bo.host_only, k.group_id(5))
    bo_tmp1 = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(6))
    bo_trace = pyxrt.bo(d, 4, pyxrt.bo.host_only, k.group_id(7))

    bo_instr.write(instr.tobytes(), 0);                 bo_instr.sync(TO)
    bo_a.write(np.ascontiguousarray(A).tobytes(), 0);   bo_a.sync(TO)
    bo_b.write(np.ascontiguousarray(B).tobytes(), 0);   bo_b.sync(TO)

    # warmup + timed runs
    def once():
        r = k(3, bo_instr, instr.size, bo_a, bo_b, bo_c, bo_tmp1, bo_trace)
        r.wait()
    once()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_c.sync(FROM)
    C = np.frombuffer(bo_c.read(C_ref.nbytes, 0), dtype=np.int32).reshape(M, N)
    ok = np.array_equal(C, C_ref)
    nbad = int((C != C_ref).sum())
    macs = 2.0 * M * K * N
    print(f"[run] exact i32 GEMM correct: {'PASS' if ok else 'FAIL'}  mismatches={nbad}/{M*N}")
    print(f"[run] device time/iter: {dt*1e3:.3f} ms  ->  {macs/dt/1e9:.2f} GOPS")
    print(f"[run] C[0,:5]={C[0,:5]}  ref={C_ref[0,:5]}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
