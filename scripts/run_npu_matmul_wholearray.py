#!/usr/bin/env python3
"""Validate a bf16 -> f32 whole_array (multi-column) matmul on the XDNA2 NPU via pyxrt.

This is the multi-column sibling of run_npu_matmul_bf16.py. The single_core path
uses only 1 of 8 NPU columns and is the block perf bottleneck; whole_array spreads
the GEMM across all 8 columns (4 rows x 8 cols = 32 compute cores).

Build first (no NPU). The kernel object is named mm_${m}x${k}x${n}.o (tile only,
NOT dtype) so a stale i16/bf16 object is silently reused -- rm it before a bf16
build, then verify symbols:
  source scripts/iron_env.sh
  MM=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
  rm -f $MM/build/mm_*.o
  make -C $MM NPU2=1 M=$M K=$K N=$N dtype_in=bf16 dtype_out=f32 \
       n_aie_cols=8 use_iron=1
  nm -C $MM/build/mm_*.o | grep -E 'matmul_bf16_f32|zero_f32'   # must print both

Host memory layout (whole_array README "Tiling and Data Layout Transformations"):
ALL host buffers are PLAIN ROW-MAJOR. The per-column tiling/packing is done
entirely by the DMA descriptors inside the runtime sequence (TensorTiler2D), not
by the host. So there is NO host-side packing -- A=[M,K] bf16 row-major,
B=[K,N] bf16 row-major (b_col_maj=0), C=[M,N] f32 row-major. The numpy reference
is therefore just C = A.f32 @ B.f32, identical to single_core.

Host ABI (matrix_multiplication/test.cpp, shared by single_core and whole_array):
  opcode=3; kernel(opcode, instr[gid1,cacheable], n, A[gid3], B[gid4],
                   C[gid5], tmp[gid6], trace[gid7])

Artifact naming differs from single_core: whole_array appends the column count,
  final_${M}x${K}x${N}_${m}x${k}x${n}_${cols}c.xclbin
  insts_${M}x${K}x${N}_${m}x${k}x${n}_${cols}c.txt

Usage:
  .venv-iron/bin/python scripts/run_npu_matmul_wholearray.py -M 512 -K 768 -N 768 --dry
  .venv-iron/bin/python scripts/run_npu_matmul_wholearray.py -M 512 -K 768 -N 768
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

EXD = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=512)
    ap.add_argument("-K", type=int, default=768)
    ap.add_argument("-N", type=int, default=768)
    ap.add_argument("--tile", default="32x32x32")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    M, K, N = a.M, a.K, a.N
    suffix = f"{M}x{K}x{N}_{a.tile}_{a.cols}c"
    xclbin = f"{EXD}/final_{suffix}.xclbin"
    insts = f"{EXD}/insts_{suffix}.txt"

    rng = np.random.RandomState(0)
    A = rng.uniform(-1, 1, size=(M, K)).astype(bfloat16)
    B = rng.uniform(-1, 1, size=(K, N)).astype(bfloat16)
    ref = A.astype(np.float32) @ B.astype(np.float32)   # bf16 in, f32 accumulate (NPU convention)
    print(f"[ref] A[{M},{K}] @ B[{K},{N}] -> C[{M},{N}]  bf16->f32  ({a.cols} cols)")

    if a.dry:
        # whole_array host layout is plain row-major for A, B, C -- no packing.
        # The per-column tiling lives in the DMA descriptors, not the host buffers.
        print(f"[dry] xclbin={xclbin}")
        print(f"[dry] insts ={insts}")
        print(f"[dry] artifacts present: "
              f"xclbin={os.path.exists(xclbin)} insts={os.path.exists(insts)}")
        print(f"[dry] host A layout: row-major [{M},{K}] bf16 (uint16 view), "
              f"contiguous, nbytes={A.size*2}")
        print(f"[dry] host B layout: row-major [{K},{N}] bf16 (uint16 view), "
              f"contiguous, nbytes={B.size*2}  (b_col_maj=0)")
        print(f"[dry] host C layout: row-major [{M},{N}] f32, "
              f"nbytes={M*N*4}  (c_col_maj=0) -- NO unpack needed")
        print(f"[dry] ref[0,:4]={ref[0,:4]}")
        print(f"[dry] ref dtype={ref.dtype} shape={ref.shape} "
              f"max|ref|={np.abs(ref).max():.4f}")
        # self-consistency: round-trip A,B through the exact uint16 view the
        # runner sends, recompute, confirm it matches the reference bit-for-bit.
        Ab = np.ascontiguousarray(A).view(np.uint16)
        Bb = np.ascontiguousarray(B).view(np.uint16)
        A2 = Ab.view(bfloat16).reshape(M, K)
        B2 = Bb.view(bfloat16).reshape(K, N)
        ref2 = A2.astype(np.float32) @ B2.astype(np.float32)
        same = np.array_equal(ref, ref2)
        print(f"[dry] uint16 round-trip of A/B reproduces ref exactly: {same}")
        return 0 if same else 1
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

    # All host buffers plain row-major: A=[M,K] bf16, B=[K,N] bf16, C=[M,N] f32.
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
    # C comes back plain row-major [M,N] f32 -- no unpack.
    C = np.frombuffer(bo_c.read(M * N * 4, 0), np.float32).reshape(M, N)

    d_ = np.abs(C - ref)
    rel = d_.max() / (np.abs(ref).max() + 1e-9)
    fro = np.linalg.norm(d_) / (np.linalg.norm(ref) + 1e-9)
    macs = 2.0 * M * K * N
    ok = rel < 0.03 and not np.isnan(C).any()
    print(f"[run] device time/iter: {dt*1e3:.3f} ms -> {macs/dt/1e9:.1f} GFLOP/s")
    print(f"[run] vs f32(bf16 in): max|Δ|={d_.max():.4e}  max_rel={rel:.3e}  frob_rel={fro:.3e}")
    print(f"[run] C[0,:4]={C[0,:4]}  ref={ref[0,:4]}")
    print(f"[run] bf16 whole_array matmul {M}x{K}x{N} ({a.cols} cols) on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
