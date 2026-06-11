#!/usr/bin/env python3
"""Run + validate the WHOLE-ARRAY (8-column) FUSED matmul+bias(+SiLU) on XDNA2.

ONE xclbin computes, across all 8 NPU2 columns (4 rows x 8 cols = 32 cores):
    silu mode  (FFN linear1):   out = silu(A @ B + bias)     bf16 in/out
    bias mode  (FFN linear2):   out =      A @ B + bias      bf16 in/out
bf16 inputs, f32 in-core accumulate, bf16 out, no host post-processing. The
f32->bf16 downconvert (and SiLU, in silu mode) happens on chip in the epilogue.

This is the whole-array sibling of scripts/run_npu_mm_silu.py (single core). The
whole-array design splits N across the 8 columns (n*n_aie_cols = 256 per pass,
3072/256 = 12 row-blocks, well under the 64-BD repeat limit) so N=3072 builds in
ONE xclbin -- NO N-splitting, unlike the single-core path.

Design:   mlir-aie/.../matrix_multiplication/whole_array/whole_array_silu_iron.py
Makefile: same dir, Makefile.silu
Epilogue: mlir-aie/aie_kernels/aie2p/mm_silu_epilogue.cc
            mm_silu_epilogue_f32_bf16   (silu mode)
            mm_narrow_epilogue_f32_bf16 (bias mode, pure f32->bf16 narrow)

BIAS via K-AUGMENTATION (the NPU2 compute tile has only 2 input DMA channels, so
a third "bias" stream does not fit). Append ONE extra k-block (size k) to K:
  A_aug = [ A | E ]   E is M x k     with column 0 = 1, the rest 0
  B_aug = [ B ; F ]   F is k x N     with row 0 = bias, the rest 0
  A_aug @ B_aug = A@B + (E @ F) = A@B + (1_col @ bias_row) = A@B + bias
i.e. bias is added to every output row. The device sees a plain whole-array
matmul of inner dim K_aug = K + k, then applies the epilogue. Host buffers are
plain ROW-MAJOR; all per-column tiling lives in the DMA descriptors.

Host ABI (matrix_multiplication/test.cpp, shared by single_core + whole_array):
  opcode=3; kernel(opcode, instr[gid1,cacheable], n, A[gid3], B[gid4],
                   C[gid5], tmp[gid6], trace[gid7])
  A=[M,Kaug] bf16, B=[Kaug,N] bf16 (row-major), C=[M,N] bf16 (post-epilogue).

Artifact naming (whole_array Makefile.silu):
  final_${M}x${Kaug}x${N}_${m}x${k}x${n}_${cols}c_${mode}.xclbin   (mode=silu|bias)
  insts_${M}x${Kaug}x${N}_${m}x${k}x${n}_${cols}c_${mode}.txt

Build (no NPU). Kernel .o files are named by tile size, NOT dtype/mode, so remove
stale objects first, then verify symbols:
  source scripts/iron_env.sh
  MM=$PWD/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
  rm -f $MM/build/mm_32x32x32.o $MM/build/mm_silu_epilogue_32x32x32.o
  # linear1 (silu): Kaug = 768 + 32 = 800
  make -f $MM/Makefile.silu -C $MM NPU2=1 M=512 K=800 N=3072 n_aie_cols=8 \
       build/final_512x800x3072_32x32x32_8c_silu.xclbin
  # linear2 (bias): Kaug = 3072 + 32 = 3104
  make -f $MM/Makefile.silu -C $MM NPU2=1 M=512 K=3104 N=768 n_aie_cols=8 no_silu=1 \
       build/final_512x3104x768_32x32x32_8c_bias.xclbin
  nm -C $MM/build/mm_32x32x32.o          | grep -E 'matmul_bf16_f32|zero_f32'
  nm -C $MM/build/mm_silu_epilogue_*.o   | grep -E 'mm_(silu|narrow)_epilogue_f32_bf16'

Usage:
  # linear1 (silu)
  .venv-iron/bin/python scripts/run_npu_mm_silu_wa.py -M 512 -K 768 -N 3072 --silu --dry
  # linear2 (bias, no activation)
  .venv-iron/bin/python scripts/run_npu_mm_silu_wa.py -M 512 -K 3072 -N 768 --bias --dry
  # drop --dry to dispatch on the NPU (single-tenant; main session validates)
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

EXD = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"


def silu_ref_tanh_bf16(z_f32):
    """Mirror the device SiLU epilogue: bf16-round the accumulator, then SiLU via
    the tanh identity sigmoid(x)=0.5*(1+tanh(x/2)) with bf16 rounding at each step
    (matches mm_silu_epilogue.cc / silu.cc)."""

    def bf(x):
        return x.astype(bfloat16).astype(np.float32)

    x = bf(z_f32)  # accumulator narrowed to bf16 (a.to_vector<bfloat16>())
    half_x = bf(x * 0.5)
    tanh_half_x = bf(np.tanh(half_x.astype(np.float32)))  # tanh<bf16> over f32
    tanh_p1 = bf(tanh_half_x + 1.0)
    sig = bf(tanh_p1 * 0.5)
    out = bf(x * sig)
    return out  # f32 holding bf16-representable values


def narrow_ref_bf16(z_f32):
    """Mirror the device narrow epilogue: just bf16-round the f32 accumulator."""
    return z_f32.astype(bfloat16).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=512)
    ap.add_argument("-K", type=int, default=768, help="real inner dim (bias adds one k-block)")
    ap.add_argument("-N", type=int, default=3072)
    ap.add_argument("--tile", default="32x32x32")
    ap.add_argument("--cols", type=int, default=8)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--silu", action="store_true", help="silu(A@B+bias) [linear1, default]")
    mode.add_argument("--bias", action="store_true", help="A@B+bias, no activation [linear2]")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    M, K, N = a.M, a.K, a.N
    m, k, n = (int(v) for v in a.tile.split("x"))
    Kaug = K + k  # one extra k-block carries the bias
    do_silu = not a.bias  # silu is the default unless --bias requested
    mode_tag = "silu" if do_silu else "bias"

    suffix = f"{M}x{Kaug}x{N}_{a.tile}_{a.cols}c_{mode_tag}"
    xclbin = f"{EXD}/final_{suffix}.xclbin"
    insts = f"{EXD}/insts_{suffix}.txt"

    rng = np.random.RandomState(0)
    A = rng.uniform(-1, 1, size=(M, K)).astype(bfloat16)
    B = rng.uniform(-1, 1, size=(K, N)).astype(bfloat16)
    bias = rng.uniform(-1, 1, size=(N,)).astype(bfloat16)

    # Reference: epilogue(A@B + bias), f32 accumulate (bf16 inputs).
    z = A.astype(np.float32) @ B.astype(np.float32) + bias.astype(np.float32)
    if do_silu:
        ref_b = silu_ref_tanh_bf16(z)              # device-matched
        ref_clean = z / (1.0 + np.exp(-z))         # clean fp32 silu, sanity
        print(f"[ref] silu(A[{M},{K}] @ B[{K},{N}] + bias[{N}]) -> C[{M},{N}] bf16  ({a.cols} cols)")
    else:
        ref_b = narrow_ref_bf16(z)                 # device-matched (narrow only)
        ref_clean = z                              # clean fp32 A@B+bias
        print(f"[ref] (A[{M},{K}] @ B[{K},{N}] + bias[{N}]) -> C[{M},{N}] bf16  ({a.cols} cols, no act)")
    print(f"[ref] mode={mode_tag}  ref_b[0,:4]={ref_b[0,:4]}  clean[0,:4]={ref_clean[0,:4]}")

    # Build the K-augmented operands (data prep, not post-processing).
    A_aug = np.zeros((M, Kaug), dtype=bfloat16)
    A_aug[:, :K] = A
    A_aug[:, K] = bfloat16(1.0)   # extra k-block, col 0 = 1
    B_aug = np.zeros((Kaug, N), dtype=bfloat16)
    B_aug[:K, :] = B
    B_aug[K, :] = bias            # extra k-block, row 0 = bias
    chk = A_aug.astype(np.float32) @ B_aug.astype(np.float32)
    aug_err = np.abs(chk - z).max() / (np.abs(z).max() + 1e-9)
    print(f"[ref] K-aug fold check: max_rel(A_aug@B_aug vs A@B+bias)={aug_err:.2e} (Kaug={Kaug})")

    if a.dry:
        print(f"[dry] xclbin={xclbin}")
        print(f"[dry] insts ={insts}")
        for p in (xclbin, insts):
            print(f"[dry]   {'OK ' if os.path.exists(p) else 'MISSING'} {p}")
        # whole_array host layout is plain row-major; per-column tiling is in the
        # DMA descriptors, not the host buffers.
        print(f"[dry] host A_aug: row-major [{M},{Kaug}] bf16, nbytes={A_aug.nbytes}")
        print(f"[dry] host B_aug: row-major [{Kaug},{N}] bf16, nbytes={B_aug.nbytes} (b_col_maj=0)")
        print(f"[dry] host C    : row-major [{M},{N}] bf16, nbytes={M*N*2} (post-epilogue, NO unpack)")
        nrep = N // n // a.cols
        print(f"[dry] B per-col repeat N/n/cols={nrep} (<=64 BD limit: {'OK' if nrep <= 64 else 'OVER'})")
        # self-consistency: round-trip A_aug/B_aug through the exact uint16 view
        # the runner sends, recompute, confirm bit-for-bit match.
        Ab = np.ascontiguousarray(A_aug).view(np.uint16)
        Bb = np.ascontiguousarray(B_aug).view(np.uint16)
        A2 = Ab.view(bfloat16).reshape(M, Kaug)
        B2 = Bb.view(bfloat16).reshape(Kaug, N)
        z2 = A2.astype(np.float32) @ B2.astype(np.float32)
        same = np.array_equal(chk, z2)
        print(f"[dry] uint16 round-trip of A_aug/B_aug reproduces fold exactly: {same}")
        print("[dry] not touching the NPU.")
        return 0 if same else 1

    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build it (see header; Kaug={Kaug}, mode={mode_tag})")
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

    Ab = np.ascontiguousarray(A_aug).view(np.uint16)
    Bb = np.ascontiguousarray(B_aug).view(np.uint16)
    bo_i = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bo_a = pyxrt.bo(d, Ab.nbytes, pyxrt.bo.host_only, kk.group_id(3))
    bo_b = pyxrt.bo(d, Bb.nbytes, pyxrt.bo.host_only, kk.group_id(4))
    bo_c = pyxrt.bo(d, M * N * 2, pyxrt.bo.host_only, kk.group_id(5))  # bf16 out
    bo_tmp = pyxrt.bo(d, 1, pyxrt.bo.host_only, kk.group_id(6))
    bo_tr = pyxrt.bo(d, 4, pyxrt.bo.host_only, kk.group_id(7))
    bo_i.write(instr.tobytes(), 0); bo_i.sync(TO)
    bo_a.write(Ab.tobytes(), 0); bo_a.sync(TO)
    bo_b.write(Bb.tobytes(), 0); bo_b.sync(TO)

    def once():
        kk(3, bo_i, instr.size, bo_a, bo_b, bo_c, bo_tmp, bo_tr).wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters
    bo_c.sync(FROM)
    C = np.frombuffer(bo_c.read(M * N * 2, 0), dtype=np.uint16).view(bfloat16).reshape(M, N)
    Cf = C.astype(np.float32)

    d_b = np.abs(Cf - ref_b)
    rel_b = d_b.max() / (np.abs(ref_b).max() + 1e-9)
    d_c = np.abs(Cf - ref_clean)
    denom = np.maximum(np.abs(ref_clean), 1e-2)
    rel_c = (d_c / denom)
    macs = 2.0 * M * Kaug * N
    ok = (rel_b < 0.03) and (rel_c.mean() < 0.03) and not np.isnan(Cf).any()
    print(f"[run] device time/iter: {dt*1e3:.3f} ms -> {macs/dt/1e9:.1f} GFLOP/s")
    print(f"[run] vs device-matched ref: max|Δ|={d_b.max():.4e}  max_rel={rel_b:.3e}")
    print(f"[run] vs clean fp32 ref:     mean_rel={rel_c.mean():.4f}  p99_rel={np.percentile(rel_c,99):.4f}")
    print(f"[run] C[0,:4]={Cf[0,:4]}  ref_b={ref_b[0,:4]}")
    print(f"[run] fused {mode_tag} whole_array {M}x{K}x{N} ({a.cols} cols) on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
