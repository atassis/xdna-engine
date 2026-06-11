#!/usr/bin/env python3
"""Run + validate the FUSED matmul+bias+SiLU on the XDNA2 NPU via pyxrt.

ONE xclbin computes   out = silu(A @ B + bias)   in bf16 (f32 accumulate),
with the bias add and the SiLU done on-chip and the f32->bf16 downconvert fused
into the epilogue. The host does NO post-processing.

Design: mlir-aie/programming_examples/ml/mm_silu_fused/mm_silu_fused_iron.py
Epilogue kernel: mlir-aie/aie_kernels/aie2p/mm_silu_epilogue.cc (pure SiLU+cast)

BIAS via K-AUGMENTATION (the NPU2 compute tile has only 2 input DMA channels, so
a third "bias" stream does not fit). We append ONE extra k-block (size k) to K so
  A_aug = [ A | E ]   E is M x k  with column 0 = 1, the rest 0
  B_aug = [ B ; F ]   F is k x N  with row 0 = bias, the rest 0
  A_aug @ B_aug = A@B + (E @ F) = A@B + (1_col @ bias_row) = A@B + bias
i.e. bias is added to every output row. The device sees a plain matmul of inner
dim K_aug = K + k, then applies SiLU. Host buffers are plain row-major; the DMA
descriptors do all tiling (same as single_core / whole_array).

Build (no NPU). Kernel .o files are named by tile size, NOT dtype, so remove
stale objects first:
  source scripts/iron_env.sh
  MM=mlir-aie/programming_examples/ml/mm_silu_fused
  rm -f $MM/build/mm_32x32x32.o $MM/build/mm_silu_epilogue_32x32x32.o
  make -C $MM NPU2=1 M=$M K=$Kaug N=$N m=32 k=32 n=32 \
       build/final_${M}x${Kaug}x${N}_32x32x32.xclbin
  nm -C $MM/build/mm_32x32x32.o          | grep -E 'matmul_bf16_f32|zero_f32'
  nm -C $MM/build/mm_silu_epilogue_*.o   | grep mm_silu_epilogue_f32_bf16
  # Kaug = K + k   (e.g. K=768, k=32 -> Kaug=800)

Host ABI (matrix_multiplication/test.cpp): opcode=3;
  kernel(opcode, instr[gid1,cacheable], n, A[gid3], B[gid4], C[gid5],
         tmp[gid6], trace[gid7])
A=[M,Kaug] bf16, B=[Kaug,N] bf16 (row-major), C=[M,N] bf16 (post-SiLU).

NOTE on N: the single-core design's A DMA repeats N/n times and the hardware BD
repeat limit is 64, so N/n <= 64, i.e. N <= 2048 for n=32. The FFN linear1
N=3072 must be split into two N=1536 halves (two xclbins / two dispatches);
this script handles a single N at a time. N=1536 is the largest single-shot
power-of-two-friendly size that builds.

Usage:
  .venv-iron/bin/python scripts/run_npu_mm_silu.py -M 256 -K 224 -N 256 --dry
  .venv-iron/bin/python scripts/run_npu_mm_silu.py -M 512 -K 768 -N 1536 --dry
  .venv-iron/bin/python scripts/run_npu_mm_silu.py -M 512 -K 768 -N 1536   # REAL
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

EXD = "mlir-aie/programming_examples/ml/mm_silu_fused/build"


def silu_ref_tanh_bf16(z_f32):
    """Mirror the device epilogue: bf16-round the accumulator, then SiLU via the
    tanh identity sigmoid(x)=0.5*(1+tanh(x/2)) with bf16 rounding at each step,
    matching mm_silu_epilogue.cc / silu.cc."""

    def bf(x):
        return x.astype(bfloat16).astype(np.float32)

    x = bf(z_f32)  # accumulator narrowed to bf16 (a.to_vector<bfloat16>())
    half_x = bf(x * 0.5)
    tanh_half_x = bf(np.tanh(half_x.astype(np.float32)))  # tanh<bf16> over f32
    tanh_p1 = bf(tanh_half_x + 1.0)
    sig = bf(tanh_p1 * 0.5)
    out = bf(x * sig)
    return out  # f32 holding bf16-representable values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=512)
    ap.add_argument("-K", type=int, default=768, help="real inner dim (bias adds one k-block)")
    ap.add_argument("-N", type=int, default=1536)
    ap.add_argument("--tile", default="32x32x32")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    M, K, N = a.M, a.K, a.N
    m, k, n = (int(v) for v in a.tile.split("x"))
    Kaug = K + k  # one extra k-block carries the bias

    suffix = f"{M}x{Kaug}x{N}_{a.tile}"
    xclbin = f"{EXD}/final_{suffix}.xclbin"
    insts = f"{EXD}/insts_{suffix}.txt"

    rng = np.random.RandomState(0)
    A = rng.uniform(-1, 1, size=(M, K)).astype(bfloat16)
    B = rng.uniform(-1, 1, size=(K, N)).astype(bfloat16)
    bias = rng.uniform(-1, 1, size=(N,)).astype(bfloat16)

    # Reference: silu(A@B + bias), accumulate in f32 (bf16 inputs).
    z = A.astype(np.float32) @ B.astype(np.float32) + bias.astype(np.float32)
    ref_b = silu_ref_tanh_bf16(z)  # device-matched (tanh approx + bf16 rounding)
    ref_clean = z / (1.0 + np.exp(-z))  # clean fp32 silu, for sanity
    print(f"[ref] silu(A[{M},{K}] @ B[{K},{N}] + bias[{N}]) -> C[{M},{N}] bf16")
    print(f"[ref] device-matched ref_b[0,:4]={ref_b[0,:4]}  clean[0,:4]={ref_clean[0,:4]}")

    # Build the K-augmented operands (this is data prep, not post-processing).
    A_aug = np.zeros((M, Kaug), dtype=bfloat16)
    A_aug[:, :K] = A
    A_aug[:, K] = bfloat16(1.0)  # extra k-block, col 0 = 1
    B_aug = np.zeros((Kaug, N), dtype=bfloat16)
    B_aug[:K, :] = B
    B_aug[K, :] = bias  # extra k-block, row 0 = bias
    # sanity: A_aug@B_aug == A@B + bias
    chk = (A_aug.astype(np.float32) @ B_aug.astype(np.float32))
    aug_err = np.abs(chk - z).max() / (np.abs(z).max() + 1e-9)
    print(f"[ref] K-aug fold check: max_rel(A_aug@B_aug vs A@B+bias)={aug_err:.2e} (Kaug={Kaug})")

    if a.dry:
        print(f"[dry] xclbin={xclbin}")
        print(f"[dry] insts ={insts}")
        for p in (xclbin, insts):
            print(f"[dry]   {'OK ' if os.path.exists(p) else 'MISSING'} {p}")
        print(f"[dry] A_aug={A_aug.nbytes}B B_aug={B_aug.nbytes}B C={M*N*2}B (bf16)")
        if N // n > 64:
            print(f"[dry] WARNING: N/n={N//n} > 64 BD-repeat limit; this N will NOT build "
                  f"in single_core (split N).")
        print("[dry] not touching the NPU.")
        return 0

    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build it (see header; Kaug={Kaug})")
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

    # Compare against the device-matched reference (tanh approx + bf16 rounding).
    d_b = np.abs(Cf - ref_b)
    rel_b = d_b.max() / (np.abs(ref_b).max() + 1e-9)
    # Also vs clean fp32 silu (looser; tanh approx + bf16 error).
    d_c = np.abs(Cf - ref_clean)
    denom = np.maximum(np.abs(ref_clean), 1e-2)
    rel_c = (d_c / denom)
    macs = 2.0 * M * (K + k) * N
    ok = (rel_b < 0.03) and (rel_c.mean() < 0.03) and not np.isnan(Cf).any()
    print(f"[run] device time/iter: {dt*1e3:.3f} ms -> {macs/dt/1e9:.1f} GFLOP/s")
    print(f"[run] vs device-matched ref: max|Δ|={d_b.max():.4e}  max_rel={rel_b:.3e}")
    print(f"[run] vs clean fp32 silu:    mean_rel={rel_c.mean():.4f}  p99_rel={np.percentile(rel_c,99):.4f}")
    print(f"[run] C[0,:4]={Cf[0,:4]}  ref_b={ref_b[0,:4]}")
    print(f"[run] fused silu(A@B+bias) {M}x{K}x{N} on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
