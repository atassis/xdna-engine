#!/usr/bin/env python3
"""Run + validate LayerNorm [rows=400, cols=768] on the XDNA2 NPU via pyxrt.

The IRON layernorm design (mlir-aie/programming_examples/ml/layernorm) splits the
400 rows across 8 cores; each core normalizes one row at a time. The bf16 kernel
entry `layer_norm(bfloat16 *in, bfloat16 *out, int32_t cols)` in
aie_kernels/aie2p/layer_norm.cc is NORMALIZE-ONLY: gamma=1.0 / beta=0.0 are
hardcoded constants, so there is NO learned affine. The math per row is

    mean = sum(x) / cols
    var  = sum(x^2) / cols - mean^2          # biased (divide by cols)
    y    = (x - mean) / sqrt(var + eps)      # eps = 1e-5, gamma=1, beta=0

Accumulation is fp32 in-kernel; we mirror that in numpy (upcast bf16->fp32, single
bf16 round on the output) so the comparison is tight, not a loose tolerance. The
learned affine (gamma/beta from the GigaAM block's LayerNorm) is applied
separately on the host in the block integration — NOT here.

IRON host ABI (from layernorm.py runtime sequence `(a_in, c_out)` + the 2-buffer
path in runtime_lib/test_lib/xrt_test_wrapper.h):
  opcode=3; kernel(opcode, instr[gid1,cacheable], n_instr,
                   IN[gid3], OUT[gid4], tmp[gid5], ctrlpkts[gid6], trace[gid7])
The cols value is baked into the design at build time (rows=400 cols=768); it is
NOT a runtime BO arg. tmp/ctrlpkts/trace are dummy placeholders (0-size segfaults,
so they are tiny live buffers).

Usage:
  .venv-iron/bin/python scripts/run_npu_layernorm.py --dry   # validate refs, no NPU
  .venv-iron/bin/python scripts/run_npu_layernorm.py         # REAL run (NPU must be free)
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

ROWS, COLS = 400, 768
EPS = 1e-5
EX = "mlir-aie/programming_examples/ml/layernorm/build"


def layernorm_ref_fp32(x_bf16):
    """Normalize-only LN in fp32 (input is the bf16 values actually sent to NPU).
    x_bf16: [ROWS,COLS] bf16. Biased variance (/cols), eps=1e-5, gamma=1/beta=0.
    Returns [ROWS,COLS] fp32 (matches the kernel's fp32-accumulate math)."""
    x = x_bf16.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    # biased variance, computed the same way the kernel does: E[x^2] - E[x]^2
    mean_sq = (x * x).mean(axis=1, keepdims=True)
    var = mean_sq - mean * mean
    inv_std = 1.0 / np.sqrt(var + EPS)
    return (x - mean) * inv_std


def bf16_ulp_distance(a_bf16, b_bf16):
    """Number of representable bf16 steps between two bf16 arrays (sign-magnitude
    ordered to a monotonic int key, then abs difference)."""
    def key(v):
        u = v.view(np.uint16).astype(np.int32)
        neg = (u & 0x8000) != 0
        return np.where(neg, 0x8000 - (u & 0x7FFF), 0x8000 + (u & 0x7FFF))
    return np.abs(key(a_bf16) - key(b_bf16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    rng = np.random.RandomState(0)
    # match test.cpp's input range roughly: random_bfloat16(8.0, -4.0) -> [-4, 8)
    x = rng.uniform(-4.0, 8.0, size=(ROWS, COLS)).astype(bfloat16)

    ref_f = layernorm_ref_fp32(x)           # normalize-only LN in fp32
    ref_b = ref_f.astype(bfloat16)          # bf16-rounded truth
    print(f"[ref] normalize-only LN (gamma=1,beta=0) x{list(x.shape)} eps={EPS} "
          f"-> out[{ROWS},{COLS}] (fp32 truth)")

    X = np.ascontiguousarray(x).reshape(-1)
    if a.dry:
        print(f"[dry] X={X.nbytes}B Y={X.nbytes}B; "
              f"ref_fp32[0,:5]={ref_f[0,:5]}  row0 mean(out)={ref_f[0].mean():.4f} "
              f"std(out)={ref_f[0].std():.4f}")
        print("[dry] not touching the NPU.")
        return 0

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build: source scripts/iron_env.sh && "
                     f"make -C mlir-aie/programming_examples/ml/layernorm NPU2=1 "
                     f"rows={ROWS} cols={COLS}")
    instr = np.fromfile(a.insts, dtype=np.uint32)

    import pyxrt
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}")

    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(4))
    # dummy placeholders (0-size segfaults), matching xrt_test_wrapper.h 2-buf path
    bo_tmp = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(5))
    bo_ctrl = pyxrt.bo(d, 8, pyxrt.bo.host_only, k.group_id(6))
    bo_trace = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(7))

    bo_instr.write(instr.tobytes(), 0);              bo_instr.sync(TO)
    bo_x.write(X.view(np.uint16).tobytes(), 0);      bo_x.sync(TO)

    def once():
        r = k(3, bo_instr, instr.size, bo_x, bo_y, bo_tmp, bo_ctrl, bo_trace)
        r.wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_y.sync(FROM)
    Y = np.frombuffer(bo_y.read(X.nbytes, 0), dtype=np.uint16).view(bfloat16).reshape(ROWS, COLS)

    yf = Y.astype(np.float32)
    adiff = np.abs(yf - ref_f)                       # NPU vs fp32 truth
    denom = np.maximum(np.abs(ref_f), 1e-3)
    rel = adiff / denom
    ulp = bf16_ulp_distance(Y, ref_b)               # NPU vs bf16-rounded truth
    exact = int((ulp == 0).sum())
    within2 = int((ulp <= 2).sum())
    within4 = int((ulp <= 4).sum())
    N = ROWS * COLS

    # per-row check: a data-movement bug (a row fed to the wrong core / wrong cols)
    # shows as a few rows with large error; pure rounding is spread ~uniformly.
    per_row_max = adiff.max(axis=1)
    bad_rows = int((per_row_max > 0.1).sum())

    print(f"[run] device time/iter: {dt*1e3:.3f} ms  ({ROWS} rows x {COLS} cols)")
    print(f"[run] vs fp32 truth:  max|Δ|={adiff.max():.5f}  mean|Δ|={adiff.mean():.6f}  max_rel={rel.max():.4f}")
    print(f"[run] vs bf16 truth:  exact={exact}/{N} ({100*exact/N:.1f}%)  ≤2ulp={100*within2/N:.2f}%  ≤4ulp={100*within4/N:.2f}%  maxulp={int(ulp.max())}")
    print(f"[run] per-row:        rows with max|Δ|>0.1: {bad_rows}/{ROWS}  (0 => no mis-fed row)")
    print(f"[run] Y[0,:5]={yf[0,:5]}  ref={ref_f[0,:5]}")
    # PASS: the kernel uses a vectorized invsqrt approximation + bf16 intermediates,
    # so it's looser than dwconv's ~1 ULP. Match test.cpp's own bar (abs diff < 0.1)
    # plus a small relative bound and no systematically-wrong row.
    ok = (adiff.max() < 0.1) and (bad_rows == 0) and (np.median(rel) < 0.03)
    print(f"[run] LayerNorm [{ROWS},{COLS}] on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
