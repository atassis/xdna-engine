#!/usr/bin/env python3
"""Run + validate bf16 softmax on the XDNA2 NPU via pyxrt.

The mlir-aie softmax example (programming_examples/ml/softmax) softmaxes the input
PER TILE: the tensor of N elements is split into contiguous tiles of length n=1024
(2 cores x 128 tiles), and each tile is softmaxed independently over its 1024 keys.
It is NOT a whole-buffer softmax and it is NOT (yet) a length-400 row softmax — see
the ATTENTION CAVEAT below.

We mirror the kernel math in numpy at bf16 fidelity: scale by log2e, subtract the
per-tile max, exp2, sum in fp32, then multiply each element by inv(sum) rounded to
bf16 (kernels/aie2p/softmax.cc). Comparison is row-relative with the same tolerance
the upstream test.cpp uses (rel 0.04 / abs 0.001).

IRON host ABI (from softmax/test.cpp): opcode=3; softmax has only IN and OUT (no
weight buffer, unlike dwconv1d):
  kernel(opcode, instr[gid1,cacheable], n_instr, IN[gid3], OUT[gid4])

ATTENTION CAVEAT (GigaAM [16*400, 400] = [6400, 400] row softmax):
  The attention need is row-softmax over length 400, for 6400 rows. This kernel's
  per-tile length n is hardcoded to 1024 in softmax.py, AND the kernel reduces in
  steps of SM_VEC_LEN=32 (elem_iters = vector_size / 32). 400 % 32 != 0, so a
  length-400 tile is impossible without changing the kernel. This runner therefore
  validates the EXISTING (n=1024) granularity. To serve attention you must either
  (a) edit softmax.py's `n` to a %32 length and pad/reshape 400->that length
  (changes the softmax denominator unless masked), or (b) modify softmax.cc to
  handle a remainder tail. See report.

Usage:
  .venv-iron/bin/python scripts/run_npu_softmax.py --dry   # validate refs, no NPU
  .venv-iron/bin/python scripts/run_npu_softmax.py         # REAL run (NPU must be free)
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

# Must match softmax.py (N total, n per-tile reduction length). The xclbin is built
# for these exact sizes; changing them here without rebuilding the xclbin will fail.
N = 262144     # total elements in the buffer (softmax.py --size)
ROW = 1024     # per-tile softmax length (n in softmax.py); 400-row attention != this
LOG2E = 1.4453125  # the bf16-rounded log2e the kernel actually uses (softmax.cc)
EX = "mlir-aie/programming_examples/ml/softmax/build"


def softmax_ref_bf16(x_bf16):
    """Per-row (per-tile) softmax mirroring softmax.cc at bf16 fidelity.

    x_bf16: [rows, ROW] bf16. Returns ([rows, ROW] fp32 truth, [rows, ROW] bf16 truth).
    Math (per row): scale s = x * log2e (fp32), m = max(s), e = exp2(s - m) rounded to
    bf16, S = sum(e) in fp32, out = e * bf16(1/S) rounded to bf16. This matches the
    kernel's exp2/log2e formulation rather than the plain exp() in test.cpp; the two
    differ only by bf16 rounding."""
    x = x_bf16.astype(np.float32)
    s = (x * LOG2E)                          # scale (the kernel keeps this in fp/accum)
    m = s.max(axis=1, keepdims=True)
    e = np.exp2(s - m).astype(bfloat16).astype(np.float32)  # bf16-rounded exp2
    S = e.sum(axis=1, keepdims=True)         # fp32 accumulate
    inv = (1.0 / S).astype(bfloat16).astype(np.float32)     # bf16 reciprocal (col_sum_inv)
    out_f = (e * inv)
    out_b = out_f.astype(bfloat16)
    return out_f.astype(np.float32), out_b


def main():
    rows = N // ROW
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    rng = np.random.RandomState(0)
    # bf16 inputs in a realistic attention-score range (test.cpp uses [-512,512] on
    # NPU2; we keep it modest so exp2 doesn't saturate the bf16 reference trivially).
    x = rng.uniform(-8.0, 8.0, size=(rows, ROW)).astype(bfloat16)

    ref_f, ref_b = softmax_ref_bf16(x)       # per-row softmax (length ROW)
    print(f"[ref] x[{rows},{ROW}] bf16 -> per-row softmax over {ROW} keys ({rows} rows)")
    print(f"[ref] row0 sum(out)={ref_f[0].sum():.4f} (should be ~1.0)")

    X = np.ascontiguousarray(x).reshape(-1)
    if a.dry:
        print(f"[dry] N={N} ROW={ROW} rows={rows}  X={X.nbytes}B Y={X.nbytes}B")
        print(f"[dry] ref_fp32[0,:5]={ref_f[0,:5]}  bf16[0,:5]={ref_b[0,:5].astype(np.float32)}")
        print("[dry] NOTE granularity = per-tile softmax over ROW=1024, NOT length-400.")
        print("[dry] not touching the NPU.")
        return 0

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build: (cd {os.path.dirname(EX)} && make NPU2=1)")
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

    # softmax ABI: IN at gid3, OUT at gid4 (no weight buffer).
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(4))

    bo_instr.write(instr.tobytes(), 0);              bo_instr.sync(TO)
    bo_x.write(X.view(np.uint16).tobytes(), 0);      bo_x.sync(TO)

    def once():
        r = k(3, bo_instr, instr.size, bo_x, bo_y)
        r.wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_y.sync(FROM)
    Y = np.frombuffer(bo_y.read(X.nbytes, 0), dtype=np.uint16).view(bfloat16).reshape(rows, ROW)

    yf = Y.astype(np.float32)
    adiff = np.abs(yf - ref_f)
    denom = np.maximum(np.abs(ref_f), 1e-3)
    rel = adiff / denom
    # per-row sum check: every row should sum to ~1 (a broken tile shows up here)
    row_sums = yf.sum(axis=1)
    bad_rows = int((np.abs(row_sums - 1.0) > 0.05).sum())
    # upstream test.cpp tolerance: nearly_equal(rel=0.04, abs=0.001)
    fail_elems = int(((adiff > 0.001) & (rel > 0.04)).sum())
    tot = rows * ROW

    print(f"[run] device time/iter: {dt*1e3:.3f} ms  ({rows} rows x {ROW} keys)")
    print(f"[run] vs bf16 truth:  max|Δ|={adiff.max():.6f}  mean|Δ|={adiff.mean():.7f}  max_rel={rel.max():.4f}")
    print(f"[run] tolerance fails (rel>0.04 & abs>0.001): {fail_elems}/{tot}")
    print(f"[run] per-row sum:    rows with |sum-1|>0.05: {bad_rows}/{rows}  (0 => all rows normalized)")
    print(f"[run] Y[0,:5]={yf[0,:5]}  ref={ref_f[0,:5]}")
    ok = (fail_elems == 0) and (bad_rows == 0)
    print(f"[run] bf16 softmax (per-tile, ROW={ROW}) on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
