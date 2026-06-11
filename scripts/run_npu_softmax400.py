#!/usr/bin/env python3
"""Run + validate per-row length-400 bf16 softmax on the XDNA2 NPU via pyxrt.

GigaAM attention needs a per-row softmax over length 400 for [16 heads * 400 queries,
400 keys] = [6400, 400] (6400 independent rows, each length 400). The mlir-aie softmax
kernel reduces in 32-wide vectors (SM_VEC_LEN=32) and 400 % 32 != 0, so a length-400
tile would truncate to 384 keys (wrong denominator).

FIX (no kernel change): pad each row's key dim 400 -> 416 (= 13*32) with -1e30. Then
exp2(pad*log2e - rowmax) underflows to 0 in bf16, so the padded columns add nothing to
the softmax denominator and contribute 0 to the output. The first 400 columns of a
length-416 softmax are therefore bit-exact equal to a true length-400 softmax (proven
below in --dry). The xclbin is built for per-tile (== per-row) softmax with ROW=416;
the host pads on the way in and slices cols[:400] on the way out.

The xclbin/insts come from programming_examples/ml/softmax400 (softmax400.py), which is
softmax.py with the per-tile reduction length set to ROW=416 instead of 1024. Same
kernel (aie_kernels/aie2p/softmax.cc), invoked as softmax_bf16(in, out, 416).

Tiling: ROW=416, n_cores=2, total N = ROW * 6400 = 2,662,400 bf16 elements => all
6400 attention rows in ONE dispatch (3200 rows per core). To process fewer rows, build
with `size=ROW*R` (R divisible by 2) and pass --rows R here.

IRON host ABI (mirrors softmax/test.cpp and run_npu_softmax.py): opcode=3; softmax has
only IN and OUT (no weight buffer):
  kernel(opcode=3, instr[gid1, cacheable], n_instr, IN[gid3], OUT[gid4])
bf16 buffers are written/read as uint16 (.view(np.uint16)).

Usage:
  .venv-iron/bin/python scripts/run_npu_softmax400.py --dry  # validate refs, no NPU
  .venv-iron/bin/python scripts/run_npu_softmax400.py        # REAL run (NPU must be free)
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

REAL = 400          # real key dim of the attention scores
ROW = 416           # padded per-row softmax length (13*32); must match softmax400 build
PAD = ROW - REAL    # 16 pad columns
PADVAL = -1e30      # large negative so exp2(pad - max) underflows to 0
ROWS = 6400         # total attention rows the default xclbin covers (16 heads * 400 q)
LOG2E = 1.4453125   # bf16-rounded log2e the kernel actually uses (softmax.cc)
EX = "mlir-aie/programming_examples/ml/softmax400/build"


def softmax_ref_bf16(x_bf16, length):
    """Per-row softmax over the first `length` columns, mirroring softmax.cc at bf16
    fidelity. x_bf16: [rows, cols] bf16. Reduces only over cols[:length].
    Math (per row): s = x*log2e (fp32), m = max(s[:length]), e = exp2(s - m) bf16-rounded,
    S = sum(e[:length]) fp32, out = e * bf16(1/S) bf16-rounded."""
    x = x_bf16.astype(np.float32)
    s = x * LOG2E
    m = s[:, :length].max(axis=1, keepdims=True)
    e = np.exp2(s - m).astype(bfloat16).astype(np.float32)
    S = e[:, :length].sum(axis=1, keepdims=True)
    inv = (1.0 / S).astype(bfloat16).astype(np.float32)
    out_f = (e * inv)
    return out_f.astype(np.float32), out_f.astype(bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--rows", type=int, default=ROWS,
                    help=f"rows to process (default {ROWS}; must match the xclbin size=ROW*rows)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    R = a.rows

    rng = np.random.RandomState(0)
    # bf16 attention scores in a realistic range; modest so exp2 doesn't trivially saturate.
    scores = rng.uniform(-8.0, 8.0, size=(R, REAL)).astype(bfloat16)

    # Host PAD: 400 -> 416 with -1e30 on the 16 pad columns.
    pad = np.full((R, PAD), PADVAL, dtype=np.float32).astype(bfloat16)
    padded = np.concatenate([scores, pad], axis=1)  # [R, 416] bf16

    # numpy reference computed two ways:
    #  true400  = softmax over the 400 REAL keys (length-400 reduction)
    #  pad416   = softmax over the full padded ROW=416, then sliced cols[:400]
    true_f, true_b = softmax_ref_bf16(scores, REAL)           # [R, 400]
    pad_f, pad_b = softmax_ref_bf16(padded, ROW)              # [R, 416]
    pad_sliced_f = pad_f[:, :REAL]                            # [R, 400]

    # Equivalence check: padded-416 cols[:400] must equal true length-400 softmax.
    equiv_diff = np.abs(pad_sliced_f - true_f)
    equiv = bool(np.array_equal(pad_sliced_f, true_f))
    pad_tail_max = float(np.abs(pad_f[:, REAL:]).max())       # should be ~0

    print(f"[ref] scores[{R},{REAL}] bf16 -> pad to [{R},{ROW}] with {PADVAL} on {PAD} cols")
    print(f"[ref] true length-400 softmax: row0 sum={true_f[0].sum():.4f} (should be ~1.0)")
    print(f"[ref] pad-416 softmax cols[:400]: row0 sum={pad_sliced_f[0].sum():.4f}")
    print(f"[ref] pad tail cols[400:416] max|val|={pad_tail_max:.3e} (should be ~0)")
    print(f"[ref] pad-416 vs true-400: max|Δ|={equiv_diff.max():.3e}  bitwise_equal={equiv}")

    if a.dry:
        print(f"[dry] ROW={ROW} REAL={REAL} PAD={PAD} rows={R}  "
              f"in={padded.nbytes}B out={padded.nbytes}B")
        print(f"[dry] true400[0,:5]   ={true_f[0,:5]}")
        print(f"[dry] pad416[0,:5]    ={pad_sliced_f[0,:5]}")
        print(f"[dry] EQUIVALENCE (pad-416 cols[:400] == length-400 softmax): "
              f"{'CONFIRMED' if equiv else 'MISMATCH'}")
        print("[dry] not touching the NPU.")
        return 0 if equiv else 1

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build: (cd {os.path.dirname(EX)}/.. && "
                     f"make NPU2=1 row={ROW} size={ROW*R} build/final.xclbin)")
    instr = np.fromfile(a.insts, dtype=np.uint32)

    # Flatten padded [R,416] row-major; the design softmaxes each contiguous 416-tile.
    X = np.ascontiguousarray(padded).reshape(-1)

    import pyxrt
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size} X_elems={X.size}")

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
    Y = np.frombuffer(bo_y.read(X.nbytes, 0), dtype=np.uint16).view(bfloat16).reshape(R, ROW)
    yf = Y.astype(np.float32)

    # UNPAD: keep only the first 400 columns of each row.
    y400 = yf[:, :REAL]

    adiff = np.abs(y400 - true_f)
    denom = np.maximum(np.abs(true_f), 1e-3)
    rel = adiff / denom
    row_sums = y400.sum(axis=1)
    bad_rows = int((np.abs(row_sums - 1.0) > 0.05).sum())
    fail_elems = int(((adiff > 0.001) & (rel > 0.04)).sum())
    tot = R * REAL

    print(f"[run] device time/iter: {dt*1e3:.3f} ms  ({R} rows x {REAL} real keys, ROW={ROW})")
    print(f"[run] vs length-400 truth: max|Δ|={adiff.max():.6f} mean|Δ|={adiff.mean():.7f} max_rel={rel.max():.4f}")
    print(f"[run] tolerance fails (rel>0.04 & abs>0.001): {fail_elems}/{tot}")
    print(f"[run] per-row sum (cols[:400]): rows with |sum-1|>0.05: {bad_rows}/{R} (0 => all normalized)")
    print(f"[run] Y[0,:5]={y400[0,:5]}  ref={true_f[0,:5]}")
    ok = (fail_elems == 0) and (bad_rows == 0)
    print(f"[run] per-row length-400 softmax (pad-416) on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
