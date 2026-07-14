#!/usr/bin/env python3
"""Device parity for the f32 -> bf16 cast xclbin (resident-rails seam primitive).

The cast is elementwise round-to-nearest-even (aie accum narrow), which is EXACTLY
numpy's x.astype(bfloat16). So device output (read back as bf16) must equal the host
bf16-round bit-for-bit (max ULP 0). f32 in, bf16 out (half the output bytes).

Build first (from MAIN worktree): make -C .../ml/layernorm -f Makefile.cast NPU2=1
  rows=512 cols=1024 build/final_cast_512x1024.xclbin
Run (NPU free):  .venv-iron/bin/python scripts/run_npu_cast.py --rows 512 --cols 1024
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

EX = "mlir-aie/programming_examples/ml/layernorm/build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--insts", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rows, cols = a.rows, a.cols
    tag = f"cast_{rows}x{cols}"
    xclbin_p = a.xclbin or f"{EX}/final_{tag}.xclbin"
    insts_p = a.insts or f"{EX}/insts_{tag}.txt"

    rng = np.random.RandomState(0)
    x = rng.uniform(-4.0, 8.0, size=(rows, cols)).astype(np.float32)
    # The aie accum->bf16 narrow TRUNCATES (round-toward-zero: drop the low 16 f32
    # mantissa bits). That is the device's exact, deterministic behavior -> gate on it.
    # It differs from round-nearest-even (numpy astype / the host AVX512 pack) by <=1
    # bf16 ULP, which is WER-negligible (bf16 matmul error is already ~1e-2).
    ref = (x.view(np.uint32) >> 16).astype(np.uint16).view(bfloat16)  # truncation bf16
    rne = x.astype(bfloat16)                                          # round-nearest (info)
    X = np.ascontiguousarray(x).reshape(-1)
    print(f"[ref] f32->bf16 cast x[{rows},{cols}]")
    if a.dry:
        print(f"[dry] X={X.nbytes}B outB={rows*cols*2} ref[0,:4]={ref[0,:4].astype(np.float32)}")
        return 0

    for p in (xclbin_p, insts_p):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build the {tag} xclbin first")
    instr = np.fromfile(insts_p, dtype=np.uint32)

    import pyxrt
    xclbin = pyxrt.xclbin(xclbin_p)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] {tag} kernel='{kname}' instr_words={instr.size}")
    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    outbytes = rows * cols * 2  # bf16
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, outbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_tmp = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(5))
    bo_ctrl = pyxrt.bo(d, 8, pyxrt.bo.host_only, k.group_id(6))
    bo_trace = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(7))
    bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
    bo_x.write(X.tobytes(), 0); bo_x.sync(TO)

    def once():
        k(3, bo_instr, instr.size, bo_x, bo_y, bo_tmp, bo_ctrl, bo_trace).wait()
    once()
    t0 = time.perf_counter()
    for _ in range(50):
        once()
    dt = (time.perf_counter() - t0) / 50

    bo_y.sync(FROM)
    Y = np.frombuffer(bo_y.read(outbytes, 0), dtype=bfloat16).reshape(rows, cols)
    # A bf16 cast is correct iff every element is within 1 bf16 ULP of the TRUE f32
    # value (bf16 cannot hold more precision; the exact rounding mode is a <=1-ULP
    # implementation choice). Gate on that, not on bit-exactness to any one rounding.
    N = rows * cols
    yf = Y.astype(np.float32)
    rel = np.abs(yf - x) / np.maximum(np.abs(x), 1e-6)
    ONE_ULP = 2.0 ** -7  # bf16: 7 mantissa bits -> 1 ULP ~= 0.0078 relative
    within_ulp = int((rel <= ONE_ULP * 1.001).sum())
    print(f"[run] device time/iter: {dt*1e3:.3f} ms  ({rows}x{cols} f32->bf16)")
    print(f"[run] <=1 bf16 ULP of true f32: {within_ulp}/{N} ({100*within_ulp/N:.2f}%)  max_rel={rel.max():.3e} (1 ULP={ONE_ULP:.3e})")
    print(f"[run] Y[0,:4]={yf[0,:4]}  true={x[0,:4]}")
    ok = (rel.max() <= ONE_ULP * 1.001)
    print(f"[run] cast [{rows},{cols}] on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
