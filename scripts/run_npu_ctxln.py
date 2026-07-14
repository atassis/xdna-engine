#!/usr/bin/env python3
"""Device parity for the f32 two-pass ctxLN xclbin (resident-rails Task 2, D-parameterized).

Unlike run_npu_layernorm.py (bf16 `layer_norm`, E[x^2]-mean^2), this drives the f32
`layer_norm_2pass_f32` kernel (route_b_kernels/aie_kernels/ln_2pass.cc): NORMALIZE-ONLY,
per-row TWO-PASS centered variance, f32 in / f32 out. Matches the host reference exactly
(npu-asr-host layer_norm_normalize); ctx_ln.rs measured rel ~7.8e-7 at D=768.

Build the xclbin first (from the MAIN worktree, CPU-only):
  source scripts/iron_env.sh
  make -C mlir-aie/programming_examples/ml/layernorm -f Makefile.ctxln NPU2=1 \
       rows=ROWS cols=COLS build/final_ctxln_ROWSxCOLS.xclbin

Usage (NPU must be free):
  .venv-iron/bin/python scripts/run_npu_ctxln.py --rows 512 --cols 1024
  .venv-iron/bin/python scripts/run_npu_ctxln.py --rows 512 --cols 1024 --dry   # no NPU
"""
import argparse, os, sys, time
import numpy as np

EPS = 1e-5
EX = "mlir-aie/programming_examples/ml/layernorm/build"


def ln_2pass_ref(x):
    """f32 two-pass centered normalize-only LN (gamma=1, beta=0), the ln_2pass.cc math."""
    x = x.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)   # centered two-pass, /cols
    inv = 1.0 / np.sqrt(var + EPS)
    return (x - mean) * inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--insts", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rows, cols = a.rows, a.cols
    tag = f"ctxln_{rows}x{cols}"
    xclbin_p = a.xclbin or f"{EX}/final_{tag}.xclbin"
    insts_p = a.insts or f"{EX}/insts_{tag}.txt"

    rng = np.random.RandomState(0)
    x = rng.uniform(-4.0, 8.0, size=(rows, cols)).astype(np.float32)
    ref = ln_2pass_ref(x)
    X = np.ascontiguousarray(x).reshape(-1)
    print(f"[ref] f32 2-pass normalize-only LN x[{rows},{cols}] eps={EPS}")

    if a.dry:
        print(f"[dry] X={X.nbytes}B ref[0,:5]={ref[0,:5]} row0 mean={ref[0].mean():.4f} std={ref[0].std():.4f}")
        print("[dry] not touching the NPU.")
        return 0

    for p in (xclbin_p, insts_p):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build the {tag} xclbin first (see module docstring)")
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

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_tmp = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(5))
    bo_ctrl = pyxrt.bo(d, 8, pyxrt.bo.host_only, k.group_id(6))
    bo_trace = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(7))

    bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
    bo_x.write(X.tobytes(), 0); bo_x.sync(TO)          # f32 bytes

    def once():
        k(3, bo_instr, instr.size, bo_x, bo_y, bo_tmp, bo_ctrl, bo_trace).wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_y.sync(FROM)
    Y = np.frombuffer(bo_y.read(X.nbytes, 0), dtype=np.float32).reshape(rows, cols)

    adiff = np.abs(Y - ref)
    rel = adiff / np.maximum(np.abs(ref), 1e-3)
    per_row_max = adiff.max(axis=1)
    bad_rows = int((per_row_max > 1e-2).sum())
    print(f"[run] device time/iter: {dt*1e3:.3f} ms  ({rows} rows x {cols} cols, f32)")
    print(f"[run] vs f32 2-pass ref:  max|Δ|={adiff.max():.3e}  mean|Δ|={adiff.mean():.3e}  max_rel={rel.max():.3e}")
    print(f"[run] per-row: rows with max|Δ|>1e-2: {bad_rows}/{rows} (0 => no mis-fed row)")
    print(f"[run] Y[0,:4]={Y[0,:4]}  ref={ref[0,:4]}")
    ok = (rel.max() < 1e-2) and (bad_rows == 0)
    print(f"[run] ctxLN [{rows},{cols}] f32 on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
