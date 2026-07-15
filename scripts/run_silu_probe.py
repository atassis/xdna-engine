#!/usr/bin/env python3
"""Standalone device validation for the SiLU brick (silu_brick.cc / silu_iron.py).

Runs the [rows,cols] f32 silu xclbin (built by Makefile.silu2 at a chosen SILU_MODE)
on-device and compares vs the EXACT fp32 silu golden  silu(x) = x/(1+e^-x), splitting
the per-row error EVEN vs ODD. For rows a multiple of (16*8), rows_per_core is even so
the GLOBAL row parity == each core's depth-2 output-objectfifo ping-pong parity -- the
signature the alt-channel/heavy-body miscompile corrupts (even rows = ping-pong buf 0).

Detects both faces of the codegen bug:
  HANG    : kernel does not complete within --timeout-ms (wait aborts).
  CORRUPT : EVEN rows garbage, ODD bit-exact (or vice versa).
CLEAN + accurate: EVEN 0 / ODD 0, rel-L2 << 7e-3 (below the bf16-tanh floor).

ABI mirrors silu_iron.py: opcode 3, instr[gid1,cacheable], in[gid3], out[gid4].

Usage (NPU must be QUIESCED):
  build:  source ../xdna-engine/scripts/iron_env.sh
          rm -f <layernorm>/build/silu_brick.o
          make -C <layernorm> -f Makefile.silu2 NPU2=1 rows=1024 cols=400 silu_mode=8 \
               build/final_silu_1024x400.xclbin
  run:    ../xdna-engine/.venv-iron/bin/python scripts/run_silu_probe.py \
               --xclbin <layernorm>/build/final_silu_1024x400.xclbin \
               --insts  <layernorm>/build/insts_silu_1024x400.txt --rows 1024 --cols 400
"""
import argparse, os, sys, time
import numpy as np


def silu_exact(x):
    x = x.astype(np.float64)
    return (x / (1.0 + np.exp(-x))).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=400)
    ap.add_argument("--mode", type=int, default=-1, help="SILU_MODE built (report only)")
    ap.add_argument("--dist", default="normal", choices=("normal", "uniform6", "wide"))
    ap.add_argument("--timeout-ms", type=int, default=8000)
    a = ap.parse_args()

    rng = np.random.RandomState(0)
    if a.dist == "normal":
        x = rng.standard_normal(size=(a.rows, a.cols)).astype(np.float32)
    elif a.dist == "uniform6":
        x = rng.uniform(-6.0, 6.0, size=(a.rows, a.cols)).astype(np.float32)
    else:  # wide: N(0,1) body + heavy tail to stress sigmoid saturation
        x = (rng.standard_normal(size=(a.rows, a.cols)) * 2.5).astype(np.float32)
    ref = silu_exact(x)
    X = np.ascontiguousarray(x).reshape(-1)

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    instr = np.fromfile(a.insts, dtype=np.uint32)

    import pyxrt
    xb = pyxrt.xclbin(a.xclbin)
    kname = xb.get_kernels()[0].get_name()
    print(f"[artifacts] mode={a.mode} kernel='{kname}' instr_words={instr.size} "
          f"shape={a.rows}x{a.cols} dist={a.dist}")
    d = pyxrt.device(0)
    d.register_xclbin(xb)
    ctx = pyxrt.hw_context(d, xb.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    nbytes = X.nbytes
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
    bo_x.write(X.tobytes(), 0);         bo_x.sync(TO)

    t0 = time.perf_counter()
    r = k(3, bo_instr, instr.size, bo_x, bo_y)
    try:
        state = r.wait(a.timeout_ms)
    except RuntimeError as e:
        print(f"[run] r.wait raised: {e}  => kernel did NOT complete (HANG/abort)")
        print(f"[result] mode={a.mode}: HANG")
        return 2
    dt = time.perf_counter() - t0
    scode = str(state)
    completed = ("COMPLETED" in scode) or (scode in ("4", "ert_cmd_state.ERT_CMD_STATE_COMPLETED"))
    if not completed:
        print(f"[run] wait state={scode} within {a.timeout_ms} ms => HANG")
        print(f"[result] mode={a.mode}: HANG")
        return 2

    bo_y.sync(FROM)
    Y = np.frombuffer(bo_y.read(nbytes, 0), dtype=np.float32).reshape(a.rows, a.cols).astype(np.float64)
    adiff = np.abs(Y - ref)
    per_row = adiff.mean(axis=1)
    relL2 = np.linalg.norm(Y - ref) / max(np.linalg.norm(ref), 1e-12)
    thr = 0.05 * max(np.abs(ref).mean(), 1e-3) + 0.02
    even_bad = int((per_row[0::2] > thr).sum())
    odd_bad = int((per_row[1::2] > thr).sum())
    print(f"[run] completed {dt*1e3:.3f} ms. rel-L2={relL2:.4e}  max|d|={adiff.max():.4f}")
    print(f"[run] per-row mean|d|>{thr:.3g}: EVEN {even_bad}/{a.rows//2}  ODD {odd_bad}/{a.rows//2}")
    print(f"[run] Y[0,:4]={Y[0,:4]}  ref={ref[0,:4]}")
    print(f"[run] Y[1,:4]={Y[1,:4]}  ref={ref[1,:4]}")
    clean = (even_bad == 0 and odd_bad == 0)
    accurate = relL2 < 7e-3
    if clean and accurate:
        print(f"[result] mode={a.mode}: CLEAN+EXACT (rel-L2 {relL2:.2e}, EVEN 0 ODD 0)")
        return 0
    if clean:
        print(f"[result] mode={a.mode}: CLEAN but rel-L2 {relL2:.2e} >= 7e-3 (not exact)")
        return 3
    print(f"[result] mode={a.mode}: CORRUPT  EVEN {even_bad}  ODD {odd_bad}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
