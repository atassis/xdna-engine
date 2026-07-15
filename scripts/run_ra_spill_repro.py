#!/usr/bin/env python3
"""Standalone device validation for the minimal RA/spill-around-call repro.

Mirrors run_npu_silu.py's ABI (opcode 3, in=gid3, out=gid4). Runs the tiny
[rows,cols] f32 kernel built from route_b_kernels/probes and compares vs an fp32
host golden, splitting the per-row error EVEN vs ODD (even row == ping-pong buffer
0 of the depth-2 output objectfifo). Detects the two faces of the bug:

  --hold 1  (REPRO)   : expect HANG (wait timeout) OR even-row corruption.
  --hold 0  (CONTROL) : expect COMPLETE + clean (matches golden within f32-emul noise).

The kernel's f32 mul/add on aie2p go through bf16 mac chains, so a "clean" run
still carries ~1e-2 relative noise; a miscompile blows past O(1) or hangs.

Usage (NPU must be QUIESCED):
  .venv-iron/bin/python scripts/run_ra_spill_repro.py --hold 1
  .venv-iron/bin/python scripts/run_ra_spill_repro.py --hold 0
"""
import argparse, os, sys, time
import numpy as np

EX = "route_b_kernels/probes/build"


def heavy16(x):
    a = x * x
    b = a + x
    c = b * a
    return c + b


def golden(x, hold):
    x = x.astype(np.float32)
    s = heavy16(x)
    if hold:
        r0 = 1.0 / (1.0 + s)
        num = np.where(x < 2.0, s, 1.0)   # aie::select(one, s, lt(x,two)) == (x<2)?s:1
        sig = num * r0
        return x * sig
    else:
        return 1.0 / (1.0 + s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=1, choices=(0, 1))
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--timeout-ms", type=int, default=8000)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    tag = f"raspill_h{a.hold}_{a.rows}x{a.cols}"
    xclbin = f"{EX}/final_{tag}.xclbin"
    insts = f"{EX}/insts_{tag}.txt"

    rng = np.random.RandomState(0)
    x = rng.standard_normal(size=(a.rows, a.cols)).astype(np.float32)  # x ~ N(0,1)
    ref = golden(x, a.hold)
    print(f"[ref] x[{a.rows},{a.cols}] f32 -> golden (hold={a.hold})")

    X = np.ascontiguousarray(x).reshape(-1)
    if a.dry:
        for p in (xclbin, insts):
            print(f"[dry] {'OK ' if os.path.exists(p) else 'MISSING'} {p}")
        return 0
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build: source ../xdna-engine/scripts/iron_env.sh && "
                     f"make -C route_b_kernels/probes -f Makefile.raspill NPU2=1 HOLD={a.hold} all")
    instr = np.fromfile(insts, dtype=np.uint32)

    import pyxrt
    xb = pyxrt.xclbin(xclbin)
    kname = xb.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}")
    d = pyxrt.device(0)
    d.register_xclbin(xb)
    ctx = pyxrt.hw_context(d, xb.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    nbytes = X.nbytes  # f32
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
    bo_x.write(X.tobytes(), 0);         bo_x.sync(TO)

    r = k(3, bo_instr, instr.size, bo_x, bo_y)
    # hang detection: bounded wait. pyxrt run.wait(timeout_ms) returns the ert state;
    # if it does not reach COMPLETED within the budget we call it a HANG.
    state = None
    try:
        state = r.wait(a.timeout_ms)   # newer XRT: ms timeout, returns ert_cmd_state
    except TypeError:
        # older binding: wait() blocks; wrap with wall-clock guard is not possible here.
        t0 = time.perf_counter()
        r.wait()
        state = "COMPLETED" if (time.perf_counter() - t0) * 1e3 < a.timeout_ms else "TIMEOUT"
    except RuntimeError as e:
        # XRT aborts the command to an error/timeout state (e.g. "qds_device::wait()
        # unexpected command state") -- the kernel did NOT reach COMPLETED = the hang face.
        print(f"[run] r.wait raised: {e}  => kernel did NOT complete (HANG/abort)")
        print(f"[result] hold={a.hold}: HANG")
        return 2
    scode = str(state)
    completed = ("COMPLETED" in scode) or (scode in ("4", "ert_cmd_state.ERT_CMD_STATE_COMPLETED"))
    if not completed:
        print(f"[run] wait state={scode} within {a.timeout_ms} ms  => HANG (kernel did not complete)")
        print(f"[result] hold={a.hold}: HANG")
        return 2

    bo_y.sync(FROM)
    Y = np.frombuffer(bo_y.read(nbytes, 0), dtype=np.float32).reshape(a.rows, a.cols)
    adiff = np.abs(Y - ref)
    per_row = adiff.mean(axis=1)
    relL2 = np.linalg.norm(Y - ref) / max(np.linalg.norm(ref), 1e-9)
    thr = 0.05 * max(np.abs(ref).mean(), 1e-3) + 0.05  # loose: >> f32-emul noise
    even_bad = int((per_row[0::2] > thr).sum())
    odd_bad = int((per_row[1::2] > thr).sum())
    print(f"[run] completed. rel-L2={relL2:.4e}  max|d|={adiff.max():.4f}")
    print(f"[run] per-row mean|d|>{thr:.3g}:  EVEN {even_bad}/{a.rows//2}   ODD {odd_bad}/{a.rows//2}")
    print(f"[run] Y[0,:4]={Y[0,:4]}  ref={ref[0,:4]}")
    print(f"[run] Y[1,:4]={Y[1,:4]}  ref={ref[1,:4]}")
    if even_bad == 0 and odd_bad == 0:
        print(f"[result] hold={a.hold}: CLEAN (rel-L2 {relL2:.2e})")
        return 0
    print(f"[result] hold={a.hold}: CORRUPT  EVEN {even_bad}/{a.rows//2}  ODD {odd_bad}/{a.rows//2}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
