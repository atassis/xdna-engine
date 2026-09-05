#!/usr/bin/env python3
"""Round 30/31 minimal repro for the energy-phase stall
(decode energy stalls in poll after the quiesce gate, measured 2026-08-31). Builds
decode_step_resident ONCE, then dispatches it under one of three isolated variants, each changing
exactly ONE thing relative to the Round 30 baseline (2 bare calls, no refill, resident only --
REFUTED, both calls returned). Prints and flushes around every dispatch: the diagnostic value is
entirely in knowing which call, if any, does not return.

Usage: python3 min_repro_resident_twice.py --variant {refill,n,hostloop,sensor} [--order after|before]
  refill   : 2 calls, `_fill(hwt, hw0)` before EACH -- matches what the real run() loop does that
             the Round 30 baseline did not (isolates the re-fill variable alone, still N=2).
  n        : N=32 bare calls (no refill, same buffer reused) -- isolates dispatch COUNT alone.
  hostloop : build decode_step_hostloop (JIT + its own warm dispatch) between resident's two bare
             calls, mirroring energy()'s `runners = {}` dict-literal build order -- isolates
             whether a second design's build/warm-up in the same process is the interferer.
  sensor   : construct npu_sensors.SensorReader() (a second fd on /dev/accel/accel0, then one
             ioctl-based power_mw() sample) around an already-warmed resident context, printing
             open/sample/dispatch as three SEPARATE steps so a hang localizes to exactly one.
             --order after (default, matches energy()'s own call order): warm resident, THEN
             open+sample+dispatch-again. --order before: open+sample on a bare device BEFORE any
             design is built/warmed, then build+warm+dispatch -- isolates whether the reader is
             harmless without a context and only fatal once one exists.
"""
import argparse, sys, time
import numpy as np
import ml_dtypes

from bricklib import GEN, iron  # noqa: E402
from probe_decode_loop_onchip import (  # noqa: E402
    _build, _fill, make_tables, pack_hw, PACK, STEPS, HW_N, EMB_N, D,
)


def flush_print(*a):
    print(*a, flush=True)
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=("refill", "n", "hostloop", "sensor"))
    ap.add_argument("--order", default="after", choices=("after", "before", "concurrent"),
                    help="variant=sensor only: reader before or after the resident context exists")
    ap.add_argument("--n", type=int, default=32, help="variant=n: how many bare calls")
    a = ap.parse_args()

    flush_print("[min-repro] variant=%s" % a.variant)
    flush_print("[min-repro] building decode_step_resident (STEPS=%d PACK=%d)..." % (STEPS, PACK))
    d = _build("decode_step_resident", STEPS, PACK, True)
    flush_print("[min-repro] build done")

    emb, weight, hidden0 = make_tables(20260817)
    hw0 = pack_hw(hidden0, weight)
    emb_t = iron.tensor(np.ascontiguousarray(np.asarray(emb).reshape(-1)),
                        dtype=ml_dtypes.bfloat16, device="npu")
    hwt = iron.tensor(np.ascontiguousarray(hw0), dtype=ml_dtypes.bfloat16, device="npu")
    ot = iron.zeros((STEPS * PACK,), dtype=np.float32, device="npu")
    flush_print("[min-repro] tensors ready")

    if a.variant == "refill":
        for i in range(2):
            flush_print("[min-repro] refill %d" % i)
            _fill(hwt, hw0)
            flush_print("[min-repro] dispatching call %d (with refill)" % i)
            t0 = time.monotonic()
            d(hwt, emb_t, ot)
            flush_print("[min-repro] CALL %d RETURNED after %.3fs" % (i, time.monotonic() - t0))
        flush_print("[min-repro] ALL CALLS COMPLETED (refill variant) -- hypothesis REFUTED")

    elif a.variant == "n":
        for i in range(a.n):
            flush_print("[min-repro] dispatching bare call %d/%d" % (i, a.n))
            t0 = time.monotonic()
            d(hwt, emb_t, ot)
            flush_print("[min-repro] CALL %d RETURNED after %.3fs" % (i, time.monotonic() - t0))
        flush_print("[min-repro] ALL %d CALLS COMPLETED (n variant) -- hypothesis REFUTED" % a.n)

    elif a.variant == "hostloop":
        flush_print("[min-repro] dispatching resident call 0 (bare)")
        t0 = time.monotonic()
        d(hwt, emb_t, ot)
        flush_print("[min-repro] CALL 0 RETURNED after %.3fs" % (time.monotonic() - t0))

        flush_print("[min-repro] building decode_step_hostloop (mirrors energy()'s runners={})...")
        d2 = _build("decode_step_hostloop", 1, PACK + D, False)
        flush_print("[min-repro] hostloop build done, constructing its tensors + warm dispatch")
        hwt2 = iron.tensor(np.ascontiguousarray(pack_hw(hidden0, weight)),
                           dtype=ml_dtypes.bfloat16, device="npu")
        ob = iron.zeros((PACK + D,), dtype=np.float32, device="npu")
        t1 = time.monotonic()
        d2(hwt2, emb_t, ob)
        flush_print("[min-repro] HOSTLOOP WARM DISPATCH RETURNED after %.3fs" % (time.monotonic() - t1))

        flush_print("[min-repro] dispatching resident call 1 (bare, after hostloop is built+warm)")
        t2 = time.monotonic()
        d(hwt, emb_t, ot)
        flush_print("[min-repro] CALL 1 RETURNED after %.3fs" % (time.monotonic() - t2))
        flush_print("[min-repro] ALL CALLS COMPLETED (hostloop variant) -- hypothesis REFUTED")


    elif a.variant == "sensor":
        from probe_decode_loop_onchip import npu_sensors  # noqa: E402  (path set up by that import)

        def open_sample_dispatch(label):
            flush_print("[min-repro] [%s] opening SensorReader (os.open on /dev/accel/accel0)..." % label)
            t_open = time.monotonic()
            npu = npu_sensors.SensorReader()
            flush_print("[min-repro] [%s] SensorReader OPENED after %.3fs" % (label, time.monotonic() - t_open))

            flush_print("[min-repro] [%s] taking one power_mw() sample (the ioctl)..." % label)
            t_ioctl = time.monotonic()
            v = npu.power_mw()
            flush_print("[min-repro] [%s] power_mw() RETURNED (%s) after %.3fs" % (label, v, time.monotonic() - t_ioctl))
            npu.close()
            flush_print("[min-repro] [%s] SensorReader closed" % label)

            flush_print("[min-repro] [%s] dispatching resident (the call AFTER the reader)..." % label)
            t_d = time.monotonic()
            d(hwt, emb_t, ot)
            flush_print("[min-repro] [%s] DISPATCH RETURNED after %.3fs" % (label, time.monotonic() - t_d))

        if a.order == "before":
            # Reader BEFORE any design is built/warmed -- no hw-context exists yet.
            flush_print("[min-repro] --order before: sampling on a bare device, no context yet")
            flush_print("[min-repro] opening SensorReader (bare device)...")
            t_open = time.monotonic()
            npu = npu_sensors.SensorReader()
            flush_print("[min-repro] SensorReader OPENED after %.3fs" % (time.monotonic() - t_open))
            flush_print("[min-repro] taking one power_mw() sample (bare device)...")
            t_ioctl = time.monotonic()
            v = npu.power_mw()
            flush_print("[min-repro] power_mw() RETURNED (%s) after %.3fs" % (v, time.monotonic() - t_ioctl))
            npu.close()
            flush_print("[min-repro] SensorReader closed (bare-device phase done, no hang)")
            flush_print("[min-repro] NOW building+warming resident (context created AFTER a clean bare-device sample)")
            d(hwt, emb_t, ot)
            flush_print("[min-repro] warm dispatch (post-bare-sample) RETURNED")
            flush_print("[min-repro] ALL STEPS COMPLETED (sensor variant, order=before) -- hypothesis REFUTED")
        elif a.order == "concurrent":
            # Closest match to _PowerLog's actual mechanism: ONE SensorReader, left OPEN, a
            # background thread sampling power_mw() on a tight interval, CONCURRENT with
            # dispatches on the main thread -- the one axis no prior variant (or the sequential
            # sensor tests earlier this round) exercised.
            import threading
            flush_print("[min-repro] --order concurrent: one persistent SensorReader + a sampling thread")
            npu = npu_sensors.SensorReader()
            flush_print("[min-repro] SensorReader opened (left open for the whole test)")
            stop = threading.Event()
            samples = []
            def sampler():
                while not stop.is_set():
                    samples.append(npu.power_mw())
                    time.sleep(0.003)
            th = threading.Thread(target=sampler, daemon=True)
            th.start()
            flush_print("[min-repro] sampler thread started, dispatching resident x5 while it runs")
            for i in range(5):
                t0 = time.monotonic()
                d(hwt, emb_t, ot)
                flush_print("[min-repro] concurrent dispatch %d RETURNED after %.3fs (samples so far: %d)"
                           % (i, time.monotonic() - t0, len(samples)))
            stop.set()
            th.join(timeout=5)
            npu.close()
            flush_print("[min-repro] sampler thread stopped (%d samples), SensorReader closed" % len(samples))
            flush_print("[min-repro] ALL STEPS COMPLETED (sensor variant, order=concurrent) -- hypothesis REFUTED")
        else:
            # order == "after": matches energy()'s own call order -- warm the context FIRST,
            # THEN construct the reader, THEN dispatch again.
            flush_print("[min-repro] --order after: warming resident BEFORE the reader (matches energy())")
            t0 = time.monotonic()
            d(hwt, emb_t, ot)
            flush_print("[min-repro] warm dispatch RETURNED after %.3fs" % (time.monotonic() - t0))
            open_sample_dispatch("after")
            flush_print("[min-repro] ALL STEPS COMPLETED (sensor variant, order=after) -- hypothesis REFUTED")


if __name__ == "__main__":
    main()
