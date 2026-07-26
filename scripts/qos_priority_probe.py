#!/usr/bin/env python3
"""Does amdxdna QoS `priority` affect SCHEDULING, or only the DPM clock?

THE LOGICAL LEVER: in aie2_solver.c set_dpm_level(), each context computes its own level but
the driver then programs `set_dft_dpm_level(dev_level)` where dev_level is the MAX across all
active sessions. So two concurrent contexts necessarily share ONE device clock. Therefore any
throughput ASYMMETRY between two concurrently-running contexts cannot be a DPM effect -- it
has to be scheduling. That is what this probe measures.

DESIGN: two independent worker PROCESSES (separate hw contexts), each looping a fixed dispatch
for a fixed wall time and counting completions. Scenarios:

    solo         one worker alone                     -> the uncontended baseline
    NORMAL/NORMAL both workers at NORMAL priority      -> symmetric-sharing control
    REALTIME/LOW  worker A realtime, worker B low
    LOW/REALTIME  swapped -- controls for process start-order and CPU-side asymmetry

READING IT: if priority drives scheduling, the REALTIME worker keeps a larger share of solo
throughput than the LOW one, and the effect SWAPS when the priorities swap. If the asymmetry
does not swap, it is an artifact of launch order, not priority. If both scenarios look like
NORMAL/NORMAL, priority does not reach the scheduler on this part.

Usage (device must be quiesced -- stop npu services, fuser-clean):
    scripts/qos_priority_probe.py --xclbin artifacts/parakeet/lpddr_bw/<file>.xclbin
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_power_mode import require_pinned  # noqa: E402

# include/uapi/drm/amdxdna_accel.h
PRIORITY = {
    "realtime": 0x100,
    "high":     0x180,
    "normal":   0x200,
    "low":      0x280,
}


def worker(tag, priority_name, xclbin_path, insts_path, seconds, nbufs, buf_bytes, out_q):
    """Loop dispatches for `seconds`, report completions. Runs in its own process."""
    try:
        import pyxrt
        dev = pyxrt.device(0)
        instr = np.fromfile(insts_path, dtype=np.uint32)
        xb = pyxrt.xclbin(xclbin_path)
        dev.register_xclbin(xb)

        cfg = {"priority": PRIORITY[priority_name]}
        ctx = pyxrt.hw_context(dev, xb.get_uuid(), cfg)
        k = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())

        TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
        bo_i.write(instr.tobytes(), 0)
        bo_i.sync(TO)
        bufs = [pyxrt.bo(dev, buf_bytes, pyxrt.bo.host_only, k.group_id(3 + i))
                for i in range(nbufs)]

        k(3, bo_i, instr.size, *bufs).wait()          # warm: xclbin load, first touch
        out_q.put(("ready", tag))

        n, t0 = 0, time.perf_counter()
        deadline = t0 + seconds
        while time.perf_counter() < deadline:
            k(3, bo_i, instr.size, *bufs).wait()
            n += 1
        elapsed = time.perf_counter() - t0
        out_q.put(("result", {"tag": tag, "priority": priority_name,
                              "dispatches": n, "seconds": elapsed,
                              "rate_hz": n / elapsed}))
    except Exception as e:  # noqa: BLE001 - report, do not hang the coordinator
        out_q.put(("error", {"tag": tag, "error": repr(e)}))


def run_scenario(name, prios, args, barrier_wait=2.0):
    """Launch one worker per entry in `prios` (dict tag->priority); return their results."""
    ctxmp = mp.get_context("spawn")
    q = ctxmp.Queue()
    procs = []
    for tag, p in prios.items():
        pr = ctxmp.Process(target=worker,
                           args=(tag, p, args.xclbin, args.insts, args.seconds,
                                 args.nbufs, args.buf_bytes, q))
        pr.start()
        procs.append(pr)

    results, errors, ready = [], [], 0
    deadline = time.time() + args.seconds + 180
    while len(results) + len(errors) < len(procs) and time.time() < deadline:
        try:
            kind, payload = q.get(timeout=5)
        except Exception:
            continue
        if kind == "ready":
            ready += 1
        elif kind == "result":
            results.append(payload)
        else:
            errors.append(payload)
    for pr in procs:
        pr.join(timeout=10)
        if pr.is_alive():
            pr.terminate()
    return {"scenario": name, "results": results, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--insts", default=None, help="defaults to <xclbin>.insts.bin")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--nbufs", type=int, default=2, help="rdwr c1 = 2 runtime buffers")
    ap.add_argument("--buf-bytes", type=int, default=16777216)
    ap.add_argument("--allow-unpinned", action="store_true")
    a = ap.parse_args()
    if a.insts is None:
        a.insts = a.xclbin.replace(".xclbin", ".insts.bin")
    if a.allow_unpinned:
        os.environ["NPU_ALLOW_UNPINNED"] = "1"
    require_pinned()

    scenarios = [
        ("solo",            {"A": "normal"}),
        ("NORMAL/NORMAL",   {"A": "normal",   "B": "normal"}),
        ("REALTIME/LOW",    {"A": "realtime", "B": "low"}),
        ("LOW/REALTIME",    {"A": "low",      "B": "realtime"}),
    ]
    out = []
    for name, prios in scenarios:
        print(f"\n=== {name} ===", flush=True)
        r = run_scenario(name, prios, a)
        for e in r["errors"]:
            print(f"  ERROR {e['tag']}: {e['error']}")
        for res in sorted(r["results"], key=lambda x: x["tag"]):
            print(f"  {res['tag']} ({res['priority']:>8}): "
                  f"{res['dispatches']:5d} dispatches in {res['seconds']:.2f}s "
                  f"= {res['rate_hz']:7.2f} Hz")
        out.append(r)

    # Verdict
    solo = next((x["results"][0]["rate_hz"] for x in out
                 if x["scenario"] == "solo" and x["results"]), None)
    print("\n=== share of solo throughput ===")
    if not solo:
        print("  solo baseline missing; cannot normalise")
        return 1
    for r in out[1:]:
        by = {x["priority"]: x["rate_hz"] for x in r["results"]}
        if len(by) < 2:
            continue
        parts = "  ".join(f"{p}={100*v/solo:5.1f}%" for p, v in sorted(by.items()))
        print(f"  {r['scenario']:>14}: {parts}")
    rl = next((r for r in out if r["scenario"] == "REALTIME/LOW"), None)
    lr = next((r for r in out if r["scenario"] == "LOW/REALTIME"), None)
    if rl and lr and len(rl["results"]) == 2 and len(lr["results"]) == 2:
        def rate(r, prio):
            return next(x["rate_hz"] for x in r["results"] if x["priority"] == prio)
        r1 = rate(rl, "realtime") / rate(rl, "low")
        r2 = rate(lr, "realtime") / rate(lr, "low")
        print(f"\n  realtime:low throughput ratio -> {r1:.2f}x and {r2:.2f}x (swapped)")
        if r1 > 1.15 and r2 > 1.15:
            print("  VERDICT: priority AFFECTS SCHEDULING (favours realtime, survives the swap)")
        elif abs(r1 - 1) < 0.15 and abs(r2 - 1) < 0.15:
            print("  VERDICT: priority does NOT reach the scheduler (symmetric both ways)")
        else:
            print("  VERDICT: INCONCLUSIVE -- asymmetry does not survive the swap, so it is")
            print("           launch-order/CPU-side artifact, not priority. Re-run quiesced.")
    json.dump(out, open("/tmp/qos_priority_probe.json", "w"), indent=2)
    print("\nwrote /tmp/qos_priority_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
