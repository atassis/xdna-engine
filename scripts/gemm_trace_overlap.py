#!/usr/bin/env python3
"""Overlap two tracks of a core trace -- does the core stall WHILE its feed runs, or in the gaps?

The per-arm summary gives each event as a percent of span, which cannot answer this. LOCK_STALL 44%
and PORT_RUNNING_0 36% are consistent both with 'the core waits exactly while data streams in' (the
feed is the constraint) and with 'the core waits while nothing is being delivered' (a scheduling gap
-- the feed has headroom and is not being asked to run). Those have opposite fixes, so the aggregate
must not be read as either.

Both tracks come from the same core in the same run, so their intervals are directly comparable.

Usage: gemm_trace_overlap.py <trace_gemm_*.json> [--stall LOCK_STALL] [--port PORT_RUNNING_0]
"""
import argparse, json, sys


def intervals(events, name):
    """Chrome-trace B/E pairs on one track -> merged [start, end) list."""
    out, depth, start = [], 0, None
    for e in events:
        if e.get("name") != name:
            continue
        if e.get("ph") == "B":
            if depth == 0:
                start = e["ts"]
            depth += 1
        elif e.get("ph") == "E":
            depth -= 1
            if depth == 0 and start is not None:
                out.append((start, e["ts"]))
                start = None
    return merge(out)


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def overlap(a, b):
    """Total measure of a INTERSECT b, both merged and sorted."""
    i = j = acc = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            acc += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--stall", default="LOCK_STALL")
    ap.add_argument("--port", default="PORT_RUNNING_0")
    args = ap.parse_args()

    hdr = (f"{'arm':<26}{'rep':>4}{'span':>8}{'stall':>9}{'port':>9}"
           f"{'concurrent':>11}{'gap':>8}{'%gap':>7}")
    print(hdr)
    print("-" * len(hdr))
    by_arm = {}
    for path in args.traces:
        d = json.load(open(path))
        ev = [e for e in d if e.get("ph") in ("B", "E")]
        if not ev:
            print(f"{path}: no B/E events", file=sys.stderr)
            continue
        span = max(e["ts"] for e in ev)
        st = intervals(ev, args.stall)
        pr = intervals(ev, args.port)
        ts, tp = total(st), total(pr)
        # Split the stall by what the feed was doing: concurrent = the core waited while the port
        # was actively streaming (a rate shortfall -- more buffers cannot fix it); gap = it waited
        # with the port idle (a start/handshake latency -- buffering CAN hide that).
        conc = overlap(st, pr)
        gap = ts - conc
        arm = path.split("/")[-1].replace("trace_gemm_", "").replace(".json", "")
        arm = arm.replace("512x1024x1024_", "")
        by_arm.setdefault(arm, []).append((span, ts, tp, conc, gap))

    for arm in sorted(by_arm):
        for i, (span, ts, tp, conc, gap) in enumerate(by_arm[arm], 1):
            pct = 100.0 * gap / ts if ts else 0.0
            print(f"{arm:<26}{i:>4}{span:>8}{ts:>9}{tp:>9}{conc:>11}{gap:>8}{pct:>6.1f}%")

    # Per-rep values are printed above and are the reportable form: a mean over few reps with no
    # spread quoted is not a measurement on this box.
    if any(len(v) > 1 for v in by_arm.values()):
        print()
        print(f"{'arm':<26}{'n':>3}{'concurrent lo..hi':>22}{'gap lo..hi':>20}")
        for arm in sorted(by_arm):
            v = by_arm[arm]
            c = [x[3] for x in v]
            g = [x[4] for x in v]
            print(f"{arm:<26}{len(v):>3}{f'{min(c)}..{max(c)}':>22}{f'{min(g)}..{max(g)}':>20}")


if __name__ == "__main__":
    main()
