#!/usr/bin/env python3
"""Overlap two tracks of a core trace -- does the core stall WHILE its feed runs, or in the gaps?

The per-arm summary gives each event as a percent of span, which cannot answer this. LOCK_STALL 44%
and PORT_RUNNING_0 36% are consistent both with 'the core waits exactly while data streams in' (the
feed is the constraint) and with 'the core waits while nothing is being delivered' (a scheduling gap
-- the feed has headroom and is not being asked to run). Those have opposite fixes, so the aggregate
must not be read as either.

Both tracks come from the same core in the same run, so their intervals are directly comparable.

--port may be repeated. This design feeds A on core DMA S2MM ch0 and B on ch1, so one port is
half the feed: with two, the ports' own overlap says whether the two streams run CONCURRENTLY or
SERIALIZE against each other, and the stall is split against their UNION rather than one channel.

Usage: gemm_trace_overlap.py <trace_gemm_*.json> [--stall LOCK_STALL] [--port PORT_RUNNING_0]...
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


def union(*ivs):
    """Merged union of several already-merged interval lists."""
    return merge([iv for lst in ivs for iv in lst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--stall", default="LOCK_STALL")
    ap.add_argument("--port", action="append", default=None)
    args = ap.parse_args()
    ports = args.port or ["PORT_RUNNING_0"]

    hdr = (f"{'arm':<26}{'rep':>4}{'span':>8}{'stall':>9}{'port':>9}"
           f"{'concurrent':>11}{'gap':>8}{'%gap':>7}")
    print(hdr)
    print("-" * len(hdr))
    by_arm = {}
    per_port = {}
    for path in args.traces:
        d = json.load(open(path))
        ev = [e for e in d if e.get("ph") in ("B", "E")]
        if not ev:
            print(f"{path}: no B/E events", file=sys.stderr)
            continue
        span = max(e["ts"] for e in ev)
        st = intervals(ev, args.stall)
        prs = [intervals(ev, name) for name in ports]
        feed = union(*prs)
        ts, tp = total(st), total(feed)
        # Split the stall by what the feed was doing: concurrent = the core waited while the feed
        # was actively streaming (a rate shortfall -- more buffers cannot fix it); gap = it waited
        # with the feed idle (a start/handshake latency -- buffering CAN hide that).
        conc = overlap(st, feed)
        gap = ts - conc
        arm = path.split("/")[-1].replace("trace_gemm_", "").replace(".json", "")
        arm = arm.replace("512x1024x1024_", "")
        by_arm.setdefault(arm, []).append((span, ts, tp, conc, gap))
        per_port.setdefault(arm, []).append(
            ([total(x) for x in prs], [overlap(st, x) for x in prs],
             overlap(prs[0], prs[1]) if len(prs) > 1 else None)
        )

    for arm in sorted(by_arm):
        for i, (span, ts, tp, conc, gap) in enumerate(by_arm[arm], 1):
            pct = 100.0 * gap / ts if ts else 0.0
            print(f"{arm:<26}{i:>4}{span:>8}{ts:>9}{tp:>9}{conc:>11}{gap:>8}{pct:>6.1f}%")

    if len(ports) > 1:
        print()
        hdr2 = f"{'arm':<26}{'rep':>4}" + "".join(f"{n[-12:]:>16}" for n in ports)
        hdr2 += f"{'ports overlap':>15}{'%of smaller':>12}"
        print(hdr2)
        print("-" * len(hdr2))
        for arm in sorted(per_port):
            for i, (tots, concs, ov) in enumerate(per_port[arm], 1):
                row = f"{arm:<26}{i:>4}" + "".join(f"{t:>16}" for t in tots)
                # Two feeds on separate DMA channels CAN run at once. Their overlap as a fraction
                # of the smaller one is the direct read: ~100% = concurrent, ~0% = serialized.
                pct = 100.0 * ov / min(tots) if ov is not None and min(tots) else 0.0
                row += f"{ov:>15}{pct:>11.1f}%"
                print(row)
        print()
        print(f"{'arm':<26}{'rep':>4}" + "".join(f"{'stall^' + n[-9:]:>16}" for n in ports))
        for arm in sorted(per_port):
            for i, (tots, concs, ov) in enumerate(per_port[arm], 1):
                print(f"{arm:<26}{i:>4}" + "".join(f"{c:>16}" for c in concs))

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
