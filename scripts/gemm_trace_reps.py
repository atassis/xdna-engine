#!/usr/bin/env python3
"""Fold a multi-rep gemm_k_sweep_device log into a per-arm mean +/- spread table.

The device script writes every arm of a session to one log, so passing the arm list twice gives
interleaved reps in that one log. This box carries a per-session shift that lands specifically on
acquire wait -- the same design has read up to 19.8pp apart in LOCK_STALL across sessions while two
interleaved reps within a session agree to ~2pp -- so a multi-arm series is only meaningful when
every arm ran in ONE session. This reads a single log and refuses to merge two, by construction.

Usage: gemm_trace_reps.py <device.log> [event ...]
"""
import re, sys, collections

ARM = re.compile(r"^-{10} (\S+) -{10}")
SPAN = re.compile(r"span=(\d+) cyc")
EV = re.compile(r"^\s{4}(\w+)\s+(\d+) cyc\s+([\d.]+)%\s+n=(\d+)\s+mean=([\d.]+)")
RELL2 = re.compile(r"rel-L2 = (\S+)\s+(\w+)")

def parse(path):
    reps = collections.defaultdict(list)   # arm -> [ {span, rel_l2, events{}} ]
    cur = None
    for line in open(path):
        m = ARM.match(line)
        if m:
            cur = {"arm": m.group(1), "span": None, "rel_l2": None, "ev": {}}
            reps[m.group(1)].append(cur)
            continue
        if cur is None:
            continue
        m = RELL2.search(line)
        if m and cur["rel_l2"] is None:
            cur["rel_l2"] = (m.group(1), m.group(2))
        m = SPAN.search(line)
        if m and cur["span"] is None:
            cur["span"] = int(m.group(1))
        m = EV.match(line)
        if m:
            cur["ev"][m.group(1)] = {
                "cyc": int(m.group(2)), "pct": float(m.group(3)),
                "n": int(m.group(4)), "mean": float(m.group(5)),
            }
    return reps

def spread(vals):
    if len(vals) < 2:
        return 0.0
    lo, hi = min(vals), max(vals)
    return 0.0 if lo == 0 else (hi - lo) / lo * 100.0

def main():
    path = sys.argv[1]
    want = sys.argv[2:] or ["LOCK_STALL", "MEMORY_STALL", "STREAM_STALL", "INSTR_VECTOR"]
    reps = parse(path)
    print(f"{path}\n")
    hdr = f"{'arm':<46} {'reps':>4} {'span mean':>10} {'spr%':>6}"
    for e in want:
        hdr += f" | {e[:12]:>12} {'spr(pp)':>7} {'n':>9}"
    print(hdr)
    print("-" * len(hdr))
    for arm, rs in reps.items():
        spans = [r["span"] for r in rs if r["span"]]
        if not spans:
            continue
        row = f"{arm:<46} {len(rs):>4} {sum(spans)/len(spans):>10.0f} {spread(spans):>6.2f}"
        for e in want:
            pcts = [r["ev"][e]["pct"] for r in rs if e in r["ev"]]
            ns = [r["ev"][e]["n"] for r in rs if e in r["ev"]]
            if pcts:
                row += f" | {sum(pcts)/len(pcts):>11.2f}% {max(pcts)-min(pcts):>7.2f} {'/'.join(map(str, ns)):>9}"
            else:
                row += f" | {'-':>12} {'-':>7} {'-':>9}"
        print(row)
    print()
    for arm, rs in reps.items():
        bad = [r["rel_l2"] for r in rs if r["rel_l2"] and r["rel_l2"][1] != "PASS"]
        if bad:
            print(f"!! {arm}: rel-L2 not PASS: {bad}")
    print("rel-L2: " + ", ".join(
        f"{arm.split('_4c_')[-1] or arm}={rs[0]['rel_l2'][0]}({rs[0]['rel_l2'][1]})"
        for arm, rs in reps.items() if rs[0]["rel_l2"]))

main()
