#!/usr/bin/env python3
"""Paired within-rep delta between two arms of one device log.

Arms are interleaved inside a rep, so arm B always runs immediately after arm A under the same
thermal/session state. Differencing WITHIN a rep cancels the drift that makes a raw mean-vs-mean
comparison unreadable -- this session's k=32 arm moved 13.71 -> 9.90% LOCK between its own two
reps, which is larger than any depth effect being tested.

Usage: gemm_trace_paired.py <device.log> <arm_a> <arm_b> [event]
"""
import re, sys, collections

ARM = re.compile(r"^-{10} (\S+) -{10}")
SPAN = re.compile(r"span=(\d+) cyc")
EV = re.compile(r"^\s{4}(\w+)\s+(\d+) cyc\s+([\d.]+)%\s+n=(\d+)")

def parse(path):
    out = collections.defaultdict(list)
    cur = None
    for line in open(path):
        m = ARM.match(line)
        if m:
            cur = {"span": None, "ev": {}}
            out[m.group(1)].append(cur)
            continue
        if cur is None:
            continue
        m = SPAN.search(line)
        if m and cur["span"] is None:
            cur["span"] = int(m.group(1))
        m = EV.match(line)
        if m:
            cur["ev"][m.group(1)] = (float(m.group(3)), int(m.group(4)))
    return out

log, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
ev = sys.argv[4] if len(sys.argv) > 4 else "LOCK_STALL"
d = parse(log)
ra, rb = d[a], d[b]
n = min(len(ra), len(rb))
if n == 0:
    sys.exit(f"no paired reps for {a} / {b}")
print(f"{log}\n  A = {a}\n  B = {b}\n  paired reps: {n}, event: {ev}\n")
print(f"  {'rep':>3} | {'A span':>8} {'B span':>8} {'d span%':>8} | {'A '+ev:>16} {'B '+ev:>16} {'d pp':>7} | {'A n':>6} {'B n':>6}")
ds, de = [], []
for i in range(n):
    sa, sb = ra[i]["span"], rb[i]["span"]
    pa, na = ra[i]["ev"].get(ev, (float("nan"), 0))
    pb, nb = rb[i]["ev"].get(ev, (float("nan"), 0))
    dspan = (sb - sa) / sa * 100
    dev = pb - pa
    ds.append(dspan); de.append(dev)
    print(f"  {i+1:>3} | {sa:>8} {sb:>8} {dspan:>+8.2f} | {pa:>15.2f}% {pb:>15.2f}% {dev:>+7.2f} | {na:>6} {nb:>6}")
mean = lambda v: sum(v) / len(v)
print(f"\n  mean d span {mean(ds):+.2f}%  (per-rep {', '.join(f'{x:+.2f}' for x in ds)})")
print(f"  mean d {ev} {mean(de):+.2f}pp  (per-rep {', '.join(f'{x:+.2f}' for x in de)})")
print(f"  sign-consistent: span {all(x<0 for x in ds) or all(x>0 for x in ds)}, "
      f"{ev} {all(x<0 for x in de) or all(x>0 for x in de)}")
