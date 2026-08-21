#!/usr/bin/env python3
"""Paired per-rep statistics for the IN-ENCODER restream cost.

Consumes the `same-xclbin dispatches by PREDECESSOR` table that
`NPU_DISPATCH_LOG=1 parakeet_encode_npu` prints for the last (warm) clip of each
invocation, one file per rep, and reports the per-rep paired difference

    d = mean_ms(prev = a different stream, same xclbin) - mean_ms(prev = the same stream)

Pairing is WITHIN a row, which is what makes this a measurement rather than an
inference: a row is one (xclbin, instruction stream), so both populations run
identical work and interleave in dispatch order within the one clip. The rep is
the pairing unit; the CI is Student-t over reps.

The prior "~0.6-0.7 ms/restream" figure was a residual backed out of an
end-to-end delta, not a paired measurement -- see the task
in-encoder-restream-direct-measure.
"""

import argparse
import math
import re
import sys
from collections import defaultdict

# `    {label:<30} {insts:>8}  {same:>18}  {restream:>18}  {xclbin:>18}{async}`
# where each of the three cells is either `-` or `x<n> <mean_ms>`.
ROW = re.compile(r"^\s{4}(\S+#\d+@0x[0-9a-f]+)\s+(\d+)\s+(.*)$")
CELL = re.compile(r"x(\d+)\s+([0-9.]+)")


def parse_cells(rest):
    """Left-to-right: `-` is an empty cell, `x<n> <mean>` is a populated one."""
    toks = rest.split("(")[0].split()
    cells, i = [], 0
    while i < len(toks) and len(cells) < 3:
        if toks[i] == "-":
            cells.append(None)
            i += 1
        elif toks[i].startswith("x") and i + 1 < len(toks):
            cells.append((int(toks[i][1:]), float(toks[i + 1])))
            i += 2
        else:
            i += 1
    while len(cells) < 3:
        cells.append(None)
    return cells  # (same, restream, other-xclbin)


def parse_report(path):
    """-> {row_key: (insts, same, restream, xclbin)} for the one table in `path`."""
    rows, in_table = {}, False
    for line in open(path, errors="replace"):
        if "dispatches by PREDECESSOR" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("transitions (from"):
                break
            m = ROW.match(line.rstrip("\n"))
            if m:
                key, insts, rest = m.group(1), int(m.group(2)), m.group(3)
                # The stream ADDRESS is not stable across invocations; the ordinal is.
                rows[key.split("@")[0]] = (insts, *parse_cells(rest))
    return rows


def t_crit(df):
    """Two-sided 95% Student-t. Table to 30 df, normal beyond -- avoids a scipy dep."""
    tbl = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
           8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
           15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
           21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
           27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
    return tbl.get(df, 1.960) if df >= 1 else float("nan")


def summarise(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return n, mean, float("nan"), float("nan"), sum(1 for d in diffs if d > 0)
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half = t_crit(n - 1) * math.sqrt(var / n)
    return n, mean, mean - half, mean + half, sum(1 for d in diffs if d > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--label", default="")
    ap.add_argument("--min-reps", type=int, default=3)
    a = ap.parse_args()

    # Two contrasts, both paired within a row and within a rep:
    #   vs_same = restream - same    what a stream change costs over repeating the stream
    #   vs_xcl  = restream - xclbin  what a stream change SAVES against a full transition.
    # The second is the load-bearing one: mode-switching only pays if it is < 0.
    vs_same, vs_xcl = defaultdict(list), defaultdict(list)
    counts = defaultdict(lambda: [0, 0, 0])
    for p in a.reports:
        for row, (insts, same, restr, xc) in parse_report(p).items():
            if restr and same:
                vs_same[row].append(restr[1] - same[1])
            if restr and xc:
                vs_xcl[row].append(restr[1] - xc[1])
            if restr:
                counts[row][0] += same[0] if same else 0
                counts[row][1] += restr[0]
                counts[row][2] += xc[0] if xc else 0

    if not vs_same and not vs_xcl:
        print("no row carried a prev=other-stream sample at all", file=sys.stderr)
        return 1

    print(f"\n=== in-encoder restream, paired per rep {a.label} ===")
    print(f"reports: {len(a.reports)}")
    ranked = sorted(counts.items(), key=lambda kv: -kv[1][1])

    def table(title, data, note):
        print(f"\n  {title}   ({note})")
        print(f"  {'xclbin#stream':<58} {'reps':>4} {'mean ms':>9} {'95% CI':>22} {'>0':>7}")
        for row, _ in ranked:
            diffs = data.get(row, [])
            if len(diffs) < a.min_reps:
                continue
            n, mean, lo, hi, pos = summarise(diffs)
            print(f"  {row:<58} {n:>4} {mean:>+9.3f} "
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>22} {f'{pos}/{n}':>7}")

    table("restream - same-stream", vs_same, "cost of changing the instruction stream")
    table("restream - xclbin transition", vs_xcl,
          "SAVING vs a full context switch; must be < 0 for mode-switching to pay")

    top = ranked[0][0]
    print(f"\nheadline row {top}")
    print(f"  dispatch counts over all reps: same={counts[top][0]} "
          f"restream={counts[top][1]} xclbin={counts[top][2]}")
    for name, data in (("vs same-stream", vs_same), ("vs xclbin transition", vs_xcl)):
        if len(data.get(top, [])) >= 2:
            n, mean, lo, hi, pos = summarise(data[top])
            print(f"  {name:<22} {mean:+.3f} ms  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"{pos}/{n} reps positive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
