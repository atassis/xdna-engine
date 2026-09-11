#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive the batched-vs-per-token prefill crossover from scripts/time_prefill.sh's own logs and
write it into the swept prefill artifact's meta.json, as `dims.prefill_break_even_tokens`.

Closes the failure mode that shipped a stale threshold 2026-09-11: generator.rs used to carry the
crossover as a hand-edited Rust constant, measured against one artifact and silently wrong after
the next rebuild (a 13-token prompt paid ~1.6x what per-token priming would have cost). The
constant still exists as a fallback for pre-2026-09-11 artifacts; this is what an artifact's OWN
number should come from going forward -- the fixed batched-dispatch cost and the per-token
stepwise rate are each roughly FLAT across prompt length (measured, not assumed -- this script
checks that and refuses to write a number it does not trust), so `fixed_ms / per_token_ms` is the
crossover, no curve-fit needed.

    bash scripts/time_prefill.sh 3 3 4,8,12,16,20,24,28,32,48,64   # writes into $OUT
    python3 scripts/ingest_prefill_break_even.py --timing-dir /mnt/data/xdna/scratch/prefill/timing \\
        --prefill-art artifacts/qwen3-0.6b/prefill4096

Reads every `r*_arm0.log` (pertok) / `r*_arm1.log` (batched) in --timing-dir -- time_prefill.sh's
own naming, unchanged. Refuses to write (exit 1, no meta.json touched) if either arm's values are
not flat within --tolerance, since a non-flat arm means the fixed-cost/flat-rate model this script
assumes does not hold for that sweep and a derived number would be a guess wearing a measurement's
clothes.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$"
)  # P  batched  ms_median  ms/token  dispatches


def parse_log(path):
    """[(P, ms_median, ms_per_token), ...] for every table row in one time_prefill.sh round log."""
    rows = []
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if m:
            p, _batched, ms_median, ms_per_token, _dispatches = m.groups()
            rows.append((int(p), float(ms_median), float(ms_per_token)))
    return rows


def check_flat(label, values, tolerance):
    lo, hi, med = min(values), max(values), statistics.median(values)
    spread = (hi - lo) / med if med else float("inf")
    if spread > tolerance:
        print(f"ERROR: {label} is not flat -- min={lo:.2f} max={hi:.2f} median={med:.2f} "
              f"spread={spread:.1%} (tolerance {tolerance:.1%}). The fixed-cost/flat-rate model "
              f"this script assumes does not hold; not writing a derived number.", file=sys.stderr)
        return None
    return med


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing-dir", required=True, help="dir of r*_arm{0,1}.log from time_prefill.sh")
    ap.add_argument("--prefill-art", required=True, help="prefill artifact dir whose meta.json to patch")
    ap.add_argument("--tolerance", type=float, default=0.15,
                     help="max (max-min)/median allowed within an arm before refusing (default 0.15)")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, do not write meta.json")
    args = ap.parse_args()

    timing_dir = Path(args.timing_dir)
    pertok_logs = sorted(timing_dir.glob("r*_arm0.log"))
    batched_logs = sorted(timing_dir.glob("r*_arm1.log"))
    if not pertok_logs or not batched_logs:
        sys.exit(f"ERROR: {timing_dir} has no r*_arm0.log/r*_arm1.log -- run scripts/time_prefill.sh first")

    pertok_ms_per_tok = [row[2] for log in pertok_logs for row in parse_log(log)]
    batched_ms_fixed = [row[1] for log in batched_logs for row in parse_log(log)]
    if not pertok_ms_per_tok or not batched_ms_fixed:
        sys.exit(f"ERROR: parsed 0 rows from {timing_dir} -- log format changed?")

    stepwise = check_flat("pertok ms/token", pertok_ms_per_tok, args.tolerance)
    fixed = check_flat("batched ms_median (fixed dispatch cost)", batched_ms_fixed, args.tolerance)
    if stepwise is None or fixed is None:
        sys.exit(1)

    import math
    crossover = math.ceil(fixed / stepwise)
    print(f"[ingest] stepwise={stepwise:.2f} ms/token (n={len(pertok_ms_per_tok)})  "
          f"batched_fixed={fixed:.2f} ms (n={len(batched_ms_fixed)})  "
          f"crossover=ceil({fixed:.2f}/{stepwise:.2f})={crossover}")

    meta_path = Path(args.prefill_art) / "meta.json"
    if not meta_path.is_file():
        sys.exit(f"ERROR: no meta.json at {meta_path}")
    meta = json.loads(meta_path.read_text())
    old = meta.get("dims", {}).get("prefill_break_even_tokens")
    if args.dry_run:
        print(f"[ingest] --dry-run: would write dims.prefill_break_even_tokens = {crossover} "
              f"(was {old!r}) into {meta_path}")
        return
    meta.setdefault("dims", {})["prefill_break_even_tokens"] = crossover
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[ingest] wrote dims.prefill_break_even_tokens = {crossover} (was {old!r}) -> {meta_path}")


if __name__ == "__main__":
    main()
