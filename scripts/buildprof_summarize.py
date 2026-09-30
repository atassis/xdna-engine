#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn S0 build-profile output into one TSV: artifact, stage, ms, peak_mib.

Input: <s0-dir>/<artifact>/aiecc-logs/aiecc-*.log (buildprof_shim.sh) and
<s0-dir>/<artifact>/time.txt (/usr/bin/time -v). OUTSIDE_AIECC is total wall minus the summed
aiecc wall of that artifact: generator, IRON, packing and anything else not inside aiecc.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

# First match wins; "package" precedes "per-core" so full.elf is not read as a core ELF.
STAGE_RULES = [
    ("control-code", ("npu_", "materialized", "dma_lowered", "insts")),
    ("package", ("cdo", "pdi", "bif", "partition", "xclbin", "full_elf", "full.elf",
                 "kernels", "mem_topology")),
    ("per-core", ("percore", "lowered_", "llvmir_", "peano-", "opted_", "core_",
                  "ld.script", "probescripts")),
    ("front", ("input", "placed", "physical", "traced", "stack", "params", "symbols",
               "ctrlpkt")),
]
ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S.*)$")


def stage_of(edge):
    e = edge.lower()
    for stage, keys in STAGE_RULES:
        if any(k in e for k in keys):
            return stage
    return "other"


def parse_profile(text):
    rows, inside = [], False
    for line in text.splitlines():
        if line.startswith("aiecc: profile"):
            inside = True
            continue
        if not inside:
            continue
        m = ROW.match(line)
        if not m:
            continue
        ms, _drss, peak, edge = m.groups()
        if edge.strip() == "total":
            inside = False
            continue
        rows.append((edge.strip(), int(ms), float(peak) if peak != "-" else 0.0))
    return rows


def _time_v(path):
    wall_ms, rss_kb = 0, 0
    for line in path.read_text().splitlines():
        if "Elapsed (wall clock)" in line:
            parts = [float(p) for p in line.rsplit(": ", 1)[1].split(":")]
            secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
            wall_ms = int(round(secs * 1000))
        elif "Maximum resident set size" in line:
            rss_kb = int(line.rsplit(": ", 1)[1])
    return wall_ms, rss_kb


def summarize(s0_dir):
    out = ["artifact\tstage\tms\tpeak_mib"]
    for art in sorted(p for p in Path(s0_dir).iterdir() if p.is_dir()):
        ms, peak = defaultdict(int), defaultdict(float)
        aiecc_wall = 0
        for log in sorted((art / "aiecc-logs").glob("aiecc-*.log")):
            text = log.read_text()
            for edge, e_ms, e_peak in parse_profile(text):
                st = stage_of(edge)
                ms[st] += e_ms
                peak[st] = max(peak[st], e_peak)
            m = re.search(r"^wall_ms: (\d+)$", text, re.M)
            aiecc_wall += int(m.group(1)) if m else 0
        for st in sorted(ms):
            out.append(f"{art.name}\t{st}\t{ms[st]}\t{peak[st]:.1f}")
        if (art / "time.txt").exists():
            wall, rss = _time_v(art / "time.txt")
            out.append(f"{art.name}\tTOTAL_WALL\t{wall}\t{rss / 1024:.1f}")
            out.append(f"{art.name}\tOUTSIDE_AIECC\t{max(wall - aiecc_wall, 0)}\t")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(summarize(sys.argv[1]))
