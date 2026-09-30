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

# Built 2026-09-30 by reading every edge-name literal aiecc.cpp actually passes to
# .map/.split/.join/.filter, splitPerDevice and buildNpuProgramSubgraph (see
# test_stage_of_covers_every_real_edge_name) -- the earlier version guessed keys and
# misclassified real names (e.g. "mem_topology" never matched "memTopology_{0}.json").
# Matched by exact lowercase name or lowercase prefix/suffix, never by substring: a
# substring "partition" would also catch npu_partition_{0}.mlir, which is control-code.
STAGE_RULES = [
    ("front", {
        "input.mlir", "input_with_symbols.mlir", "input_physical.mlir",
        "input_with_addresses.mlir", "placed.mlir", "traced.mlir",
        "default_stack_size.mlir", "params.txt", "physical_with_elfs.mlir",
        "measured_stack_sizes.mlir", "measured_data_sizes.mlir",
        "checked_bank_placement.mlir", "checked_lut_banks.mlir",
        "devicecachelookup", "sequenceplacement", "perdevicematching",
    }, ("perdevice_",), ()),
    ("per-core", {
        "percoreindevice", "percorecompile", "prebakedcores", "placedcorecompile",
        "perdevicecompilematching",
    }, (
        "llvmir_", "chess-compat_", "chesslinked_", "peano-compat_", "peano-linked_",
        "opted_", "percore_", "prebakedelfs_", "percorestackspace_", "percorearches_",
        "percoreirlinkfiles_", "probescripts_", "probeelfs_", "elfs_", "placedcore_",
        "ldscripts_", "linkwith_", "perdevicecompile_", "perdevicearches_", "lowered_",
        "objects_",
    ), (".bcf",)),
    ("control-code", {
        "perseqmatching", "ctrlpktseqs", "noctrlpktseqs", "fullelfctrlpktnonempty",
        "perdevicenpuloweredmatching",
    }, ("npu_", "ctrlpkt_", "perdevicenpulowered_"), ()),
    ("package", {
        "full.elf", "aie.xclbin", "full_elf_config.json",
        "sim/reports/graph.xpe", "sim/arch/aieshim_solution.aiesol",
        "sim/config/scsim_config.json", "sim/.target", "aiesim.sh", "sim/ps/ps.so",
        "aiesim.stamp", "aie_inc.cpp",
    }, (
        "kernels_", "memtopology_", "partition_", "merged_partition_",
        "input_aie_partition_", "cdo_", "bif_", "full_elf_",
    ), (".pdi", ".elf", ".xclbin")),
]
ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S.*)$")


def stage_of(edge):
    e = edge.lower()
    for stage, exact, prefixes, suffixes in STAGE_RULES:
        if e in exact or any(e.startswith(p) for p in prefixes) or \
           any(e.endswith(s) for s in suffixes):
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
        # The total row has an empty dRSS column (3 tokens: ms, peak, "total"), so ROW --
        # which requires 4 -- never matches it; detect it by its last token instead, before
        # trying ROW, or the sentinel is never cleared and a later digit-led line is misread
        # as still being inside the block.
        if line.split()[-1:] == ["total"]:
            inside = False
            continue
        m = ROW.match(line)
        if not m:
            continue
        ms, _drss, peak, edge = m.groups()
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
