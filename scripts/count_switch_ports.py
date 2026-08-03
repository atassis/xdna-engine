#!/usr/bin/env python3
"""Count stream-switch DESTINATION port usage per tile from routed AIE MLIR.

Check 2 of mode-switched-multi-program-xclbin's pre-build list, the one never done:
both topologies' routes live in the PDI simultaneously, so if two designs are to
co-reside as SEPARATE topologies their switchbox connections add up per tile and
per bundle, the same way locks do.

Feed it MLIR that has been through routing, e.g.

    aie-opt --aie-canonicalize-device --aie-place-tiles --aie-assign-lock-ids \\
            --aie-objectFifo-stateful-transform --aie-create-pathfinder-flows \\
            design.mlir -o routed.mlir

Budgets are per (tile class, bundle), read from AIE2TargetModel::
getNumDestSwitchboxConnections in the pinned instance -- AIE2P inherits them, it
declares no override.
"""
import argparse
import collections
import re
import sys

TILE_RE = re.compile(r"(%[\w]+)\s*=\s*aie\.tile\((\d+),\s*(\d+)\)")
SWITCH_RE = re.compile(r"aie\.switchbox\((%[\w]+)\)")
CONNECT_RE = re.compile(r"aie\.connect<([A-Za-z]+)\s*:\s*(\d+)\s*,\s*([A-Za-z]+)\s*:\s*(\d+)>")

# AIE2/AIE2P destination-port budgets. Edge tiles get 0 for the outward direction;
# we score against the interior maximum, which is the permissive reading -- an
# edge violation would be caught by the router itself, not by this census.
BUDGET = {
    "core": {"Core": 1, "DMA": 2, "FIFO": 1, "North": 6, "West": 4, "South": 4, "East": 4,
             "TileControl": 1},
    "mem":  {"DMA": 6, "North": 6, "South": 4, "TileControl": 1},
    "shim": {"FIFO": 1, "North": 6, "West": 4, "South": 6, "East": 4, "TileControl": 1},
}


def tile_class(row):
    return "shim" if row == 0 else ("mem" if row == 1 else "core")


def parse(path):
    text = open(path).read()
    tiles = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in TILE_RE.finditer(text)}
    used = collections.defaultdict(collections.Counter)   # (col,row) -> bundle -> n
    cur = None
    for line in text.splitlines():
        m = SWITCH_RE.search(line)
        if m:
            cur = tiles.get(m.group(1))
            continue
        if cur is None:
            continue
        for c in CONNECT_RE.finditer(line):
            used[cur][c.group(3)] += 1          # destination bundle
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir", nargs="+", help="routed MLIR; several are SUMMED per tile")
    ap.add_argument("--label", action="append", default=None)
    args = ap.parse_args()

    per_design = [parse(p) for p in args.mlir]
    labels = args.label or [p.split("/")[-1] for p in args.mlir]

    for lab, used in zip(labels, per_design):
        worst = collections.defaultdict(collections.Counter)
        for (col, row), b in used.items():
            for bundle, n in b.items():
                k = tile_class(row)
                worst[k][bundle] = max(worst[k][bundle], n)
        print(f"\n=== {lab}: worst destination-port use of each tile class ===")
        for k in ("core", "mem", "shim"):
            if not worst[k]:
                continue
            cells = ", ".join(f"{b} {n}/{BUDGET[k].get(b,'?')}" for b, n in sorted(worst[k].items()))
            print(f"  {k:5s} {cells}")

    if len(per_design) > 1:
        total = collections.defaultdict(collections.Counter)
        for used in per_design:
            for tile, b in used.items():
                total[tile].update(b)
        print("\n=== SUM (co-resident as SEPARATE topologies) ===")
        over = []
        worst = collections.defaultdict(collections.Counter)
        for (col, row), b in total.items():
            k = tile_class(row)
            for bundle, n in b.items():
                worst[k][bundle] = max(worst[k][bundle], n)
                lim = BUDGET[k].get(bundle)
                if lim is not None and n > lim:
                    over.append(f"({col},{row}) {k} {bundle} {n}/{lim}")
        for k in ("core", "mem", "shim"):
            if not worst[k]:
                continue
            cells = ", ".join(f"{b} {n}/{BUDGET[k].get(b,'?')}" for b, n in sorted(worst[k].items()))
            print(f"  {k:5s} {cells}")
        print("\n  VERDICT: " + ("FITS -- no bundle over budget" if not over
                                 else "OVER BUDGET at " + "; ".join(sorted(set(over)))))


if __name__ == "__main__":
    main()
