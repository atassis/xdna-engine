#!/usr/bin/env python3
"""Count the per-tile hardware resources an AIE design actually configures.

Two questions need this, and both were about to be answered from an ESTIMATE:

  * mode-switched-multi-program-xclbin: can two topologies' CONFIGURATION coexist in one xclbin?
    Data memory time-multiplexes, but locks, BDs and buffers do not -- they are configuration, and
    configuration for every mode lives in the PDI simultaneously. So the budget question is
    per-tile locks and BDs, not L1 bytes.
  * conveyor-fit-sizing: what fits on the array at all.

AIE2P budgets (aie/include/aie/Dialect/AIE/IR/AIETargetModel.h):
    core tile     16 locks, 16 BDs, 64 KB data memory
    mem tile      64 locks, 48 BDs, 512 KB
    shim tile     16 locks, 16 BDs

Feed it MLIR that has been through objectFIFO lowering, e.g.

    aie-opt --aie-canonicalize-device --aie-place-tiles --aie-assign-lock-ids \\
            --aie-objectFifo-stateful-transform design.mlir -o lowered.mlir
    python3 scripts/count_tile_resources.py lowered.mlir

Counting the LOWERED form matters: at the objectFIFO level one `ObjectFifo(depth=2)` is a single op,
but it lowers to 2 locks and 2 BDs per endpoint, and it is the lowered numbers that the hardware has
to hold.
"""
import argparse
import collections
import re
import sys

LOCK_RE = re.compile(r"aie\.lock\((%[\w]+)\s*,\s*(\d+)\)")
LOCK_NOID_RE = re.compile(r"aie\.lock\((%[\w]+)\)")
TILE_RE = re.compile(r"(%[\w]+)\s*=\s*aie\.tile\((\d+),\s*(\d+)\)")
BUF_RE = re.compile(r"aie\.buffer\((%[\w]+)\)[^\n]*?memref<([^>]+)>")
BLOCK_RE = re.compile(r"aie\.(mem|memtile_dma|shim_dma)\((%[\w]+)\)")

# AIE2P per-tile budgets. Source: AIETargetModel.h (getNumLocks / getNumBDs / getLocalMemorySize).
BUDGET = {
    "core": {"locks": 16, "bds": 16, "mem": 64 * 1024},
    "mem": {"locks": 64, "bds": 48, "mem": 512 * 1024},
    "shim": {"locks": 16, "bds": 16, "mem": 0},
}

DTYPE_BYTES = {"bf16": 2, "f32": 4, "i32": 4, "i8": 1, "i16": 2, "f16": 2, "i64": 8, "f64": 8,
               "ui8": 1, "si8": 1, "bfp16ebs8": 1}


def memref_bytes(spec):
    """'64x32xbf16' -> bytes. Returns 0 for a shape it cannot parse rather than guessing."""
    parts = spec.split("x")
    try:
        dt = parts[-1]
        n = 1
        for p in parts[:-1]:
            n *= int(p)
        return n * DTYPE_BYTES.get(dt, 0)
    except (ValueError, IndexError):
        return 0


def tile_class(row):
    return "shim" if row == 0 else ("mem" if row == 1 else "core")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--quiet", action="store_true", help="only the summary lines")
    args = ap.parse_args()

    text = open(args.mlir).read()

    tiles = {}  # ssa name -> (col, row)
    for m in TILE_RE.finditer(text):
        tiles[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    if not tiles:
        print("no aie.tile ops -- is this placed MLIR? run --aie-place-tiles first")
        sys.exit(2)

    locks = collections.Counter()
    for m in LOCK_RE.finditer(text):
        locks[m.group(1)] += 1
    unassigned = len(LOCK_NOID_RE.findall(text))

    buffers = collections.Counter()
    buf_bytes = collections.Counter()
    for m in BUF_RE.finditer(text):
        buffers[m.group(1)] += 1
        buf_bytes[m.group(1)] += memref_bytes(m.group(2))

    # BDs are counted inside the dma block that owns them, attributed to that block's tile.
    bds = collections.Counter()
    starts = collections.Counter()
    lines = text.splitlines()
    cur = None
    depth = 0
    for line in lines:
        bm = BLOCK_RE.search(line)
        if bm and cur is None:
            cur = bm.group(2)
            depth = 0
        if cur is not None:
            depth += line.count("{") - line.count("}")
            if "aie.dma_bd(" in line:
                bds[cur] += 1
            if "aie.dma_start(" in line:
                starts[cur] += 1
            if depth <= 0 and ("{" in line or "}" in line):
                cur = None

    rows = []
    for ssa, (col, row) in sorted(tiles.items(), key=lambda kv: (kv[1][1], kv[1][0])):
        cls = tile_class(row)
        rows.append({
            "tile": f"({col},{row})", "class": cls,
            "locks": locks.get(ssa, 0), "bds": bds.get(ssa, 0),
            "starts": starts.get(ssa, 0), "buffers": buffers.get(ssa, 0),
            "bytes": buf_bytes.get(ssa, 0),
        })

    if not args.quiet:
        print(f"{'tile':>8} {'class':>5} {'locks':>12} {'BDs':>12} {'dma_start':>9} "
              f"{'buffers':>8} {'L1/L2 bytes':>14}")
        for r in rows:
            b = BUDGET[r["class"]]
            lk = f"{r['locks']}/{b['locks']}"
            bd = f"{r['bds']}/{b['bds']}"
            mem = (f"{r['bytes']}/{b['mem']} ({100.0*r['bytes']/b['mem']:.0f}%)"
                   if b["mem"] else "-")
            print(f"{r['tile']:>8} {r['class']:>5} {lk:>12} {bd:>12} {r['starts']:>9} "
                  f"{r['buffers']:>8} {mem:>14}")
        print()

    print("=== worst tile of each class (this is the budget that binds) ===")
    print(f"{'class':>6} {'tiles':>6} {'max locks':>16} {'max BDs':>16} {'max bytes':>22}")
    summary = {}
    for cls in ("core", "mem", "shim"):
        sub = [r for r in rows if r["class"] == cls]
        if not sub:
            continue
        b = BUDGET[cls]
        ml = max(r["locks"] for r in sub)
        mb = max(r["bds"] for r in sub)
        mm = max(r["bytes"] for r in sub)
        memstr = (f"{mm}/{b['mem']} ({100.0*mm/b['mem']:.0f}%)" if b["mem"] else "-")
        print(f"{cls:>6} {len(sub):>6} {f'{ml}/{b[chr(108)+chr(111)+chr(99)+chr(107)+chr(115)]}':>16}"
              f" {f'{mb}/{b[chr(98)+chr(100)+chr(115)]}':>16} {memstr:>22}")
        summary[cls] = {"tiles": len(sub), "max_locks": ml, "max_bds": mb, "max_bytes": mm,
                        "lock_budget": b["locks"], "bd_budget": b["bds"], "mem_budget": b["mem"]}

    print()
    print("=== headroom for a SECOND co-resident program (configuration is not time-shared) ===")
    for cls in ("core", "mem"):
        if cls not in summary:
            continue
        s = summary[cls]
        print(f"  {cls:>4} tile: {s['lock_budget'] - s['max_locks']:>2} locks and "
              f"{s['bd_budget'] - s['max_bds']:>2} BDs free "
              f"(this design uses {s['max_locks']} / {s['max_bds']})")
    if unassigned:
        print(f"\nNOTE: {unassigned} aie.lock ops carry no ID -- lock IDs were not assigned, so the "
              f"lock counts above are op counts, not hardware allocations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
