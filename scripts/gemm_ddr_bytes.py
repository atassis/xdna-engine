#!/usr/bin/env python3
# DDR byte accounting for the whole_array GEMM, read out of each design's own shim BDs
# (task gemm-offcore-residue-occupancy, item 1 -- splitting the width-independent floor).
#
# WHY THIS EXISTS: the width series fits T(cols) = S + P/cols and reads S as a per-command serial
# floor. That reading assumes the DDR traffic is the same at every width. It is not. Each shim
# `aie.dma_bd` carries a 4-D access pattern whose OUTER dim is a repeat count, and A's outer repeat
# falls 8 -> 2 -> 1 as cols goes 1 -> 4 -> 8, so A's DDR reads fall 8 MB -> 2 MB -> 1 MB while the
# design moves the identical operands. Anything that varies with cols cannot be part of S, so the
# byte count has to be taken before the fit means anything.
#
# THE len CONVENTION, verified rather than assumed: on every BD in every arm checked, `len` equals
# the product of the INNER THREE sizes, not all four -- e.g. sizes = [8, 32, 32, 128] carries
# len = 131072 = 32*32*128. So the outer entry is an iteration count applied on top of len, and DDR
# traffic for one BD is outer * len * sizeof(elem). A stride-0 outer dim re-reads the same address
# range: the shim DMA has no cache, so those are real repeated DDR reads, which is exactly how A
# costs 8 MB at cols=1 for 1 MB of unique data.
#
# Device-free. Run:  .venv-iron/bin/python scripts/gemm_ddr_bytes.py --suffixes <a> <b> ...
import argparse
import json
import os
import re
import sys

ELEM_BYTES = {"bf16": 2, "f32": 4, "i8": 1}

# aie.dma_bd(%argN : memref<LENxTYPE> offset = O len = L sizes = [a, b, c, d] strides = [...])
BD_RE = re.compile(
    r"aie\.dma_bd\(%(\w+)\s*:\s*memref<(\d+)x(bf16|f32|i8)>"
    r"[^)]*?len\s*=\s*(\d+)\s*sizes\s*=\s*\[([^\]]*)\]\s*strides\s*=\s*\[([^\]]*)\]"
)


def mlir_path(build_dir, suffix):
    """Top-level generated MLIR. PROFILE=production keeps no .prj, and both carry the same shim BDs."""
    for p in (os.path.join(build_dir, f"aie_{suffix}.mlir"),
              os.path.join(build_dir, f"aie_{suffix}.mlir.prj", "input_with_addresses.mlir")):
        if os.path.exists(p):
            return p
    sys.exit(f"{suffix}: no generated MLIR under {build_dir}")


def account(build_dir, suffix):
    path = mlir_path(build_dir, suffix)
    with open(path) as f:
        body = f.read()
    seq = body[body.index("aie.runtime_sequence("):]

    per_arg, bds, viol = {}, 0, []
    for m in BD_RE.finditer(seq):
        arg, total_elems, ty, ln = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        sizes = [int(x) for x in m.group(5).split(",")]
        strides = [int(x) for x in m.group(6).split(",")]
        inner = 1
        for s in sizes[1:]:
            inner *= s
        # The convention this instrument rests on. Check it per BD rather than trusting it: a BD
        # where len is the product of ALL four sizes would be double-counted by the outer factor.
        if inner != ln:
            viol.append({"arg": arg, "len": ln, "inner_product": inner, "sizes": sizes})
        outer = sizes[0]
        e = per_arg.setdefault(arg, {"type": ty, "unique_elems": total_elems, "ddr_elems": 0,
                                     "bds": 0, "outer_reps": set(), "stride0_outer": False})
        e["ddr_elems"] += outer * ln
        e["bds"] += 1
        e["outer_reps"].add(outer)
        if strides[0] == 0:
            e["stride0_outer"] = True
        bds += 1

    out, total_ddr, total_unique = {}, 0, 0
    for arg, e in sorted(per_arg.items()):
        eb = ELEM_BYTES[e["type"]]
        ddr, uniq = e["ddr_elems"] * eb, e["unique_elems"] * eb
        total_ddr += ddr
        total_unique += uniq
        out[arg] = {
            "type": e["type"], "bds": e["bds"],
            "outer_reps": sorted(e["outer_reps"]), "stride0_outer": e["stride0_outer"],
            "unique_mib": round(uniq / 2**20, 3), "ddr_mib": round(ddr / 2**20, 3),
            "amplification": round(ddr / uniq, 3) if uniq else None,
        }
    return {
        "suffix": suffix, "mlir": os.path.relpath(path, build_dir), "shim_bds": bds,
        "per_operand": out,
        "unique_mib": round(total_unique / 2**20, 3),
        "ddr_mib": round(total_ddr / 2**20, 3),
        "len_convention_violations": viol,
    }


def main(o):
    rows = [account(o.build_dir, s) for s in o.suffixes]
    for r in rows:
        print(f"\n---------- {r['suffix']} ----------")
        print(f"  {r['shim_bds']} shim BDs   unique {r['unique_mib']} MiB   "
              f"DDR {r['ddr_mib']} MiB   amplification {round(r['ddr_mib']/r['unique_mib'], 2)}x")
        for arg, d in r["per_operand"].items():
            s0 = " stride0-outer" if d["stride0_outer"] else ""
            print(f"    {arg:6s} {d['type']:4s} bds={d['bds']:2d} outer={d['outer_reps']}{s0}"
                  f"  unique {d['unique_mib']:8.3f}  DDR {d['ddr_mib']:8.3f} MiB"
                  f"  x{d['amplification']}")
        if r["len_convention_violations"]:
            print(f"    WARN: {len(r['len_convention_violations'])} BD(s) where len != product of "
                  f"inner 3 sizes -- byte counts for this arm are NOT trustworthy:")
            for v in r["len_convention_violations"][:4]:
                print(f"      {v}")
    os.makedirs(o.artifacts, exist_ok=True)
    out = os.path.join(o.artifacts, "gemm_ddr_bytes.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="mlir-aie/programming_examples/basic/"
                                          "matrix_multiplication/whole_array/build")
    p.add_argument("--suffixes", nargs="+", required=True)
    p.add_argument("--artifacts", default="artifacts")
    main(p.parse_args())
