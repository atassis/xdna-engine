#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Enumerate, filter, build and record GEMM tilings -- the machinery that fills gemm_tiles.json.

Four modes, and only one of them needs the NPU:

  --plan        census only. For each prefill shape, how many candidate tilings are LEGAL and
                which rule killed each of the rest. Needs no toolchain and no device; it is pure
                arithmetic over `llm_decode_spec.gemm_tiling_rejection`.
  (default)     census, then build every surviving candidate device-free through
                `gen_gemm_tile_arm.py`, and emit a manifest of artifacts to time.
  --ingest      read the device timings back, pick a winner per shape, write gemm_tiles.json.
  --seed*       write an entry WITHOUT a measurement, marked as such.

The census is not a by-product. "How much freedom do we actually have" is the question that decides
whether tile choice is worth sweeping at all, and it is answerable before a single build: the
kernel's own `static_assert`s, `op.py`'s divisibility rules, the 64 KB core L1 and the 512 KB
MemTile between them cut a 4-dimensional grid down hard, and they cut it differently per shape.
`ctx` (Nout=128) is the extreme case -- it is why the generators already carried a hand-written
`tile_n=16` override for that one op.

WHY THE BUILDS GET ONE WORK DIRECTORY EACH. IRON's artifact cache is filename+mtime keyed
(`iron/common/compilation/base.py::is_available_in_filesystem`), and `GEMM.name` -- the per-operator
MLIR filename -- omits `emulate_bf16_mmul_with_bfp16`, `prio_accuracy` and `round_conv_even`, all
declared `repr=False`. Two arms differing only in those flags therefore share a `.mlir` filename and
the second links the first's design. This is not hypothetical: it happened twice on this rail on
2026-09-08, both times producing byte-identical ELFs that measured nothing. Per-arm work dirs remove
the sharing entirely; the arm's sequence NAME carries every knob as well; and every ELF is md5'd,
with a duplicate treated as a build failure rather than a curiosity.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemm_tile_registry import Registry, key_of  # noqa: E402
from llm_decode_spec import (  # noqa: E402
    SPECS, gemm_tile_census, gemm_tile_grid, largest_valid_tile_n,
)

HERE = Path(__file__).resolve().parent
ARM_GEN = HERE / "gen_gemm_tile_arm.py"
DEFAULT_OUT = Path("/mnt/data/xdna/scratch/gemm_sweep")

# The grid to BUILD from, when the legal set is not capped. Coarser than the census grid on
# purpose: the census answers "what is legal" over everything the microkernel admits, the build
# answers "which of them is fastest" and each arm costs an aiecc run.
BUILD_TILE_M = (16, 32, 64)
BUILD_TILE_K = (16, 32, 64, 128)
BUILD_TILE_N = (16, 32, 64, 128)
BUILD_COLS = (4, 8)


# ---------------------------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------------------------
def prefill_shapes(spec_name: str, M: int, S: int):
    """The GEMMs `gen_llm_prefill.py` actually builds, deduplicated by (K, N, b_col_maj).

    Deliberately NOT `LlmSpec.check_prefill`'s op list: that one carries a FUSED `qkv` at
    Nout=q_dim+2*kv_dim, and the generator does not build that -- at M>1 a token's v rows sit
    between its k rows and the next token's q rows, so q and k are three separate GEMMs. Sweeping
    a shape nothing builds would fill the registry with entries no lookup ever hits.
    """
    sp = SPECS[spec_name]
    D, FF, HD = sp.d_model, sp.ffn, sp.head_dim
    ops = [
        ("q", D, sp.q_dim, True),
        ("k", D, sp.kv_dim, True),
        ("v", D, sp.kv_dim, True),
        ("o", sp.q_dim, D, True),
        ("gate", D, FF, True),
        ("up", D, FF, True),
        ("down", FF, D, True),
        ("scores", HD, S, True),
        ("ctx", S, HD, False),
    ]
    merged = OrderedDict()
    for label, K, N, bcm in ops:
        merged.setdefault((K, N, bcm), []).append(label)
    return [{"labels": labels, "label": "/".join(labels), "M": M, "K": K, "N": N,
             "b_col_maj": bcm}
            for (K, N, bcm), labels in merged.items()]


def parse_shape(text: str):
    """`MxKxN` or `MxKxN:label`."""
    body, _, label = text.partition(":")
    try:
        M, K, N = (int(v) for v in body.lower().split("x"))
    except ValueError:
        raise SystemExit(f"ERROR: --shape wants MxKxN[:label], got {text!r}")
    return {"labels": [label] if label else [], "label": label or f"{M}x{K}x{N}",
            "M": M, "K": K, "N": N, "b_col_maj": True}


# ---------------------------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------------------------
def census_for(shape, *, emulate, prio_accuracy, grid=None, check_memtile=True):
    legal, rejected = gemm_tile_census(shape["M"], shape["K"], shape["N"], bfp16=emulate,
                                       prio_accuracy=prio_accuracy, check_memtile=check_memtile,
                                       grid=grid)
    return {
        "shape": [shape["M"], shape["K"], shape["N"]],
        "b_col_maj": shape["b_col_maj"],
        "grid": len(legal) + sum(len(v) for v in rejected.values()),
        "legal": len(legal),
        "legal_candidates": [list(c) for c in legal],
        "rejected": {code: len(v) for code, v in sorted(rejected.items())},
        "rejected_example": {code: v[0][1] for code, v in sorted(rejected.items())},
    }


def print_census(name, cen, *, verbose_examples=True):
    pct = 100.0 * cen["legal"] / cen["grid"] if cen["grid"] else 0.0
    M, K, N = cen["shape"]
    print(f"\n=== {name}  M={M} K={K} N={N} b_col_maj={cen['b_col_maj']} ===")
    print(f"  legal: {cen['legal']} / {cen['grid']} candidates ({pct:.1f}%)")
    for code, n in sorted(cen["rejected"].items(), key=lambda kv: -kv[1]):
        print(f"    rejected {n:>6}  [{code}]")
        if verbose_examples:
            print(f"                     e.g. {cen['rejected_example'][code]}")
    if cen["legal"]:
        best = sorted(cen["legal_candidates"])
        print(f"  legal tile_m: {sorted({c[0] for c in best})}")
        print(f"  legal tile_k: {sorted({c[1] for c in best})}")
        print(f"  legal tile_n: {sorted({c[2] for c in best})}")
        print(f"  legal cols  : {sorted({c[3] for c in best})}")


def subsample(candidates, cap):
    """A STRATIFIED subsample of the legal set -- a fixed stride through it, in grid order.

    Not "the most promising": any ranking here would be the guess this whole file exists to
    delete. Sorting by (cols, tile_m, tile_k, tile_n) and taking a fixed stride spreads the sample
    over every axis, which is what a first sweep wants; drop --max-per-shape to build them all.
    """
    ordered = sorted(candidates)
    if not cap or len(ordered) <= cap:
        return ordered
    step = len(ordered) / float(cap)
    picked = [ordered[min(len(ordered) - 1, int(i * step))] for i in range(cap)]
    seen, out = set(), []
    for c in picked:
        if tuple(c) not in seen:
            seen.add(tuple(c))
            out.append(c)
    return out


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
def build_one(arm_dir, shape, cand, *, emulate, prio_accuracy, round_conv_even, python, timeout):
    tile_m, tile_k, tile_n, cols = cand
    arm_dir.mkdir(parents=True, exist_ok=True)
    work = arm_dir / "work"
    work.mkdir(exist_ok=True)
    cmd = [python, str(ARM_GEN), "--out", str(arm_dir),
           "-M", str(shape["M"]), "-K", str(shape["K"]), "-N", str(shape["N"]),
           "--tile-m", str(tile_m), "--tile-k", str(tile_k), "--tile-n", str(tile_n),
           "--cols", str(cols), "--label", shape["label"]]
    if not shape["b_col_maj"]:
        cmd.append("--no-b-col-maj")
    if not emulate:
        cmd.append("--no-emulate")
    if prio_accuracy:
        cmd.append("--prio-accuracy")
    if not round_conv_even:
        cmd.append("--floor-rounding")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True, timeout=timeout)
    log = arm_dir / "build.log"
    log.write_text(f"$ {' '.join(cmd)}\n(cwd {work})\n\n{proc.stdout}\n{proc.stderr}")
    return proc.returncode, time.time() - t0


def run_build(args, shapes, registry_path):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    build_grid = gemm_tile_grid(bfp16=not args.no_emulate,
                                tile_ms=args.tile_m, tile_ks=args.tile_k,
                                tile_ns=args.tile_n, colss=args.cols)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sweep_root": str(out),
        "spec": args.spec, "batch": args.batch, "seq": args.seq,
        "dtype": "bf16>bf16",
        "emulate": not args.no_emulate,
        "prio_accuracy": bool(args.prio_accuracy),
        "round_conv_even": not args.floor_rounding,
        "registry": str(registry_path),
        "census_grid_size": len(gemm_tile_grid(bfp16=not args.no_emulate)),
        "build_grid_size": len(build_grid),
        "census": {}, "arms": [],
    }
    md5s = {}
    for shape in shapes:
        full = census_for(shape, emulate=not args.no_emulate,
                          prio_accuracy=args.prio_accuracy,
                          check_memtile=not args.no_memtile_filter)
        print_census(shape["label"], full)
        manifest["census"][shape["label"]] = full

        buildable = census_for(shape, emulate=not args.no_emulate,
                               prio_accuracy=args.prio_accuracy,
                               check_memtile=not args.no_memtile_filter,
                               grid=build_grid)
        cands = subsample(buildable["legal_candidates"], args.max_per_shape)
        print(f"  build grid: {buildable['legal']}/{buildable['grid']} legal"
              f" -> building {len(cands)}")
        if args.plan:
            continue
        for cand in cands:
            tile_m, tile_k, tile_n, cols = cand
            sub = f"tm{tile_m}_tk{tile_k}_tn{tile_n}_c{cols}"
            arm_dir = out / f"{shape['label'].replace('/', '-')}_m{shape['M']}k{shape['K']}n{shape['N']}" / sub
            meta_path = arm_dir / "meta.json"
            if meta_path.is_file() and (arm_dir / "gemm_arm.elf").is_file() and not args.force:
                meta = json.loads(meta_path.read_text())
                rc, secs, status = 0, 0.0, "cached"
            else:
                rc, secs = build_one(arm_dir, shape, cand, emulate=not args.no_emulate,
                                     prio_accuracy=args.prio_accuracy,
                                     round_conv_even=not args.floor_rounding,
                                     python=args.python, timeout=args.timeout)
                meta = json.loads(meta_path.read_text()) if rc == 0 and meta_path.is_file() else {}
                status = "built" if rc == 0 else "FAILED"
            entry = {
                "label": shape["label"], "labels": shape["labels"],
                "M": shape["M"], "K": shape["K"], "N": shape["N"],
                "b_col_maj": shape["b_col_maj"],
                "tile": list(cand[:3]), "cols": cols,
                "dir": str(arm_dir), "status": status, "build_s": round(secs, 1),
                "elf_md5": meta.get("elf_md5"), "elf_bytes": meta.get("elf_bytes"),
                "sequence_name": meta.get("sequence_name"),
                "budget": meta.get("budget"), "bytes": meta.get("bytes"),
                "macs": meta.get("macs"),
                "key": key_of(shape["M"], shape["K"], shape["N"],
                              emulate=not args.no_emulate, prio_accuracy=args.prio_accuracy),
            }
            if entry["elf_md5"]:
                if entry["elf_md5"] in md5s:
                    entry["status"] = "MD5-COLLISION"
                    entry["collides_with"] = md5s[entry["elf_md5"]]
                else:
                    md5s[entry["elf_md5"]] = f"{shape['label']}/{sub}"
            manifest["arms"].append(entry)
            print(f"    [{entry['status']:>13}] {sub:<28} {secs:6.1f}s  "
                  f"md5={entry['elf_md5']}")

    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    built = [a for a in manifest["arms"] if a["status"] in ("built", "cached")]
    failed = [a for a in manifest["arms"] if a["status"] == "FAILED"]
    collided = [a for a in manifest["arms"] if a["status"] == "MD5-COLLISION"]
    print(f"\n[sweep] {len(built)} arms ready, {len(failed)} failed, "
          f"{len(collided)} md5 collisions -> {mpath}")
    if collided:
        print("  *** md5 COLLISION: two arms produced the same ELF. Their measurements would be "
              "the same binary twice. ***")
        for a in collided:
            print(f"      {a['dir']} == {a['collides_with']}")
    if failed:
        for a in failed[:10]:
            print(f"      FAILED {a['dir']} (see build.log)")
    print(f"\n[next] time them on the device:\n"
          f"    bash scripts/time_gemm_tiles.sh {mpath}")
    return 1 if (failed or collided) else 0


# ---------------------------------------------------------------------------------------------
# device timing -- the ONLY mode here that opens /dev/accel
# ---------------------------------------------------------------------------------------------
def run_time_manifest(args, registry_path):
    """Dispatch every built arm through fused_elf_probe and write timings.jsonl.

    Two pieces of measurement discipline, both paid for elsewhere on this rail. The power mode is
    GATED before the first dispatch -- a sequential sweep on an unpinned NPU aliases the swept
    variable with the DVFS ramp, which is how a 1.17x DMA fan-out once read as 3.05x
    (`scripts/npu_power_mode.py`). And the arms are ROUND-ROBINED across shapes rather than run
    shape by shape, so a box that drifts over the run drifts across the whole set instead of
    landing entirely on whichever shape happened to go last.
    """
    sys.path.insert(0, str(HERE.parent.parent / "scripts"))
    from npu_power_mode import require_pinned  # noqa: E402  -- scripts/, not this package
    mode = require_pinned()

    manifest = json.loads(Path(args.time_manifest).read_text())
    arms = [a for a in manifest["arms"] if a["status"] in ("built", "cached")]
    if not arms:
        raise SystemExit(f"ERROR: no built arms in {args.time_manifest}")
    out_dir = Path(args.time_manifest).parent
    timings = out_dir / "timings.jsonl"
    probe = Path(args.probe)
    if not probe.is_file():
        raise SystemExit(f"ERROR: no fused_elf_probe at {probe} "
                         f"(cargo build --release -p npu-probes --bin fused_elf_probe)")
    env = dict(os.environ)
    if args.ld_library_path:
        env["LD_LIBRARY_PATH"] = args.ld_library_path

    # Interleave: round-robin over shapes, so shape order is not confounded with time.
    by_label = OrderedDict()
    for a in arms:
        by_label.setdefault(a["label"], []).append(a)
    order = []
    while any(by_label.values()):
        for lbl in list(by_label):
            if by_label[lbl]:
                order.append(by_label[lbl].pop(0))

    rows = []
    with timings.open("w") as fh:
        for n, a in enumerate(order, 1):
            proc = subprocess.run([str(probe), a["dir"], "--warmup", str(args.warmup),
                                   "--iters", str(args.iters)],
                                  capture_output=True, text=True, env=env)
            text = proc.stdout + proc.stderr
            (Path(a["dir"]) / "probe.log").write_text(text)
            row = dict(a)
            row.update(parse_probe(text))
            row.update({"warmup": args.warmup, "iters": args.iters, "power_mode": mode,
                        "date": time.strftime("%Y-%m-%d")})
            meta = json.loads((Path(a["dir"]) / "meta.json").read_text())
            row["emulate"] = meta["dims"]["emulate"]
            row["prio_accuracy"] = meta["dims"]["prio_accuracy"]
            row["b_col_maj"] = meta["dims"]["b_col_maj"]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            print(f"  [{n:>3}/{len(order)}] {'PASS' if row['pass'] else 'FAIL'} "
                  f"{a['label']:<10} {a['tile']}@{a['cols']}c  "
                  f"min={row['warm_min_us']} avg={row['warm_avg_us']} us  "
                  f"rel-L2={row['rel_l2']}")
    print(f"\n[time] {len(rows)} arms -> {timings}  (power mode {mode})")
    if args.no_ingest:
        print(f"[time] --no-ingest: registry untouched. Ingest with\n"
              f"    designs/decode_fused/sweep_gemm_tiles.py --ingest {timings}")
        return 0
    args.ingest = str(timings)
    return run_ingest(args, registry_path)


def parse_probe(text):
    """Pull the numbers out of one fused_elf_probe run. Missing means the arm did not get there."""
    import re

    def grab(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else None

    return {
        "warm_avg_us": grab(r"warm dispatch [^\n]*avg=([0-9.]+)"),
        "warm_min_us": grab(r"warm dispatch [^\n]*min=([0-9.]+)"),
        "single_shot_us": grab(r"single-shot \(first dispatch\): ([0-9.]+)"),
        "rel_l2": grab(r"rel-L2 = ([0-9.]+)"),
        "pass": "*** PASS" in text,
    }


# ---------------------------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------------------------
def pick_winner(rows, prefer):
    """Winner among the arms that PASSED, by a stated rule.

    `time`     lowest warm-dispatch minimum, tie-broken by the average. The minimum is the right
               statistic for a shared, DVFS-variable box: it is the least contaminated sample.
    `accuracy` lowest rel-L2 among the arms within 5% of the best time -- accuracy inside a time
               budget, not accuracy at any cost, because a tile that is 3x slower and marginally
               more accurate is not a tile anyone wants.
    """
    ok = [r for r in rows if r.get("pass") and r.get("warm_min_us")]
    if not ok:
        return None
    if prefer == "accuracy":
        best_t = min(r["warm_min_us"] for r in ok)
        near = [r for r in ok if r["warm_min_us"] <= best_t * 1.05]
        return min(near, key=lambda r: (r.get("rel_l2", float("inf")), r["warm_min_us"]))
    return min(ok, key=lambda r: (r["warm_min_us"], r.get("warm_avg_us", float("inf"))))


def run_ingest(args, registry_path):
    rows = []
    for line in Path(args.ingest).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"ERROR: {args.ingest} has no timing rows")
    reg = Registry.load(registry_path)
    by_key = {}
    for r in rows:
        by_key.setdefault(r["key"], []).append(r)
    for key, group in sorted(by_key.items()):
        win = pick_winner(group, args.prefer)
        if win is None:
            print(f"  [skip] {key}: no arm passed its gate ({len(group)} timed)")
            continue
        labels = sorted({lb for r in group for lb in r.get("labels", []) if lb})
        reg.record(win["M"], win["K"], win["N"], *win["tile"], win["cols"],
                   emulate=win.get("emulate", True),
                   prio_accuracy=win.get("prio_accuracy", False),
                   b_col_maj=win.get("b_col_maj"),
                   source="sweep", labels=labels,
                   candidates=len(group),
                   measured={
                       "warm_min_us": win.get("warm_min_us"),
                       "warm_avg_us": win.get("warm_avg_us"),
                       "single_shot_us": win.get("single_shot_us"),
                       "rel_l2": win.get("rel_l2"),
                       "ddr_bytes": win.get("bytes", {}).get("total") if win.get("bytes") else None,
                       "macs": win.get("macs"),
                       "iters": win.get("iters"), "warmup": win.get("warmup"),
                       "date": win.get("date") or time.strftime("%Y-%m-%d"),
                       "power_mode": win.get("power_mode"),
                       "probe": "fused_elf_probe",
                       "selection": args.prefer,
                       "runner_up_us": sorted(r["warm_min_us"] for r in group
                                              if r.get("pass") and r.get("warm_min_us"))[1:2] or None,
                   })
        span = [r["warm_min_us"] for r in group if r.get("pass") and r.get("warm_min_us")]
        spread = (max(span) / min(span)) if len(span) > 1 else 1.0
        print(f"  [win] {key}: {win['tile']}@{win['cols']}cols  "
              f"{win['warm_min_us']:.1f} us  (best of {len(span)} passing, "
              f"{spread:.2f}x spread over the swept set)")
    path = reg.save(registry_path)
    print(f"\n[ingest] wrote {path}  status={reg.status()}")
    return 0


# ---------------------------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------------------------
#: What the generators hardcoded before this file existed: 64/64/64 at 8 columns for every GEMM,
#: with `ctx` already overridden to tile_n=16 because 128 % (64*8) fails. Seeded so the wiring
#: change is INERT -- the build that consults the registry must produce the same ELF as the build
#: that carried the module constants, and only a sweep may move it.
SEED_TILES = (64, 64, 64, 8)
SEED_CTX_TILES = (64, 64, 16, 8)


def run_seed_current(args, registry_path):
    reg = Registry.load(registry_path) if Path(registry_path).is_file() else Registry({}, Path(registry_path))
    n = 0
    for spec_name in args.seed_specs:
        for shape in prefill_shapes(spec_name, args.batch, args.seq):
            tiles = SEED_CTX_TILES if "ctx" in shape["labels"] else SEED_TILES
            # The generators' own pick_tile_n: the widest legal tile_n, which is what they used.
            for emulate in (True, False):
                for prio in (False, True):
                    tm, tk, tn, cols = tiles
                    if shape["N"] % (tn * cols):
                        tn = largest_valid_tile_n(shape["N"], cols, emulate)
                    reg.record(shape["M"], shape["K"], shape["N"], tm, tk, tn, cols,
                               emulate=emulate, prio_accuracy=prio, source="seed",
                               b_col_maj=shape["b_col_maj"],
                               labels=[f"{spec_name}:{lb}" for lb in shape["labels"]])
                    n += 1
    path = reg.save(registry_path)
    print(f"[seed] {n} entries at the pre-registry values -> {path}  status={reg.status()}")
    return 0


def run_seed_one(args, registry_path):
    shape = parse_shape(args.seed)
    tm, tk, tn = (int(v) for v in args.tile.split(","))
    reg = Registry.load(registry_path) if Path(registry_path).is_file() else Registry({}, Path(registry_path))
    key = reg.record(shape["M"], shape["K"], shape["N"], tm, tk, tn, args.seed_cols,
                     emulate=not args.no_emulate, prio_accuracy=args.prio_accuracy,
                     source=args.source, b_col_maj=shape["b_col_maj"],
                     labels=shape["labels"])
    path = reg.save(registry_path)
    print(f"[seed] {key} = {tm}x{tk}x{tn}@{args.seed_cols}cols source={args.source} -> {path}")
    return 0


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default="qwen3-0.6b", choices=sorted(SPECS))
    ap.add_argument("--batch", type=int, default=256, help="token batch M")
    ap.add_argument("--seq", type=int, default=2048, help="compiled KV window S")
    ap.add_argument("--shape", action="append", default=[],
                    help="MxKxN[:label]; repeatable. Overrides the spec-derived shape set.")
    ap.add_argument("--label", default=None, help="label for a single --shape")
    ap.add_argument("--out", default=str(DEFAULT_OUT / time.strftime("%Y%m%d")),
                    help="sweep root; NVMe, never /tmp (tmpfs = RAM)")
    ap.add_argument("--registry", default=None, help="gemm_tiles.json to read/write")
    ap.add_argument("--plan", action="store_true", help="census only, build nothing")
    ap.add_argument("--force", action="store_true", help="rebuild arms that already have an ELF")
    ap.add_argument("--max-per-shape", type=int, default=0,
                    help="cap arms per shape (0 = every legal candidate in the build grid); the "
                         "cap takes a stratified stride over the legal set, not a ranking")
    ap.add_argument("--tile-m", type=lambda s: [int(v) for v in s.split(",")], default=list(BUILD_TILE_M))
    ap.add_argument("--tile-k", type=lambda s: [int(v) for v in s.split(",")], default=list(BUILD_TILE_K))
    ap.add_argument("--tile-n", type=lambda s: [int(v) for v in s.split(",")], default=list(BUILD_TILE_N))
    ap.add_argument("--cols", type=lambda s: [int(v) for v in s.split(",")], default=list(BUILD_COLS))
    ap.add_argument("--no-emulate", action="store_true",
                    help="build the plain-bf16 arm ((r,s,t)=(4,8,8)) instead of the bfp16 default")
    ap.add_argument("--prio-accuracy", action="store_true", help="f32 L1 accumulator")
    ap.add_argument("--floor-rounding", action="store_true")
    ap.add_argument("--no-memtile-filter", action="store_true",
                    help="do not reject on the 512 KB MemTile budget. That budget is DERIVED from "
                         "design.py's three L3<->L2 ObjectFifos, not measured; if a build the "
                         "filter rejects turns out to compile, drop the filter and say so.")
    ap.add_argument("--python", default=sys.executable, help="the IRON venv python")
    ap.add_argument("--timeout", type=int, default=1800, help="per-arm build timeout (s)")
    ap.add_argument("--ingest", default=None, help="timings .jsonl from --time-manifest")
    ap.add_argument("--time-manifest", default=None,
                    help="ON DEVICE: dispatch every built arm of this manifest and ingest the "
                         "result. The only mode here that opens /dev/accel; it gates on a pinned "
                         "power mode first (scripts/npu_power_mode.py).")
    ap.add_argument("--probe", default="rust/target/release/fused_elf_probe")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ld-library-path", default=None)
    ap.add_argument("--no-ingest", action="store_true",
                    help="--time-manifest: write timings.jsonl but leave the registry alone")
    ap.add_argument("--prefer", default="time", choices=("time", "accuracy"),
                    help="winner rule at ingest")
    ap.add_argument("--seed", default=None, help="MxKxN[:label] -- write ONE unmeasured entry")
    ap.add_argument("--tile", default="64,64,64", help="tile_m,tile_k,tile_n for --seed")
    ap.add_argument("--seed-cols", type=int, default=8, help="cols for --seed")
    ap.add_argument("--source", default="assumed", choices=("seed", "assumed"),
                    help="provenance for --seed; 'sweep' is reserved for --ingest")
    ap.add_argument("--seed-current", action="store_true",
                    help="write the pre-registry values for every prefill shape of --seed-specs")
    ap.add_argument("--seed-specs", type=lambda s: s.split(","), default=["qwen3-0.6b"])
    a = ap.parse_args()

    registry_path = Path(a.registry or os.environ.get("GEMM_TILES_JSON")
                         or (HERE / "gemm_tiles.json"))
    if a.seed_current:
        return run_seed_current(a, registry_path)
    if a.seed:
        return run_seed_one(a, registry_path)
    if a.time_manifest:
        return run_time_manifest(a, registry_path)
    if a.ingest:
        return run_ingest(a, registry_path)

    if a.shape:
        shapes = [parse_shape(s) for s in a.shape]
        if a.label and len(shapes) == 1:
            shapes[0]["label"] = a.label
            shapes[0]["labels"] = [a.label]
    else:
        shapes = prefill_shapes(a.spec, a.batch, a.seq)
    return run_build(a, shapes, registry_path)


if __name__ == "__main__":
    sys.exit(main())
