#!/usr/bin/env python3
"""Export every design the S2 decoder chain builds as a `final.xclbin` + `insts.bin` artifact pair
the Rust engine can load -- the bridge from the Python bring-up scaffold (`iron.jit`, in-process)
to the pre-built-artifact model `npu-engine` already uses for every other kernel.

HOW THE DESIGN SET IS ENUMERATED, and why not hardcoded. `decoder_chain.run_head/run_stage/
run_tail` are the real forward pass (head -> stage1..4 -> tail); each op call bottoms out in
`window_driver._run`, which builds ONE `CallableDesign` per distinct (symbol, shim source, shapes,
flags, resident_depth) and memoises it in `window_driver._DESIGNS`/`_BUILD_LOG` (see
window_driver.py's 2026-09-01 `_run`/`_get_design` split). Driving `run_head`/`run_stage`/
`run_tail` with `window_driver.BUILD_ONLY = True` walks that exact real graph -- real GGUF weight
shapes, real per-stage `ci_chunk`/`resident_depth` from `stage_shapes.plan()` -- and collects every
design it builds, WITHOUT a device tensor or a dispatch anywhere in window_driver.py: `_run`
returns zeros instead of calling the design. The input data is a dummy all-zero latent (only long
enough to clear every op's `M > 0` window-length assertion, via `decoder_chain.chain_offset`) --
irrelevant, since a design's shape/symbol/shim never depends on the window offset or the data,
only on channel counts and the fixed op parameters, which come from the real GGUF and the real
`plan()`. That is what makes this "byte-identical to the Python path": the shim text handed to
`design.compile()` is the exact string `window_driver._conv_chunk`/`_conv_transpose_chunk`/`snake`
would hand to `_run` on a real device dispatch.

ONE `name` (window_driver's own dispatch tag, e.g. "conv_s1u0_s11x1_k128") can belong to TWO
designs: `conv()`'s ci_chunk loop reuses the same tag for every input-channel chunk of one op, but
the FIRST chunk carries the residual `add` and later chunks don't, which is a different `symbol`
and shim body -- so `symbol` (unique per design, confirmed 68/68 on the real chain) is what names
the exported directory, not `name`.

WHY `iron.set_current_device(NPU2())` runs before anything else: `bricklib._build_streamed`
(route_b_kernels/bricks/_verify/bricklib.py) targets whatever `iron.get_current_device()` returns,
and that function's default path PROBES the live NPU runtime the first time nothing has bound a
device explicitly -- exactly the /dev/accel0 touch this export must never make (device-free, NPU is
single-tenant and busy with the main session's job). Binding NPU2() up front short-circuits that
probe (`hostruntime._CURRENT_DEVICE is not None` wins outright). This box's NPU is npu2, Gorgon
Point, 8 columns (scripts/toolchain_up.sh), the same architecture the probe would have resolved to,
so the generated MLIR and compiled xclbin are unaffected -- this is what every other route_b_kernels
generator script already does explicitly (`from aie.iron.device import NPU2; Program(NPU2(), ...)`);
bricklib's verify harness is the one place in this codebase that leans on the implicit probe instead,
because it always ran with a live, already-locked device.

    python3 export_codec_artifacts.py <out_dir> [--only SUBSTRING]

`--only` filters by substring against the design's directory name / op role / graph group (e.g.
`--only stage4` or `--only tail`) -- every design is still ENUMERATED (cheap: no compile happens
during enumeration), only the compile step is skipped for non-matches, so `manifest.json` always
reflects what is actually ON DISK in `<out_dir>` after the run, not what --only was passed this time.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent          # this worktree
WS = REPO.parent


def _toolchain_env():
    """Reproduce bricks/_verify/run.sh's env (instance from THIS worktree's toolchain.lock, venv
    from wherever a `.venv-iron` exists) without its NPU lock or `exec` -- this process never
    dispatches, so it never needs the device queue. See run.sh's own comment for why instance and
    venv have to come from two different places."""
    inst = subprocess.check_output(
        [str(REPO / "scripts" / "toolchain_up.sh")], text=True).strip()
    if not inst:
        raise RuntimeError("toolchain_up.sh returned no instance dir")
    venv = os.environ.get("BRICK_VENV")
    if not venv:
        for cand in [REPO / ".venv-iron"] + sorted(WS.glob("*/.venv-iron")):
            if (cand / "bin" / "python").exists():
                venv = str(cand)
                break
    if not venv:
        raise RuntimeError("no .venv-iron found; set BRICK_VENV")
    peano = next((p for p in Path(venv, "lib").glob("python*/site-packages/llvm-aie")), None)
    if peano is None:
        raise RuntimeError(f"no llvm-aie under {venv}/lib/python*/site-packages/")
    os.environ["PATH"] = f"{venv}/bin:{venv}/cc-shim:{os.environ.get('PATH', '')}"
    os.environ["AIECC_PATH"] = f"{inst}/bin/aiecc"
    os.environ["PEANO_INSTALL_DIR"] = str(peano)
    os.environ.setdefault("XRT_INC_DIR", "/usr/include")
    os.environ.setdefault("XRT_LIB_DIR", "/usr/lib")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, f"{inst}/python")
    return inst, venv


INST, VENV = _toolchain_env()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))

import aie.iron as iron                        # noqa: E402
from aie.iron.device import NPU2                # noqa: E402

iron.set_current_device(NPU2())                 # see module docstring: must precede any compile()

import window_driver as wd                      # noqa: E402
import decoder_chain as dc                       # noqa: E402
import codec_paths                                # noqa: E402
import gguf_extract as gx                          # noqa: E402


# ---- enumerate the real design set --------------------------------------------------------------

def _min_latent_len():
    """Smallest latent length that survives every stage's `M > 0` window assertion, found by
    walking `decoder_chain.chain_offset` (the SAME forward arithmetic `run_chain` uses) rather than
    re-deriving the per-stage context budget here."""
    L = 64
    while True:
        try:
            dc.chain_offset(0, L)
            return L
        except AssertionError:
            L += 32


def enumerate_designs():
    """Drive the real head->stage1..4->tail graph in BUILD_ONLY mode and return a list of dicts,
    one per distinct design, in build order.

    The input is an all-zero dummy latent; only its LENGTH matters (see module docstring). Weights
    are the real GGUF tensors -- decoder_chain.run_* need them for real shapes anyway, and BUILD_ONLY
    never reads their values.
    """
    gguf_path = codec_paths.gguf()
    cache = {}

    def g(name):
        if name not in cache:
            cache[name] = gx.load(gguf_path, name).astype(np.float32)
        return cache[name]

    z = np.zeros((1024, _min_latent_len()), np.float32)
    wd.BUILD_ONLY = True
    wd._BUILD_LOG.clear()

    groups = []  # (label, build_log slice)

    def call(label, fn, *a, **kw):
        start = len(wd._BUILD_LOG)
        out = fn(*a, **kw)
        groups.append((label, start, len(wd._BUILD_LOG)))
        return out

    chain = call("head", dc.run_head, z, g)
    for stage in (1, 2, 3, 4):
        chain = call(f"stage{stage}", dc.run_stage, chain, stage, g, f"s{stage}")
    call("tail", dc.run_tail, chain, g)

    group_of = [None] * len(wd._BUILD_LOG)
    for label, start, end in groups:
        group_of[start:end] = [label] * (end - start)

    entries = []
    for i, (name, key, design, op_meta) in enumerate(wd._BUILD_LOG):
        (symbol, shim_text, n_tiles, in_tile, out_numel, resident_len, flags,
         resident_depth) = key
        group = group_of[i]
        op, stage, unit = _classify(group, op_meta)
        entries.append(dict(
            design_name=symbol[3:] if symbol.startswith("wd_") else symbol,
            name=name, symbol=symbol, shim_text=shim_text, group=group, op=op, stage=stage,
            unit=unit, op_meta=op_meta, design=design, n_tiles=n_tiles, in_tile=in_tile,
            out_numel=out_numel, resident_len=resident_len, flags=list(flags),
            resident_depth=resident_depth,
        ))
    return entries


# ---- op classification (labelling only -- never affects what gets compiled) ---------------------
# Mirrors the tag conventions decoder_chain.py/window_driver.py build: "s{stage}up_s{stage}sn"
# (upsample's snake), "s{stage}up_s{stage}ct_k{chunk}" (its conv_transpose), "s{stage}u{unit}_
# s{stage}{a,dil,b,1x1}[_k{chunk}]" (a residual unit's snake/dilated-conv/snake/pointwise-conv).
# Falls back to an "unknown:" label rather than raising -- a labelling miss must never abort an
# export whose xclbin/insts are otherwise correct.
_UP_SNAKE = re.compile(r'^s\d+up_s\d+sn$')
_RES_SNAKE_A = re.compile(r'^s\d+u(\d+)_s\d+a$')
_RES_SNAKE_B = re.compile(r'^s\d+u(\d+)_s\d+b$')
_RES_CONV = re.compile(r'^s\d+u(\d+)_s\d+(?:dil|1x1)_k\d+$')


def _classify(group, meta):
    kind, tag = meta.get("kind"), meta.get("tag", "")
    if group == "head":
        return "head_conv", None, None
    if group == "tail":
        return ("tail_snake" if kind == "snake" else "tail_conv"), None, None
    stage = int(group[len("stage"):])
    if kind == "conv_transpose":
        return f"stage{stage}_upsample_convtranspose", stage, None
    if kind == "snake":
        if _UP_SNAKE.match(tag):
            return f"stage{stage}_upsample_snake", stage, None
        m = _RES_SNAKE_A.match(tag) or _RES_SNAKE_B.match(tag)
        sub = "snake_a" if _RES_SNAKE_A.match(tag) else "snake_b"
        if m:
            return f"stage{stage}_res{m.group(1)}_{sub}", stage, int(m.group(1))
    if kind == "conv":
        m = _RES_CONV.match(tag)
        unit = int(m.group(1)) if m else None
        sub = "dil" if meta.get("k") == 7 else "1x1"
        return f"stage{stage}_res{unit}_{sub}", stage, unit
    return f"unknown:{group}:{kind}:{tag}", stage, None


# ---- compile + write artifacts -------------------------------------------------------------------

def _read_toolchain_pin():
    text = (REPO / "toolchain.lock").read_text()
    out = {}
    for var in ("MLIR_AIE_FORK_COMMIT", "PEANO_FORK_COMMIT"):
        m = re.search(rf'^{var}=(\S+)', text, re.M)
        out[var] = m.group(1) if m else None
    return out


_CB_RE = re.compile(r"cb \d+")


def _canonical_shim(text):
    """window_driver's shim_text embeds `_CB` (`time.time()`-derived) as a `cb {_CB}` comment
    token, PURELY as a per-process cache-buster for bricklib's own JIT key -- so every fresh
    process produces a byte-different shim for an otherwise-identical design. Strip it before
    hashing for idempotency, or a plain sha256 of `shim_text` would never match across runs and
    "compare the shim sha256" (the constraint's own prescribed check) could never hold."""
    return _CB_RE.sub("cb 0", text, count=1)


def _up_to_date(meta_path, xclbin_path, inst_path, symbol, flags, shim_sha):
    """True if a previous export of this exact design is already on disk -- compared against the
    recorded (canonicalised) shim sha256, not file mtimes or presence alone, so any real change to
    the generated shim (shape, flags, kernel body) forces a rebuild."""
    if not (meta_path.exists() and xclbin_path.exists() and inst_path.exists()):
        return False
    try:
        prev = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (prev.get("shim_sha256") == shim_sha and prev.get("symbol") == symbol
            and prev.get("compile_flags") == flags)


def export_one(entry, out_dir):
    ddir = out_dir / entry["design_name"]
    ddir.mkdir(parents=True, exist_ok=True)
    xclbin_path, inst_path, meta_path = ddir / "final.xclbin", ddir / "insts.bin", ddir / "meta.json"
    shim_sha = hashlib.sha256(_canonical_shim(entry["shim_text"]).encode()).hexdigest()

    if _up_to_date(meta_path, xclbin_path, inst_path, entry["symbol"], entry["flags"], shim_sha):
        meta = json.loads(meta_path.read_text())
        return dict(entry, build_s=0.0, skipped=True, xclbin_bytes=xclbin_path.stat().st_size,
                    out_bytes=meta["buffer_bytes"]["out"])

    (ddir / "shim.cc").write_text(entry["shim_text"])
    t0 = time.time()
    entry["design"].compile(xclbin_path=xclbin_path, inst_path=inst_path)
    build_s = time.time() - t0

    n_tiles, in_tile = entry["n_tiles"], entry["in_tile"]
    out_numel, resident_len = entry["out_numel"], entry["resident_len"]
    in_bytes, out_bytes = n_tiles * in_tile * 4, n_tiles * out_numel * 4
    resident_bytes = resident_len * 4
    insts_bytes = inst_path.stat().st_size
    xclbin_bytes = xclbin_path.stat().st_size

    meta = dict(
        design_name=entry["design_name"], symbol=entry["symbol"], op=entry["op"],
        stage=entry["stage"], unit=entry["unit"], group=entry["group"],
        dispatch_tag=entry["name"], op_params=entry["op_meta"],
        n_tiles=n_tiles, in_tile=in_tile, out_numel=out_numel, resident_len=resident_len,
        resident_depth=entry["resident_depth"], compile_flags=entry["flags"],
        dtypes={"in": "float32", "out": "float32",
                "resident": ("float32" if resident_len else None)},
        buffer_bytes={"in": in_bytes, "out": out_bytes, "resident": resident_bytes},
        insts_bytes=insts_bytes, insts_words=insts_bytes // 4, xclbin_bytes=xclbin_bytes,
        shim_sha256=shim_sha,
        xclbin_sha256=hashlib.sha256(xclbin_path.read_bytes()).hexdigest(),
        build_seconds=build_s,
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return dict(entry, build_s=build_s, skipped=False, xclbin_bytes=xclbin_bytes,
                out_bytes=out_bytes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--only", default=None,
                    help="substring filter against design_name/op/group; matches are compiled, "
                         "everything else is still enumerated (for manifest.json) but skipped")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries = enumerate_designs()
    print(f"enumerated {len(entries)} distinct designs from the real decoder chain", flush=True)

    rows = []
    for e in entries:
        haystack = f"{e['design_name']} {e['op']} {e['group']}"
        if args.only and args.only not in haystack:
            continue
        rows.append(export_one(e, args.out_dir))
        r = rows[-1]
        tag = "SKIP(up-to-date)" if r["skipped"] else f"{r['build_s']:6.1f}s"
        print(f"  [{tag:>16s}] {r['design_name']:38s} {r['n_tiles']:5d}x{r['in_tile']:<6d} "
              f"out={r['out_bytes']:7d}B xclbin={r['xclbin_bytes']:8d}B", flush=True)

    # Built from whatever `meta.json`s are ON DISK, not just the designs `rows` touched this run --
    # a narrow --only must not make the manifest forget designs a PRIOR invocation already exported.
    manifest_designs = []
    for md in sorted(args.out_dir.glob("*/meta.json")):
        try:
            m = json.loads(md.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        manifest_designs.append(dict(
            name=m["design_name"], dir=md.parent.name, op=m["op"], stage=m["stage"],
            group=m["group"], xclbin_bytes=m["xclbin_bytes"], build_seconds=m["build_seconds"]))
    manifest = dict(toolchain=_read_toolchain_pin(), designs=manifest_designs)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{'design':38s} {'tiles':>12s} {'out B':>9s} {'xclbin B':>10s} {'build s':>8s}")
    for r in rows:
        print(f"{r['design_name']:38s} {r['n_tiles']:5d}x{r['in_tile']:<6d} {r['out_bytes']:9d} "
              f"{r['xclbin_bytes']:10d} {r['build_s']:8.2f}")
    total_s = sum(r["build_s"] for r in rows)
    print(f"\n{len(rows)}/{len(entries)} designs exported to {args.out_dir}, "
          f"{total_s:.1f}s total build time")


if __name__ == "__main__":
    main()
