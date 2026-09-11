#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-derive the precision plane's per-site byte census from a built fused decode MLIR.

`designs/decode_fused/precision.py` carries the census as constants, because a plan has to be
priced before anything is built. A constant nobody can refresh is how a figure survives past the
graph it described, so this is the refresh, and `--check` is the staleness detector:

  python scripts/precision_census.py <fused.mlir>            # print the table
  python scripts/precision_census.py <fused.mlir> --check    # exit 1 if the constants drifted

Reads the same shim BDs scripts/decode_ddr_bytes.py reads, and attributes them to SITES by the
decode layer's argument ORDER -- the list decode_layer_dp/op.py::get_arg_spec declares, which is
that order's single owner.

ONE RUNG ONLY. A rung-ladder build carries several decode_layer_dp designs, one per attention
window, and exactly one runs per token; summing them prices a token nobody dispatches. The widest
is taken by default -- it is what `main:sequence` falls back to -- and --window picks another.
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "designs", "decode_fused"))
import precision as P  # noqa: E402

ELEM = {"bf16": 2, "f32": 4, "i8": 1, "i32": 4}
BD = re.compile(r"aie\.dma_bd\(%(\w+)\s*:\s*memref<((?:\d+x)+)(bf16|f32|i8|i32)>[^)]*?"
                r"len\s*=\s*(\d+)\s+sizes\s*=\s*\[([^\]]*)\]\s+strides\s*=\s*\[([^\]]*)\]")
DEV = re.compile(r"aie\.device\(\w+\)\s*@(\w+)\s*\{")
CFG = re.compile(r"aiex\.configure\s+@(\w+)\s*\{")

# decode_layer_dp/op.py::get_arg_spec, in order. Only the sited ones are named here; the rest
# are activations and scratch and land in `unsited`.
LAYER_ARG_SITE = {2: "qkv", 4: "kv", 5: "kv", 8: "attn_o", 9: "mlp", 10: "mlp", 11: "mlp"}
LAYER_ARGS = 15


def device_bodies(src):
    out = {}
    for m in DEV.finditer(src):
        depth = 0
        for j in range(m.end() - 1, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
        out[m.group(1)] = src[m.end():j]
    return out


def arg_bytes(body):
    per = collections.Counter()
    for b in BD.finditer(body):
        arg, _n, ty, ln, sizes, _st = b.groups()
        sz = [int(x.strip()) for x in sizes.split(",")]
        outer = sz[0] if len(sz) == 4 else 1
        inner = 1
        for v in sz[1:]:
            inner *= v
        per[arg] += outer * int(ln) * ELEM[ty]
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--window", type=int, default=None,
                    help="pick a rung by its S; default is the widest")
    ap.add_argument("--tol", type=float, default=0.5, help="MB/token tolerance for --check")
    a = ap.parse_args()

    src = open(a.mlir).read()
    devs = device_bodies(src)
    top = src[src.rindex("aie.device(npu2) {"):]
    runs, cur = collections.Counter(), None
    for line in top.splitlines():
        m = CFG.search(line)
        if m:
            cur = m.group(1)
        elif "aiex.run" in line and cur:
            runs[cur] += 1

    layers = {op: n for op, n in runs.items() if "DecodeLayer" in op}
    if not layers:
        sys.exit(f"{a.mlir}: no DecodeLayerDataParallel invocations -- this tool reads the fused "
                 "decode layer's argument order and has nothing to attribute without it")
    # Each rung is a separate device; the widest window moves the most KV bytes per run.
    chosen = max(layers, key=lambda op: sum(arg_bytes(devs[op]).values()))
    if a.window is not None:
        cands = [op for op in layers if f"_S{a.window}_" in op]
        if not cands:
            sys.exit(f"no rung at window {a.window}; have {sorted(layers)}")
        chosen = cands[0]
    n_layers = layers[chosen]

    per = arg_bytes(devs[chosen])
    if len(per) > LAYER_ARGS:
        sys.exit(f"{chosen} has {len(per)} argument buffers, expected at most {LAYER_ARGS} -- the "
                 "argument order this tool attributes by has changed; re-read get_arg_spec")
    site_mb = collections.Counter()
    for arg, b in per.items():
        idx = int(arg.removeprefix("arg"))
        site_mb[LAYER_ARG_SITE.get(idx, "unsited")] += b * n_layers / 1e6

    # The head is its own GEMV, run once, and its weight is the one argument that is not small.
    for op, n in runs.items():
        if "DecodeLayer" in op:
            continue
        pa = arg_bytes(devs[op])
        if not pa:
            continue
        big, bb = pa.most_common(1)[0]
        # Every non-layer op here is the lm-head GEMV or a norm; only the head moves a weight.
        site_mb["head" if bb > 1e6 else "unsited"] += bb / 1e6
        site_mb["unsited"] += (sum(pa.values()) - bb) / 1e6

    total = sum(site_mb.values())
    print(f"{os.path.basename(a.mlir)}\n  rung {chosen}\n  {n_layers} layers\n")
    print(f"  {'site':8} {'MB/token':>9} {'share':>7}   {'recorded':>9}  {'delta':>8}")
    drift = []
    for key in sorted(P.SITES, key=lambda k: -site_mb[k]):
        got, rec = site_mb[key], P.SITES[key].mb_per_token
        print(f"  {key:8} {got:9.2f} {100 * got / total:6.1f}%   {rec:9.2f}  {got - rec:+8.2f}")
        if abs(got - rec) > a.tol:
            drift.append((key, rec, got))
    print(f"  {'unsited':8} {site_mb['unsited']:9.2f}")
    print(f"  {'TOTAL':8} {total:9.2f}            {P.CENSUS_TOKEN_MB:9.2f}  "
          f"{total - P.CENSUS_TOKEN_MB:+8.2f}")

    if a.check:
        if abs(total - P.CENSUS_TOKEN_MB) > a.tol:
            drift.append(("TOTAL", P.CENSUS_TOKEN_MB, total))
        if drift:
            print("\nDRIFT -- precision.py's census no longer describes this build:")
            for key, rec, got in drift:
                print(f"  {key}: recorded {rec:.2f}, measured {got:.2f}")
            print("Update SITES/CENSUS_TOKEN_MB and CENSUS_DATE, and re-price anything ranked "
                  "off the old table.")
            return 1
        print("\ncheck: the recorded census describes this build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
