#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add (or refresh) the TIER 1 `gate` block on a prefill artifact that is already built.

The generators now emit it themselves, so this is for the artifacts that already exist -- and for
re-gating one whose tolerance rule has moved without paying for an aiecc run. It needs numpy and
ml_dtypes and nothing else: no IRON, no toolchain instance, no device. That is deliberate. An
artifact is a build product that outlives the toolchain that made it, and a correctness reference
you can only regenerate by reproducing a compiler is not a reference you can rely on.

The reference is computed from the artifact's OWN `buffers/*.bin` -- the exact bytes the device was
handed -- not from a re-run of the generator's RNG. Same seed, same numbers, but only one of the
two is a fact about this artifact.

  python3 scripts/refresh_prefill_goldens.py /mnt/data/xdna/scratch/prefill/mlp_m256
  python3 scripts/refresh_prefill_goldens.py --weights <npy dir> <full prefill artifact>

Kind is read off `meta.json`'s `output` field: `out` = MLP block, `cx` = attention block,
`xout` = the full layer stack (the only one that needs --weights, because its weights live in the
decode artifact's arena rather than in its own buffers/).
"""
import argparse
import json
import os
import sys

import numpy as np
import ml_dtypes

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "designs", "decode_fused"))
from llm_decode_spec import SPECS  # noqa: E402
from gate_numeric import ATOL_MARGIN  # noqa: E402
from prefill_ref import (attn_block, bf16, f32, gate_block, layer_stack,  # noqa: E402
                         merge_gate, mlp_block, npy_weights)

BF16 = ml_dtypes.bfloat16

# Both arms of every dataflow, always. The f32 arm is what the device is judged against; the bf16
# arm is what a faithful host implementation of the same steps costs, and that is what sizes atol.
# Recomputed here rather than read out of buffers/: the shipped bf16 golden was written by whatever
# generator revision built the artifact, and an atol sized off a stale one is a silent wrong answer.
ARMS = (("f32", f32), ("bf16", bf16))


def buf(art, name, shape=None, dtype=BF16):
    p = os.path.join(art, "buffers", f"{name}.bin")
    if not os.path.isfile(p):
        raise SystemExit(f"ERROR: {p} is missing -- this artifact cannot be re-gated from its own "
                         f"inputs. Rebuild it; do not synthesise the input.")
    a = np.fromfile(p, dtype=dtype)
    if shape is not None:
        want = int(np.prod(shape))
        if a.size != want:
            raise SystemExit(f"ERROR: {p} holds {a.size} elements, meta.json implies {want}")
        a = a.reshape(shape)
    return a


def spec_of(meta):
    # The two block generators put `spec` under `dims`; the full-stack one puts it at the top
    # level. Read both rather than picking one and calling the other malformed.
    name = meta["dims"].get("spec") or meta.get("spec")
    if name not in SPECS:
        raise SystemExit(f"ERROR: meta.json names spec {name!r}, which is not a known spec "
                         f"({sorted(SPECS)}); the reference needs eps/act/head counts")
    return SPECS[name]


def refs_mlp(art, meta):
    d = meta["dims"]
    sp = spec_of(meta)
    M, D, FF, L = d["M"], d["d_model"], d["ffn"], d["layer"]
    args = (buf(art, "x", (M, D)), buf(art, "n_pf", (D,)), buf(art, "Wg", (FF, D)),
            buf(art, "Wu", (FF, D)), buf(art, "Wd", (D, FF)), sp.eps, sp.act)
    print(f"[refresh] mlp block L{L} M={M} D={D} FF={FF}")
    return tuple({"out": np.asarray(mlp_block(*args, rnd), np.float32).reshape(M, D)}
                 for _, rnd in ARMS)


def refs_attn(art, meta):
    d = meta["dims"]
    M, S, Hq, Hkv, HD = d["M"], d["S"], d["q_heads"], d["kv_heads"], d["head_dim"]
    if d.get("causal", False):
        raise SystemExit("ERROR: this attention artifact declares causal=true, and the reference "
                         "here is the non-causal one gen_llm_prefill_attn.py builds")
    args = (buf(art, "q", (Hq, M, HD)), buf(art, "kc", (Hkv, S, HD)), buf(art, "vc", (Hkv, S, HD)))
    print(f"[refresh] attn block M={M} S={S} Hq={Hq} Hkv={Hkv} HD={HD}")
    return tuple({"cx": np.asarray(attn_block(*args, rnd), np.float32).reshape(Hq, M, HD)}
                 for _, rnd in ARMS)


def refs_stack(art, meta, weights_dir, base_arg):
    if not weights_dir or not os.path.isdir(weights_dir):
        raise SystemExit(f"ERROR: the full prefill stack needs --weights <npy dir>; its weights "
                         f"live in the decode artifact's shared arena, not in its own buffers/ "
                         f"(meta.weights_from = {meta.get('weights_from')})")
    d = meta["dims"]
    sp = spec_of(meta)
    NL, M, S, D, HD = d["layers"], d["M"], d["S"], d["d_model"], d["head_dim"]
    Hq, Hkv = d["q_heads"], d["kv_heads"]
    causal = meta.get("causal_mode", "rows" if meta.get("causal") else "none")

    # `base` is not recorded in meta, but the causal arm's own widths encode it: row 0 attends
    # base+1 positions. Recover it rather than trusting a flag, and cross-check the whole vector,
    # because a base mismatch produces a reference that is wrong in a plausible-looking way.
    widths = None
    if causal == "rows":
        w = buf(art, "sm_widths", (Hq * M,), np.int32)
        base = int(w[0]) - 1
        want = np.clip(base + np.tile(np.arange(M) + 1, Hq), 1, S).astype(np.int32)
        if not np.array_equal(w, want):
            raise SystemExit("ERROR: buffers/sm_widths.bin is not clamp(base+i+1, 1, S) repeated "
                             "per head; the causal rule this reference implements is not the one "
                             "the artifact was built with")
        widths = w[:M]
    else:
        base = base_arg

    X = buf(art, "x", (M, D))
    table = buf(art, "rope", (M, HD))
    print(f"[refresh] full stack {sp.name} L={NL} M={M} S={S} base={base} causal={causal} "
          f"-- TWO {NL}-layer CPU forwards, this takes a while")
    W = npy_weights(weights_dir)
    out = []
    for tag, rnd in ARMS:
        print(f"[refresh]   {tag} arm ...", flush=True)
        x, slabs = layer_stack(sp, W, NL, M, S, base, X, table, widths, rnd)
        d = {"xout": np.asarray(x, np.float32).reshape(M, D)}
        for l, (k, v) in enumerate(slabs):
            d[f"L{l}_kc"] = np.asarray(k, np.float32).reshape(Hkv, M, HD)
            d[f"L{l}_vc"] = np.asarray(v, np.float32).reshape(Hkv, M, HD)
        out.append(d)
    return tuple(out)


KINDS = {"out": refs_mlp, "cx": refs_attn, "xout": refs_stack}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", nargs="+")
    ap.add_argument("--weights", default=None, help="dumped .npy weight dir (full stack only)")
    ap.add_argument("--base", type=int, default=0,
                    help="chunk's first absolute position; only consulted for --causal none, "
                         "where there are no widths to recover it from")
    ap.add_argument("--margin", type=float, default=None,
                    help=f"atol headroom over the measured bf16 floor (default "
                         f"{ATOL_MARGIN}, sized to admit bfp16 block-float operands). 1.0 is the "
                         f"strictest value that a faithful bf16 host implementation still passes; "
                         f"it is recorded in meta.json as gate.atol_margin either way.")
    a = ap.parse_args()

    for art in a.artifact:
        mp = os.path.join(art, "meta.json")
        if not os.path.isfile(mp):
            raise SystemExit(f"ERROR: no meta.json in {art}")
        meta = json.load(open(mp))
        out = meta.get("output")
        if out not in KINDS:
            raise SystemExit(f"ERROR: {mp} output={out!r} is not a prefill artifact this knows "
                             f"how to reference ({sorted(KINDS)})")
        refs, floors = (KINDS[out](art, meta, a.weights, a.base) if out == "xout"
                        else KINDS[out](art, meta))
        gate = gate_block(art, refs, floors,
                          ATOL_MARGIN if a.margin is None else a.margin)
        merge_gate(mp, gate)
        for n, t in gate["tensors"].items():
            print(f"[refresh]   {n:<10} shape {t['shape']}  rms {t['rms']:.4g}  "
                  f"bf16 floor {t['bf16_floor']:.4e}  atol {t['atol']:.4e}")
        print(f"[refresh] wrote {len(refs)} float32 reference(s) + gate block "
              f"(rtol {gate['rtol']:g}, atol margin {gate['atol_margin']:g}x the bf16 floor) "
              f"-> {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
