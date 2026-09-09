#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TIER 2P's references: one per prompt length, from the pinned long prompt. No device, no IRON.

WHY LENGTHS AND NOT A LENGTH. The batched path's failure modes are all at chunk boundaries -- an
off-by-one in `kv_off`, a mask width computed from the wrong base, a padded tail that is not
actually masked. A single prompt length exercises exactly one of those geometries. The set below
spans them deliberately:

    64   under one chunk           -- mostly padding; if pad rows leaked, this breaks first
    255  one chunk, pad 1          -- the smallest non-zero padding
    256  exactly one chunk         -- no padding at all, the only clean case
    257  two chunks, pad 255       -- the first cross-chunk KV handoff, worst padding
    512  exactly two chunks        -- clean multi-chunk
    600  three chunks, pad 168     -- ragged multi-chunk
    768  exactly three chunks      -- clean, and the longest the pinned prompt admits

The prompt is real prose, pinned with its ids in tests/refs/<spec>/gate_prompt_long.json. Top-k
inclusion is only meaningful on in-distribution logits: on a synthetic id sequence every step is a
near-tie, and a gate that cannot distinguish a tie from a defect is not a gate.

  python3 scripts/make_prefill_refs.py --weights <npy dir>
"""
import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(REPO, "designs", "decode_fused"))
from llm_decode_spec import SPECS  # noqa: E402

LENGTHS = (64, 255, 256, 257, 512, 600, 768)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b", choices=sorted(SPECS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lengths", default=",".join(str(n) for n in LENGTHS))
    a = ap.parse_args()

    refdir = os.path.join(REPO, "tests", "refs", a.spec)
    pinned = os.path.join(refdir, "gate_prompt_long.json")
    if not os.path.isfile(pinned):
        raise SystemExit(f"ERROR: no pinned prompt at {pinned}")
    ids = json.load(open(pinned))["ids"]
    out = os.path.join(refdir, "prefill")
    os.makedirs(out, exist_ok=True)

    lens = [int(x) for x in a.lengths.split(",")]
    too_long = [n for n in lens if n > len(ids)]
    if too_long:
        raise SystemExit(f"ERROR: the pinned prompt is {len(ids)} tokens; {too_long} exceed it. "
                         f"Lengthen the prompt rather than truncating the gate.")
    for n in lens:
        dst = os.path.join(out, f"ref_P{n}.json")
        print(f"[refs] P={n} -> {dst}", flush=True)
        subprocess.run([sys.executable, os.path.join(_HERE, "gate_llm_reference.py"),
                        "--spec", a.spec, "--weights", a.weights,
                        "--prompt-ids", ",".join(str(i) for i in ids[:n]),
                        "--tokens", str(a.tokens), "--k", str(a.k), "--out", dst],
                       check=True)
        # `gate_llm_reference.py` stamps its own default prompt STRING when only ids are given, so
        # the file would claim "The capital of France is" beside 512 ids of something else -- a
        # reference that misreports its own subject, which is worse than one with no label. Name the
        # provenance instead. (Consequence: --backend hf cannot regenerate these, because its
        # tokenizer cross-check compares tokenizer(prompt) against prompt_ids and a slice has no
        # faithful string. Anchoring the numpy backend against hf is done on the canonical prompt.)
        j = json.load(open(dst))
        j["prompt"] = f"<gate_prompt_long.json ids[:{n}]>"
        j["prompt_source"] = {"file": "gate_prompt_long.json", "slice": n, "of": len(ids)}
        json.dump(j, open(dst, "w"), indent=1)
    print(f"[refs] {len(lens)} reference(s) in {out}")


if __name__ == "__main__":
    sys.exit(main())
