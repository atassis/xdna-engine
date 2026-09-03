#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Greedy token-parity gate for a fused decode ELF, on device.

Drives the SAME graph gen_llm_decode.py built (via build_graph, not a re-typed runlist) one token at
a time through the deep-C constant-ELF protocol -- host writes `x`, the RoPE angle row and the two
scratchpad params, then ONE dispatch -- and compares the greedy token sequence against a HuggingFace
bf16 reference captured off-device (scripts/... -> refs/greedy_ref.json).

The graph is decode-only: there is no prefill, so the prompt is fed one token at a time through the
same path (each step appends to the KV cache) and generation continues free-running from the last
prompt token. Parity is judged on the FREE-RUNNING tokens.

  python route_b_kernels/decode_fused/verify_llm_decode.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --ref artifacts/qwen3-0.6b/refs/greedy_ref.json

Single-tenant: stop npu serve first.
"""
import argparse
import json
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph  # noqa: E402

BF16 = ml_dtypes.bfloat16


def rope_row(pos, head_dim, theta):
    """One position's angle row in the layout iron's RoPE op expects.

    iron/operators/rope/reference.py: `angles` holds INTERLEAVED [cos, sin, cos, sin, ...] along the
    last dim (length head_dim), and method_type=0 rotates two halves
    (y1 = x1*cos - x2*sin, y2 = x2*cos + x1*sin) -- the same convention as HF's rotate_half.
    NOTE this is NOT the half-split [cos..., sin...] packing mlir-air's examples use.
    """
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64)[:half] / head_dim))
    ang = pos * inv
    row = np.empty(head_dim, dtype=np.float32)
    row[0::2] = np.cos(ang)
    row[1::2] = np.sin(ang)
    return row.astype(BF16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ref", required=True, help="greedy_ref.json from the HF reference")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=None, help="free-running tokens to compare")
    ap.add_argument("--dump-logits", default=None, help="write step-0 logits to this .npy for offline compare")
    a = ap.parse_args()

    ref = json.load(open(a.ref))
    prompt_ids, gen_ids = ref["prompt_ids"], ref["gen_ids"]
    margins = ref.get("margins")
    hf_ids = ref.get("hf_f32_gen_ids")
    steps = a.steps if a.steps is not None else len(gen_ids)

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    NL, S = md["NL"], md["S"]
    HD, D, VOCAB = sp.head_dim, sp.d_model, sp.vocab
    print(f"[verify] {sp.name}: {NL} layers, S={S}, vocab={VOCAB}")

    c = fused.get_callable()
    params = c.params
    if params is None:
        raise SystemExit("[verify] no runtime parameters bound -- params.txt missing from the build; "
                         "the ELF cannot be driven per-token")
    print("[verify] ParameterScratchpad bound (kv_off, sm_mask)")

    for name, arr in weights.items():
        buf = c.get_buffer(name)
        np.copyto(buf.data, np.asarray(arr, BF16).reshape(-1))
    print(f"[verify] {len(weights)} weight buffers loaded")

    # embed_tokens doubles as the tied lm-head; the host gathers the row for the current token.
    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0

    xin = c.get_buffer("x")
    rope_buf = c.get_buffer("rope_global")
    out = c.get_buffer("logits")

    fed = list(prompt_ids)
    produced = []
    tok = fed[0]
    for pos in range(len(fed) + steps - 1):
        np.copyto(xin.data, np.asarray(embed[tok] * scale, BF16).reshape(-1))
        np.copyto(rope_buf.data, rope_row(pos, HD, sp.rope_theta_global).reshape(-1))
        params.write("kv_off", int(pos * HD))
        params.write("sm_mask", int(pos + 1))
        params.sync()
        c()
        lg = np.asarray(out.data[:VOCAB], dtype=np.float32)
        if a.dump_logits and pos == 0:
            np.save(a.dump_logits, lg)
            print(f"[verify] step-0 logits dumped to {a.dump_logits}")
        nxt = int(np.argmax(lg))
        if pos + 1 < len(fed):
            tok = fed[pos + 1]              # teacher-force through the prompt
        else:
            produced.append(nxt)
            tok = nxt
        if len(produced) >= steps:
            break

    n = min(len(produced), len(gen_ids))
    match = sum(1 for i in range(n) if produced[i] == gen_ids[i])
    print(f"\n[verify] oracle  : {gen_ids[:n]}")
    print(f"[verify] NPU     : {produced[:n]}")
    if hf_ids:
        print(f"[verify] HF f32  : {hf_ids[:n]}   (contrast only -- NOT the gate)")
    if margins:
        print(f"[verify] margins : {['%.4f' % m for m in margins[:n]]}")
    print(f"\n[verify] greedy token parity vs the oracle: {match}/{n}")
    # A mismatch at a margin near the device's own logit error is a tie the precision cannot
    # resolve, not a defect. Say which kind each one is instead of leaving it to be argued.
    for i in range(n):
        if produced[i] != gen_ids[i]:
            m = margins[i] if margins else float("nan")
            kind = "KNIFE-EDGE" if margins and m < 0.25 else "REAL"
            print(f"           step {i}: oracle {gen_ids[i]} vs NPU {produced[i]}, "
                  f"margin {m:.4f} -> {kind}")
    print("*** PARITY PASS ***" if match == n else f"*** {n-match} MISMATCH ***")
    return 0 if match == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
