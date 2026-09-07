#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Greedy token-parity gate for a fused decode ELF, on device.

Drives the SAME graph gen_llm_decode.py built (via build_graph, not a re-typed runlist) one token at
a time through the deep-C constant-ELF protocol -- host writes `x`, the RoPE angle row and the two
scratchpad params, then ONE dispatch -- and compares the greedy token sequence against a HuggingFace
bf16 reference captured off-device (scripts/llm_decode_bf16_oracle.py -> refs/bf16_oracle.json).
NOTE tests/refs/<model>/ holds TWO refs and only one of them is a legitimate gate for a bf16
device: greedy_ref.json is HF **f32** and has no `margins`; bf16_oracle.json is the bf16
oracle and carries them. Gating bf16 silicon against the f32 ref charges it for ties it
cannot win -- Qwen3-0.6B step 5 is exactly that, and every dataflow arm 'fails' it.

The graph is decode-only: there is no prefill, so the prompt is fed one token at a time through the
same path (each step appends to the KV cache) and generation continues free-running from the last
prompt token. Parity is judged on the FREE-RUNNING tokens.

  python designs/decode_fused/verify_llm_decode.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --ref tests/refs/qwen3-0.6b/bf16_oracle.json

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
from gen_llm_decode import build_graph, report_artifact_freshness, load_weight_buffer, isolate_build_dir  # noqa: E402

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


def bf16_ulp(x):
    """One bf16 quantum at magnitude x.

    bf16 keeps 7 explicit mantissa bits, so the spacing at magnitude x is 2**(exp - 7). Two logits
    closer than this are the SAME number in the dtype the device emits: which one wins the argmax
    is decided by index order, not by the model. That makes it a derived threshold rather than the
    hardcoded 0.25 this classifier used to carry, which had no derivation and no owner.
    """
    import math

    if not math.isfinite(x) or x == 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ref", required=True,
                    help="bf16_oracle.json -- the MATCHING-PRECISION oracle. greedy_ref.json "
                         "is HF f32 and is a contrast, not a gate: a bf16 device cannot win a "
                         "step where f32 and bf16 legitimately disagree.")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=None, help="free-running tokens to compare")
    ap.add_argument("--dump-logits", default=None, help="write step-0 logits to this .npy for offline compare")
    ap.add_argument("--teacher-force", action="store_true",
                    help="feed the ORACLE's tokens instead of the device's own, so each step is "
                         "judged independently. Free-running conflates one bad token with the "
                         "trajectory it then drags behind it.")
    a = ap.parse_args()
    isolate_build_dir("verify")

    report_artifact_freshness(a.weights)

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
        load_weight_buffer(buf, arr)
    # FLUSH SCRATCH. Every weight and both KV caches live in the scratch arena, and the callable
    # syncs only input (host->device) and output (device->host) -- scratch in NEITHER direction,
    # deliberately, because it is large and "whoever loads it" is supposed to sync it. Nobody did.
    # So these writes sat in dirty host cache lines over DRAM the device then read, and what the
    # device saw depended on which lines the CPU had happened to write back.
    #
    # MEASURED 2026-09-07 with probe_decode_first_divergence.py, two identical passes of the
    # 2-layer decode: without this flush BOTH arms are nondeterministic -- the TMV arm first
    # diverges at L0_q (the Q projection, runlist index 1, 95/114 snapshots differing and a
    # different token by step 2) and the kv arm at L0_kr by 64 elements = exactly 2 x 64-byte
    # cache lines. With it, both arms are bit-identical across passes and agree token for token.
    # The "kv arm is 0/336" that this defect was localised against was luck, not a property.
    # The Rust rail has always done this -- rust/npu-engine/src/llm/npu_decode.rs:113, one bulk
    # arena.sync_to_device() after the weight load, with the write -> sync_input -> dispatch ->
    # sync_from_device contract in that module's doc. Only the Python path was missing it.
    c.scratch_buffer.device = "cpu"
    c.scratch_buffer.to("npu")
    print(f"[verify] {len(weights)} weight buffers loaded and scratch flushed to the device")

    # embed_tokens doubles as the tied lm-head; the host gathers the row for the current token.
    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0

    xin = c.get_buffer("x")
    rope_buf = c.get_buffer("rope_global")
    out = c.get_buffer("logits")

    fed = list(prompt_ids)
    produced = []
    # Per produced step: (device top-1 logit, logit the device gave the ORACLE's token).
    # This is what classifies a mismatch. The oracle's stored `margins` describe a DIFFERENT
    # implementation's forward pass; the device's own gap describes this one.
    step_logits = []
    tok = fed[0]
    for pos in range(len(fed) + steps - 1):
        np.copyto(xin.data, np.asarray(embed[tok] * scale, BF16).reshape(-1))
        np.copyto(rope_buf.data, rope_row(pos, HD, sp.rope_theta_global).reshape(-1))
        params.write("kv_off", int(pos * HD))
        params.write("sm_mask", int(pos + 1))
        params.sync()
        # ONE dispatch per position. The duplicate that used to sit here worked around
        # _sync_inputs() trusting a coherence map this harness never updates; that is fixed at the
        # source now (iron/common/sequence.py forces host residency, mirroring _sync_outputs), so
        # a second dispatch would only double the cost and mask a regression in the real fix.
        c()
        lg = np.asarray(out.data[:VOCAB], dtype=np.float32)
        if a.dump_logits and pos == 0:
            np.save(a.dump_logits, lg)
            print(f"[verify] step-0 logits dumped to {a.dump_logits}")
        nxt = int(np.argmax(lg))
        if pos + 1 < len(fed):
            tok = fed[pos + 1]              # teacher-force through the prompt
        else:
            i = len(produced)
            produced.append(nxt)
            want = gen_ids[i] if i < len(gen_ids) else nxt
            step_logits.append((float(lg[nxt]), float(lg[want])))
            # Free-running: one wrong token puts every later step on a different trajectory, so a
            # single flip reads as N failures. Teacher-forcing feeds the oracle's token instead,
            # which makes each step an independent test of the forward pass.
            tok = gen_ids[i] if (a.teacher_force and i < len(gen_ids)) else nxt
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
        if produced[i] == gen_ids[i]:
            continue
        # Classify from THIS device's own logits. The oracle's `margins` come from a different
        # forward pass and cannot say whether this device saw a tie; its own gap can. A gap at or
        # below one bf16 quantum means the two tokens are the SAME number in the emitted dtype and
        # the argmax was decided by index order -- not a defect the model can be held to.
        if i < len(step_logits):
            got_lg, want_lg = step_logits[i]
            gap = got_lg - want_lg
            ulp = bf16_ulp(got_lg)
            if gap <= ulp:
                kind = f"TIE (gap {gap:.4f} <= one bf16 ulp {ulp:.4f} at {got_lg:.4f})"
            elif gap <= 2 * ulp:
                kind = f"NEAR-TIE (gap {gap:.4f}, {gap / ulp:.1f} ulp)"
            else:
                kind = f"REAL (gap {gap:.4f} = {gap / ulp:.1f} ulp, well above the dtype quantum)"
        else:
            # No logits captured for this step -- say so rather than defaulting to the most
            # confident verdict, which is what the old `else "REAL"` branch did whenever the ref
            # simply had no margins.
            kind = "UNCLASSIFIED (no device logits captured for this step)"
        extra = f", host margin {margins[i]:.4f}" if margins and i < len(margins) else ""
        print(f"           step {i}: oracle {gen_ids[i]} vs NPU {produced[i]}{extra} -> {kind}")
    print("*** PARITY PASS ***" if match == n else f"*** {n-match} MISMATCH ***")
    return 0 if match == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
