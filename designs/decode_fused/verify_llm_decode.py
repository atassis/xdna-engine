#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Greedy token-parity gate for a fused decode ELF, on device.

Drives the SAME graph gen_llm_decode.py built (via build_graph, not a re-typed runlist) one token at
a time through the deep-C constant-ELF protocol -- host writes `x`, the RoPE angle row and the
scratchpad params (kv_off/sm_mask, plus attn_window when the build declares a dynamic window), then
ONE dispatch -- and compares the greedy token sequence against a HuggingFace
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
from redispatch_check import assert_redispatch_identical  # noqa: E402
from iron.common.kv_layout import KVLayout  # noqa: E402

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


def window_len(pos, granule):
    """The attended length for `pos`: positions the cache holds (pos+1), rounded UP to `granule`.

    Mirrors rust/npu-engine/src/llm/npu_decode.rs::window_len exactly -- same rounding, no clamp
    (the caller mins against dims.S, same as the Rust call site does against bucket.window). Two
    implementations of one formula that disagree is precisely what this parity gate exists to
    catch, and it would present as a mismatch blamed on the kernel rather than on the host.
    """
    need = pos + 1
    return -(-need // granule) * granule


# Self-check at import time, not a separate test file: mirrors the Rust unit test
# (window_len_rounds_the_attended_length_up_to_the_granule) verbatim, plus the dims.S clamp the
# Rust side applies at its call site. A second granule (96, not just 128) is required -- a
# granule-independent bug (e.g. a hardcoded 128) would still pass the first block.
assert window_len(0, 128) == 128, "pos 0 needs 1 position -- the first granule"
assert window_len(127, 128) == 128, "pos 127 needs exactly 128 -- still fits"
assert window_len(128, 128) == 256, "pos 128 needs 129 -- one past, next granule"
assert window_len(0, 96) == 96
assert window_len(95, 96) == 96
assert window_len(96, 96) == 192
assert min(window_len(200, 96), 192) == 192, "clamp: 288 > dims.S=192 caps to the built window"


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
    ap.add_argument("--emit-topk", default=None,
                    help="TIER 2 capture: write this run's per-step top-K token ids and logits to "
                         "a JSON for scripts/gate_token_set.py. With this set the script CAPTURES "
                         "rather than judges -- its own 1:1 parity line stays as a note and the "
                         "exit status reports whether the DEVICE RUN worked, not whether the "
                         "tokens matched, because the verdict is the token-set gate's to give.")
    ap.add_argument("--topk", type=int, default=5,
                    help="how many candidates per step to record (GATE_K in gate_llm_reference.py)")
    ap.add_argument("--teacher-force", action="store_true",
                    help="feed the ORACLE's tokens instead of the device's own, so each step is "
                         "judged independently. Free-running conflates one bad token with the "
                         "trajectory it then drags behind it.")
    ap.add_argument("--redispatch-check", action="store_true",
                    help="redispatch check: write step-0's "
                         "inputs once, dispatch twice with nothing rewritten in between, and "
                         "require byte-identical logits. Runs instead of the parity loop.")
    a = ap.parse_args()
    isolate_build_dir("verify")

    report_artifact_freshness(a.weights)

    ref = json.load(open(a.ref))
    prompt_ids, gen_ids = ref["prompt_ids"], ref["gen_ids"]
    margins = ref.get("margins")
    hf_ids = ref.get("hf_f32_gen_ids")
    steps = a.steps if a.steps is not None else len(gen_ids)

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    NL, S, T = md["NL"], md["S"], md["T"]
    # None unless build_graph actually built decode_layer_dp with window_parameter="attn_window"
    # (DYNAMIC_WINDOW=1 at build time AND the spec eligible) -- see gen_llm_decode.py's own comment
    # on this field: gated on both, never on the env var alone, so a stray DYNAMIC_WINDOW cannot
    # claim a param that was never actually wired into the graph.
    window_granule = md.get("window_granule")
    HD, D, VOCAB = sp.head_dim, sp.d_model, sp.vocab
    kv_layout = KVLayout(Hkv=sp.n_kv_heads, S=S, HD=HD, T=T)
    print(f"[verify] {sp.name}: {NL} layers, S={S}, kv_block={T}, vocab={VOCAB}")

    c = fused.get_callable()
    params = c.params
    if params is None:
        raise SystemExit("[verify] no runtime parameters bound -- params.txt missing from the build; "
                         "the ELF cannot be driven per-token")
    print("[verify] ParameterScratchpad bound (kv_off, sm_mask"
          + (", attn_window" if window_granule is not None else "") + ")")

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

    if a.redispatch_check:
        tok0 = prompt_ids[0]
        with xin.overwrite() as _buf:
            _buf[:] = np.asarray(embed[tok0] * scale, BF16).reshape(-1)
        with rope_buf.overwrite() as _buf:
            _buf[:] = rope_row(0, HD, sp.rope_theta_global).reshape(-1)
        params.write("kv_off", 0)
        params.write("sm_mask", 1)
        if window_granule is not None:
            # No manual << 2 here: params.write() already resolves attn_window's "core" kind from
            # params.txt and pre-shifts internally, exactly as it does for sm_mask above -- adding
            # a second shift on top would silently double it (<<4, not <<2), the wrong-window
            # class of bug the header warns about.
            params.write("attn_window", min(window_len(0, window_granule), S))
        params.sync()
        assert_redispatch_identical(c, out, label=sp.name, vocab=VOCAB)
        return

    fed = list(prompt_ids)
    produced = []
    topk_ids, topk_logits = [], []
    # Per produced step: (device top-1 logit, logit the device gave the ORACLE's token).
    # This is what classifies a mismatch. The oracle's stored `margins` describe a DIFFERENT
    # implementation's forward pass; the device's own gap describes this one.
    step_logits = []
    tok = fed[0]
    for pos in range(len(fed) + steps - 1):
        with xin.overwrite() as _buf:
            _buf[:] = np.asarray(embed[tok] * scale, BF16).reshape(-1)
        with rope_buf.overwrite() as _buf:
            _buf[:] = rope_row(pos, HD, sp.rope_theta_global).reshape(-1)
        params.write("kv_off", int(kv_layout.kv_off(pos)))
        params.write("sm_mask", int(pos + 1))
        if window_granule is not None:
            # Same shift-free call as sm_mask above -- params.write() pre-shifts "core"-kind
            # params by name from params.txt, so passing the raw length here is correct, not an
            # omission. Clamp to S (this build's dims.S): window_len alone can round past the
            # window an unbucketed build was compiled for, and attending past what the ELF's own
            # taps cover is not a smaller bug than attending short of it.
            params.write("attn_window", min(window_len(pos, window_granule), S))
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
            # The device's own top-K, captured whether or not this step matched: the token-set gate
            # needs the candidates AT the first divergence, which is not knowable in advance.
            top = np.argpartition(-lg, a.topk)[:a.topk]
            top = top[np.argsort(-lg[top])]
            topk_ids.append([int(t) for t in top])
            topk_logits.append([float(lg[t]) for t in top])
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
    if a.emit_topk:
        json.dump({
            "spec": sp.name, "backend": f"npu fused decode, {NL} layers, S={S}",
            "prompt": ref.get("prompt"), "prompt_ids": prompt_ids,
            "n_tokens": len(produced), "k": a.topk, "gen_ids": produced,
            "topk_ids": topk_ids, "topk_logits": topk_logits,
            "teacher_forced": bool(a.teacher_force),
            "note": "Device capture for scripts/gate_token_set.py. The 1:1 parity line this run "
                    "also printed is the OLD gate and is kept as a note: it demands byte-identical "
                    "tokens against one particular host implementation, which stops being "
                    "achievable as soon as a rail has two implementations of an op.",
        }, open(a.emit_topk, "w"), indent=1)
        print(f"[verify] top-{a.topk} capture -> {a.emit_topk}; the VERDICT is "
              f"scripts/gate_token_set.py's, not this line's")
        return 0
    return 0 if match == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
