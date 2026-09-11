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
from gen_llm_decode import (build_graph, report_artifact_freshness, load_weight_buffer,  # noqa: E402
                            isolate_build_dir, QUANT_MLP_DTYPE, QUANT_MLP_GROUP)
from redispatch_check import assert_redispatch_identical  # noqa: E402
from iron.common.kv_layout import KVLayout  # noqa: E402

BF16 = ml_dtypes.bfloat16


def rope_row(pos, head_dim, theta, partial=None):
    """One position's angle row in the layout iron's RoPE op expects.

    iron/operators/rope/reference.py: `angles` holds INTERLEAVED [cos, sin, cos, sin, ...] along the
    last dim (length head_dim), and method_type=0 rotates two halves
    (y1 = x1*cos - x2*sin, y2 = x2*cos + x1*sin) -- the same convention as HF's rotate_half.
    NOTE this is NOT the half-split [cos..., sin...] packing mlir-air's examples use.

    `partial` is the checkpoint's partial_rotary_factor for rope_type "proportional": zero the
    inverse frequency past `int(f * head_dim // 2)` pairs, keeping the FULL head_dim width, because
    a zero frequency is the identity rotation. The exponent's denominator stays head_dim -- that is
    what makes it "proportional" rather than ordinary partial rotary, which divides by the rotated
    width. Same rule as rust/npu-engine/src/llm/npu_decode.rs::rope_row.

    THIS IS THE THIRD COPY of this arithmetic (here, the Rust host, and transformers itself). The
    two Python ones cannot merge with the Rust one, and the proportional/default distinction is
    exactly the kind of detail that drifts between copies, so the Rust side is gated against
    transformers directly and this one is gated against the Rust side by producing the same bytes.
    """
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64)[:half] / head_dim))
    if partial is not None:
        inv[int(partial * head_dim // 2):] = 0.0
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
    ap.add_argument("--trace-residual", action="store_true",
                    help="after the first dispatch, print the RMS of every layer's residual buffer "
                         "x0..xNL. The host maths keeps this O(1) at every depth -- each layer's "
                         "input_layernorm renormalises the carry and re-injects O(1) contributions, "
                         "so it CANNOT compound. A device stream that instead tracks "
                         "prod(layer_scalar) is the defect, and this says which layer it starts at.")
    ap.add_argument("--tokenizer", default=None,
                    help="path to tokenizer.json, read directly for --smoke-prompt decoding")
    ap.add_argument("--smoke-prompt", default=None,
                    help="run WITHOUT a reference: comma-separated PROMPT IDS to generate from and print the tokens "
                         "and their detokenized text. A weak gate on purpose -- it cannot catch a "
                         "near-miss the way token parity can -- but it needs NO reference model in "
                         "memory, where a full-depth bf16 oracle for a 12B is ~23 GB and will swap "
                         "a shared box into the ground. Coherent text still rules out the failure "
                         "modes seen here: a holed lm-head output and a collapsed residual both "
                         "produce garbage, not sentences.")
    ap.add_argument("--ref", required=False, default=None,
                    help="bf16_oracle.json -- the MATCHING-PRECISION oracle. greedy_ref.json "
                         "is HF f32 and is a contrast, not a gate: a bf16 device cannot win a "
                         "step where f32 and bf16 legitimately disagree.")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=None, help="free-running tokens to compare")
    ap.add_argument("--dump-logits", default=None, help="write step-0 logits to this .npy for offline compare")
    ap.add_argument("--host-lm-head", action="store_true",
                    help="compute the logits on the HOST from the device's own `xf`, instead of "
                         "reading the on-device lm-head output. `xf` is verified correct at every "
                         "depth (the per-node bisect puts it at 2.79e-02 at 6 layers, the same as "
                         "the 5-layer arm that passes) while the on-device lm-head output is not: "
                         "it comes back with whole runs unwritten and a few elements wildly wrong, "
                         "in specific column blocks, from 6 layers up. This isolates that defect "
                         "so the REST of the model can be gated, and is slow -- a 262144x3840 "
                         "matvec per token -- so it is a diagnostic, not a shipping path.")
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

    if a.ref is None and not a.smoke_prompt:
        raise SystemExit("[verify] need --ref (token parity) or --smoke-prompt (no-reference smoke run)")
    ref = json.load(open(a.ref)) if a.ref else {"prompt": a.smoke_prompt, "gen_ids": [], "margins": []}
    # A reference captured at a DIFFERENT depth is not a reference for this build. It presents as a
    # token mismatch at step 0, which reads as a device defect; refuse instead. `layers` is absent
    # on refs captured before it was recorded, and absent means full depth.
    ref_layers = ref.get("layers")
    if a.smoke_prompt:
        # No reference model, and no `transformers` either -- it is not in the build venv and adding
        # it to reach a tokenizer would be the wrong dependency for the wrong reason. The prompt ids
        # come from an existing ref (same prompt, same tokenizer, so the same ids), and decoding
        # reads tokenizer.json's vocab directly: it is a plain id->piece map plus SentencePiece's
        # U+2581 word-boundary convention.
        ref["prompt_ids"] = [int(x) for x in a.smoke_prompt.split(",")]
        _vocab = json.load(open(a.tokenizer))["model"]["vocab"]
        _inv = {v: k for k, v in (_vocab.items() if isinstance(_vocab, dict)
                                  else ((t, i) for i, t in enumerate(_vocab)))}

        def _decode(ids):
            return "".join(_inv.get(int(i), f"<{i}>") for i in ids).replace("\u2581", " ")

        _tok = type("T", (), {"decode": staticmethod(
            lambda ids, skip_special_tokens=False: _decode(ids))})()
        print(f"[verify] SMOKE RUN, no reference. prompt ids {ref['prompt_ids']}", file=sys.stderr)
    elif ref_layers != a.layers:
        raise SystemExit(
            f"reference was captured at layers={ref_layers} but this build is layers={a.layers}. "
            f"Re-capture with scripts/llm_hf_bf16_ref.py --layers {a.layers}.")
    # The reference's WEIGHT FORMAT, checked the same way and for the same reason as its depth: a
    # format disagreement presents as a token mismatch, which reads as a device defect. It is not.
    # An int4 arm gated against a bf16 oracle cost this project a bisect for a bug that was not
    # there -- symmetric RTN int4 is 12.3% weight rel-L2 against a gate whose margins are 4.0/2.0/
    # 0.0. `quant` is absent on refs captured before it was recorded, and absent means bf16.
    if not a.smoke_prompt:
        rq = ref.get("quant") or {"dtype": "bf16", "group": 0}
        # Same authority order build_graph uses: a packed dump's quant.json wins over the env,
        # because the bytes are already on disk in it. Read here rather than importing the
        # generator's module globals, which the manifest only overrides once build_graph runs --
        # and the point of this check is to fire BEFORE the build is paid for.
        _qm = os.path.join(a.weights, "quant.json")
        _j = json.load(open(_qm)) if os.path.isfile(_qm) else None
        # PACKED, not the file's existence, is what makes the manifest the authority -- build_graph
        # gates on `if PACKED:` for the same reason. dump_llm_weights.py writes quant.json
        # UNCONDITIONALLY, so a plain bf16 dump carries {"dtype": "bf16", "packed": []}, and reading
        # that as an authority makes an int8 build look like a bf16 one and refuses a correct run.
        bq = ({"dtype": _j["dtype"], "group": int(_j["group_size"])}
              if _j and _j.get("packed") else {"dtype": QUANT_MLP_DTYPE, "group": QUANT_MLP_GROUP})
        if (rq["dtype"], int(rq["group"])) != (bq["dtype"], int(bq["group"])) and not (
                rq["dtype"] == "bf16" and bq["dtype"] == "bf16"):
            raise SystemExit(
                f"reference is {rq['dtype']} g{rq['group']} but this build's MLP weight format is "
                f"{bq['dtype']} g{bq['group']}. Exact token parity across two weight formats is "
                f"not a defect test -- it charges the implementation for the format's own error. "
                f"Re-capture with scripts/llm_hf_bf16_ref.py --quant-dtype {bq['dtype']} "
                f"--quant-group {bq['group']}, or build at the reference's format.")

    prompt_ids, gen_ids = ref["prompt_ids"], ref["gen_ids"]
    margins = ref.get("margins")
    hf_ids = ref.get("hf_f32_gen_ids")
    steps = a.steps if a.steps is not None else len(gen_ids)
    # A SMOKE RUN HAS NO gen_ids TO SIZE ITSELF FROM, so this used to come out 0: the harness built
    # the graph, loaded every weight, dispatched nothing, printed "generated ids : []" and exited 0.
    # Measured 2026-09-10: a full-depth Gemma-4 run cost 3 minutes and 11.3 GB to produce exactly
    # that, and it reads as a pass at a glance. Default it instead, and refuse zero outright --
    # a run that generates nothing is never what the caller meant.
    if a.smoke_prompt and a.steps is None:
        steps = 8
    if steps <= 0:
        raise SystemExit(f"[verify] steps resolved to {steps}: nothing would be generated. Pass "
                         f"--steps N, or use a --ref whose gen_ids are non-empty.")

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    n_weight_buffers = len(weights)   # the loop below empties `weights` to halve peak residency
    NL, S, T = md["NL"], md["S"], md["T"]
    # None unless build_graph actually built decode_layer_dp with window_parameter="attn_window"
    # (DYNAMIC_WINDOW=1 at build time AND the spec eligible) -- gated on both, never on the env
    # var alone, so a stray DYNAMIC_WINDOW cannot claim a param never wired into the graph.
    window_granule = md.get("window_granule")
    HD, D, VOCAB = sp.head_dim, sp.d_model, sp.vocab
    kv_layout = KVLayout(Hkv=sp.n_kv_heads, S=S, HD=HD, T=T)
    print(f"[verify] {sp.name}: {NL} layers, S={S}, kv_block={T}, vocab={VOCAB}")

    # THE STACK IS A LIST OF DISPATCHES, length 1 unless DECODE_SEGMENTS cut it. The arena limit
    # that forces the cut is per DISPATCH (aiex.npu.address_patch's I32 arg_plus, 4 GiB), so a
    # 48-layer Gemma-4 whose single arena is 9.13 GiB fits as three small ones. Each segment owns
    # its own layers' weights and KV caches; only the residual crosses, 7680 bytes a seam a token.
    segs = md["segments"]
    stack = []
    for si, sg in enumerate(segs):
        sc = sg["seq"].get_callable()
        if sc.params is None:
            raise SystemExit(f"[verify] segment {si} ({sg['seq'].name}): no runtime parameters "
                             f"bound -- params.txt missing from the build; the ELF cannot be "
                             f"driven per-token")
        # Each segment's weights come off the GRAPH's own buffer list, not off a filter applied
        # here. A weight that belongs to no segment is then a loud KeyError below rather than a
        # silent skip -- the property the lm-head split's `head_only` partition was protecting.
        for name in sg["weights"]:
            # POP, not index. load_weight_buffer COPIES into the device buffer (`dst[:] = ...`)
            # and keeps no reference, so holding the numpy array afterwards keeps every weight
            # resident TWICE -- once as host anon memory and once as the shmem-backed XRT BO the
            # amdxdna driver allocates. At int8 that is ~11 GB of duplicate for Gemma-4-12B, and
            # it is what a 48-layer build died of: total-vm 27.9 GB on a 30 GB box, with the
            # driver reporting `amdxdna_insert_pages: Failed shmem mmap -12`. The device-side
            # requirement is irreducible; this duplicate is not.
            load_weight_buffer(sc.get_buffer(name), weights.pop(name))
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
        sc.scratch_buffer.device = "cpu"
        sc.scratch_buffer.to("npu")
        stack.append(dict(c=sc, params=sc.params, sg=sg,
                          inlet=(sc.get_buffer(sg["inlet"]) if si else None),
                          outlet=sc.get_buffer(sg["outlet"]),
                          # Rope angles are an INPUT to every segment that has a layer reading
                          # them, so the host writes the row into each one. Driving only the first
                          # segment's would leave every later layer rotating against zeros.
                          rope_g=(sc.get_buffer("rope_global")
                                  if "rope_global" in sg["inputs"] else None),
                          rope_l=(sc.get_buffer("rope_local")
                                  if "rope_local" in sg["inputs"] else None)))
    c = stack[0]["c"]
    params = stack[0]["params"]
    print("[verify] ParameterScratchpad bound (kv_off, sm_mask"
          + (", attn_window" if window_granule is not None else "") + ")")
    if len(stack) > 1:
        for si, st in enumerate(stack):
            la, lb = st["sg"]["layers"]
            print(f"[verify] segment {si}: layers {la}..{lb - 1}, "
                  f"{st['sg']['inlet']} -> {st['sg']['outlet']}, "
                  f"{len(st['sg']['weights'])} weights", file=sys.stderr)
    print(f"[verify] {n_weight_buffers} weight buffers loaded and scratch flushed to the device")

    # embed_tokens doubles as the tied lm-head; the host gathers the row for the current token.
    # The name comes off the SPEC, like every other tensor name: `weight_prefix` is "model." on a
    # text-only checkpoint and "model.language_model." on Gemma-4-12B, whose text stack sits beside
    # a vision and an audio embedder. Hardcoding it made this the last single-model assumption in
    # the harness, and it failed AFTER the graph built and 208 buffers had reached the device.
    embed_npy = os.path.join(a.weights, f"{sp.weight_prefix}embed_tokens.weight.npy")
    embed = np.load(embed_npy).astype(np.float32)
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0

    xin = c.get_buffer("x")
    # Drive whichever angle buffers the BUILD declares, not a fixed pair. Two reasons, and the
    # second is a silent accuracy fault rather than a crash: a truncated dual-theta stack can hold
    # only sliding layers (Gemma-3 goes global every 6th), so at --layers 2 there is no
    # rope_global to fetch; and a dual-theta spec declares a SECOND angle input, where the
    # sliding-attention layers rotate at rope_theta_local and the global ones at
    # rope_theta_global, with which layer reads which baked into the ELF. Driving only
    # rope_global leaves every sliding layer rotating against a buffer the host never writes.
    # The production path already does this (rust/npu-engine/src/llm/npu_decode.rs); this
    # harness was the half still missing it, so a gate run here would have mis-rotated most
    # layers and presented as a device divergence.
    declared = set(md["inputs"])
    # ONE SLOT PER DISTINCT head_dim, from the graph metadata build_graph returns -- NOT from a
    # meta.json, which this harness never reads: it rebuilds the graph rather than loading an
    # artifact. `md["kv_slots"]` is [(param_name, head_dim)], the same list the generator turns into
    # meta.json's `scratchpad.kv_params`.
    kv_slots = md["kv_slots"]
    rope_buf = stack[0]["rope_g"]
    rope_loc_buf = stack[0]["rope_l"]
    # The logits come out of the LAST segment (or, under SPLIT_LM_HEAD, out of the head graph --
    # `lg` is taken from there below). Reading them off segment 0 was right only while there was
    # exactly one segment, and would have returned a buffer the stack never writes once there are
    # three.
    # ... and under SPLIT_LM_HEAD it is in NEITHER: the stack ends at `xf` and `logits` belongs to
    # the head graph's arena, so asking the last segment for it is a KeyError. Only the unsplit-head
    # path reads `out` (the free-running arm below and --redispatch-check); the split path takes its
    # logits off head_c.
    out = None if md.get("split_lm_head") else stack[-1]["c"].get_buffer("logits")

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
        if len(stack) > 1:
            raise SystemExit("[verify] --redispatch-check drives ONE dispatch and compares it with "
                             "itself; across a segmented stack it would re-run only segment 0 and "
                             "report determinism for a fraction of the model. Run it unsegmented.")
        if out is None:
            raise SystemExit("[verify] --redispatch-check needs the logits in the graph it "
                             "re-dispatches; under SPLIT_LM_HEAD they are in the head graph, so it "
                             "would compare a buffer this dispatch never writes. Run it with "
                             "SPLIT_LM_HEAD=0.")
        assert_redispatch_identical(c, out, label=sp.name, vocab=VOCAB)
        return

    fed = list(prompt_ids)
    produced = []
    # mmap, never np.load: this is the 262144x3840 f32 table, 3.75 GiB, and astype copies even when
    # the dtype already matches (see gen_llm_decode.py::npy for what that cost).
    head_w = (np.load(os.path.join(a.weights, f"{sp.weight_prefix}embed_tokens.weight.npy"),
                      mmap_mode="r") if a.host_lm_head else None)
    # The split-lm-head arm: a second graph holding only the lm-head. Its W_head is loaded into its
    # OWN arena, and `xf` crosses between the two through the host -- 7680 bytes a token.
    head_seq = md.get("head")
    head_c = None
    if head_seq is not None:
        head_c = head_seq.get_callable()
        load_weight_buffer(head_c.get_buffer("W_head"), weights.pop("W_head"))
        head_c.scratch_buffer.device = "cpu"
        head_c.scratch_buffer.to("npu")
        print(f"[verify] lm-head split into its own dispatch ({head_seq.name})", file=sys.stderr)
    topk_ids, topk_logits = [], []
    # Per produced step: (device top-1 logit, logit the device gave the ORACLE's token).
    # This is what classifies a mismatch. The oracle's stored `margins` describe a DIFFERENT
    # implementation's forward pass; the device's own gap describes this one.
    step_logits = []
    tok = fed[0]
    for pos in range(len(fed) + steps - 1):
        with xin.overwrite() as _buf:
            _buf[:] = np.asarray(embed[tok] * scale, BF16).reshape(-1)
        if a.trace_residual and pos == 0:
            _pending_trace = True
        # ONE PASS PER SEGMENT, in order, the residual handed forward through the host. At one
        # segment this is exactly the single dispatch it always was; the loop body is per-segment
        # because EVERY input of every segment has to be driven, not just the first one's.
        for _si, _st in enumerate(stack):
            _sc, _sp_ = _st["c"], _st["params"]
            if _si:
                # The seam. The previous segment's outlet is a declared OUTPUT, so it is synced
                # back for us; this copies it into the next segment's inlet, D*2 = 7680 bytes.
                _res = np.asarray(stack[_si - 1]["outlet"].data, BF16)
                with _st["inlet"].overwrite() as _b:
                    _b[:] = _res.reshape(-1)
            # WIDTH PER BUFFER, not one spec-wide HD. Gemma-4-12B's global layers rotate at head_dim
            # 512 and its sliding layers at 256, so the two rows differ in width as well as theta;
            # taking both from `sp.head_dim` writes the wrong number of angles into one of them.
            if _st["rope_g"] is not None:
                with _st["rope_g"].overwrite() as _buf:
                    _buf[:] = rope_row(pos, _buf.size, sp.rope_theta_global,
                                       sp.rope_partial_rotary).reshape(-1)
            if _st["rope_l"] is not None:
                # Sliding layers are rope_type "default" -- nothing narrowed.
                with _st["rope_l"].overwrite() as _buf:
                    _buf[:] = rope_row(pos, _buf.size, sp.rope_theta_local).reshape(-1)
            # ONE WRITE PER DISTINCT head_dim, off the artifact's own kv_params. `pos * head_dim` is
            # two different byte offsets under per-layer geometry, and a single write silently hands
            # the global layers the sliding layers' KV offset.
            # THIS SEGMENT'S slots, not the whole graph's: the names are per geometry, so a
            # segment with no global layer has no `kv_off1` and writing one raises.
            for slot_name, slot_hd in _st["sg"]["kv_slots"]:
                # `kv_layout.kv_off` owns this formula (main), but it needs kv_heads for the block
                # stride and `kv_slots` carries only head_dim. At T == S the block term vanishes
                # (block 0, within == pos) and the two agree exactly, which is every artifact built
                # today. Refuse loudly rather than write a plausible wrong offset: a KV offset that
                # is wrong per geometry is silent, and this chain has already paid for one.
                if T != S and len(_st["sg"]["kv_slots"]) > 1:
                    raise SystemExit(
                        f"[verify] blocked KV (T={T}, S={S}) with per-layer geometry is not wired: "
                        f"kv_slots carry head_dim but not kv_heads, and the block stride needs both. "
                        f"Emit kv_heads per slot in gen_llm_decode.py before using this combination.")
                _kvl = KVLayout(Hkv=sp.n_kv_heads, S=S, HD=slot_hd, T=T)
                _sp_.write(slot_name, int(_kvl.kv_off(pos)))
            _sp_.write("sm_mask", int(pos + 1))
            if window_granule is not None:
                # params.write() pre-shifts "core"-kind params by name, so the raw length is
                # correct here. Clamp to this build's S: window_len can round past the window the
                # ELF's taps cover, and over-attending is not a smaller bug than under-attending.
                _sp_.write("attn_window", min(window_len(pos, window_granule), S))
            _sp_.sync()
            # ONE dispatch per position per segment. The duplicate that used to sit here worked
            # around _sync_inputs() trusting a coherence map this harness never updates; that is
            # fixed at the source now (iron/common/sequence.py forces host residency, mirroring
            # _sync_outputs), so a second dispatch would only double the cost and mask a regression
            # in the real fix.
            _sc()
        if a.trace_residual and pos == 0:
            for _st in stack:
                _st["c"].scratch_buffer.device = "npu"
                _st["c"].scratch_buffer.to("cpu")
            print("[trace] per-layer residual RMS (host keeps this O(1) at every depth):",
                  file=sys.stderr)
            # SEARCH EVERY SEGMENT for each residual. Under a split, x0..x15 live in segment 0's
            # arena and x16..x31 in segment 1's, at offsets that restart near zero -- so a walk over
            # one segment would report the rest as absent and read as a truncated stack. The segment
            # index is printed because "off 1.2 GiB" means a different thing in each arena.
            for _l in range(md["NL"] + 1):
                for _si, _st in enumerate(stack):
                    try:
                        _v = np.asarray(_st["c"].get_buffer(f"x{_l}").data, np.float32)
                    except Exception:
                        continue
                    try:
                        _t, _off, _len = _st["sg"]["seq"].get_layout_for_buffer(f"x{_l}")
                    except Exception:
                        _t, _off, _len = "?", -1, -1
                    _r = float(np.sqrt((_v.astype(np.float64)**2).mean()))
                    print(f"[trace]   x{_l:<3} seg{_si} RMS {_r:11.4e}  arena {_t} "
                          f"off {_off:>13,} ({_off/2**30:7.3f} GiB)"
                          f"{'  <-- UNWRITTEN' if _r == 0.0 else ''}", file=sys.stderr)
                    break
        # EXPLICIT FLAG WINS. --host-lm-head is the substitution arm that localises a fault to the
        # lm-head, and SPLIT_LM_HEAD is a build default -- so an `elif` here silently ignored the
        # flag whenever the head had been split out, which is exactly the configuration you want to
        # bisect at depth. Measured 2026-09-10: a full-depth diagnostic run reported the device
        # head's output while claiming to be the host arm, and the two were indistinguishable
        # because they were the same path.
        if head_c is not None and not a.host_lm_head:
            # stack -> xf (a declared OUTPUT, so it is synced), then the head graph -> logits.
            _xf = np.asarray(stack[-1]["outlet"].data, BF16)
            if pos == 0:
                _f = np.asarray(_xf, np.float32)
                print(f"[split] xf: size={_f.size} nonzero={int((_f!=0).sum())} "
                      f"norm={float(np.linalg.norm(_f)):.6g}", file=sys.stderr)
            with head_c.get_buffer("xf").overwrite() as _b:
                _b[:] = _xf.reshape(-1)
            head_c()
            lg = np.asarray(head_c.get_buffer("logits").data[:VOCAB], dtype=np.float32)
        elif a.host_lm_head:
            # `xf` lives in the SCRATCH arena, and the device is non-coherent: the host's copy is
            # whatever was last written from this side unless it is pulled back. The output arena is
            # synced for us, scratch is not -- which is why this read returned all zeros at 12
            # layers while working at 6, a depth-dependent lie rather than a model result.
            _last = stack[-1]["c"]
            _last.scratch_buffer.device = "npu"
            _last.scratch_buffer.to("cpu")
            xf = np.asarray(_last.get_buffer("xf").data, dtype=np.float32)
            if pos == 0:
                print(f"[host-lm-head] xf: size={xf.size} nonzero={int((xf!=0).sum())} "
                      f"norm={float(np.linalg.norm(xf)):.6g}", file=sys.stderr)
            lg = np.empty(VOCAB, np.float32)
            for lo in range(0, VOCAB, 16384):
                hi = min(lo + 16384, VOCAB)
                lg[lo:hi] = np.asarray(head_w[lo:hi], np.float32) @ xf
        else:
            lg = np.asarray(out.data[:VOCAB], dtype=np.float32)
        # final_logit_softcapping, the same transform rust/npu-engine applies after readback. It
        # changes the ARGMAX (tanh saturates), so a harness that skips it does not gate the model
        # the engine runs.
        #
        # WHETHER to apply it comes from the REFERENCE, never from a flag here. A truncated stack
        # needs it off (its logits are ~1168 against a cap of 30, so 28.9% of the vocab ties at the
        # cap and the argmax is index order); a full-depth one needs it on. Two independent flags
        # would present a disagreement as a token mismatch instead of a configuration error.
        if sp.logit_softcap is not None and not ref.get("logit_softcap_disabled"):
            lg = np.tanh(lg / sp.logit_softcap) * sp.logit_softcap
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

    if a.smoke_prompt:
        txt = _tok.decode(produced, skip_special_tokens=False)
        print(f"\n[verify] generated ids : {produced}")
        print(f"[verify] generated text: {txt!r}")
        print("[verify] SMOKE RUN -- coherent text is evidence, NOT a gate. Token parity against a "
              "matching-precision oracle is the gate; this exists because that oracle is ~23 GB "
              "for a 12B and will swap a shared box into the ground.")
        return 0

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
