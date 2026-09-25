#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced perplexity of a fused decode ELF, on device -- the QUALITY gate.

Exists because the shipped correctness gate cannot see what a weight-format change does. The
bf16 oracle is ONE prompt and EIGHT free-running tokens judged by argmax, and two of its steps sit
at 0.0203 and 0.154 logit margins where f32 and bf16 already disagree; an argmax gate over 8 tokens
is blind to a format that shifts every logit slightly. Perplexity reads the whole distribution at
every position, so it sees exactly the damage a quantized weight stream does.

Method: drive the same graph build_graph() builds, one token per dispatch, feeding the TRUE next
token every step (never the model's own), and accumulate -log softmax(logits)[true_next]. So this
measures the forward pass at each position independently -- it never lets one bad token drag a
trajectory, which is the failure free-running comparison has.

Read it as a DELTA between two arms on the same text and the same graph, never as an absolute:
the absolute depends on the corpus, and this harness has no calibration against a reference
implementation of the same model.

  python designs/decode_fused/eval_llm_perplexity.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --text some.txt --max-tokens 1024

Single-tenant; arm is selected by the same env flags gen_llm_decode.py reads.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from verify_llm_decode import window_len, rope_row  # noqa: E402 -- one owner for each
from gen_llm_decode import (build_graph, report_artifact_freshness,  # noqa: E402
                            load_weight_buffer, isolate_build_dir)
from decode_flash_ref import flash_slot_writes  # noqa: E402
from iron.common.kv_layout import KVLayout  # noqa: E402
from llm_decode_spec import SPECS  # noqa: E402

BF16 = ml_dtypes.bfloat16


def log_softmax_at(logits, idx):
    """Stable log-softmax evaluated at ONE index, in f64.

    f64 deliberately: the sum runs over 151936 exponentials and this is an accuracy instrument,
    so the harness must not contribute error comparable to what it is measuring.
    """
    x = np.asarray(logits, dtype=np.float64)
    m = x.max()
    return float(x[idx] - m - math.log(np.exp(x - m).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--text", default=None, help="UTF-8 corpus file, tokenized here with QwenBPE "
                                                 "-- only for specs in qwen_bpe's "
                                                 "TEXT_TOKENIZER_HINT; others must use --ids")
    ap.add_argument("--ids", default=None, help="pre-tokenized ids (tokenize_corpus.py's json "
                                                "output). The only route for a spec QwenBPE "
                                                "cannot read, e.g. Gemma's Split pretokenizer.")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (default: the HF cache)")
    ap.add_argument("--ref", default=None, help="oracle json; if given, the tokenizer self-tests "
                                                "against its prompt_ids before anything runs")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--dump-nll", default=None, help="per-position NLL .npy -- the arms share a\n                    corpus and positions, so a PAIRED comparison is available and a difference of\n                    two means is not the right test")
    a = ap.parse_args()
    isolate_build_dir("ppl")
    report_artifact_freshness(a.weights)

    if bool(a.text) == bool(a.ids):
        raise SystemExit("[ppl] give exactly one of --text or --ids")
    if a.ids:
        ids = json.load(open(a.ids))
        corpus_name = os.path.basename(a.ids)
        tokenizer_prov = f"pre-tokenized ({corpus_name})"
    else:
        from qwen_bpe import QwenBPE, self_test, text_tokenizer_hint
        tj = a.tokenizer or text_tokenizer_hint(a.spec)
        if tj is None:
            raise SystemExit(
                f"[ppl] --text cannot read {a.spec}: QwenBPE parses a Qwen-style tokenizer.json "
                f"and this spec's is not one. Pre-tokenize with tokenize_corpus.py and pass "
                f"--ids. (Defaulting to another spec's tokenizer is what scored Gemma text "
                f"against a Qwen vocabulary: NLL 19.836 against a 12.477 uniform, top-1 0.0.)")
        if os.path.isdir(tj):
            tj = os.path.join(tj, sorted(os.listdir(tj))[0], "tokenizer.json")
        if a.ref:
            self_test(tj, a.ref)
            print(f"[ppl] tokenizer self-test PASS against {os.path.basename(a.ref)}")
        tok = QwenBPE(tj)
        ids = tok.encode(open(a.text, encoding="utf-8").read())
        corpus_name = os.path.basename(a.text)
        tokenizer_prov = tj
    # An id the model has no embedding row for is a corpus/spec mismatch, and the run that
    # follows would look like a quality result rather than a wrong one.
    _vocab = SPECS[a.spec].vocab
    if max(ids) >= _vocab:
        raise SystemExit(f"[ppl] corpus {corpus_name} has id {max(ids)} against {a.spec}'s "
                         f"vocab {_vocab} -- wrong tokenizer for this spec")
    print(f"[ppl] corpus {corpus_name}: {len(ids)} tokens, tokenizer {tokenizer_prov}")

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    S, HD, D, VOCAB = md["S"], sp.head_dim, sp.d_model, sp.vocab
    # None unless build_graph actually built decode_layer_dp with window_parameter="attn_window".
    window_granule = md.get("window_granule")
    kv_layout = KVLayout(Hkv=sp.n_kv_heads, S=S, HD=HD, T=md["T"])
    # One KV slot per fed token, and the last fed token's logits predict nothing we score.
    n = min(a.max_tokens, len(ids) - 1, S - 1)
    if n < 2:
        raise SystemExit(f"[ppl] corpus too short: {len(ids)} tokens against S={S}")
    print(f"[ppl] {sp.name}: {md['NL']} layers, S={S}; scoring {n} positions")

    c = fused.get_callable()
    params = c.params
    if params is None:
        raise SystemExit("[ppl] no runtime parameters bound -- params.txt missing from the build")
    for name, arr in weights.items():
        load_weight_buffer(c.get_buffer(name), arr)
    # Weights and both KV caches live in the scratch arena, which the callable syncs in NEITHER
    # direction. Without this the device reads whatever the CPU happened to write back.
    c.scratch_buffer.device = "cpu"
    c.scratch_buffer.to("npu")
    print(f"[ppl] {len(weights)} weight buffers loaded, scratch flushed")

    embed = np.load(os.path.join(a.weights, f"{sp.weight_prefix}embed_tokens.weight.npy")).astype(np.float32)
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0
    xin, out = c.get_buffer("x"), c.get_buffer("logits")
    # Every per-position input this graph declares, driven the way verify_llm_decode.py drives it.
    # A dual-theta spec declares a SECOND angle buffer for its sliding layers, and under per-layer
    # geometry the two differ in WIDTH as well as theta -- so the row comes off each buffer's own
    # size, never off sp.head_dim (a single global head_dim is wrong for Gemma-4's mixed
    # sliding/global geometry -- writing only rope_global at sp.head_dim left 40 of 48 layers
    # rotating against a buffer the host never wrote).
    rope_g = c.get_buffer("rope_global") if "rope_global" in md["inputs"] else None
    rope_l = c.get_buffer("rope_local") if "rope_local" in md["inputs"] else None
    # One (kv_off, sm_mask) pair per distinct GEOMETRY, off geom_slots -- not kv_slots/mask_slots
    # alone, which cannot be zipped positionally (mask_slots is keyed by distinct WINDOW, so two
    # geometries sharing one window collapse to its single entry while kv_slots still has two; see
    # gen_llm_decode.py's geom_slots comment). Falls back to the pre-existing single-slot form when
    # the build predates geom_slots (older artifacts, or SLIDING_KV_CIRCULAR-unaware paths).
    geom_slots = md.get("geom_slots") or [("kv_off", HD, S, "sm_mask", min(md["T"], S), sp.n_kv_heads)]
    # Per-geometry capacity/block/kv-heads -- see bench_llm_decode.py's note.
    geoms = [(nm, KVLayout(Hkv=khv, S=cap, HD=hd, T=blk), cap, mn)
             for nm, hd, cap, mn, blk, khv in geom_slots]
    flash_slots = md.get("flash_slots") or []

    nll, t0, top1_hits, n_sat = [], time.perf_counter(), 0, 0
    for pos in range(n):
        with xin.overwrite() as _buf:
            _buf[:] = np.asarray(embed[ids[pos]] * scale, BF16).reshape(-1)
        if rope_g is not None:
            with rope_g.overwrite() as _buf:
                _buf[:] = rope_row(pos, _buf.size, sp.rope_theta_global,
                                   sp.rope_partial_rotary).reshape(-1)
        if rope_l is not None:
            with rope_l.overwrite() as _buf:
                _buf[:] = rope_row(pos, _buf.size, sp.rope_theta_local).reshape(-1)
        # SLIDING_KV_CIRCULAR: this geometry's capacity is ww, not the build's S. The cache wraps
        # (pos % ww) and the mask clamps to the same bound -- exact, not approximate, because
        # softmax is order-independent and RoPE is written against the ABSOLUTE position above, so
        # a rotated slot ordering downstream is not observable. See gen_llm_decode.py's
        # SLIDING_KV_CIRCULAR doc.
        for _slot, _kvl, _ww, _mask in geoms:
            params.write(_slot, int(_kvl.kv_off(pos % _ww)))
            params.write(_mask, min(pos + 1, _ww))
        for _fn, _fv in flash_slot_writes(flash_slots, pos):  # see decode_flash_ref.flash_slot_writes
            params.write(_fn, _fv)
        if window_granule is not None:
            # A dynamic-window build reads its attended length from this parameter every dispatch.
            # Omitting it does NOT fail -- the core reads whatever the scratchpad happens to hold,
            # attends a garbage window, and returns NaN logits. That is silent for the whole run:
            # this harness reported `running ppl nan` for 20000 positions over 34 minutes before
            # anyone looked at why. Clamped to S exactly as verify_llm_decode.py does.
            params.write("attn_window", min(window_len(pos, window_granule), S))
        params.sync()
        c()
        lg = np.asarray(out.data[:VOCAB], dtype=np.float32)
        if sp.logit_softcap is not None:
            # final_logit_softcapping, applied after readback -- this IS the model's output
            # distribution. Skipping it on Gemma-4-12B reads mean NLL 19.9, above ln(vocab): a
            # different model, not an approximation of this one.
            lg = np.tanh(lg / sp.logit_softcap) * sp.logit_softcap
            n_sat += int((np.abs(lg) >= sp.logit_softcap * (1 - 1e-6)).sum())
        nll.append(-log_softmax_at(lg, ids[pos + 1]))
        top1_hits += int(np.argmax(lg) == ids[pos + 1])
        if (pos + 1) % 256 == 0:
            print(f"[ppl]   {pos+1}/{n}  running ppl {math.exp(np.mean(nll)):.4f}", flush=True)
            if a.dump_nll:
                # Checkpoint. This harness cannot RESUME -- every position depends on the KV cache
                # the previous ones built, so there is no cheap way to skip ahead -- but a killed
                # run can still leave a usable paired sample. Cost is one small write per 256
                # dispatches. Measured need: a run stopped at 1024/2000 to hand the device over
                # was worth keeping and nearly wasn't.
                np.save(a.dump_nll, np.asarray(nll, dtype=np.float64))
    wall = time.perf_counter() - t0

    if sp.logit_softcap is not None and n_sat > 0.001 * n * VOCAB:
        print(f"[ppl] WARNING: {n_sat/(n*VOCAB):.1%} of logits saturate the "
              f"{sp.logit_softcap} softcap -- the argmax is index order there and this run does "
              "not gate the model. A truncated stack does this; a full-depth one does not.",
              file=sys.stderr)
    mean_nll = float(np.mean(nll))
    res = {
        "spec": sp.name, "layers": md["NL"], "text": corpus_name,
        # Which tokenizer produced the ids. Absent from every run recorded before 2026-09-14, so
        # a wrong-vocabulary run could not be told from a bad-weights one after the fact.
        "tokenizer": tokenizer_prov,
        "logit_softcap": sp.logit_softcap, "softcap_saturated_frac": n_sat / (n * VOCAB),
        "n_scored": n, "mean_nll": mean_nll, "perplexity": math.exp(mean_nll),
        "top1_acc": top1_hits / n, "median_nll": float(np.median(nll)),
        # NOT a benchmark, and named so it cannot be quoted as one. No warmup, no alternation,
        # no repetition, and every position pays a host-side f64 log-softmax over the whole vocab
        # that a real decode never does. MEASURED: the SAME arm read 56.8 and 48.2 ms/token on two
        # runs of an identical job -- a 15% swing on an unchanged binary. Use bench_llm_decode.py
        # for timing; this is a progress indicator.
        "wall_s": wall, "harness_ms_per_position_NOT_A_BENCHMARK": 1000.0 * wall / n,
        "env": {k: os.environ.get(k) for k in
                ("QUANT_MLP_DTYPE", "QUANT_MLP_GROUP", "FUSE_MLP_DP", "FUSE_QKV_DP")},
    }
    print(f"\n[ppl] mean NLL {mean_nll:.6f}   PERPLEXITY {res['perplexity']:.4f}   "
          f"top-1 {100*res['top1_acc']:.2f}%   ({n} positions, "
          f"{res['harness_ms_per_position_NOT_A_BENCHMARK']:.1f} ms/pos -- harness rate, NOT a "
          f"decode benchmark: no warmup, and a host f64 log-softmax per position)")
    if a.dump_nll:
        np.save(a.dump_nll, np.asarray(nll, dtype=np.float64))
        print(f"[ppl] per-position NLL -> {a.dump_nll}")
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1)
        print(f"[ppl] wrote {a.out_json}")


if __name__ == "__main__":
    sys.exit(main())
