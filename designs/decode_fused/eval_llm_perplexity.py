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
from gen_llm_decode import (build_graph, report_artifact_freshness,  # noqa: E402
                            load_weight_buffer, isolate_build_dir)
from qwen_bpe import QwenBPE  # noqa: E402

BF16 = ml_dtypes.bfloat16


def rope_row(pos, head_dim, theta):
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64)[:half] / head_dim))
    ang = pos * inv
    row = np.empty(head_dim, dtype=np.float32)
    row[0::2] = np.cos(ang)
    row[1::2] = np.sin(ang)
    return row.astype(BF16)


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
    ap.add_argument("--text", required=True, help="UTF-8 corpus file")
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

    tj = a.tokenizer or os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
    if os.path.isdir(tj):
        tj = os.path.join(tj, sorted(os.listdir(tj))[0], "tokenizer.json")
    if a.ref:
        from qwen_bpe import self_test
        self_test(tj, a.ref)
        print(f"[ppl] tokenizer self-test PASS against {os.path.basename(a.ref)}")
    tok = QwenBPE(tj)
    ids = tok.encode(open(a.text, encoding="utf-8").read())
    print(f"[ppl] corpus {a.text}: {len(ids)} tokens")

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    S, HD, D, VOCAB = md["S"], sp.head_dim, sp.d_model, sp.vocab
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

    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0
    xin, rope_buf, out = c.get_buffer("x"), c.get_buffer("rope_global"), c.get_buffer("logits")

    nll, t0, top1_hits = [], time.perf_counter(), 0
    for pos in range(n):
        np.copyto(xin.data, np.asarray(embed[ids[pos]] * scale, BF16).reshape(-1))
        np.copyto(rope_buf.data, rope_row(pos, HD, sp.rope_theta_global).reshape(-1))
        params.write("kv_off", int(pos * HD))
        params.write("sm_mask", int(pos + 1))
        params.sync()
        c()
        lg = np.asarray(out.data[:VOCAB], dtype=np.float32)
        nll.append(-log_softmax_at(lg, ids[pos + 1]))
        top1_hits += int(np.argmax(lg) == ids[pos + 1])
        if (pos + 1) % 256 == 0:
            print(f"[ppl]   {pos+1}/{n}  running ppl {math.exp(np.mean(nll)):.4f}", flush=True)
    wall = time.perf_counter() - t0

    mean_nll = float(np.mean(nll))
    res = {
        "spec": sp.name, "layers": md["NL"], "text": os.path.basename(a.text),
        "n_scored": n, "mean_nll": mean_nll, "perplexity": math.exp(mean_nll),
        "top1_acc": top1_hits / n, "median_nll": float(np.median(nll)),
        "wall_s": wall, "ms_per_token": 1000.0 * wall / n,
        "env": {k: os.environ.get(k) for k in
                ("QUANT_MLP_DTYPE", "QUANT_MLP_GROUP", "FUSE_MLP_DP", "FUSE_QKV_DP")},
    }
    print(f"\n[ppl] mean NLL {mean_nll:.6f}   PERPLEXITY {res['perplexity']:.4f}   "
          f"top-1 {100*res['top1_acc']:.2f}%   ({n} positions, {res['ms_per_token']:.1f} ms/token)")
    if a.dump_nll:
        np.save(a.dump_nll, np.asarray(nll, dtype=np.float64))
        print(f"[ppl] per-position NLL -> {a.dump_nll}")
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1)
        print(f"[ppl] wrote {a.out_json}")


if __name__ == "__main__":
    sys.exit(main())
