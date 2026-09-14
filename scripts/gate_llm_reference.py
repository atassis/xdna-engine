#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TIER 2's reference: what the MODEL answers, for the canonical prompt, N tokens deep.

Tier 1 says each block computes the right numbers. It cannot say the assembled rail generates the
right text, and no per-block tolerance implies that -- a correct block wired to the wrong KV row
passes Tier 1 every time. So the end-to-end gate is on TOKENS, and this writes the tokens to gate
against, along with the per-step top-k and the top1-top2 margin that says whether a disagreement was
ever decidable.

WHY THE GATE IS SET INCLUSION AND NOT EQUALITY. Greedy decode is chaotic wherever the top two
logits are close, and this model has such a step on this prompt: at step 5 the margin is 0.0203,
about one bf16 quantum at that magnitude. Measured here, transformers 4.57.6 in float32 flips that
step relative to the f32 sequence recorded in `tests/refs/qwen3-0.6b/greedy_ref.json` -- so the
reference disagrees with an earlier run of the SAME reference implementation on the SAME prompt. A
gate demanding token equality would be charging silicon for a coin flip. Set inclusion at k=5 is
what the domain uses (it is the shape of vLLM's `check_logprobs_close`) and is what
`scripts/gate_token_set.py` judges.

BACKENDS. Both write the same JSON, so either can be the reference and they can be diffed:

  numpy  (default) float32 arithmetic on the bf16 weights the device holds, from the dumped .npy.
         No new dependency: the rail already carries this forward pass
         (scripts/llm_decode_host_ref.py, scripts/llm_decode_bf16_oracle.py), and this generalises
         it to N tokens and top-k rather than adding a second implementation of the model.
  hf     transformers, from the HF cache. NOT installed in `.venv-iron` (the toolchain env, which
         must not grow dependencies) -- it IS installed in `.venv-export`, so run this backend with
         that interpreter. This is the soundness ANCHOR for the numpy backend, not a per-run
         dependency: run it once, diff the two files, and the numpy reference is validated as the
         model rather than merely as our reading of it.

  .venv-iron/bin/python   scripts/gate_llm_reference.py --weights <npy dir> --out ref.json
  .venv-export/bin/python scripts/gate_llm_reference.py --backend hf --out ref_hf.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "designs", "decode_fused"))
from llm_decode_spec import SPECS  # noqa: E402

# How many free-running tokens the gate judges. 32 rather than the 8 the existing oracles carry:
# 8 tokens of one prompt is a handful of argmaxes, and this rail has already had a dataflow arm
# that agreed on the first token and diverged by the third. 32 is the length the reference
# implementations in this domain use for the same gate.
GATE_N_TOKENS = 32

# The token-set width. The reference's token must be within the device's top-K at the first
# disagreement. k=5 is the domain's convention; it is wide enough to absorb a bf16-vs-f32 tie and
# narrow enough that a genuinely wrong distribution cannot hide in it.
GATE_K = 5

# One prompt, fixed, so a reference file is comparable across runs. The ids are Qwen3's BPE for the
# string and are checked against it whenever a tokenizer is available (the hf backend).
CANONICAL_PROMPT = "The capital of France is"
CANONICAL_PROMPT_IDS = [785, 6722, 315, 9625, 374]

HF_MODEL = {"qwen3-0.6b": "Qwen/Qwen3-0.6B"}


def topk(logits, k):
    idx = np.argpartition(-logits, k)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [int(i) for i in idx], [float(logits[i]) for i in idx]


# ------------------------------------------------------------------------------------------------
# numpy backend: float32 arithmetic on the bf16 weights the device holds.
# ------------------------------------------------------------------------------------------------
def run_numpy(sp, weights_dir, prompt_ids, n_tokens, k):
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16

    # Refuse rather than approximate: an axis this reference does not know the shape of (a rope_type
    # other than the one Gemma-4 actually uses) is exactly the same-name-different-meaning trap the
    # sandwich/dual-theta/sliding-window/per-layer-geometry axes below already were.
    if sp.rope_type_global not in (None, "proportional"):
        raise SystemExit(f"ERROR: spec {sp.name} has rope_type_global={sp.rope_type_global!r}, "
                         f"which this reference only implements for 'proportional'.")
    NL, D = sp.n_layers, sp.d_model
    dual_rope = sp.rope_theta_local is not None

    def npy(n):
        """bf16-quantise on load: the reference must start from the weights the DEVICE holds, so
        that what it measures is the arithmetic and not the checkpoint's own rounding."""
        a = np.load(os.path.join(weights_dir, f"{n}.npy")).astype(np.float32)
        return np.asarray(np.asarray(a, BF16), np.float32)

    def npy_bf16(n):
        """Like `npy`, but returns an mmap VIEW instead of a materialised bf16 array -- see `mm()`.

        Gemma-4-12B's dump is 45GB of float32 .npy on disk across 48 layers; holding every layer's
        projection weights bf16-resident at once (~22.5GB) does not survive this box's real
        headroom once production/desktop overhead is accounted for -- crashed the run twice.
        mmap defers the bf16-round-then-widen to `mm()`'s per-call cast: the OS backs the float32 read with
        reclaimable page cache instead of pinned anonymous memory, so peak RSS is bounded by one
        weight matrix's transient cast, not all 48 layers' worth. Same rounding as before, paid per
        matmul call instead of once (~30 min of cast overhead over an 800-position run, per the
        prior measurement this replaces).
        """
        return np.load(os.path.join(weights_dir, f"{n}.npy"), mmap_mode="r")

    def mm(w_view, v):
        """`w_view` is an mmap'd float32 view (see `npy_bf16`) -- round to bf16 THEN widen, so the
        arithmetic matches `npy()`'s bf16-quantise-on-load contract exactly, just deferred."""
        return np.asarray(np.asarray(w_view, BF16), np.float32) @ v

    def rms(x, w=None):
        """`w=None` is the v_norm case: gainless (with_scale=False in the checkpoint, so there is no
        weight tensor to load -- multiplying by 1.0 is the operator's own definition, not a stand-in
        for a missing one)."""
        g = 1.0 if w is None else ((1.0 + w) if sp.norm_gain == "one_plus_w" else w)
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + sp.eps) * g

    def act(x):
        if sp.act == "silu":
            return x / (1.0 + np.exp(-np.clip(x, -60, 60)))
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    def rope(v, pos, hd, theta, partial):
        """`partial` zeroes the inverse frequency past `int(partial*hd//2)` pairs -- a zero
        frequency is the identity rotation -- matching gen_llm_prefill.rope_table's "proportional"
        rule exactly (same derivation, one position instead of a row per chunk)."""
        inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64)[:hd // 2] / hd))
        if partial is not None:
            inv[int(partial * hd // 2):] = 0.0
        c, s = np.cos(pos * inv).astype(np.float32), np.sin(pos * inv).astype(np.float32)
        v = v.reshape(-1, hd)
        x1, x2 = v[:, :hd // 2], v[:, hd // 2:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).reshape(-1)

    def rope_for(l, v, pos):
        """Dual-theta: the global geometry rotates its OWN head_dim (which may differ from the
        sliding one -- 512 vs 256 on Gemma-4) and, on Gemma-4, only a fraction of it; the sliding
        geometry always rotates its full head_dim at the local theta. Single-theta specs get
        rope_theta_global everywhere and no partial rotary, matching gen_llm_prefill's non-dual arm."""
        hd = sp.head_dim_for(l)
        if not dual_rope:
            return rope(v, pos, hd, sp.rope_theta_global, None)
        g = sp.is_global(l)
        theta = sp.rope_theta_global if g else sp.rope_theta_local
        partial = sp.rope_partial_rotary if g else None
        return rope(v, pos, hd, theta, partial)

    embed = npy_bf16(f"{sp.weight_prefix}embed_tokens.weight")
    n_final = npy(f"{sp.weight_prefix}norm.weight")

    Wt = {}
    for l in range(NL):
        p = f"{sp.weight_prefix}layers.{l}."
        w = {k_: npy(v) for k_, v in sp.norm_weight_names(l).items()}
        w["Wq"] = npy_bf16(p + "self_attn.q_proj.weight")
        w["Wk"] = npy_bf16(p + "self_attn.k_proj.weight")
        if sp.has_v_proj(l):
            w["Wv"] = npy_bf16(p + "self_attn.v_proj.weight")
        w["Wo"] = npy_bf16(p + "self_attn.o_proj.weight")
        w["Wg"] = npy_bf16(p + "mlp.gate_proj.weight")
        w["Wu"] = npy_bf16(p + "mlp.up_proj.weight")
        w["Wd"] = npy_bf16(p + "mlp.down_proj.weight")
        if sp.layer_scalar:
            w["ls"] = float(npy(sp.layer_scalar_name(l)).reshape(-1)[0])
        Wt[l] = w

    S = len(prompt_ids) + n_tokens + 1
    kc = [np.zeros((sp.n_kv_heads_for(l), S, sp.head_dim_for(l)), np.float32) for l in range(NL)]
    vc = [np.zeros((sp.n_kv_heads_for(l), S, sp.head_dim_for(l)), np.float32) for l in range(NL)]
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0
    attn_scale = sp.attn_scale

    produced, tops, margins = [], [], []
    tok = prompt_ids[0]
    for pos in range(len(prompt_ids) + n_tokens - 1):
        x = np.asarray(np.asarray(embed[tok], BF16), np.float32) * scale
        for l in range(NL):
            w = Wt[l]
            hd, kvh = sp.head_dim_for(l), sp.n_kv_heads_for(l)
            grp, has_v, is_glob = sp.n_q_heads // kvh, sp.has_v_proj(l), sp.is_global(l)
            window = None if (sp.sliding_window is None or is_glob) else sp.sliding_window

            h = rms(x, w["n_in"])
            q, k_ = mm(w["Wq"], h), mm(w["Wk"], h)
            v = mm(w["Wv"], h) if has_v else None
            if sp.v_norm:
                # attention_k_eq_v: v_norm reads the RAW k projection -- before qk-norm and RoPE,
                # which mutate q/k_ below -- and its output IS v; mirrors gen_llm_prefill.py's
                # ordering exactly (op_vn runs before qn_runs/op_kn in the emitted op list).
                src = k_ if not has_v else v
                v = np.concatenate([rms(src.reshape(kvh, hd)[i]) for i in range(kvh)])
            if sp.qk_norm:
                q = np.concatenate([rms(q.reshape(sp.n_q_heads, hd)[i], w["n_qn"])
                                    for i in range(sp.n_q_heads)])
                k_ = np.concatenate([rms(k_.reshape(kvh, hd)[i], w["n_kn"]) for i in range(kvh)])
            q, k_ = rope_for(l, q, pos), rope_for(l, k_, pos)
            kc[l][:, pos, :] = k_.reshape(kvh, hd)
            vc[l][:, pos, :] = v.reshape(kvh, hd)
            qh = q.reshape(sp.n_q_heads, hd)
            ctx = np.empty((sp.n_q_heads, hd), np.float32)
            lo = 0 if window is None else max(0, pos - window + 1)
            for hh in range(sp.n_q_heads):
                kvi = hh // grp
                sc = (kc[l][kvi, lo:pos + 1] @ qh[hh]) * attn_scale
                sc = np.exp(sc - sc.max())
                ctx[hh] = (sc / sc.sum()) @ vc[l][kvi, lo:pos + 1]
            a_out = mm(w["Wo"], ctx.reshape(-1))
            if sp.sandwich_norms:
                a_out = rms(a_out, w["n_pa"])
            x = x + a_out
            hf = rms(x, w["n_pf"])
            d_out = mm(w["Wd"], act(mm(w["Wg"], hf)) * mm(w["Wu"], hf))
            if sp.sandwich_norms:
                d_out = rms(d_out, w["n_pff"])
            x = x + d_out
            if sp.layer_scalar:
                x = x * w["ls"]
        lg = mm(embed, rms(x, n_final))        # tied lm head
        if sp.logit_softcap is not None:
            c = sp.logit_softcap
            lg = c * np.tanh(lg / c)
        ids, vals = topk(lg, k)
        if pos + 1 < len(prompt_ids):
            tok = prompt_ids[pos + 1]         # teacher-force through the prompt
        else:
            produced.append(ids[0])
            tops.append((ids, vals))
            margins.append(vals[0] - vals[1])
            tok = ids[0]
        if len(produced) >= n_tokens:
            break
    return produced, tops, margins


# ------------------------------------------------------------------------------------------------
# hf backend: the anchor. Same JSON, so `gate_token_set.py --ref a --npu b` diffs the two.
# ------------------------------------------------------------------------------------------------
def run_hf(model_id, prompt, prompt_ids, n_tokens, k, dtype):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            f"ERROR: {e}. The hf backend needs transformers+torch, which are NOT in .venv-iron and "
            f"must not be added to it -- that is the toolchain env. Use the export venv:\n"
            f"  ../xdna-engine/.venv-export/bin/python scripts/gate_llm_reference.py --backend hf ...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids[0].tolist() != list(prompt_ids):
        raise SystemExit(f"ERROR: the tokenizer gives {ids[0].tolist()} for {prompt!r}, the "
                         f"reference file pins {list(prompt_ids)}. One of the two is wrong and "
                         f"guessing which would put a silently different prompt into the gate.")
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]).eval()
    produced, tops, margins = [], [], []
    with torch.no_grad():
        for _ in range(n_tokens):
            lg = m(ids).logits[0, -1].to(torch.float32).numpy()
            tk, vals = topk(lg, k)
            produced.append(tk[0])
            tops.append((tk, vals))
            margins.append(vals[0] - vals[1])
            ids = torch.cat([ids, torch.tensor([[tk[0]]])], dim=1)
    return produced, tops, margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b", choices=sorted(SPECS))
    ap.add_argument("--backend", default="numpy", choices=("numpy", "hf"))
    ap.add_argument("--weights", default=None, help="dumped .npy dir (numpy backend)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=GATE_N_TOKENS)
    ap.add_argument("--k", type=int, default=GATE_K)
    ap.add_argument("--hf-dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--prompt", default=CANONICAL_PROMPT)
    ap.add_argument("--prompt-ids", default=None, help="comma-separated; defaults to the canonical")
    a = ap.parse_args()

    sp = SPECS[a.spec]
    pids = ([int(x) for x in a.prompt_ids.split(",")] if a.prompt_ids
            else list(CANONICAL_PROMPT_IDS))
    if a.backend == "numpy":
        if not a.weights or not os.path.isdir(a.weights):
            raise SystemExit("ERROR: --weights <dumped .npy dir> is required for the numpy backend")
        gen, tops, margins = run_numpy(sp, a.weights, pids, a.tokens, a.k)
        backend = "numpy float32 arithmetic on bf16 weights"
    else:
        mid = HF_MODEL.get(sp.name)
        if not mid:
            raise SystemExit(f"ERROR: no HF model id known for spec {sp.name}")
        gen, tops, margins = run_hf(mid, a.prompt, pids, a.tokens, a.k, a.hf_dtype)
        backend = f"transformers {a.hf_dtype}"

    out = {
        "spec": sp.name, "backend": backend, "prompt": a.prompt, "prompt_ids": pids,
        "n_tokens": len(gen), "k": a.k, "gen_ids": gen,
        "topk_ids": [t[0] for t in tops], "topk_logits": [t[1] for t in tops],
        "margins": margins,
        "note": "TIER 2 reference for scripts/gate_token_set.py. `margins` is top1-top2 per step: "
                "a step whose margin is near the device's own logit error was never decidable, "
                "which is why the gate is top-k set inclusion and not equality.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    tight = [i for i, m in enumerate(margins) if m < 0.05]
    print(f"[ref] {backend}: {len(gen)} tokens")
    print(f"[ref] gen_ids : {gen}")
    print(f"[ref] margins < 0.05 at steps {tight} -- knife-edge steps, the reason for top-{a.k}"
          if tight else "[ref] no step has a margin under 0.05")
    print(f"[ref] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
