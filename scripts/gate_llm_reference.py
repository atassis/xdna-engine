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

    # Refuse rather than approximate. Every one of these axes is a same-name-different-meaning trap
    # that llm_decode_spec.py's header already catalogues, and a reference that quietly runs the
    # wrong one is worse than no reference: it fails the device for the reference's bug.
    for flag, why in ((sp.rope_theta_local is not None, "local/global split RoPE theta"),
                      (sp.sliding_window is not None, "sliding-window attention"),
                      (sp.sandwich_norms, "sandwich norms (the pre/post-FFN weight NAMES move "
                                          "with them -- Gemma's post_attention_layernorm is a "
                                          "different tensor from Qwen3's)")):
        if flag:
            raise SystemExit(f"ERROR: spec {sp.name} has {why}, which this reference does not "
                             f"implement. Implement it; do not let a plainer forward pass stand in.")
    NL, D, HD = sp.n_layers, sp.d_model, sp.head_dim
    Hq, Hkv, grp = sp.n_q_heads, sp.n_kv_heads, sp.gqa_group

    def npy(n):
        """bf16-quantise on load: the reference must start from the weights the DEVICE holds, so
        that what it measures is the arithmetic and not the checkpoint's own rounding."""
        a = np.load(os.path.join(weights_dir, f"{n}.npy")).astype(np.float32)
        return np.asarray(np.asarray(a, BF16), np.float32)

    def rms(x, w):
        g = (1.0 + w) if sp.norm_gain == "one_plus_w" else w
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + sp.eps) * g

    def act(x):
        if sp.act == "silu":
            return x / (1.0 + np.exp(-np.clip(x, -60, 60)))
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    inv = 1.0 / (sp.rope_theta_global ** (np.arange(0, HD, 2, dtype=np.float64)[:HD // 2] / HD))

    def rope(v, pos):
        c, s = np.cos(pos * inv).astype(np.float32), np.sin(pos * inv).astype(np.float32)
        v = v.reshape(-1, HD)
        x1, x2 = v[:, :HD // 2], v[:, HD // 2:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).reshape(-1)

    embed, n_final = npy("model.embed_tokens.weight"), npy("model.norm.weight")
    names = [("n_in", "input_layernorm.weight"), ("n_pf", "post_attention_layernorm.weight"),
             ("Wq", "self_attn.q_proj.weight"), ("Wk", "self_attn.k_proj.weight"),
             ("Wv", "self_attn.v_proj.weight"), ("Wo", "self_attn.o_proj.weight"),
             ("Wg", "mlp.gate_proj.weight"), ("Wu", "mlp.up_proj.weight"),
             ("Wd", "mlp.down_proj.weight")]
    if sp.qk_norm:
        names += [("n_qn", "self_attn.q_norm.weight"), ("n_kn", "self_attn.k_norm.weight")]
    Wt = {l: {k_: npy(f"model.layers.{l}.{v}") for k_, v in names} for l in range(NL)}

    S = len(prompt_ids) + n_tokens + 1
    kc = [np.zeros((Hkv, S, HD), np.float32) for _ in range(NL)]
    vc = [np.zeros((Hkv, S, HD), np.float32) for _ in range(NL)]
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0

    produced, tops, margins = [], [], []
    tok = prompt_ids[0]
    for pos in range(len(prompt_ids) + n_tokens - 1):
        x = embed[tok] * scale
        for l in range(NL):
            w = Wt[l]
            h = rms(x, w["n_in"])
            q, k_, v = w["Wq"] @ h, w["Wk"] @ h, w["Wv"] @ h
            if sp.qk_norm:
                q = np.concatenate([rms(q.reshape(Hq, HD)[i], w["n_qn"]) for i in range(Hq)])
                k_ = np.concatenate([rms(k_.reshape(Hkv, HD)[i], w["n_kn"]) for i in range(Hkv)])
            q, k_ = rope(q, pos), rope(k_, pos)
            kc[l][:, pos, :] = k_.reshape(Hkv, HD)
            vc[l][:, pos, :] = v.reshape(Hkv, HD)
            qh = q.reshape(Hq, HD)
            ctx = np.empty((Hq, HD), np.float32)
            for hh in range(Hq):
                kv = hh // grp
                sc = (kc[l][kv, :pos + 1] @ qh[hh]) * sp.attn_scale
                sc = np.exp(sc - sc.max())
                ctx[hh] = (sc / sc.sum()) @ vc[l][kv, :pos + 1]
            x = x + w["Wo"] @ ctx.reshape(-1)
            hf = rms(x, w["n_pf"])
            x = x + w["Wd"] @ (act(w["Wg"] @ hf) * (w["Wu"] @ hf))
        lg = embed @ rms(x, n_final)          # tied lm head
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
