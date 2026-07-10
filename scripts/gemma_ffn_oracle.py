#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture ONE Gemma FFN sub-block (I/O + weights) as the r1-spike correctness oracle (host CPU).

The FFN sub-block is the resident unit the on-NPU kernel reproduces:

    x_in -> pre_feedforward_layernorm -> mlp(gate/up -> GELU(gate)*up -> down)
         -> post_feedforward_layernorm -> (+ x_in residual) -> x_out

i.e. x_out = x_in + post_ff_norm( down( gelu(gate(pre_ff_norm(x_in))) * up(pre_ff_norm(x_in)) ) ).

We hook the pre-norm INPUT (= sub-block input on the residual stream) and the post-norm OUTPUT, and
reconstruct x_out = x_in + post_norm_out. We also dump the three FFN weight matrices + both RMSNorm
gammas + eps so the numpy golden and the NPU kernel consume the exact same weights. ONE model load
(matters for the 10GB E2B). NPU-first engine: never the dGPU.

Usage:
  CUDA_VISIBLE_DEVICES="" ~/gemma4-ref-venv/bin/python scripts/gemma_ffn_oracle.py \
      --model google/gemma-4-E2B --out artifacts/gemma4-e2b/ffn_oracle [--layer -1] [--dtype bfloat16]
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # NPU-first engine; the oracle is host CPU only

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _find_layers(model):
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers", "language_model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj, path
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers; inspect model with print(model)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=-1, help="decoder layer idx; -1 = auto middle layer")
    ap.add_argument("--out", default="artifacts/gemma-ffn/ffn_oracle")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="bf16 is memory-safe (~10GB E2B) and matches the NPU kernel; fp32 needs ~20GB")
    a = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    torch.manual_seed(0)
    print(f"[oracle] loading {a.model} dtype={a.dtype}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    layers, layer_path = _find_layers(model)
    layer_idx = (len(layers) // 2) if a.layer < 0 else a.layer
    L = layers[layer_idx]
    for attr in ("pre_feedforward_layernorm", "mlp"):
        if not hasattr(L, attr):
            raise RuntimeError(f"layer {layer_idx} has no {attr}; this script assumes the Gemma pre-norm FFN")
    print(f"[oracle] layers at {layer_path!r} (n={len(layers)}); sub-block at layer {layer_idx}")

    # Resident sub-block boundary = pre_feedforward_layernorm INPUT (x, the residual stream) -> mlp OUTPUT.
    # We deliberately EXCLUDE post_feedforward_layernorm, the residual add, and (Gemma-4 E2B) the MoE
    # combine / per-layer-input (PLE) block / layer_scalar -- those are layer-level plumbing outside the
    # resident GEMM core. Hooking `mlp` directly gives the PURE dense gated-GeGLU on every model.
    cap = {}
    h1 = L.pre_feedforward_layernorm.register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("x_in", inp[0].detach().float().cpu().numpy()))
    h2 = L.mlp.register_forward_hook(
        lambda m, inp, out: cap.__setitem__("mlp_out",
                                            (out[0] if isinstance(out, tuple) else out).detach().float().cpu().numpy()))
    ids = tok(a.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids)
    h1.remove(); h2.remove()
    x_in = cap["x_in"]       # sub-block input on the residual stream (before pre_norm)
    x_out = cap["mlp_out"]   # sub-block output = dense gated-GeGLU result (before post_norm/residual/PLE)

    def effective_gamma(norm):
        """The per-channel multiplier the RMSNorm actually applies, so the golden/kernel stay
        convention-agnostic: Gemma3 does normed*(1+weight); Gemma4 does normed*weight (if with_scale)."""
        w = norm.weight.detach().float().cpu().numpy()
        if hasattr(norm, "with_scale"):            # Gemma4-style
            return w if getattr(norm, "with_scale", True) else np.ones_like(w)
        return 1.0 + w                             # Gemma3-style

    os.makedirs(a.out, exist_ok=True)
    wdir = os.path.join(a.out, "weights"); os.makedirs(wdir, exist_ok=True)
    np.save(f"{a.out}/ffn_in.npy", x_in)
    np.save(f"{a.out}/ffn_out.npy", x_out)
    for k, v in {"gate_proj": L.mlp.gate_proj.weight, "up_proj": L.mlp.up_proj.weight,
                 "down_proj": L.mlp.down_proj.weight}.items():
        np.save(f"{wdir}/{k}.npy", v.detach().float().cpu().numpy())
    np.save(f"{wdir}/pre_norm.npy", effective_gamma(L.pre_feedforward_layernorm))   # effective multiplier
    np.save(f"{wdir}/post_norm.npy", effective_gamma(L.post_feedforward_layernorm))
    norm_conv = "gemma4_weight" if hasattr(L.pre_feedforward_layernorm, "with_scale") else "gemma3_1plus_weight"

    meta = {
        "model": a.model, "layer": int(layer_idx), "layer_path": layer_path,
        "d_model": int(cfg.hidden_size), "intermediate": int(cfg.intermediate_size),
        "act": str(getattr(cfg, "hidden_activation", getattr(cfg, "hidden_act", "?"))),
        "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-6)),
        "norm_convention": norm_conv,  # pre_norm/post_norm.npy already hold the EFFECTIVE gamma
        "dtype": a.dtype, "prompt": a.prompt, "seed": 0,
        "in_shape": list(x_in.shape), "out_shape": list(x_out.shape),
        "weights": {k: list(v.shape) for k, v in W.items()},
    }
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2)
    print("[oracle]", json.dumps(meta))


if __name__ == "__main__":
    main()
