#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump ONE Gemma-4-E2B FFN sub-block I/O as the r1-spike correctness oracle (host CPU, float32).

Registers a forward hook on one decoder layer's MLP (FFN) submodule and captures its input hidden
state and output on a fixed prompt. This is the ground-truth the on-NPU resident FFN sub-block is
gated against (rel-L2 <= 0.08 + corr >= 0.99). NPU-first engine: never the dGPU.

Usage:
  CUDA_VISIBLE_DEVICES="" ~/gemma4-ref-venv/bin/python scripts/gemma_ffn_oracle.py \
      --model google/gemma-4-E2B [--layer -1] [--prompt "..."] [--out artifacts/gemma4-e2b/ffn_oracle]
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # NPU-first engine; the oracle is host CPU only

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _find_layers(model):
    """Return the decoder layer list across the possible Gemma-4 module nestings."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "language_model.layers"):
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
    ap.add_argument("--out", default="artifacts/gemma4-e2b/ffn_oracle")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="bf16 is memory-safe (~10GB) and matches the NPU kernel dtype; fp32 needs ~20GB RAM")
    a = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    torch.manual_seed(0)
    print(f"[oracle] loading {a.model} on CPU dtype={a.dtype} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    layers, layer_path = _find_layers(model)
    layer_idx = (len(layers) // 2) if a.layer < 0 else a.layer
    mlp = layers[layer_idx].mlp
    print(f"[oracle] layers at {layer_path!r} (n={len(layers)}); hooking layer {layer_idx} .mlp = {type(mlp).__name__}")

    cap = {}

    def hook(mod, inp, out):
        cap["in"] = inp[0].detach().float().cpu().numpy()
        cap["out"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu().numpy()

    h = mlp.register_forward_hook(hook)
    ids = tok(a.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids)
    h.remove()

    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/ffn_in.npy", cap["in"])
    np.save(f"{a.out}/ffn_out.npy", cap["out"])
    meta = {
        "model": a.model, "layer": int(layer_idx), "layer_path": layer_path,
        "mlp_type": type(mlp).__name__,
        "d_model": int(cfg.hidden_size), "intermediate": int(cfg.intermediate_size),
        "act": str(getattr(cfg, "hidden_activation", getattr(cfg, "hidden_act", "?"))),
        "prompt": a.prompt, "seed": 0,
        "in_shape": list(cap["in"].shape), "out_shape": list(cap["out"].shape),
    }
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2)
    print("[oracle]", json.dumps(meta))


if __name__ == "__main__":
    main()
