#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump a HuggingFace decoder-LLM checkpoint to the flat .npy tree gen_llm_decode.py reads.

One .npy per tensor, named by its HF key, f32. Only the tensors the spec's graph actually consumes
are written, and every one is CHECKED against the spec's dims before writing -- a silently
transposed or mis-shaped projection is the failure mode that survives every build gate and shows up
only as drifting token parity.

  python scripts/dump_llm_weights.py --spec qwen3-0.6b --out artifacts/qwen3-0.6b/weights
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "route_b_kernels", "decode_fused"))
from llm_decode_spec import SPECS  # noqa: E402

HF_REPO = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "gemma3-270m": "unsloth/gemma-3-270m-it"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None, help="override the HF repo id")
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    sp = SPECS[a.spec]
    repo = a.repo or HF_REPO[a.spec]
    NL = a.layers if a.layers is not None else sp.n_layers
    os.makedirs(a.out, exist_ok=True)

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
    shards = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    index = {}
    for s in shards:
        # framework="pt": these checkpoints are bf16, which safetensors' numpy backend cannot read
        # ("data type 'bfloat16' not understood"). torch reads it and .float() widens losslessly.
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                index[k] = s
    print(f"{repo}: {len(index)} tensors across {len(shards)} shard(s)")

    def get(key):
        if key not in index:
            raise KeyError(f"{key!r} not in checkpoint (have e.g. {sorted(index)[:3]})")
        with safe_open(index[key], framework="pt") as f:
            return f.get_tensor(key).float().numpy()

    want = {}
    for l in range(NL):
        want.update({v: None for v in sp.norm_weight_names(l).values()})
        for t in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            want[f"model.layers.{l}.{t}.weight"] = None
    want["model.norm.weight"] = None
    want["model.embed_tokens.weight"] = None

    # expected shapes -- HF stores nn.Linear weights as [out, in]
    D, FF, HD, QD, KVD, V = sp.d_model, sp.ffn, sp.head_dim, sp.q_dim, sp.kv_dim, sp.vocab
    exp = {"q_proj": (QD, D), "k_proj": (KVD, D), "v_proj": (KVD, D), "o_proj": (D, QD),
           "gate_proj": (FF, D), "up_proj": (FF, D), "down_proj": (D, FF)}

    n = 0
    for key in sorted(want):
        w = get(key)
        leaf = key.rsplit(".", 2)[-2]
        if leaf in exp and w.shape != exp[leaf]:
            raise ValueError(f"{key}: shape {w.shape} != expected {exp[leaf]} for spec {sp.name}")
        if key.endswith("q_norm.weight") or key.endswith("k_norm.weight"):
            if w.shape != (HD,):
                raise ValueError(f"{key}: shape {w.shape} != expected ({HD},)")
        elif "layernorm" in key or key == "model.norm.weight":
            if w.shape != (D,):
                raise ValueError(f"{key}: shape {w.shape} != expected ({D},)")
        if key == "model.embed_tokens.weight" and w.shape != (V, D):
            raise ValueError(f"{key}: shape {w.shape} != expected ({V}, {D})")
        np.save(os.path.join(a.out, f"{key}.npy"), w)
        n += 1
    print(f"wrote {n} tensors to {a.out} (spec {sp.name}, {NL} layers) -- all shapes checked")


if __name__ == "__main__":
    main()
