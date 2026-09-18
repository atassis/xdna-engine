#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump Gemma-4-12B's vision + audio tower tensors to the same flat .npy tree
dump_llm_weights.py uses for the text stack -- one .npy per tensor, named by its exact HF key.

Sibling to dump_llm_weights.py rather than an extension of its `want` dict: that script's key set
is built from llm_decode_spec's per-layer geometry (SPECS[name].q_dim_for(l) etc), which has no
notion of these towers at all -- they are a fixed 11-tensor bag, not parameterized by layer count,
so folding them into that per-layer loop would be a bigger seam than a 90-line sibling.

  python scripts/dump_gemma4_towers.py --checkpoint-dir <dir with model.safetensors> --out <dir>

Every tensor's shape is checked against config.json's own vision_config/audio_config before
writing -- same discipline as dump_llm_weights.py, for the same reason (a silently transposed or
mis-shaped projection survives every build gate and shows up only as drifting output).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True,
                     help="local directory holding model.safetensors + config.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.checkpoint_dir, "config.json")))
    vc, ac = cfg["vision_config"], cfg["audio_config"]
    patch_dim = vc["patch_size"] ** 2 * 3  # model_patch_size**2 * 3, but patch_size here IS the
    # teacher patch size (16); the merge happens in preprocessing, so the tensor's own K axis is
    # patch_size**2*3 * pooling_kernel_size**2 == model_patch_size**2*3 -- checked below directly
    # against the tensor's own shape rather than re-derived, since both give 6912 and only one
    # needs to be right.
    model_patch_dim = (vc["patch_size"] * vc["pooling_kernel_size"]) ** 2 * 3
    D = vc["mm_embed_dim"]  # 3840, shared with audio's output_proj_dims and text hidden_size here

    want = {
        "model.vision_embedder.patch_ln1.weight": (model_patch_dim,),
        "model.vision_embedder.patch_ln1.bias": (model_patch_dim,),
        "model.vision_embedder.patch_dense.weight": (D, model_patch_dim),
        "model.vision_embedder.patch_dense.bias": (D,),
        "model.vision_embedder.patch_ln2.weight": (D,),
        "model.vision_embedder.patch_ln2.bias": (D,),
        "model.vision_embedder.pos_embedding": (vc["mm_posemb_size"], 2, D),
        "model.vision_embedder.pos_norm.weight": (D,),
        "model.vision_embedder.pos_norm.bias": (D,),
        "model.embed_vision.embedding_projection.weight": (D, vc["output_proj_dims"]),
        "model.embed_audio.embedding_projection.weight": (D, ac["audio_embed_dim"]),
    }
    del patch_dim  # unused beyond the comment above; kept named for the derivation it documents

    from safetensors import safe_open
    path = a.checkpoint_dir
    shards = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    index = {}
    for s in shards:
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                index[k] = s
    print(f"{path}: {len(index)} tensors across {len(shards)} shard(s)")

    os.makedirs(a.out, exist_ok=True)
    n = 0
    for key, want_shape in sorted(want.items()):
        if key not in index:
            raise KeyError(f"{key!r} not in checkpoint (have e.g. {sorted(index)[:3]})")
        with safe_open(index[key], framework="pt") as f:
            w = f.get_tensor(key).float().numpy()
        if w.shape != want_shape:
            raise ValueError(f"{key}: shape {w.shape} != expected {want_shape}")
        np.save(os.path.join(a.out, f"{key}.npy"), w)
        n += 1
    print(f"wrote {n} tensors to {a.out} -- all shapes checked against config.json")


if __name__ == "__main__":
    main()
