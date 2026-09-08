#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump a HuggingFace decoder-LLM checkpoint to the flat .npy tree gen_llm_decode.py reads.

One .npy per tensor, named by its HF key. Only the tensors the spec's graph actually consumes are
written, and every one is CHECKED against the spec's dims before writing -- a silently transposed or
mis-shaped projection is the failure mode that survives every build gate and shows up only as
drifting token parity.

  python scripts/dump_llm_weights.py --spec qwen3-0.6b --out artifacts/qwen3-0.6b/weights

`--quant` packs the PROJECTION matrices on the way out, for models whose f32 dump does not fit on
disk: a 12B is 43 GB at f32 and 6 GB at int4 g64. Packed tensors are np.int8 ON-WIRE BYTES, not
values, so the dump declares itself in `quant.json` (dtype, group_size, and the exact set of packed
names) and the generator READS that declaration. Agreeing via matching env vars instead would make a
group-size disagreement silent numerical garbage rather than a load error -- the packer and the
kernel share a byte layout, and nothing else checks it.

  python scripts/dump_llm_weights.py --spec qwen3-0.6b --out ... --quant int4 --quant-group 64
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "designs", "decode_fused"))
from llm_decode_spec import SPECS  # noqa: E402

HF_REPO = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "gemma3-270m": "unsloth/gemma-3-270m-it"}

# The projection leaves, i.e. everything that is a [out, in] matrix rather than a norm gain or the
# embedding table. Only these are packable.
EXP_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None, help="override the HF repo id")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--quant", default="bf16", choices=("bf16", "int4", "int8"),
                    help="pack the PROJECTION matrices at this width (norms and the embedding stay "
                         "f32 and readable). bf16 writes the plain f32 dump.")
    ap.add_argument("--quant-group", type=int, default=64)
    # Default: exactly the leaves gen_llm_decode.py can CONSUME packed today -- the MLP class
    # (QUANT_MLP_DTYPE) and Wo (QUANT_ATTN_DTYPE). q/k/v are excluded because their GEMVs are built
    # without quant kwargs (`gemv(QD, D, ctx)`, and the concatenated Wqkv likewise), so a packed
    # q_proj has no reader. Packing them is allowed but the build will refuse it by name rather
    # than quietly widening the bytes.
    ap.add_argument("--quant-leaves", default="gate_proj,up_proj,down_proj,o_proj",
                    help="comma-separated projection leaves to pack (default: the ones the "
                         "generator can read back)")
    a = ap.parse_args()
    sp = SPECS[a.spec]
    repo = a.repo or HF_REPO[a.spec]
    NL = a.layers if a.layers is not None else sp.n_layers
    os.makedirs(a.out, exist_ok=True)

    quant_leaves = {x for x in a.quant_leaves.split(",") if x}
    quantize_weight = None
    if a.quant != "bf16":
        unknown = quant_leaves - set(EXP_LEAVES)
        if unknown:
            ap.error(f"--quant-leaves has non-projection leaves {sorted(unknown)}; "
                     f"expected a subset of {sorted(EXP_LEAVES)}")
        # Imported from IRON rather than reimplemented here, because the on-wire row layout
        # ([n_groups x f32 scale][packed payload]) is shared with the matvec kernel. A second copy
        # of it is a seam with no owner, which is the class this whole manifest exists to close.
        from iron.operators.gemv.quant import quantize_weight

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

    n, packed = 0, []
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
        # Shapes are checked ABOVE, on the f32 array, before any packing -- a packed tensor is a
        # flat byte run and has no shape left to check.
        if quantize_weight is not None and leaf in quant_leaves:
            np.save(os.path.join(a.out, f"{key}.npy"),
                    quantize_weight(w, a.quant_group, a.quant))
            packed.append(key)
        else:
            np.save(os.path.join(a.out, f"{key}.npy"), w)
        n += 1

    manifest = {
        "dtype": a.quant,
        "group_size": a.quant_group,
        "packed": sorted(packed),
        "note": "packed arrays are np.int8 on-wire bytes, NOT values -- never .astype()",
    }
    with open(os.path.join(a.out, "quant.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    tail = ""
    if packed:
        bits = ((4 if a.quant == "int4" else 8) * a.quant_group + 32) / a.quant_group
        tail = f", {len(packed)} packed at {a.quant} g{a.quant_group} = {bits:.2f} bits/weight"
    print(f"wrote {n} tensors to {a.out} (spec {sp.name}, {NL} layers) -- all shapes checked{tail}")


if __name__ == "__main__":
    main()
