# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""GGUF q6_k -> .npy dumper for S2-Pro's Slow-AR, remapped to HF-style keys.

The GGUF's own tensor names (layers.N.attention.wqkv.weight, feed_forward.w1/w2/w3, ...) are
fish-speech's native layout, not the model.layers.N.self_attn.q_proj.weight keys
designs/decode_fused/gen_llm_prefill.py already reads (confirmed against dump_llm_weights.py). This
module bridges the two: wqkv is Q/K/V fused in one matrix (split here, per s2_model.cpp's own
Q-then-K-then-V slice order) and w1/w2/w3 are gate/down/up respectively (per s2_model.cpp's
feed-forward: gate=w1, up=w3, ff_out = w2 @ swiglu(gate, up)).

Dequantization is delegated to gguf.quants.dequantize (the library's own reference implementation
for Q6_K and friends), not re-derived here -- see test_dump_s2_gguf_weights.py's cross-check.
"""
import argparse
import os

import gguf
import numpy as np

# From S2-Pro checkpoint config: head_count=32, head_count_kv=8, head_dim=128, embedding_length=2560
D_MODEL = 2560  # embedding_length
Q_DIM = 32 * 128  # head_count * head_dim = 4096
KV_DIM = 8 * 128  # head_count_kv * head_dim = 1024
FFN_DIM = 9728  # feed_forward_length


def load_gguf_tensor_names(path: str) -> set[str]:
    reader = gguf.GGUFReader(path)
    return {t.name for t in reader.tensors}


def count_layers(tensor_dict: dict) -> int:
    """Count the number of transformer layers in a tensor dict by counting wqkv tensors."""
    return sum(1 for name in tensor_dict if name.startswith("layers.") and name.endswith("attention.wqkv.weight"))


def dequantize_tensor(tensor) -> np.ndarray:
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type)


def _get(tensor_dict: dict, name: str) -> np.ndarray:
    if name not in tensor_dict:
        raise KeyError(f"{name!r} not in GGUF")
    return dequantize_tensor(tensor_dict[name])


def dump_slow_ar_layer(tensor_dict: dict, layer: int) -> dict[str, np.ndarray]:
    """One Slow-AR layer's tensors, keyed by the HF-style name gen_llm_prefill.py expects.

    Every tensor is shape-checked against expected (out, in) or (dim,) before returning;
    a mismatch raises ValueError rather than silently writing a transposed/mis-shaped projection.
    """
    p = f"layers.{layer}."
    wqkv = _get(tensor_dict, p + "attention.wqkv.weight")
    if wqkv.shape != (Q_DIM + 2 * KV_DIM, D_MODEL):
        raise ValueError(f"layer {layer}: wqkv shape {wqkv.shape} != "
                         f"{(Q_DIM + 2 * KV_DIM, D_MODEL)}")
    hp = f"model.layers.{layer}."

    # Split and validate attention projections
    q_proj = wqkv[0:Q_DIM]
    k_proj = wqkv[Q_DIM:Q_DIM + KV_DIM]
    v_proj = wqkv[Q_DIM + KV_DIM:Q_DIM + 2 * KV_DIM]
    assert q_proj.shape == (Q_DIM, D_MODEL)
    assert k_proj.shape == (KV_DIM, D_MODEL)
    assert v_proj.shape == (KV_DIM, D_MODEL)

    o_proj = _get(tensor_dict, p + "attention.wo.weight")
    if o_proj.shape != (D_MODEL, Q_DIM):
        raise ValueError(f"layer {layer}: o_proj shape {o_proj.shape} != {(D_MODEL, Q_DIM)}")

    # Per-head norms (head_dim = 128)
    q_norm = _get(tensor_dict, p + "attention.q_norm.weight")
    if q_norm.shape != (128,):
        raise ValueError(f"layer {layer}: q_norm shape {q_norm.shape} != (128,)")
    k_norm = _get(tensor_dict, p + "attention.k_norm.weight")
    if k_norm.shape != (128,):
        raise ValueError(f"layer {layer}: k_norm shape {k_norm.shape} != (128,)")

    # Layer norms
    input_ln = _get(tensor_dict, p + "attention_norm.weight")
    if input_ln.shape != (D_MODEL,):
        raise ValueError(f"layer {layer}: input_layernorm shape {input_ln.shape} != ({D_MODEL},)")
    post_attn_ln = _get(tensor_dict, p + "ffn_norm.weight")
    if post_attn_ln.shape != (D_MODEL,):
        raise ValueError(f"layer {layer}: post_attention_layernorm shape {post_attn_ln.shape} != ({D_MODEL},)")

    # MLP projections (w1=gate, w3=up, w2=down per s2_model.cpp:1044-1047)
    gate_proj = _get(tensor_dict, p + "feed_forward.w1.weight")
    if gate_proj.shape != (FFN_DIM, D_MODEL):
        raise ValueError(f"layer {layer}: gate_proj shape {gate_proj.shape} != {(FFN_DIM, D_MODEL)}")
    up_proj = _get(tensor_dict, p + "feed_forward.w3.weight")
    if up_proj.shape != (FFN_DIM, D_MODEL):
        raise ValueError(f"layer {layer}: up_proj shape {up_proj.shape} != {(FFN_DIM, D_MODEL)}")
    down_proj = _get(tensor_dict, p + "feed_forward.w2.weight")
    if down_proj.shape != (D_MODEL, FFN_DIM):
        raise ValueError(f"layer {layer}: down_proj shape {down_proj.shape} != {(D_MODEL, FFN_DIM)}")

    return {
        hp + "self_attn.q_proj.weight": q_proj,
        hp + "self_attn.k_proj.weight": k_proj,
        hp + "self_attn.v_proj.weight": v_proj,
        hp + "self_attn.o_proj.weight": o_proj,
        hp + "self_attn.q_norm.weight": q_norm,
        hp + "self_attn.k_norm.weight": k_norm,
        hp + "input_layernorm.weight": input_ln,
        hp + "post_attention_layernorm.weight": post_attn_ln,
        hp + "mlp.gate_proj.weight": gate_proj,
        hp + "mlp.up_proj.weight": up_proj,
        hp + "mlp.down_proj.weight": down_proj,
    }


def dump(gguf_path: str, out_dir: str, layers: int | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    reader = gguf.GGUFReader(gguf_path)

    # Build raw tensor dict once to avoid repeated scans (dequantization happens lazily in _get())
    tensor_dict = {t.name: t for t in reader.tensors}

    n_layers = layers if layers is not None else count_layers(tensor_dict)
    written = 0
    for layer in range(n_layers):
        for key, arr in dump_slow_ar_layer(tensor_dict, layer).items():
            np.save(os.path.join(out_dir, key + ".npy"), arr)
            written += 1

    norm_weight = _get(tensor_dict, "norm.weight")
    np.save(os.path.join(out_dir, "model.norm.weight.npy"), norm_weight)
    embed_tokens = _get(tensor_dict, "embeddings.weight")
    np.save(os.path.join(out_dir, "model.embed_tokens.weight.npy"), embed_tokens)
    written += 2
    print(f"wrote {written} tensors ({n_layers} layers) to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="path to s2-pro-*.gguf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (bring-up)")
    args = ap.parse_args()
    dump(args.gguf, args.out, layers=args.layers)


if __name__ == "__main__":
    main()
