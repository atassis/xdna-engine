# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Fish-Audio dual-AR checkpoint -> .npy dumper, remapped to the HF-style keys the rail reads.

Serves S2-Pro and s1-mini from either source they ship in: a PyTorch `model.pth` or a GGUF. Their
tensor NAMES are the same in both (`layers.N.attention.wqkv.weight`, `feed_forward.w1/w2/w3`, ...)
-- verified bit-for-bit on s1-mini, whose GGUF conversion is a straight dump -- so only the reader
differs. `designs/decode_fused/gen_llm_prefill.py` reads `model.layers.N.self_attn.q_proj.weight`
instead, so this module bridges the two: wqkv is Q/K/V fused in one matrix (split per
s2_model.cpp's own Q-then-K-then-V slice order) and w1/w2/w3 are gate/down/up respectively.

THE ROPE PERMUTATION IS THE REASON THIS FILE IS NOT A PLAIN RENAME. These checkpoints are rotated
with ggml's GGML_ROPE_TYPE_NORMAL -- adjacent pairs (0,1),(2,3),... each taking one (cos,sin) --
and the rail's RoPE operator uses the NeoX split-half form. They are different functions, rel-L2
0.776 apart on one row. Reordering each head's rows to [0,2,4,...,1,3,5,...] makes the NeoX form
compute exactly what the adjacent-pair form does (2.5e-08, the f32 noise floor), so the conversion
happens once here rather than as a second RoPE operator on the device. q_norm/k_norm are permuted
with their features because QK-norm runs BEFORE RoPE; v and wo are untouched, since v is never
rotated and attention is invariant under a permutation q and k share.

Dequantization is delegated to gguf.quants.dequantize (the library's own reference implementation
for Q6_K and friends), not re-derived here -- see the test module's cross-check.
"""
import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Dims:
    """Geometry read off the tensors themselves, never from module constants.

    Two shapes pin all four attention numbers: `wo` is (D, n_q*head_dim) so it gives the query
    width, and `wqkv` is ((n_q + 2*n_kv)*head_dim, D) so the remainder gives the KV width. head_dim
    comes from the per-head norm's own length. A checkpoint whose shapes disagree with this fails
    in `derive_dims` rather than producing a plausible wrong split.
    """
    d_model: int
    head_dim: int
    n_q_heads: int
    n_kv_heads: int
    ffn: int
    n_layers: int
    vocab: int
    tied_embeddings: bool

    @property
    def q_dim(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


class Source:
    """A checkpoint, read by name. `.pth` goes through torch, anything else through gguf."""

    def __init__(self, path: str):
        self.path = path
        self._is_pth = path.endswith((".pth", ".pt", ".bin"))
        if self._is_pth:
            import torch
            sd = torch.load(path, map_location="cpu", weights_only=True)
            self._t = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        else:
            import gguf
            self._t = {t.name: t for t in gguf.GGUFReader(path).tensors}

    def names(self) -> set:
        return set(self._t)

    def get(self, name: str) -> np.ndarray:
        """One tensor as f32, in (out, in) order."""
        if name not in self._t:
            raise KeyError(f"{name!r} not in {self.path}")
        t = self._t[name]
        if self._is_pth:
            return t.float().numpy()
        return dequantize_tensor(t).astype(np.float32, copy=False)


def dequantize_tensor(tensor) -> np.ndarray:
    import gguf
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type)


def load_gguf_tensor_names(path: str) -> set:
    return Source(path).names()


def count_layers(names) -> int:
    """Slow-AR depth, counted from the wqkv tensors. `fast_layers.` is a different stack."""
    names = names.names() if isinstance(names, Source) else names
    return sum(1 for n in names
               if n.startswith("layers.") and n.endswith("attention.wqkv.weight"))


def derive_dims(src: Source) -> Dims:
    names = src.names()
    wqkv = src.get("layers.0.attention.wqkv.weight")
    wo = src.get("layers.0.attention.wo.weight")
    head_dim = src.get("layers.0.attention.q_norm.weight").shape[0]
    d_model = wqkv.shape[1]
    if wo.shape[0] != d_model:
        raise ValueError(f"wo out-dim {wo.shape[0]} != wqkv in-dim {d_model}")
    q_dim = wo.shape[1]
    if q_dim % head_dim:
        raise ValueError(f"q_dim {q_dim} is not a multiple of head_dim {head_dim}")
    kv_two = wqkv.shape[0] - q_dim
    if kv_two % (2 * head_dim):
        raise ValueError(f"wqkv rows {wqkv.shape[0]} leave {kv_two} for K+V, which is not "
                         f"2 * a multiple of head_dim {head_dim}")
    embed = src.get("embeddings.weight")
    return Dims(d_model=d_model, head_dim=head_dim,
                n_q_heads=q_dim // head_dim, n_kv_heads=kv_two // (2 * head_dim),
                ffn=src.get("layers.0.feed_forward.w1.weight").shape[0],
                n_layers=count_layers(names), vocab=embed.shape[0],
                tied_embeddings="output.weight" not in names)


def rope_row_permutation(head_dim: int) -> np.ndarray:
    """[0, 2, 4, ..., 1, 3, 5, ...] -- see the module docstring."""
    return np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])


def permute_heads(w: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """Reorder each head's rows of a projection, or the elements of a per-head gain vector."""
    perm = rope_row_permutation(head_dim)
    if w.ndim == 1:
        return w[perm]
    return w.reshape(n_heads, head_dim, -1)[:, perm, :].reshape(w.shape)


def dump_slow_ar_layer(src: Source, d: Dims, layer: int, permute_rope: bool = True) -> dict:
    """One Slow-AR layer, keyed by the HF-style name gen_llm_prefill.py expects.

    Every tensor is shape-checked against `d` before it is returned, so a checkpoint that disagrees
    with the geometry raises instead of silently writing a transposed or mis-split projection.
    """
    p, hp = f"layers.{layer}.", f"model.layers.{layer}."

    def need(name, want):
        a = src.get(p + name)
        if a.shape != want:
            raise ValueError(f"layer {layer}: {name} shape {a.shape} != {want}")
        return a

    wqkv = need("attention.wqkv.weight", (d.q_dim + 2 * d.kv_dim, d.d_model))
    q_proj = wqkv[:d.q_dim]
    k_proj = wqkv[d.q_dim:d.q_dim + d.kv_dim]
    v_proj = wqkv[d.q_dim + d.kv_dim:]
    q_norm = need("attention.q_norm.weight", (d.head_dim,))
    k_norm = need("attention.k_norm.weight", (d.head_dim,))
    if permute_rope:
        q_proj = permute_heads(q_proj, d.n_q_heads, d.head_dim)
        k_proj = permute_heads(k_proj, d.n_kv_heads, d.head_dim)
        q_norm = permute_heads(q_norm, 1, d.head_dim)
        k_norm = permute_heads(k_norm, 1, d.head_dim)

    return {
        hp + "self_attn.q_proj.weight": q_proj,
        hp + "self_attn.k_proj.weight": k_proj,
        hp + "self_attn.v_proj.weight": v_proj,
        hp + "self_attn.o_proj.weight": need("attention.wo.weight", (d.d_model, d.q_dim)),
        hp + "self_attn.q_norm.weight": q_norm,
        hp + "self_attn.k_norm.weight": k_norm,
        hp + "input_layernorm.weight": need("attention_norm.weight", (d.d_model,)),
        hp + "post_attention_layernorm.weight": need("ffn_norm.weight", (d.d_model,)),
        # w1=gate, w3=up, w2=down, per s2_model.cpp's feed-forward.
        hp + "mlp.gate_proj.weight": need("feed_forward.w1.weight", (d.ffn, d.d_model)),
        hp + "mlp.up_proj.weight": need("feed_forward.w3.weight", (d.ffn, d.d_model)),
        hp + "mlp.down_proj.weight": need("feed_forward.w2.weight", (d.d_model, d.ffn)),
    }


def dump(path: str, out_dir: str, layers=None, permute_rope: bool = True) -> Dims:
    os.makedirs(out_dir, exist_ok=True)
    src = Source(path)
    d = derive_dims(src)
    n_layers = d.n_layers if layers is None else layers
    written = 0
    for layer in range(n_layers):
        for key, arr in dump_slow_ar_layer(src, d, layer, permute_rope).items():
            np.save(os.path.join(out_dir, key + ".npy"), arr)
            written += 1

    np.save(os.path.join(out_dir, "model.norm.weight.npy"), src.get("norm.weight"))
    np.save(os.path.join(out_dir, "model.embed_tokens.weight.npy"), src.get("embeddings.weight"))
    written += 2
    # An untied model keeps its head apart from the gather table and BOTH are needed: the device
    # streams the head, the host gathers prompt rows from the table. A tied one has one file.
    if not d.tied_embeddings:
        np.save(os.path.join(out_dir, "model.lm_head.weight.npy"), src.get("output.weight"))
        written += 1

    manifest = dict(source=os.path.abspath(path), source_sha256=_sha256(path),
                    layers_written=n_layers, tensors_written=written,
                    rope_rows_permuted=bool(permute_rope), dims=asdict(d))
    with open(os.path.join(out_dir, "dump_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {written} tensors ({n_layers} of {d.n_layers} layers, "
          f"rope_rows_permuted={permute_rope}) to {out_dir}")
    return d


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="model.pth or *.gguf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (bring-up)")
    ap.add_argument("--no-rope-permute", action="store_true",
                    help="write q/k as the checkpoint stores them. Only for reproducing a dump "
                         "made before the permutation existed -- the rail will rotate the wrong "
                         "element pairs. See the module docstring.")
    a = ap.parse_args()
    dump(a.checkpoint, a.out, layers=a.layers, permute_rope=not a.no_rope_permute)


if __name__ == "__main__":
    main()
