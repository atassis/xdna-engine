#!/usr/bin/env python3
"""Extract openai/whisper-small DECODER weights from the exported ONNX to npy.

Companion to scripts/extract_whisper_encoder.py (which pulls the encoder from
the HF transformers checkpoint). The decoder is exported to ONNX by optimum
(`optimum-cli export onnx --model openai/whisper-small`), producing
`decoder_model.onnx` / `decoder_with_past_model.onnx`. We extract from the
no-cache `decoder_model.onnx` because both share the same weights and the
plain graph is simpler to walk.

ONNX export folds every nn.Linear weight into an anonymous `onnx::MatMul_NNNN`
initializer that feeds a MatMul node; only the BIASES keep their PyTorch names.
We recover the semantic name of each weight from the MatMul *node* name
(e.g. `/model/decoder/layers.0/self_attn/q_proj/MatMul`), so the saved files
mirror the encoder extractor's scheme exactly:

  whisper_decoder/L{i}/{q,k,v,out}.weight.npy        self-attention   [K_in,N_out]
  whisper_decoder/L{i}/{q,k,v,out}.bias.npy          (k has none -> zeros)
  whisper_decoder/L{i}/cross_{q,k,v,out}.weight.npy  cross-attention  [K_in,N_out]
  whisper_decoder/L{i}/cross_{q,k,v,out}.bias.npy    (cross k has none -> zeros)
  whisper_decoder/L{i}/fc1.weight.npy  fc1.bias.npy  [768,3072] / [3072]
  whisper_decoder/L{i}/fc2.weight.npy  fc2.bias.npy  [3072,768] / [768]
  whisper_decoder/L{i}/ln_self.{weight,bias}.npy     self_attn_layer_norm  (gamma,beta)
  whisper_decoder/L{i}/ln_cross.{weight,bias}.npy    encoder_attn_layer_norm
  whisper_decoder/L{i}/ln_final.{weight,bias}.npy    final_layer_norm (post-FFN, pre-residual)
  whisper_decoder/embed_tokens.npy        [vocab,768]   token embedding
  whisper_decoder/embed_positions.npy     [448,768]     learned positions
  whisper_decoder/proj_out.weight.npy     [768,vocab]   output projection (tied to embed_tokens.T)
  whisper_decoder/ln_post.{weight,bias}.npy             model.decoder.layer_norm (final pre-proj_out)

Linear weights keep ONNX's [K_in, N_out] layout (the MatMul rhs), i.e. already
transposed for `x @ W`, matching the encoder extractor's stored convention.

All tensors saved as float32.
"""
import os
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# Decoder ONNX lives under the scenario's [artifacts].weights dir. Default to the
# main checkout's artifacts (this worktree's artifacts/ is gitignored/regenerable).
DEFAULT_ONNX = (
    "$REPO"
    "/artifacts/whisper-small/onnx/decoder_model.onnx"
)
ONNX_PATH = Path(os.environ.get("WHISPER_DECODER_ONNX", DEFAULT_ONNX))
# Output mirrors the encoder convention; sibling to encoder's artifacts/whisper-small.
ART = Path(os.environ.get("WHISPER_OUT", str(ONNX_PATH.parent.parent)))
OUT = ART / "whisper_decoder"

N_LAYERS = 12
HIDDEN = 768
FFN = 3072

if not ONNX_PATH.exists():
    raise SystemExit(f"decoder ONNX not found: {ONNX_PATH}\n"
                     "Set WHISPER_DECODER_ONNX or export it via "
                     "`optimum-cli export onnx --model openai/whisper-small`.")

OUT.mkdir(parents=True, exist_ok=True)

print(f"loading {ONNX_PATH}")
model = onnx.load(str(ONNX_PATH))
g = model.graph
byname = {i.name: i for i in g.initializer}


def init_arr(name):
    return numpy_helper.to_array(byname[name]).astype(np.float32)


# tensor name -> nodes that consume it (to find the MatMul that uses a weight init)
consumers = {}
for node in g.node:
    for inp in node.input:
        consumers.setdefault(inp, []).append(node)

# Map each weight: semantic MatMul node name -> initializer name of its rhs weight.
# MatMul node name looks like /model/decoder/layers.0/self_attn/q_proj/MatMul
# or /proj_out/MatMul. The weight is the input that is an onnx::MatMul_* init.
weight_init_for = {}
for node in g.node:
    if node.op_type != "MatMul":
        continue
    for inp in node.input:
        if inp in byname:  # this MatMul operand is an initializer => it's the weight
            weight_init_for[node.name] = inp


manifest = []


def save(rel, t):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(t.astype(np.float32))
    np.save(p, a)
    manifest.append((str(rel), tuple(a.shape)))


def w(node_name):
    """Weight matrix [K_in, N_out] for a given MatMul node name."""
    return init_arr(weight_init_for[node_name])


def b(init_name, n):
    """Bias vector, or zeros[n] if Whisper drops it (k_proj / cross k_proj)."""
    if init_name in byname:
        return init_arr(init_name)
    return np.zeros(n, dtype=np.float32)


def ln(prefix):
    return init_arr(prefix + ".weight"), init_arr(prefix + ".bias")


for i in range(N_LAYERS):
    base = f"/model/decoder/layers.{i}"
    pbase = f"model.decoder.layers.{i}"
    L = f"L{i}"

    # --- self-attention: q/k/v/out (k has no bias) ---
    for short, proj in [("q", "q_proj"), ("k", "k_proj"),
                        ("v", "v_proj"), ("out", "out_proj")]:
        save(f"{L}/{short}.weight.npy", w(f"{base}/self_attn/{proj}/MatMul"))
        save(f"{L}/{short}.bias.npy",
             b(f"{pbase}.self_attn.{proj}.bias", HIDDEN))

    # --- cross-attention (encoder_attn): q/k/v/out (k has no bias) ---
    # k/v project the encoder output; q/out project the decoder stream.
    for short, proj in [("cross_q", "q_proj"), ("cross_k", "k_proj"),
                        ("cross_v", "v_proj"), ("cross_out", "out_proj")]:
        save(f"{L}/{short}.weight.npy", w(f"{base}/encoder_attn/{proj}/MatMul"))
        save(f"{L}/{short}.bias.npy",
             b(f"{pbase}.encoder_attn.{proj}.bias", HIDDEN))

    # --- FFN ---
    save(f"{L}/fc1.weight.npy", w(f"{base}/fc1/MatMul"))
    save(f"{L}/fc1.bias.npy", b(f"{pbase}.fc1.bias", FFN))
    save(f"{L}/fc2.weight.npy", w(f"{base}/fc2/MatMul"))
    save(f"{L}/fc2.bias.npy", b(f"{pbase}.fc2.bias", HIDDEN))

    # --- LayerNorms (gamma=weight, beta=bias) ---
    gw, gb = ln(f"{pbase}.self_attn_layer_norm")
    save(f"{L}/ln_self.weight.npy", gw)
    save(f"{L}/ln_self.bias.npy", gb)
    gw, gb = ln(f"{pbase}.encoder_attn_layer_norm")
    save(f"{L}/ln_cross.weight.npy", gw)
    save(f"{L}/ln_cross.bias.npy", gb)
    gw, gb = ln(f"{pbase}.final_layer_norm")
    save(f"{L}/ln_final.weight.npy", gw)
    save(f"{L}/ln_final.bias.npy", gb)

# --- embeddings + output projection + final layernorm ---
save("embed_tokens.npy", init_arr("model.decoder.embed_tokens.weight"))
save("embed_positions.npy", init_arr("model.decoder.embed_positions.weight"))
save("proj_out.weight.npy", w("/proj_out/MatMul"))
gw, gb = ln("model.decoder.layer_norm")
save("ln_post.weight.npy", gw)
save("ln_post.bias.npy", gb)

# --- manifest ---
print(f"\nwrote {len(manifest)} tensors -> {OUT}\n")
for name, shape in manifest:
    print(f"  {name:32s} {shape}")

# --- shape assertions against whisper-small ---
emb = np.load(OUT / "embed_tokens.npy")
vocab = emb.shape[0]
assert emb.shape == (vocab, HIDDEN), emb.shape
assert np.load(OUT / "L0/q.weight.npy").shape == (HIDDEN, HIDDEN)
assert np.load(OUT / "L0/fc1.weight.npy").shape == (HIDDEN, FFN)
assert np.load(OUT / "L0/fc2.weight.npy").shape == (FFN, HIDDEN)
assert np.load(OUT / "proj_out.weight.npy").shape == (HIDDEN, vocab)
# per layer: self q/k/v/out (w+b)=8, cross q/k/v/out (w+b)=8, fc1/fc2 (w+b)=4, 3 LN (w+b)=6 -> 26
assert sum(1 for n, _ in manifest if n.startswith("L")) == N_LAYERS * 26
print(f"\nshape checks OK: hidden={HIDDEN} ffn={FFN} vocab={vocab} layers={N_LAYERS}")

# --- sanity: extracted npy == ONNX initializer, first 5 values ---
chk_node = "/model/decoder/layers.0/self_attn/q_proj/MatMul"
chk_init = weight_init_for[chk_node]
from_npy = np.load(OUT / "L0/q.weight.npy").ravel()[:5]
from_onnx = init_arr(chk_init).ravel()[:5]
print("\nsanity (L0 self_attn q_proj weight, first 5):")
print("  npy :", from_npy)
print("  onnx:", from_onnx)
print("  equal:", np.array_equal(from_npy, from_onnx))
assert np.array_equal(from_npy, from_onnx)
