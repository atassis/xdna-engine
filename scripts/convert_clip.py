#!/usr/bin/env python3
"""Convert a transformers CLIPModel (laion/CLIP-ViT-B-32-laion2B-s34B-b79K) to engine per-tensor
.npy ([K,N] linears). This is the OpenAI-architecture CLIP in transformers format; the laion repo is
used because it ships model.safetensors (openai/clip-vit-base-patch32 ships only pytorch_model.bin,
which the npu-weights loader does not read).

CLIP-ViT-B/32: a TEXT tower (12-layer transformer, hidden 512, ffn 2048, 49408 vocab, 77 ctx) and a
VISION tower (12-layer ViT, hidden 768, ffn 3072, patch32, 50 pos), each a stack of the SAME
CLIPEncoderLayer (q/k/v/out_proj WITH bias, layer_norm1/2, mlp.fc1/fc2 with bias, gelu), plus
text_projection / visual_projection (Linear bias-free, both -> 512 joint dim) and a logit_scale
scalar. The vision patch-embedding Conv2d (no bias) is im2col-flattened + transposed into a GEMM
weight exactly as convert_vit.py. Per-tower layer counts are inferred. Output: artifacts/clip-vit-b32/."""
import os, re, numpy as np, torch
from transformers import CLIPModel

HF = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
OUT = "artifacts/clip-vit-b32"
model = CLIPModel.from_pretrained(HF).eval()
sd = model.state_dict()
g = lambda k: sd[k].cpu().numpy().astype(np.float32)
def save(p, a): os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, np.ascontiguousarray(a))

def n_layers(prefix):
    return 1 + max(int(m.group(1)) for k in sd
                   for m in [re.match(re.escape(prefix) + r"encoder\.layers\.(\d+)\.", k)] if m)

def emit_layer(tower, prefix, i):
    p = f"{prefix}encoder.layers.{i}."; d = f"{OUT}/{tower}/L{i}"
    for s, t in [("self_attn.q_proj","q"),("self_attn.k_proj","k"),("self_attn.v_proj","v"),
                 ("self_attn.out_proj","out")]:
        save(f"{d}/{t}.weight.npy", g(p+s+".weight").T.copy()); save(f"{d}/{t}.bias.npy", g(p+s+".bias"))
    save(f"{d}/ln1.weight.npy", g(p+"layer_norm1.weight")); save(f"{d}/ln1.bias.npy", g(p+"layer_norm1.bias"))
    save(f"{d}/ln2.weight.npy", g(p+"layer_norm2.weight")); save(f"{d}/ln2.bias.npy", g(p+"layer_norm2.bias"))
    save(f"{d}/fc1.weight.npy", g(p+"mlp.fc1.weight").T.copy()); save(f"{d}/fc1.bias.npy", g(p+"mlp.fc1.bias"))
    save(f"{d}/fc2.weight.npy", g(p+"mlp.fc2.weight").T.copy()); save(f"{d}/fc2.bias.npy", g(p+"mlp.fc2.bias"))

# TEXT tower
tl = n_layers("text_model.")
save(f"{OUT}/text/tok_emb.npy", g("text_model.embeddings.token_embedding.weight"))
save(f"{OUT}/text/pos_emb.npy", g("text_model.embeddings.position_embedding.weight"))
save(f"{OUT}/text/final_ln.weight.npy", g("text_model.final_layer_norm.weight"))
save(f"{OUT}/text/final_ln.bias.npy", g("text_model.final_layer_norm.bias"))
for i in range(tl): emit_layer("text", "text_model.", i)

# VISION tower
vl = n_layers("vision_model.")
save(f"{OUT}/vision/cls_emb.npy", g("vision_model.embeddings.class_embedding").reshape(-1))
pw = g("vision_model.embeddings.patch_embedding.weight")  # [768,3,32,32]
save(f"{OUT}/vision/patch_proj.weight.npy", pw.reshape(pw.shape[0], -1).T.copy())  # [K,N]
save(f"{OUT}/vision/pos_emb.npy", g("vision_model.embeddings.position_embedding.weight"))
save(f"{OUT}/vision/pre_ln.weight.npy", g("vision_model.pre_layrnorm.weight"))
save(f"{OUT}/vision/pre_ln.bias.npy", g("vision_model.pre_layrnorm.bias"))
save(f"{OUT}/vision/post_ln.weight.npy", g("vision_model.post_layernorm.weight"))
save(f"{OUT}/vision/post_ln.bias.npy", g("vision_model.post_layernorm.bias"))
for i in range(vl): emit_layer("vision", "vision_model.", i)

# joint projections (Linear bias-free) + logit scale
save(f"{OUT}/text_projection.npy", g("text_projection.weight").T.copy())     # [512,512]
save(f"{OUT}/visual_projection.npy", g("visual_projection.weight").T.copy())  # [768,512]
save(f"{OUT}/logit_scale.npy", g("logit_scale").reshape(1))
print(f"converted clip-vit-b32 (text {tl} / vision {vl} layers) -> {OUT}/")
