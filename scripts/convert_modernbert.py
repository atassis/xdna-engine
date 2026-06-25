#!/usr/bin/env python3
"""Convert answerdotai/ModernBERT-base (safetensors) to engine per-tensor .npy ([K,N] linears).

ModernBERT-base: 768 hidden / 22 layers / 1152 GeGLU intermediate, BIAS-FREE, RoPE (no learned
position embeddings), fused QKV (attn.Wqkv -> 3*768=2304), GeGLU MLP (mlp.Wi -> 2*1152=2304 gate+
value, mlp.Wo back to 768), pre-norm with LayerNorm weight-only (no bias). Layer 0's attn_norm is
nn.Identity (absent in the checkpoint) since the embedding norm already normalizes. Only the encoder
BACKBONE is baked (not the MaskedLM head/decoder), mirroring the bert/vit arches. Layer count is
inferred from the checkpoint. Output: artifacts/modernbert-base/."""
import os, re, numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download
# Resolve from whatever HF cache is configured (HF_HOME may point off ~/.cache); downloads if absent.
SRC = hf_hub_download("answerdotai/ModernBERT-base", "model.safetensors")
OUT = "artifacts/modernbert-base"
f = safe_open(SRC, framework="np")
keys = list(f.keys())
NL = 1 + max(int(m.group(1)) for k in keys
             for m in [re.match(r"model\.layers\.(\d+)\.", k)] if m)
def g(k): return f.get_tensor(k).astype(np.float32)
def save(p, a): os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, np.ascontiguousarray(a))
# embeddings: token table kept verbatim [vocab,hidden]; embedding norm (weight only)
save(f"{OUT}/emb/tok_emb.npy", g("model.embeddings.tok_embeddings.weight"))
save(f"{OUT}/emb/norm_w.npy", g("model.embeddings.norm.weight"))
save(f"{OUT}/final_norm_w.npy", g("model.final_norm.weight"))
for i in range(NL):
    p = f"model.layers.{i}."; d = f"{OUT}/L{i}"
    # attn_norm is Identity for layer 0 (no param in the checkpoint) - emit only when present
    if f"{p}attn_norm.weight" in keys:
        save(f"{d}/attn_norm_w.npy", g(p+"attn_norm.weight"))
    save(f"{d}/qkv_w.npy", g(p+"attn.Wqkv.weight").T.copy())      # [768, 2304]
    save(f"{d}/attn_out_w.npy", g(p+"attn.Wo.weight").T.copy())   # [768, 768]
    save(f"{d}/mlp_norm_w.npy", g(p+"mlp_norm.weight"))
    save(f"{d}/wi_w.npy", g(p+"mlp.Wi.weight").T.copy())          # [768, 2304] (GeGLU gate+value)
    save(f"{d}/wo_w.npy", g(p+"mlp.Wo.weight").T.copy())          # [1152, 768]
print(f"converted modernbert-base ({NL} layers) -> {OUT}/")
