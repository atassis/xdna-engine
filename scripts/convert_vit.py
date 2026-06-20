#!/usr/bin/env python3
"""Convert google/vit-base-patch16-224 (safetensors) to engine per-tensor .npy ([K,N] linears).
ViT-base: 768/12/12/3072, pre-norm, gelu, patch16. Output: artifacts/vit-base/."""
import os, glob, numpy as np
from safetensors import safe_open
SRC=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--vit-base-patch16-224/snapshots/*/model.safetensors"))[0]
OUT="artifacts/vit-base"; NL=12
f=safe_open(SRC,framework="np")
def g(k): return f.get_tensor(k).astype(np.float32)
def save(p,a): os.makedirs(os.path.dirname(p),exist_ok=True); np.save(p,np.ascontiguousarray(a))
# patch proj: Conv2d weight [768,3,16,16] -> [768, 768] (out, c*h*w) -> transpose [K=768,N=768]
pw=g("vit.embeddings.patch_embeddings.projection.weight").reshape(768,-1)  # [768, 768]
save(f"{OUT}/patch_proj.weight.npy", pw.T.copy())   # [K,N]
save(f"{OUT}/patch_proj.bias.npy", g("vit.embeddings.patch_embeddings.projection.bias"))
save(f"{OUT}/cls_token.npy", g("vit.embeddings.cls_token").reshape(768))
save(f"{OUT}/pos_emb.npy", g("vit.embeddings.position_embeddings").reshape(197,768))
save(f"{OUT}/ln_final.weight.npy", g("vit.layernorm.weight"))
save(f"{OUT}/ln_final.bias.npy", g("vit.layernorm.bias"))
save(f"{OUT}/classifier.weight.npy", g("classifier.weight").T.copy())  # [768,1000]
save(f"{OUT}/classifier.bias.npy", g("classifier.bias"))
for i in range(NL):
    p=f"vit.encoder.layer.{i}."; d=f"{OUT}/L{i}"
    for s,t in [("attention.attention.query","q"),("attention.attention.key","k"),("attention.attention.value","v")]:
        save(f"{d}/{t}.weight.npy", g(p+s+".weight").T.copy()); save(f"{d}/{t}.bias.npy", g(p+s+".bias"))
    save(f"{d}/attn_out.weight.npy", g(p+"attention.output.dense.weight").T.copy()); save(f"{d}/attn_out.bias.npy", g(p+"attention.output.dense.bias"))
    save(f"{d}/ln_before.weight.npy", g(p+"layernorm_before.weight")); save(f"{d}/ln_before.bias.npy", g(p+"layernorm_before.bias"))
    save(f"{d}/inter.weight.npy", g(p+"intermediate.dense.weight").T.copy()); save(f"{d}/inter.bias.npy", g(p+"intermediate.dense.bias"))
    save(f"{d}/out.weight.npy", g(p+"output.dense.weight").T.copy()); save(f"{d}/out.bias.npy", g(p+"output.dense.bias"))
    save(f"{d}/ln_after.weight.npy", g(p+"layernorm_after.weight")); save(f"{d}/ln_after.bias.npy", g(p+"layernorm_after.bias"))
print(f"converted vit-base -> {OUT}/")
