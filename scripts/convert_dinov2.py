#!/usr/bin/env python3
"""Convert facebook/dinov2-base (safetensors) to engine per-tensor .npy ([K,N] linears).

DINOv2-base: 768/12/12/3072, pre-norm, gelu, patch14, image 518 -> 1369 patches (+1 cls = 1370 pos).
Differs from vanilla ViT (scripts/convert_vit.py): no `vit.` prefix; per-block LayerScale
(layer_scale1/2.lambda1, baked verbatim f32); norm1/norm2 (not layernorm_before/after); mlp.fc1/fc2
(not intermediate/output.dense); a learnable mask_token; and NO classifier head (this is the backbone
Dinov2Model). The patch-embed Conv2d is im2col-flattened + transposed into a GEMM weight exactly as
vit does. Layer count is inferred from the checkpoint (small=12-dim*6, base=768*12, large=1024*24).
Output: artifacts/dinov2-base/."""
import os, glob, re, numpy as np
from safetensors import safe_open
SRC = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/*/model.safetensors"))[0]
OUT = "artifacts/dinov2-base"
f = safe_open(SRC, framework="np")
keys = list(f.keys())
NL = 1 + max(int(m.group(1)) for k in keys
             for m in [re.match(r"encoder\.layer\.(\d+)\.", k)] if m)
def g(k): return f.get_tensor(k).astype(np.float32)
def save(p, a): os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, np.ascontiguousarray(a))
# patch proj: Conv2d weight [768,3,14,14] -> [768, 3*14*14] (out, c*h*w) -> transpose [K,N]
pw = g("embeddings.patch_embeddings.projection.weight")
out_c = pw.shape[0]
save(f"{OUT}/patch_proj.weight.npy", pw.reshape(out_c, -1).T.copy())   # [K,N]
save(f"{OUT}/patch_proj.bias.npy", g("embeddings.patch_embeddings.projection.bias"))
save(f"{OUT}/cls_token.npy", g("embeddings.cls_token").reshape(-1))
save(f"{OUT}/mask_token.npy", g("embeddings.mask_token").reshape(-1))
pe = g("embeddings.position_embeddings")
save(f"{OUT}/pos_emb.npy", pe.reshape(pe.shape[-2], pe.shape[-1]))
save(f"{OUT}/ln_final.weight.npy", g("layernorm.weight"))
save(f"{OUT}/ln_final.bias.npy", g("layernorm.bias"))
for i in range(NL):
    p = f"encoder.layer.{i}."; d = f"{OUT}/L{i}"
    for s, t in [("attention.attention.query", "q"), ("attention.attention.key", "k"),
                 ("attention.attention.value", "v")]:
        save(f"{d}/{t}.weight.npy", g(p+s+".weight").T.copy()); save(f"{d}/{t}.bias.npy", g(p+s+".bias"))
    save(f"{d}/attn_out.weight.npy", g(p+"attention.output.dense.weight").T.copy())
    save(f"{d}/attn_out.bias.npy", g(p+"attention.output.dense.bias"))
    save(f"{d}/norm1.weight.npy", g(p+"norm1.weight")); save(f"{d}/norm1.bias.npy", g(p+"norm1.bias"))
    save(f"{d}/norm2.weight.npy", g(p+"norm2.weight")); save(f"{d}/norm2.bias.npy", g(p+"norm2.bias"))
    save(f"{d}/ls1.npy", g(p+"layer_scale1.lambda1")); save(f"{d}/ls2.npy", g(p+"layer_scale2.lambda1"))
    save(f"{d}/fc1.weight.npy", g(p+"mlp.fc1.weight").T.copy()); save(f"{d}/fc1.bias.npy", g(p+"mlp.fc1.bias"))
    save(f"{d}/fc2.weight.npy", g(p+"mlp.fc2.weight").T.copy()); save(f"{d}/fc2.bias.npy", g(p+"mlp.fc2.bias"))
print(f"converted dinov2-base ({NL} layers) -> {OUT}/")
