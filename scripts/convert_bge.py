#!/usr/bin/env python3
"""Convert BAAI/bge-base-en-v1.5 (safetensors) to engine per-tensor .npy ([K,N] linears).
Same core dims as whisper (768/12/12/3072), BERT post-norm. Output: artifacts/bge-base/."""
import os, glob, numpy as np
from safetensors import safe_open
SRC = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/*/model.safetensors"))[0]
OUT = "artifacts/bge-base"; NL = 12
f = safe_open(SRC, framework="np")
def g(k): return f.get_tensor(k).astype(np.float32)
def save(p, a): os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, np.ascontiguousarray(a))
E = "embeddings."
save(f"{OUT}/word_emb.npy", g(E+"word_embeddings.weight"))
save(f"{OUT}/pos_emb.npy", g(E+"position_embeddings.weight"))
save(f"{OUT}/tok_type_emb.npy", g(E+"token_type_embeddings.weight"))
save(f"{OUT}/emb_ln.weight.npy", g(E+"LayerNorm.weight"))
save(f"{OUT}/emb_ln.bias.npy", g(E+"LayerNorm.bias"))
for i in range(NL):
    p = f"encoder.layer.{i}."; d = f"{OUT}/L{i}"
    for src, dst in [("attention.self.query","q"),("attention.self.key","k"),("attention.self.value","v")]:
        save(f"{d}/{dst}.weight.npy", g(p+src+".weight").T.copy()); save(f"{d}/{dst}.bias.npy", g(p+src+".bias"))
    save(f"{d}/attn_out.weight.npy", g(p+"attention.output.dense.weight").T.copy())
    save(f"{d}/attn_out.bias.npy", g(p+"attention.output.dense.bias"))
    save(f"{d}/attn_ln.weight.npy", g(p+"attention.output.LayerNorm.weight"))
    save(f"{d}/attn_ln.bias.npy", g(p+"attention.output.LayerNorm.bias"))
    save(f"{d}/inter.weight.npy", g(p+"intermediate.dense.weight").T.copy())  # [768,3072]
    save(f"{d}/inter.bias.npy", g(p+"intermediate.dense.bias"))
    save(f"{d}/out.weight.npy", g(p+"output.dense.weight").T.copy())  # [3072,768]
    save(f"{d}/out.bias.npy", g(p+"output.dense.bias"))
    save(f"{d}/out_ln.weight.npy", g(p+"output.LayerNorm.weight"))
    save(f"{d}/out_ln.bias.npy", g(p+"output.LayerNorm.bias"))
print(f"converted bge-base -> {OUT}/ (L0..L{NL-1} + embeddings)")
