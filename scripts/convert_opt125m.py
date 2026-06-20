#!/usr/bin/env python3
"""Convert facebook/opt-125m (torch pickle) to the engine's per-tensor .npy layout (f32, [K,N] linears).

opt-125m is dimension-identical to whisper-small (768/12/12/3072/64), decoder-only, relu FFN, learned
positions (offset +2). Output: artifacts/opt-125m/{embed_tokens,embed_positions,ln_final.{weight,bias},
lm_head.weight}.npy + L0..L11/{q,k,v,out}.{weight,bias}, ln_self.{w,b}, fc1/fc2.{weight,bias}, ln_ffn.{w,b}.
Linear weights are TRANSPOSED torch[out,in] -> engine[in,out]; embeddings/LN/bias kept as-is.
"""
import os, glob, torch, numpy as np

SRC = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/*/pytorch_model.bin"))[0]
OUT = "artifacts/opt-125m"
NL = 12
sd = torch.load(SRC, map_location="cpu", weights_only=True)
def f32(t): return t.detach().to(torch.float32).numpy()
def save(path, arr): os.makedirs(os.path.dirname(path), exist_ok=True); np.save(path, np.ascontiguousarray(arr))

P = "model.decoder."
# top-level
save(f"{OUT}/embed_tokens.npy", f32(sd[P+"embed_tokens.weight"]))          # [vocab, d]
save(f"{OUT}/embed_positions.npy", f32(sd[P+"embed_positions.weight"]))    # [2050, d] (offset +2)
save(f"{OUT}/ln_final.weight.npy", f32(sd[P+"final_layer_norm.weight"]))
save(f"{OUT}/ln_final.bias.npy",   f32(sd[P+"final_layer_norm.bias"]))
save(f"{OUT}/lm_head.weight.npy",  f32(sd["lm_head.weight"]).T.copy())     # [out,in]->[in,out]=[d,vocab]

amap = {"q_proj":"q","k_proj":"k","v_proj":"v","out_proj":"out"}
for i in range(NL):
    lp = f"{P}layers.{i}."
    d = f"{OUT}/L{i}"
    for src, dst in amap.items():
        save(f"{d}/{dst}.weight.npy", f32(sd[lp+f"self_attn.{src}.weight"]).T.copy())  # [in,out]
        save(f"{d}/{dst}.bias.npy",   f32(sd[lp+f"self_attn.{src}.bias"]))
    save(f"{d}/ln_self.weight.npy", f32(sd[lp+"self_attn_layer_norm.weight"]))
    save(f"{d}/ln_self.bias.npy",   f32(sd[lp+"self_attn_layer_norm.bias"]))
    save(f"{d}/fc1.weight.npy", f32(sd[lp+"fc1.weight"]).T.copy())  # [3072,768]->[768,3072]
    save(f"{d}/fc1.bias.npy",   f32(sd[lp+"fc1.bias"]))
    save(f"{d}/fc2.weight.npy", f32(sd[lp+"fc2.weight"]).T.copy())  # [768,3072]->[3072,768]
    save(f"{d}/fc2.bias.npy",   f32(sd[lp+"fc2.bias"]))
    save(f"{d}/ln_ffn.weight.npy", f32(sd[lp+"final_layer_norm.weight"]))  # OPT's per-layer "final_layer_norm" = pre-FFN LN
    save(f"{d}/ln_ffn.bias.npy",   f32(sd[lp+"final_layer_norm.bias"]))
print(f"converted opt-125m -> {OUT}/ (L0..L{NL-1} + top-level), linears transposed to [K,N]")
