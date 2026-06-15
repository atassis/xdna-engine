#!/usr/bin/env python3
"""Export an ESM-2 model to the engine's artifact layout + an ONNX oracle + a golden fixture.
Usage: python3 scripts/export_esm.py <hf_id> <out_subdir>
  e.g. python3 scripts/export_esm.py facebook/esm2_t6_8M_UR50D esm2-8m
Outputs under artifacts/<out_subdir>/:
  encoder/emb/word_emb.npy
  encoder/Li/{ln_attn_w,ln_attn_b, q_w,q_b,k_w,k_b,v_w,v_b, attn_out_w,attn_out_b,
              ln_ffn_w,ln_ffn_b, ffn1_w,ffn1_b, ffn2_w,ffn2_b}.npy   (Linear w TRANSPOSED [in,out])
  encoder/final_ln_{w,b}.npy
  model.onnx (+ .data)        accuracy oracle
  golden.json                 {"seq":..., "ids":[...], "mean_emb":[...H floats pooled+L2...]}
"""
import os, sys, json, numpy as np, torch
from transformers import AutoModel, AutoTokenizer

HF = sys.argv[1]; SUB = sys.argv[2]
OUT = os.path.join("artifacts", SUB); ENC = os.path.join(OUT, "encoder")
def save(p, a):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.ascontiguousarray(a.astype(np.float32)))

def main():
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModel.from_pretrained(HF).eval()
    sd = model.state_dict(); g = lambda k: sd[k].cpu().numpy()
    save(f"{ENC}/emb/word_emb.npy", g("embeddings.word_embeddings.weight"))
    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        p = f"encoder.layer.{i}."; L = f"{ENC}/L{i}"
        save(f"{L}/ln_attn_w.npy", g(p+"attention.LayerNorm.weight"))
        save(f"{L}/ln_attn_b.npy", g(p+"attention.LayerNorm.bias"))
        save(f"{L}/q_w.npy", g(p+"attention.self.query.weight").T); save(f"{L}/q_b.npy", g(p+"attention.self.query.bias"))
        save(f"{L}/k_w.npy", g(p+"attention.self.key.weight").T);   save(f"{L}/k_b.npy", g(p+"attention.self.key.bias"))
        save(f"{L}/v_w.npy", g(p+"attention.self.value.weight").T); save(f"{L}/v_b.npy", g(p+"attention.self.value.bias"))
        save(f"{L}/attn_out_w.npy", g(p+"attention.output.dense.weight").T); save(f"{L}/attn_out_b.npy", g(p+"attention.output.dense.bias"))
        save(f"{L}/ln_ffn_w.npy", g(p+"LayerNorm.weight")); save(f"{L}/ln_ffn_b.npy", g(p+"LayerNorm.bias"))
        save(f"{L}/ffn1_w.npy", g(p+"intermediate.dense.weight").T); save(f"{L}/ffn1_b.npy", g(p+"intermediate.dense.bias"))
        save(f"{L}/ffn2_w.npy", g(p+"output.dense.weight").T);       save(f"{L}/ffn2_b.npy", g(p+"output.dense.bias"))
    save(f"{ENC}/final_ln_w.npy", g("encoder.emb_layer_norm_after.weight"))
    save(f"{ENC}/final_ln_b.npy", g("encoder.emb_layer_norm_after.bias"))
    # ONNX oracle
    seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    enc = tok(seq, return_tensors="pt")
    torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]),
        os.path.join(OUT, "model.onnx"), input_names=["input_ids","attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids":{0:"b",1:"s"},"attention_mask":{0:"b",1:"s"},"last_hidden_state":{0:"b",1:"s"}},
        opset_version=17)
    with torch.no_grad():
        hs = model(**enc).last_hidden_state[0]          # [seq, H]
        mean = hs.mean(0); mean = mean / mean.norm()    # mean-pool + L2
    json.dump({"seq":seq, "ids":enc["input_ids"][0].tolist(), "mean_emb":mean.tolist()},
              open(os.path.join(OUT,"golden.json"),"w"))
    print(f"exported {n_layers} layers + onnx + golden to {OUT}")

if __name__ == "__main__": main()
