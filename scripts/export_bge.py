#!/usr/bin/env python3
"""Export bge-base-en-v1.5 to the engine's artifact layout.

Outputs under artifacts/bge-base/:
  encoder/emb/{word_emb,pos_emb,type_emb,emb_ln_w,emb_ln_b}.npy
  encoder/L{i}/{q_w,q_b,k_w,k_b,v_w,v_b,attn_out_w,attn_out_b,attn_ln_w,attn_ln_b,
                ffn1_w,ffn1_b,ffn2_w,ffn2_b,out_ln_w,out_ln_b}.npy
  model.onnx        (accuracy oracle)
  tokenizer.json    (HF fast tokenizer)
Linear weights are TRANSPOSED to [in, out] (the x@W B-operand form the Rust engine expects).
"""
import os, numpy as np, torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-base-en-v1.5"
OUT = os.path.join("artifacts", "bge-base")
ENC = os.path.join(OUT, "encoder")

def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(arr.astype(np.float32)))

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    sd = model.state_dict()
    g = lambda k: sd[k].cpu().numpy()

    # embeddings
    save(f"{ENC}/emb/word_emb.npy", g("embeddings.word_embeddings.weight"))
    save(f"{ENC}/emb/pos_emb.npy",  g("embeddings.position_embeddings.weight"))
    save(f"{ENC}/emb/type_emb.npy", g("embeddings.token_type_embeddings.weight"))
    save(f"{ENC}/emb/emb_ln_w.npy", g("embeddings.LayerNorm.weight"))
    save(f"{ENC}/emb/emb_ln_b.npy", g("embeddings.LayerNorm.bias"))

    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        p = f"encoder.layer.{i}."
        L = f"{ENC}/L{i}"
        # Linear weight in torch is [out, in]; transpose to [in, out].
        save(f"{L}/q_w.npy", g(p+"attention.self.query.weight").T)
        save(f"{L}/q_b.npy", g(p+"attention.self.query.bias"))
        save(f"{L}/k_w.npy", g(p+"attention.self.key.weight").T)
        save(f"{L}/k_b.npy", g(p+"attention.self.key.bias"))
        save(f"{L}/v_w.npy", g(p+"attention.self.value.weight").T)
        save(f"{L}/v_b.npy", g(p+"attention.self.value.bias"))
        save(f"{L}/attn_out_w.npy", g(p+"attention.output.dense.weight").T)
        save(f"{L}/attn_out_b.npy", g(p+"attention.output.dense.bias"))
        save(f"{L}/attn_ln_w.npy", g(p+"attention.output.LayerNorm.weight"))
        save(f"{L}/attn_ln_b.npy", g(p+"attention.output.LayerNorm.bias"))
        save(f"{L}/ffn1_w.npy", g(p+"intermediate.dense.weight").T)  # [768,3072]
        save(f"{L}/ffn1_b.npy", g(p+"intermediate.dense.bias"))
        save(f"{L}/ffn2_w.npy", g(p+"output.dense.weight").T)        # [3072,768]
        save(f"{L}/ffn2_b.npy", g(p+"output.dense.bias"))
        save(f"{L}/out_ln_w.npy", g(p+"output.LayerNorm.weight"))
        save(f"{L}/out_ln_b.npy", g(p+"output.LayerNorm.bias"))

    tok.save_pretrained(OUT)            # writes tokenizer.json

    # ONNX oracle (opset 17, dynamic seq)
    dummy = tok("hello world", return_tensors="pt")
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        os.path.join(OUT, "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "token_type_ids": {0: "b", 1: "s"}, "last_hidden_state": {0: "b", 1: "s"}},
        opset_version=17,
    )
    print(f"exported {n_layers} layers + onnx + tokenizer to {OUT}")

if __name__ == "__main__":
    main()
