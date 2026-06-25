#!/usr/bin/env python3
"""Export a sentence-transformers BERT-family encoder to the engine's artifact layout.

Usage: python3 scripts/export_minilm.py <hf_id> <out_subdir>
  e.g. python3 scripts/export_minilm.py sentence-transformers/all-MiniLM-L6-v2 minilm-l6
       python3 scripts/export_minilm.py BAAI/bge-small-en-v1.5 bge-small
       python3 scripts/export_minilm.py intfloat/e5-small-v2 e5-small

These are plain BERT backbones (MiniLM is a 6-layer BERT distillate; bge-small/e5-small are
12-layer BERT). The pooling + L2-normalize that sentence-transformers applies on top are NOT
weights - they are runtime ops, so this script bakes only the BERT encoder, identical in layout
to export_bge.py (which the `bert` arch already mirrors).

Outputs under artifacts/<out_subdir>/:
  encoder/emb/{word_emb,pos_emb,type_emb,emb_ln_w,emb_ln_b}.npy
  encoder/L{i}/{q_w,q_b,k_w,k_b,v_w,v_b,attn_out_w,attn_out_b,attn_ln_w,attn_ln_b,
                ffn1_w,ffn1_b,ffn2_w,ffn2_b,out_ln_w,out_ln_b}.npy
  model.onnx        (accuracy oracle)
  tokenizer.json    (HF fast tokenizer)
Linear weights are TRANSPOSED to [in, out] (the x@W B-operand form the Rust engine expects).
"""
import os, sys, numpy as np, torch
from transformers import AutoModel, AutoTokenizer

HF = sys.argv[1]
SUB = sys.argv[2]
OUT = os.path.join("artifacts", SUB)
ENC = os.path.join(OUT, "encoder")


def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(arr.astype(np.float32)))


def main():
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModel.from_pretrained(HF).eval()
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
        save(f"{L}/ffn1_w.npy", g(p+"intermediate.dense.weight").T)
        save(f"{L}/ffn1_b.npy", g(p+"intermediate.dense.bias"))
        save(f"{L}/ffn2_w.npy", g(p+"output.dense.weight").T)
        save(f"{L}/ffn2_b.npy", g(p+"output.dense.bias"))
        save(f"{L}/out_ln_w.npy", g(p+"output.LayerNorm.weight"))
        save(f"{L}/out_ln_b.npy", g(p+"output.LayerNorm.bias"))

    tok.save_pretrained(OUT)            # writes tokenizer.json

    # ONNX oracle (opset 17, dynamic seq). Some multilingual backbones (e.g. multilingual-e5,
    # XLM-R tokenizer) omit token_type_ids; BertModel defaults them to zeros, so synthesize that.
    dummy = tok("hello world", return_tensors="pt")
    if "token_type_ids" not in dummy:
        dummy["token_type_ids"] = torch.zeros_like(dummy["input_ids"])
    inputs = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])
    torch.onnx.export(
        model, inputs,
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
