#!/usr/bin/env python3
"""HF bge-base reference: CLS-pooled + L2-normalized embeddings for fixed sentences (golden)."""
import json, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
M="BAAI/bge-base-en-v1.5"
tok=AutoTokenizer.from_pretrained(M); model=AutoModel.from_pretrained(M).eval()
sents=["The capital of France is Paris.","A cat sat on the mat."]
out=[]
for s in sents:
    enc=tok(s, return_tensors="pt")
    with torch.no_grad(): h=model(**enc).last_hidden_state[0,0]  # CLS
    e=torch.nn.functional.normalize(h,dim=0).numpy()
    out.append({"sent":s,"ids":enc["input_ids"][0].tolist(),"emb":e.tolist()})
    print(f"{s!r}: ids={enc['input_ids'][0].tolist()[:8]}... emb[:5]={e[:5].round(4).tolist()}")
json.dump(out, open("artifacts/bge-base/ref_embeddings.json","w"))
print("saved artifacts/bge-base/ref_embeddings.json")
