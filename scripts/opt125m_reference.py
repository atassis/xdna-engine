#!/usr/bin/env python3
"""HF opt-125m greedy-decode reference (golden) for validating the engine's OPT backend.
Saves prompt+generated token ids and text to artifacts/opt-125m/ref_generation.json.
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32).eval()
prompts = ["The capital of France is", "Once upon a time"]
out = []
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=20, do_sample=False, num_beams=1)
    new = gen[0, ids.shape[1]:].tolist()
    out.append({"prompt": p, "prompt_ids": ids[0].tolist(),
                "gen_ids": new, "gen_text": tok.decode(new),
                "full_text": tok.decode(gen[0])})
    print(f"PROMPT: {p!r}\n  -> {tok.decode(new)!r}\n  gen_ids={new}")
json.dump(out, open("artifacts/opt-125m/ref_generation.json", "w"), indent=2)
print("saved artifacts/opt-125m/ref_generation.json")
