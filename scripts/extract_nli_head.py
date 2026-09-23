#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a sequence-classification checkpoint's head as the NPU's host-side NLI readout.

  python scripts/extract_nli_head.py --checkpoint-dir <hf dir> --out artifacts/openjev-4b/nli_head

`score.bin` is the `score.weight` rows as raw little-endian bf16, [labels, d_model]; `head.json`
carries the labels in id order and the checkpoint's own prompt template. The readout is taken
from `architectures`, so a causal LM pointed here is refused rather than given a head it lacks.
"""
import argparse
import json
import os
import sys

from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.checkpoint_dir, "config.json")))
    arch = (cfg.get("architectures") or [""])[0]
    if not arch.endswith("ForSequenceClassification"):
        sys.exit(f"{arch!r} is not a *ForSequenceClassification checkpoint")
    d = (cfg.get("text_config") or cfg)["hidden_size"]
    labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
    if "entailment" not in labels:
        sys.exit(f"labels {labels} carry no 'entailment'")
    shards = [f for f in sorted(os.listdir(a.checkpoint_dir)) if f.endswith(".safetensors")]
    w = None
    for s in shards:
        with safe_open(os.path.join(a.checkpoint_dir, s), framework="pt") as f:
            if "score.weight" in f.keys():
                w = f.get_tensor("score.weight")
    if w is None or tuple(w.shape) != (len(labels), d):
        sys.exit(f"score.weight is {None if w is None else tuple(w.shape)}, want {(len(labels), d)}")
    os.makedirs(a.out, exist_ok=True)
    import torch
    w.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tofile(os.path.join(a.out, "score.bin"))
    json.dump({"architecture": arch, "labels": labels, "d_model": d,
               "template": cfg.get("nli_template") or "Premise: {premise}\nHypothesis: {hypothesis}"},
              open(os.path.join(a.out, "head.json"), "w"), indent=1)
    print(f"{a.out}: {len(labels)} x {d} bf16, labels {labels}")


if __name__ == "__main__":
    main()
