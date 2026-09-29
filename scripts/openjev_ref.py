#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""openjev's NLI readout on the Qwen3.5 reference, for pairs and for JevBench typed decisions.

The NPU's `/predict`, `/rerank` and `/v1/systemone` on an NLI-head model are gated against this:
same prompts, hypotheses and windows (scripts/nli_common.py), same S, the backbone's projections
through the int4 packer when --int4-group is set.

  python scripts/openjev_ref.py --ckpt <dir> --head <nli_head dir> --pairs pairs.jsonl --out ref.jsonl
  python scripts/openjev_ref.py --ckpt <dir> --head <nli_head dir> --tasks hard.jsonl --out ref.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nli_common as nc  # noqa: E402
from qwen35_ref import Ckpt, Model  # noqa: E402


class Head:
    def __init__(self, d):
        h = json.load(open(os.path.join(d, "head.json")))
        raw = np.fromfile(os.path.join(d, "score.bin"), dtype=np.uint16).astype(np.uint32) << 16
        self.w = torch.from_numpy(raw.view(np.float32).reshape(len(h["labels"]), h["d_model"]).copy())
        self.labels, self.template = h["labels"], h["template"]
        self.ent = self.labels.index("entailment")


def score(m, head, tok, prompts):
    ids = [tok.encode(p) for p in prompts]
    hs = m.forward_many(ids)
    out = []
    for x in hs:
        logits = (m.final(x[-1:]).double() @ head.w.double().T)[0]
        out.append(torch.softmax(logits, -1).tolist())
    return ids, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--pairs", help="jsonl of {id, premise, hypothesis[, label]}")
    ap.add_argument("--tasks", help="JevBench jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--int4-group", type=int, default=0)
    ap.add_argument("--clip-search", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    head = Head(a.head)
    ck = Ckpt(a.ckpt, a.int4_group, a.clip_search)
    n_tok = lambda s: len(tok.encode(s))
    out = open(a.out, "w")
    rows = [json.loads(l) for l in open(a.pairs or a.tasks) if l.strip()]
    rows = rows[:a.limit] if a.limit else rows
    for r in rows:
        if a.pairs:
            wins = nc.fit_windows(head.template, r["premise"], [r["hypothesis"]], n_tok, a.max_seq)
            _, probs = score(Model(ck), head, tok, [nc.prompt(head.template, w, r["hypothesis"]) for w in wins])
            best = max(probs, key=lambda p: p[head.ent])
            rec = {"id": r["id"], "labels": head.labels, "probs": best, "argmax": head.labels[int(np.argmax(best))],
                   "windows": len(wins), "label": r.get("label")}
        else:
            q = r["question"]
            state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
            hyps = nc.decide_hypotheses(q)
            wins = nc.fit_windows(head.template, state, [h for _, h in hyps], n_tok, a.max_seq)
            prompts = [nc.prompt(head.template, w, h) for w in wins for _, h in hyps]
            _, probs = score(Model(ck), head, tok, prompts)
            p_ent = [[probs[wi * len(hyps) + oi][head.ent] for oi in range(len(hyps))] for wi in range(len(wins))]
            p = nc.normalise_entailment(p_ent)
            keys = [k for k, _ in hyps]
            got = keys[int(np.argmax(p))]
            rec = {"id": r["id"], "type": q["type"], "keys": keys, "probabilities": p, "answer": got,
                   "expected": r.get("expected"), "correct": str(got) == str(r.get("expected")),
                   "windows": len(wins)}
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
