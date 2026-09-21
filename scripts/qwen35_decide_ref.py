#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SemIf's direct typed-decision readout on the Qwen3.5 f32 reference, for JevBench tasks.

SemIf (github.com/TheoLeeCJ/SemIf, semif_phase1/core.py + direct.py) prompts the frozen instruct
checkpoint with a fixed system turn and a JSON user turn naming the options by letter, then softmaxes
the last position's logits over those letter tokens only. JevBench scores it through its
`semif_direct` adapter, one question per forward. This is that path, so the NPU `decide` capability
has a reference to be gated against.

  python scripts/qwen35_decide_ref.py --tasks original.jsonl --limit 8 --out decide_ref.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_ref import Ckpt, Model  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")


def options_for(q):
    """JevBench's semif_direct.build_request: noul is true/false, score levels are indexed."""
    crit = q.get("criteria")
    if q["type"] == "noul":
        opts = [(k, (crit or {}).get(k, f"The proposition is {k}.")) for k in ("true", "false")]
    elif q["type"] == "choice":
        opts = [(k, v or k) for k, v in crit.items()]
    else:
        opts = [(str(i), lvl) for i, lvl in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in opts]


def prompt_ids(tok, task):
    opts = options_for(task["question"])
    payload = {"evidence": task["state"], "criterion": task["question"]["instructions"],
               "options": [{"letter": LETTERS[i], "description": d} for i, (_, d) in enumerate(opts)]}
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    slots = []
    for letter in LETTERS[:len(opts)]:
        (t,) = tok.encode(letter, add_special_tokens=False)
        if tok.encode(text + letter, add_special_tokens=False) != ids + [t]:
            raise ValueError(f"{task['id']}: slot {letter} changes tokenization at the boundary")
        slots.append(t)
    return ids, slots, [k for k, _ in opts], text


def answer_label(qtype, keys, probs):
    """Map the argmax option back to JevBench's label space (noul answers are yes/no)."""
    k = keys[int(torch.argmax(probs))]
    return {"true": "yes", "false": "no"}.get(k, k) if qtype == "noul" else k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/data/xdna/artifacts/qwen3.5-4b/hf")
    ap.add_argument("--tasks", required=True, help="JevBench jsonl (datasets/public/original.jsonl)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--int4-group", type=int, default=0,
                    help="run the projections through the int4 packer at this group size")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    ck = Ckpt(a.ckpt, a.int4_group)
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    tasks = tasks[:a.limit] if a.limit else tasks
    out = open(a.out, "w") if a.out else None
    enc = [prompt_ids(tok, t) for t in tasks]
    t0 = time.perf_counter()
    m = Model(ck)
    hidden = m.forward_many([ids for ids, _, _, _ in enc])
    per_task = (time.perf_counter() - t0) / len(tasks)
    right = 0
    for task, (ids, slots, keys, _), x in zip(tasks, enc, hidden):
        head = torch.stack([ck.rows("embed_tokens.weight", s, s + 1)[0] for s in slots])
        logits = (m.final(x[-1:]) @ head.T)[0]
        probs = torch.softmax(logits, -1)
        got = answer_label(task["question"]["type"], keys, probs)
        ok = got == task.get("expected")
        right += ok
        rec = {"id": task["id"], "type": task["question"]["type"], "n_tokens": len(ids),
               "option_ids": keys, "option_logits": logits.tolist(), "probabilities": probs.tolist(),
               "answer": got, "expected": task.get("expected"), "correct": bool(ok),
               "int4_group": a.int4_group, "seconds": round(per_task, 1)}
        print(json.dumps(rec), flush=True)
        if out:
            out.write(json.dumps(rec) + "\n")
    print(f"[decide-ref] {right}/{len(tasks)} match the JevBench label", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
