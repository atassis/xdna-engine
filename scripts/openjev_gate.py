#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NPU vs CPU-oracle argmax gate for an NLI-head model. Exit 1 on any mismatch.

  python scripts/openjev_gate.py --url http://127.0.0.1:11435 --model openjev-4b \
      --pairs nli_1000.jsonl --oracle ref_pairs.jsonl
  python scripts/openjev_gate.py --url ... --model openjev-4b --tasks hard.jsonl --oracle ref_hard.jsonl
"""
import argparse
import json
import sys
import urllib.request


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs")
    ap.add_argument("--tasks")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    ref = {r["id"]: r for r in map(json.loads, open(a.oracle)) if r}
    rows = [json.loads(l) for l in open(a.pairs or a.tasks) if l.strip() and json.loads(l)["id"] in ref]
    bad, rel = [], []
    out = open(a.out, "w") if a.out else None
    for r in rows:
        o = ref[r["id"]]
        if a.pairs:
            v = post(f"{a.url}/predict", {"model": a.model, "inputs": [r["premise"], r["hypothesis"]]})
            got = v[0]["label"]
            want = o["argmax"]
            npu = {x["label"]: x["score"] for x in v}
            cpu = dict(zip(o["labels"], o["probs"]))
            rel.append(max(abs(npu[l] - cpu[l]) for l in cpu))
        else:
            q = r["question"]
            v = post(f"{a.url}/v1/systemone", {"model": a.model, "state": r["state"], "questions": {"q": q}})
            ans = v["answers"]["q"]
            got = ("yes" if ans["noul"] >= 0.5 else "no") if ans["type"] == "noul" else \
                max(ans["probabilities"], key=ans["probabilities"].get)
            want = o["answer"]
        if str(got) != str(want):
            bad.append((r["id"], got, want))
        if out:
            out.write(json.dumps({"id": r["id"], "npu": got, "oracle": want}) + "\n")
    print(f"[gate] {len(rows) - len(bad)}/{len(rows)} argmax equal to the oracle"
          + (f"; max |dp| {max(rel):.3e}" if rel else ""))
    for b in bad[:20]:
        print("  mismatch", *b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
