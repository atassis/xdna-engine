#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The fixed NLI gate set: the first 500 MNLI validation_mismatched and 500 ANLI test_r3 pairs.

  python scripts/nli_eval_set.py --out /mnt/data/xdna/scratch/openjev/nli_1000.jsonl
"""
import argparse
import json

from datasets import load_dataset

LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}  # both datasets' own order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for name, cfg, split, tag in (("nyu-mll/multi_nli", None, "validation_mismatched", "mnli-mm"),
                                      ("facebook/anli", None, "test_r3", "anli-r3")):
            ds = load_dataset(name, cfg, split=split)
            kept = 0
            for i, r in enumerate(ds):
                if r["label"] not in LABELS:
                    continue
                f.write(json.dumps({"id": f"{tag}-{i}", "premise": r["premise"], "hypothesis": r["hypothesis"],
                                    "label": LABELS[r["label"]]}) + "\n")
                kept += 1
                if kept == a.n:
                    break


if __name__ == "__main__":
    main()
