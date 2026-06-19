#!/usr/bin/env python3
"""Score wer_m1_decode `path<TAB>text` output against refs.json.

Uses the SAME normalize/wer/pooled methodology as whisper_decode_wer.py so numbers
are directly comparable to the canonical M=1 gate (baseline 0.1172).

Usage: _score_m1_wer.py <label> <hyp.tsv>   (hyp.tsv = path<TAB>text lines)
"""
import json, re, sys, unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFS = json.load(open(REPO / "artifacts" / "wer_clips" / "refs.json"))
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(t):
    t = unicodedata.normalize("NFC", t or "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return (0.0 if not h else 1.0), 0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if rw == hw else 1)))
        prev = cur
    return prev[-1] / len(r), len(r)


label, tsv = sys.argv[1], sys.argv[2]
items = []  # (name, edits, nref)
for line in open(tsv):
    line = line.rstrip("\n")
    if "\t" not in line:
        continue
    path, hyp = line.split("\t", 1)
    name = Path(path).name
    ref = REFS.get(name)
    if ref is None:
        continue
    w, n = wer(normalize(ref), normalize(hyp))
    edits = round(w * n)
    items.append((name, edits, n, hyp))


def pooled(sub):
    e = sum(x[1] for x in sub)
    n = sum(x[2] for x in sub)
    return e / n if n else 0.0


print(f"=== {label} ===")
for name, edits, n, hyp in sorted(items):
    print(f"  {name:12} edits={edits:3} nref={n:3}  hyp={hyp[:70]!r}")
for split in ("en", "ru"):
    sub = [x for x in items if x[0].startswith(split)]
    if sub:
        macro = sum(x[1] / x[2] for x in sub) / len(sub)
        print(f"{split} pooled WER = {pooled(sub):.4f}  (macro {macro:.4f}, n={len(sub)})")
allv = items
macro = sum(x[1] / x[2] for x in allv) / len(allv)
print(f"ALL pooled WER = {pooled(allv):.4f}  (macro {macro:.4f}, n={len(allv)})")
