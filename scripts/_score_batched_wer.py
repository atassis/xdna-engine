#!/usr/bin/env python3
"""Score batched-decode transcripts vs refs.json (normalized word-level WER).

Reads a TSV of `path<TAB>hypothesis` lines (verify_batched_decode stdout) and the refs.json
(file -> ground-truth) map, normalizes both (lowercase, strip punct, collapse ws, NFC), computes
per-clip WER + the aggregate per-stream WER (total edits / total ref words), and gates against the
batched-decode threshold 0.1172 ([[batched-decode-engine-wer]]).

    python3 scripts/_score_batched_wer.py <transcripts.tsv> [refs.json] [--gate 0.1172]
exit 0 if aggregate WER <= gate, else 1.
"""
import json
import os
import re
import sys
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text):
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def edits(ref_words, hyp_words):
    """Word-level Levenshtein edit distance (integer)."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gate = 0.1172
    for a in sys.argv[1:]:
        if a.startswith("--gate"):
            gate = float(a.split("=", 1)[1]) if "=" in a else gate
    tsv = args[0]
    refs_path = args[1] if len(args) > 1 else os.path.join(os.path.dirname(tsv), "refs.json")
    if not os.path.isfile(refs_path):
        refs_path = "artifacts/wer_clips/refs.json"
    refs = json.load(open(refs_path, encoding="utf-8"))

    hyps = {}
    for line in open(tsv, encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        path, txt = line.split("\t", 1)
        hyps[os.path.basename(path)] = txt

    total_edits = 0
    total_words = 0
    rows = []
    worst = 0.0
    for name in sorted(refs):
        if name not in hyps:
            continue
        ref = normalize(refs[name]).split()
        hyp = normalize(hyps[name]).split()
        e = edits(ref, hyp)
        w = len(ref)
        wer = e / w if w else (0.0 if not hyp else 1.0)
        worst = max(worst, wer)
        total_edits += e
        total_words += w
        rows.append((name, e, w, wer))

    # FAIL LOUD: no scored clips means the run produced nothing (empty/garbage TSV) — never report
    # a spurious 0.0000 PASS. (Was a silent failure mode: B=128 produced clips=0 -> "PASS".)
    if not rows or total_words == 0:
        n_lines = sum(1 for _ in open(tsv, encoding="utf-8"))
        print(
            f"[WER-GATE] ERROR: 0 clips scored ({len(hyps)} hyp lines in TSV, {n_lines} raw lines, "
            f"{len(refs)} refs). The decode produced no matching transcripts -> FAILURE, not a pass.",
            file=sys.stderr,
        )
        print("[WER-GATE] aggregate per-stream WER N/A (0 clips) -> FAIL")
        sys.exit(2)

    agg = total_edits / total_words if total_words else 0.0
    print(f"{'clip':<12} {'edits':>6} {'words':>6} {'WER':>8}")
    for name, e, w, wer in rows:
        flag = "" if wer == 0 else (" *" if wer <= gate else " !!")
        print(f"{name:<12} {e:>6} {w:>6} {wer:>8.4f}{flag}")
    print(f"{'-'*36}")
    print(f"{'AGGREGATE':<12} {total_edits:>6} {total_words:>6} {agg:>8.4f}   (clips={len(rows)}, worst-clip={worst:.4f})")
    status = "PASS" if agg <= gate else "FAIL"
    print(f"[WER-GATE] aggregate per-stream WER {agg:.4f} vs gate {gate:.4f} -> {status}")
    sys.exit(0 if agg <= gate else 1)


if __name__ == "__main__":
    main()
