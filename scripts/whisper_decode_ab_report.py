#!/usr/bin/env python3
"""Score scripts/whisper_decode_ab.sh logs: per arm, token-id determinism across every transcription
of each clip, pooled WER against artifacts/wer_clips/refs.json, and decode ms per emitted token by
round (decode_ms / tokens from the [WHISPER_TIMING] line, timed passes only).

  .venv-export/bin/python scripts/whisper_decode_ab_report.py <OUT_DIR> [--json out.json]
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_wer_from_hidden import normalize, wer  # noqa: E402

CLIP = re.compile(r"\[bench\] clip=\S*/(\S+\.wav)")
IDS = re.compile(r"\[token_ids\] ([\d,]+)")
TIMING = re.compile(r"\[WHISPER_TIMING\] backend=(\S+) .*decode_ms=([\d.]+) tokens=(\d+)")


def parse(log):
    """[(clip, ids, backend, decode_ms, tokens)] in run order; each clip's first is its warmup."""
    runs, clip, ids = [], None, None
    for line in open(log, encoding="utf-8", errors="replace"):
        if m := CLIP.search(line):
            clip = m.group(1)
        elif m := IDS.search(line):
            ids = tuple(int(x) for x in m.group(1).split(","))
        elif m := TIMING.search(line):
            runs.append((clip, ids, m.group(1), float(m.group(2)), int(m.group(3))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from transformers import WhisperProcessor
    tok = WhisperProcessor.from_pretrained("artifacts/whisper-small/onnx").tokenizer
    refs = json.load(open("artifacts/wer_clips/refs.json", encoding="utf-8"))
    ids_by = defaultdict(lambda: defaultdict(set))
    ms_by = defaultdict(lambda: defaultdict(list))
    backend = {}
    for log in sorted(Path(a.out_dir).glob("r*_*.log")):
        rnd, arm = log.stem.split("_", 1)
        seen = set()
        for clip, ids, be, dec_ms, n in parse(log):
            backend[arm] = be
            ids_by[arm][clip].add(ids)
            if clip in seen:
                ms_by[arm][rnd].append(dec_ms / n)
            seen.add(clip)
    report = {}
    for arm in sorted(ids_by):
        clips = ids_by[arm]
        det = sum(len(v) == 1 for v in clips.values())
        edits = words = 0
        for clip, v in clips.items():
            w, n = wer(normalize(refs[clip]), normalize(tok.decode(sorted(v)[0], skip_special_tokens=True)))
            edits, words = edits + w * n, words + n
        rounds = {r: statistics.median(v) for r, v in sorted(ms_by[arm].items())}
        allms = [x for v in ms_by[arm].values() for x in v]
        report[arm] = dict(backend=backend[arm], clips=len(clips), deterministic=det,
                           wer=edits / words, words=words, ms_per_token_by_round=rounds,
                           ms_per_token_median=statistics.median(allms), n=len(allms))
        print(f"{arm:8s} {backend[arm]:6s} deterministic {det}/{len(clips)}  WER {edits / words:.4f} "
              f"({words} words)  ms/token median {statistics.median(allms):.2f} (n={len(allms)})  "
              f"by round " + " ".join(f"{r}={v:.2f}" for r, v in rounds.items()))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
