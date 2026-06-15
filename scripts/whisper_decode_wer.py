#!/usr/bin/env python3
"""Whisper-small WER over the wer_clips FLEURS set via the live engine_serve service.

Drives the Rust engine end-to-end (NPU encoder + whichever decoder the running service was
launched with): POSTs each WAV to /v1/audio/transcriptions, scores the returned text vs
artifacts/wer_clips/refs.json. The decoder backend is selected by the env the service was
launched with (NPU_DECODE=1 => on-NPU per-token decoder; unset => ONNX). This harness does NOT
itself open the NPU — it only talks HTTP, so it is safe to run alongside a single-tenant service.

Usage:
    python3 scripts/whisper_decode_wer.py --url http://127.0.0.1:11434/v1/audio/transcriptions \
        --label npu --out /tmp/wer_npu.json
"""
import argparse, json, os, re, sys, time, unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "artifacts" / "wer_clips"

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


def transcribe(url, wav_path, timeout):
    import requests
    with open(wav_path, "rb") as f:
        files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
        t0 = time.time()
        resp = requests.post(url, files=files, data={"model": "whisper-small"}, timeout=timeout)
        dt = time.time() - t0
    resp.raise_for_status()
    return resp.json().get("text", ""), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1/audio/transcriptions")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()

    refs = json.load(open(CLIPS / "refs.json", encoding="utf-8"))
    rows = {}
    wall = []
    for name in sorted(refs):
        wav = CLIPS / name
        if not wav.is_file():
            continue
        try:
            hyp, dt = transcribe(a.url, wav, a.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"[{a.label}] {name} FAILED: {e!r}", file=sys.stderr)
            continue
        w, n = wer(normalize(refs[name]), normalize(hyp))
        rows[name] = {"hyp": hyp, "wer": w, "nref": n, "wall_s": dt}
        wall.append(dt)
        print(f"[{a.label}] {name}: WER={w:.3f}  wall={dt:.2f}s")

    # pooled WER = Σedits / Σref_words (the corpus-level WER; headline). edits_i is recovered from the
    # per-clip wer ratio and ref-word count. macro = unweighted mean of per-clip WER (secondary).
    def pooled(items):
        sum_edits = sum(round(v["wer"] * v["nref"]) for v in items)
        sum_ref = sum(v["nref"] for v in items)
        return (sum_edits / sum_ref) if sum_ref else 0.0

    print(f"\n=== label={a.label} ===")
    for split in ("en", "ru"):
        sub = [v for k, v in rows.items() if k.startswith(split)]
        if sub:
            macro = sum(v["wer"] for v in sub) / len(sub)
            print(f"{split} pooled WER = {pooled(sub):.4f}  (macro {macro:.4f}, n={len(sub)})")
    allv = list(rows.values())
    if allv:
        macro = sum(v["wer"] for v in allv) / len(allv)
        print(f"ALL pooled WER = {pooled(allv):.4f}  (macro {macro:.4f}, n={len(allv)})")
    if wall:
        print(f"mean per-clip wall = {sum(wall)/len(wall):.2f}s  (n={len(wall)})")

    if a.out:
        json.dump({"label": a.label, "rows": rows}, open(a.out, "w"), indent=2, ensure_ascii=False)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
