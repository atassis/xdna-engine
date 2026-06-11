#!/usr/bin/env python3
"""WER evaluation harness for the NPU-ASR service.

For each clip in --clips (with refs.json mapping file -> ground-truth text), this:
  1. POSTs the WAV to the live service (multipart `file`) -> OUR hypothesis.
     Endpoint: http://127.0.0.1:11434/v1/audio/transcriptions  (same shape as
     scripts/test_npu_pipeline.py / scripts/asr_service.py; returns {"text": ...}).
  2. Runs the CPU oracle (onnx-asr GigaAM-v3, same model the service decodes with)
     -> ORACLE hypothesis.
Then computes WER for (ours vs ref), (oracle vs ref), (ours vs oracle), prints a
per-clip + aggregate table, and writes a results_placeholder.md template.

Text is normalized (lowercase + punctuation stripped + whitespace collapsed) before
scoring. WER uses jiwer if installed, else an inline word-level Levenshtein (no hard dep).

Run (full eval — needs the service up on :11434 AND the NPU free):
    ~/npuvox-asr-bench/.venv/bin/python scripts/wer_eval.py --clips artifacts/wer_clips

Smoke-test the oracle path only (NO service POST — safe to run anytime):
    ~/npuvox-asr-bench/.venv/bin/python scripts/wer_eval.py --clips artifacts/wer_clips --no-service

Flags:
    --clips DIR     clip dir containing *.wav + refs.json   (default artifacts/wer_clips)
    --no-service    skip the :11434 POST; only run the oracle (for smoke-testing)
    --url URL       override service endpoint
    --timeout SEC   per-request HTTP timeout (default 120)
"""
import argparse
import glob
import json
import os
import re
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URL = "http://127.0.0.1:11434/v1/audio/transcriptions"
SNAP = None  # resolved lazily for the oracle


# --------------------------------------------------------------------------- text normalization
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text):
    """lowercase, strip punctuation, collapse whitespace, NFC-normalize."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------- WER
def _wer_inline(ref, hyp):
    """Word-level WER via Levenshtein edit distance over word tokens.
    Returns (wer_float, n_ref_words). Empty ref -> 0.0 if hyp empty else 1.0."""
    r = ref.split()
    h = hyp.split()
    if not r:
        return (0.0 if not h else 1.0), 0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(r), len(r)


try:
    import jiwer  # noqa: F401

    _HAVE_JIWER = True
except Exception:  # noqa: BLE001
    _HAVE_JIWER = False


def wer(ref, hyp):
    """(wer_float, n_ref_words). Inputs assumed already normalized."""
    if _HAVE_JIWER:
        r = ref.split()
        try:
            score = jiwer.wer(ref, hyp) if ref.strip() else (0.0 if not hyp.strip() else 1.0)
        except Exception:  # noqa: BLE001
            score, n = _wer_inline(ref, hyp)
            return score, n
        return float(score), len(r)
    return _wer_inline(ref, hyp)


# --------------------------------------------------------------------------- service hypothesis
def transcribe_via_service(wav_path, url, timeout):
    """POST the WAV (multipart field `file`) -> {"text": ...}. Returns the text str."""
    import requests

    with open(wav_path, "rb") as f:
        files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
        data = {"model": "gigaam-v3-rnnt"}
        resp = requests.post(url, files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("text", "")


# --------------------------------------------------------------------------- oracle hypothesis
_ORACLE_MODEL = None


def _load_oracle():
    """Load the onnx-asr GigaAM-v3 model once (CPU). Same snapshot the service uses."""
    global _ORACLE_MODEL, SNAP
    if _ORACLE_MODEL is not None:
        return _ORACLE_MODEL
    import numpy as np  # noqa: F401  (onnx_asr pulls it; keep import explicit)
    import onnx_asr

    hub = os.path.expanduser("~/.cache/huggingface/hub")
    snaps = glob.glob(f"{hub}/models--istupakov--gigaam-v3-onnx/snapshots/*")
    if not snaps:
        raise RuntimeError("gigaam-v3-onnx snapshot not found in HF cache")
    SNAP = snaps[0]
    print(f"[oracle] loading gigaam-v3-rnnt from {SNAP} ...", file=sys.stderr)
    _ORACLE_MODEL = onnx_asr.load_model("gigaam-v3-rnnt", path=SNAP)
    return _ORACLE_MODEL


def _read_wav_16k(path):
    import wave

    import numpy as np

    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x


def transcribe_via_oracle(wav_path):
    """CPU onnx-asr transcription -> text str."""
    model = _load_oracle()
    return model.recognize(_read_wav_16k(wav_path))


# --------------------------------------------------------------------------- placeholder template
def write_placeholder(clips_dir, names):
    path = os.path.join(clips_dir, "results_placeholder.md")
    lines = [
        "# WER results (TEMPLATE — numbers filled by the main session run)",
        "",
        "Run to populate (needs service on :11434 + NPU free):",
        "",
        "```",
        "~/npuvox-asr-bench/.venv/bin/python scripts/wer_eval.py --clips artifacts/wer_clips",
        "```",
        "",
        "WER = word error rate (lower is better). Text normalized (lowercase, no punctuation).",
        "",
        "## Per-clip",
        "",
        "| clip | ref words | WER ours-vs-ref | WER oracle-vs-ref | WER ours-vs-oracle |",
        "|------|-----------|-----------------|-------------------|--------------------|",
    ]
    for n in names:
        lines.append(f"| {n} | _ | _ | _ | _ |")
    lines += [
        "",
        "## Aggregate",
        "",
        "| set | clips | mean WER ours-vs-ref | mean WER oracle-vs-ref | mean WER ours-vs-oracle |",
        "|-----|-------|----------------------|------------------------|-------------------------|",
        "| RU  | _ | _ | _ | _ |",
        "| EN  | _ | _ | _ | _ |",
        "| ALL | _ | _ | _ | _ |",
        "",
        "_(this is a template; do not hand-fill — re-run the harness to generate real numbers)_",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="WER eval harness for NPU-ASR.")
    ap.add_argument("--clips", default=os.path.join(REPO, "artifacts", "wer_clips"))
    ap.add_argument("--no-service", action="store_true",
                    help="skip the :11434 POST; only run the oracle (smoke-test)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    clips_dir = os.path.abspath(os.path.expanduser(a.clips))
    refs_path = os.path.join(clips_dir, "refs.json")
    if not os.path.isfile(refs_path):
        print(f"[error] no refs.json in {clips_dir}", file=sys.stderr)
        return 2
    with open(refs_path, encoding="utf-8") as f:
        refs = json.load(f)

    names = sorted(refs.keys())
    write_placeholder(clips_dir, names)

    rows = []
    for name in names:
        wav_path = os.path.join(clips_dir, name)
        if not os.path.isfile(wav_path):
            print(f"[skip] {name}: wav missing", file=sys.stderr)
            continue
        ref_n = normalize(refs[name])

        ours_n = None
        if not a.no_service:
            try:
                ours = transcribe_via_service(wav_path, a.url, a.timeout)
                ours_n = normalize(ours)
            except Exception as e:  # noqa: BLE001
                print(f"[service] {name} FAILED: {e!r}", file=sys.stderr)
                ours_n = None

        try:
            oracle = transcribe_via_oracle(wav_path)
            oracle_n = normalize(oracle)
        except Exception as e:  # noqa: BLE001
            print(f"[oracle] {name} FAILED: {e!r}", file=sys.stderr)
            oracle_n = None

        w_or_ref = wer(ref_n, oracle_n)[0] if oracle_n is not None else None
        w_ours_ref = wer(ref_n, ours_n)[0] if ours_n is not None else None
        w_ours_or = (wer(oracle_n, ours_n)[0]
                     if (ours_n is not None and oracle_n is not None) else None)
        n_ref = len(ref_n.split())

        rows.append({
            "name": name, "lang": name.split("_")[0], "n_ref": n_ref,
            "ref": ref_n, "ours": ours_n, "oracle": oracle_n,
            "ours_ref": w_ours_ref, "oracle_ref": w_or_ref, "ours_oracle": w_ours_or,
        })

    # ---- print per-clip table
    def fmt(x):
        return "  -  " if x is None else f"{x*100:5.1f}%"

    print()
    print(f"WER eval  (jiwer={'yes' if _HAVE_JIWER else 'no, inline'}; "
          f"service={'SKIPPED (--no-service)' if a.no_service else a.url})")
    print("-" * 78)
    print(f"{'clip':<12}{'refW':>5}  {'ours/ref':>9}  {'oracle/ref':>11}  {'ours/oracle':>12}")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:<12}{r['n_ref']:>5}  {fmt(r['ours_ref']):>9}  "
              f"{fmt(r['oracle_ref']):>11}  {fmt(r['ours_oracle']):>12}")

    # ---- aggregates (mean over clips with a value)
    def agg(rows_subset, key):
        vals = [r[key] for r in rows_subset if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    print("-" * 78)
    for label, sub in (("RU", [r for r in rows if r["lang"] == "ru"]),
                       ("EN", [r for r in rows if r["lang"] == "en"]),
                       ("ALL", rows)):
        if not sub:
            continue
        print(f"{('MEAN ' + label):<12}{'':>5}  "
              f"{fmt(agg(sub,'ours_ref')):>9}  {fmt(agg(sub,'oracle_ref')):>11}  "
              f"{fmt(agg(sub,'ours_oracle')):>12}   (n={len(sub)})")
    print("-" * 78)

    if a.no_service:
        print("\n[note] --no-service: 'ours' columns are blank by design. The main "
              "session runs WITHOUT --no-service (service up on :11434) for full numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
