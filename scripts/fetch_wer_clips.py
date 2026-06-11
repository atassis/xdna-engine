#!/usr/bin/env python3
"""One-shot: pull a HANDFUL of FLEURS dev clips (RU + EN) by range-streaming the
gz-compressed audio tar (no full-dataset download), convert each to 16 kHz mono
s16le WAV via ffmpeg, and write refs.json from the dataset's normalized transcript.

Reputable source: google/fleurs (FLEURS), CC-BY-4.0, ungated on the HF hub.
We only pull the head of dev.tar.gz for ru_ru / en_us, enough for ~13 RU + 4 EN clips.
"""
import csv
import io
import json
import os
import subprocess
import zlib

import requests

# Repo-relative output: artifacts/wer_clips next to this script's repo root (overridable).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("WER_CLIPS_DIR", os.path.join(_REPO, "artifacts", "wer_clips"))
RAW = os.path.join(OUT, "_raw")
os.makedirs(RAW, exist_ok=True)

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
# (lang_code, split, our_prefix, how_many, head_MB)
PLAN = [
    ("ru_ru", "dev", "ru", 13, 14),
    ("en_us", "dev", "en", 4, 6),
]


def load_tsv(lang, split):
    """filename -> normalized lowercased transcript (col 3, 0-indexed)."""
    url = f"{BASE}/{lang}/{split}.tsv"
    txt = requests.get(url, timeout=60).text
    refs = {}
    for row in csv.reader(io.StringIO(txt), delimiter="\t"):
        if len(row) < 4:
            continue
        fname, norm = row[1], row[3]
        refs[fname] = norm.strip()
    return refs


def stream_tar_members(lang, split, head_mb):
    """Range-fetch head of dev.tar.gz, gz-decompress, yield (name, bytes) for
    every tar entry that is fully present in the decompressed prefix."""
    url = f"{BASE}/{lang}/audio/{split}.tar.gz"
    n = head_mb * 1024 * 1024
    r = requests.get(url, headers={"Range": f"bytes=0-{n-1}"}, timeout=180)
    r.raise_for_status()
    raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(r.content)
    off = 0
    while off + 512 <= len(raw):
        hdr = raw[off : off + 512]
        if hdr == b"\x00" * 512:
            break
        name = hdr[0:100].split(b"\x00")[0].decode("utf-8", "replace")
        if not name:
            break
        try:
            size = int(hdr[124:136].split(b"\x00")[0].strip() or b"0", 8)
        except ValueError:
            break
        data_start = off + 512
        data_end = data_start + size
        if data_end > len(raw):
            break  # truncated entry -> stop
        if size > 0 and name.endswith(".wav"):
            yield os.path.basename(name), raw[data_start:data_end]
        off = data_start + (size + 511) // 512 * 512


def to_16k_mono(src_wav_bytes, dst_path):
    """ffmpeg -> 16 kHz mono s16le WAV. Returns duration seconds (via ffprobe)."""
    p = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0",
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst_path],
        input=src_wav_bytes, capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[:300])
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", dst_path],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(dur) if dur else 0.0


def main():
    refs_out = {}
    source_rows = []
    for lang, split, prefix, want, head_mb in PLAN:
        tsv = load_tsv(lang, split)
        got = 0
        for fname, wav_bytes in stream_tar_members(lang, split, head_mb):
            if got >= want:
                break
            ref = tsv.get(fname)
            if not ref:
                continue  # need a reference to be useful for WER
            # keep clips short-ish: skip > ~15 s (FLEURS dev are read sentences)
            out_name = f"{prefix}_{got+1:02d}.wav"
            out_path = os.path.join(OUT, out_name)
            try:
                dur = to_16k_mono(wav_bytes, out_path)
            except Exception as e:  # noqa: BLE001
                print(f"  ffmpeg fail {fname}: {e}")
                continue
            if dur > 16.0:
                os.remove(out_path)
                continue
            refs_out[out_name] = ref.lower()
            source_rows.append((out_name, lang, split, fname, f"{dur:.1f}s"))
            got += 1
            print(f"  {out_name}  {dur:5.1f}s  <- {lang}/{split}/{fname}")
        print(f"[{lang}] wrote {got}/{want}")

    with open(os.path.join(OUT, "refs.json"), "w", encoding="utf-8") as f:
        json.dump(refs_out, f, ensure_ascii=False, indent=2)
    print(f"[refs] {len(refs_out)} entries -> refs.json")

    # SOURCE.md
    lines = [
        "# WER clip set — source & provenance",
        "",
        "## Dataset",
        "- **FLEURS** (Few-shot Learning Evaluation of Universal Representations of Speech)",
        "- HF hub: `google/fleurs` (ungated, public)",
        "- License: **CC-BY-4.0**",
        "- Paper: Conneau et al., 2022 (Google).",
        "",
        "## How clips were obtained",
        "- RU = config `ru_ru`, split `dev`. EN = config `en_us`, split `dev`.",
        "- We range-streamed only the *head* of each split's `audio/<split>.tar.gz`",
        "  (~14 MB RU / ~6 MB EN of the gz) and extracted whole tar entries — NO",
        "  full-dataset download.",
        "- Reference transcript = column 4 (0-indexed col 3) of `<split>.tsv`, the",
        "  dataset's own normalized lowercased transcription. Stored lowercased in refs.json.",
        "- Each WAV re-encoded to 16 kHz / mono / s16le via ffmpeg.",
        "",
        "## Why FLEURS and not Common Voice",
        "- Task preferred Mozilla Common Voice RU, but `mozilla-foundation/common_voice_17_0`",
        "  is GATED: without an HF auth token the repo exposes only README + .gitattributes",
        "  (0 data files), and the datasets-server returns nothing usable. No HF_TOKEN is",
        "  present in this environment, so CV could not be pulled. FLEURS is the reputable,",
        "  ungated RU+EN read-speech ASR benchmark used instead.",
        "",
        "## Clips",
        "",
        "| file | lang | split | source filename | duration |",
        "|------|------|-------|-----------------|----------|",
    ]
    for r in source_rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    with open(os.path.join(OUT, "SOURCE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[source] {len(source_rows)} rows -> SOURCE.md")


if __name__ == "__main__":
    main()
