#!/usr/bin/env python3
"""Does an encoder error burst reach the TRANSCRIPT?

The shipped encoder carries contiguous runs of frames at 15-79% per-frame relative error against
f32 truth. That is a real device defect, but it only matters if it changes what the model says.
This script answers that directly and cheaply, with no device and no instrumentation: it TDT-decodes
the already-dumped encoder outputs (f32 truth and each device path) and asks, per clip:

  1. do the transcripts differ at all?
  2. for every token the decoder emits, at which encoder FRAME did it emit?  (the TDT decoder
     returns per-token timestamps, so this is read out, not inferred)
  3. are the tokens that differ located AT the burst frames, or somewhere else?

(3) is the load-bearing question. If transcripts differ but the differences are not at the bursts,
the bursts are not what is costing accuracy. If transcripts do not differ at all, the bursts are
cosmetic and the whole precision thread deprioritises.

Usage (needs the onnx-asr venv, CPU only):
  ~/npuvox-asr-bench/.venv/bin/python scripts/burst_to_transcript.py \
      --ref /tmp/enc_ref --path shipped=/tmp/enc_ship --path cand=/tmp/enc_cand
"""
import argparse
import difflib
import json
import os
import re
import sys
import unicodedata

import numpy as np
import onnx_asr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.environ.get("WER_CLIPS") or os.path.join(REPO, "artifacts", "wer_clips")
MODEL = "nemo-parakeet-tdt-0.6b-v3"

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(t):
    t = unicodedata.normalize("NFC", t or "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def wer_edits(ref, hyp):
    """(edit count, ref word count) -- plain Levenshtein over words."""
    r, h = ref.split(), hyp.split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if rw == hw else 1)))
        prev = cur
    return prev[-1], len(r)


def frame_rel_err(ref, x):
    num = np.linalg.norm(ref - x, axis=-1)
    den = np.maximum(np.linalg.norm(ref, axis=-1), 1e-12)
    return num / den


def decode(asr, enc):
    """Return (text, [(token_id, frame_index), ...]).

    The TDT decoding generator yields (tokens, timestamps, logprobs) cumulatively; the last yield
    carries the full hypothesis. `timestamps` are in ENCODER FRAMES, which is exactly the axis the
    per-frame error is measured on -- no resampling or alignment guesswork.
    """
    eo = enc[None, :, :].astype(np.float32)
    lens = np.array([enc.shape[0]], np.int64)
    toks, times = [], []
    for tok, ts, lp in asr._decoding(eo, lens):
        toks = [int(x) for x in tok]
        times = [int(x) for x in ts]
    text = asr._decode_tokens(toks, None, None).text
    return text, list(zip(toks, times))


def token_texts(asr, toks):
    """Per-token surface text, so a differing token can be named rather than numbered."""
    out = []
    for t in toks:
        try:
            out.append(asr._decode_tokens([t], None, None).text)
        except Exception:
            out.append(f"<{t}>")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="f32-truth encoder dump dir")
    ap.add_argument("--path", action="append", default=[], metavar="NAME=DIR",
                    help="a device path to compare; repeatable")
    ap.add_argument("--burst-floor", type=float, default=0.15)
    ap.add_argument("--radius", type=int, default=2,
                    help="a token counts as 'at a burst' if within this many frames of one")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    paths = []
    for p in args.path:
        name, _, d = p.partition("=")
        paths.append((name, d))
    if not paths:
        ap.error("need at least one --path NAME=DIR")

    asr = onnx_asr.load_model(MODEL, providers=["CPUExecutionProvider"]).asr
    refs = json.load(open(os.path.join(CLIPS, "refs.json"), encoding="utf-8"))
    names = sorted(refs.keys())

    result = {"burst_floor": args.burst_floor, "radius": args.radius, "paths": {}, "clips": []}
    totals = {n: {"edits_vs_ref": 0, "tokens_diff": 0, "diff_frames": 0,
                  "diff_frames_at_burst": 0, "expected_at_burst": 0.0,
                  "edits_vs_truth_text": 0, "words": 0, "clips_text_differs": 0,
                  "burst_frames": 0, "emit_frames": 0, "emit_frames_at_burst": 0}
              for n, _ in paths}

    print(f"burst floor {args.burst_floor}   token-at-burst radius +-{args.radius} frames")
    print()

    for n in names:
        stem = os.path.splitext(n)[0]
        rp = os.path.join(args.ref, f"{stem}.npy")
        if not os.path.isfile(rp):
            continue
        ref_enc = np.load(rp).astype(np.float32)
        ref_text, ref_toks = decode(asr, ref_enc)
        ref_norm = normalize(ref_text)

        row = {"clip": stem, "frames": int(ref_enc.shape[0]), "ref_text": ref_norm, "paths": {}}
        print(f"=== {stem}  T={ref_enc.shape[0]}  ref tokens={len(ref_toks)}")

        for pname, pdir in paths:
            enc = np.load(os.path.join(pdir, f"{stem}.npy")).astype(np.float32)
            e = frame_rel_err(ref_enc, enc)
            bursts = np.where(e >= args.burst_floor)[0]
            burst_set = set()
            for b in bursts:
                for d in range(-args.radius, args.radius + 1):
                    burst_set.add(int(b) + d)

            text, toks = decode(asr, enc)
            norm = normalize(text)

            # Token-level diff against the f32-truth decode, aligned by the decoder's own frames.
            ref_pairs = [(t, f) for t, f in ref_toks]
            cnd_pairs = [(t, f) for t, f in toks]
            sm = difflib.SequenceMatcher(
                a=[t for t, _ in ref_pairs], b=[t for t, _ in cnd_pairs], autojunk=False)
            diff_frames, ndiff = set(), 0
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                ndiff += max(i2 - i1, j2 - j1)
                diff_frames.update(f for _, f in ref_pairs[i1:i2])
                diff_frames.update(f for _, f in cnd_pairs[j1:j2])
            diff_frames = sorted(diff_frames)
            at_burst = sum(1 for f in diff_frames if f in burst_set)

            # Null model: if the differing tokens were placed uniformly at random over the frames
            # the decoder actually emits at, how many would land on a burst by chance? Without this
            # a low at-burst count is "no evidence", not "evidence of no effect".
            emit_frames = sorted({f for _, f in ref_pairs} | {f for _, f in cnd_pairs})
            covered = sum(1 for f in emit_frames if f in burst_set)
            p_chance = covered / max(len(emit_frames), 1)
            exp_at_burst = p_chance * len(diff_frames)

            ed_ref, _ = wer_edits(ref_norm, norm)
            ed_truth, nw = wer_edits(normalize(refs[n]), norm)

            t = totals[pname]
            t["edits_vs_ref"] += ed_ref
            t["tokens_diff"] += ndiff
            t["diff_frames"] += len(diff_frames)
            t["diff_frames_at_burst"] += at_burst
            t["expected_at_burst"] += exp_at_burst
            t["edits_vs_truth_text"] += ed_truth
            t["words"] += nw
            t["clips_text_differs"] += int(norm != ref_norm)
            t["burst_frames"] += int(len(bursts))
            t["emit_frames"] += len(emit_frames)
            t["emit_frames_at_burst"] += covered

            burst_str = ",".join(str(b) for b in bursts[:12]) + ("..." if len(bursts) > 12 else "")
            print(f"  {pname:10} bursts@[{burst_str}] ({len(bursts)})  "
                  f"tokens_diff={ndiff} at-frames={len(diff_frames)} (at-burst {at_burst}, "
                  f"chance {exp_at_burst:.2f})  word-edits-vs-f32={ed_ref}  "
                  f"text {'SAME' if norm == ref_norm else 'DIFFERS'}")
            if norm != ref_norm:
                print(f"      f32 : {ref_norm}")
                print(f"      {pname:4}: {norm}")
                if diff_frames:
                    print(f"      differing tokens at frames {diff_frames}"
                          f"   burst frames {sorted(bursts.tolist())}")

            row["paths"][pname] = {
                "burst_frames": bursts.tolist(), "n_burst": int(len(bursts)),
                "tokens_diff": ndiff, "diff_frames_at_burst": at_burst,
                "expected_at_burst": exp_at_burst,
                "diff_frames": [int(f) for f in diff_frames],
                "word_edits_vs_f32": ed_ref, "word_edits_vs_truth": ed_truth,
                "text": norm, "text_differs": norm != ref_norm,
            }
        result["clips"].append(row)

    print()
    print("=" * 100)
    print(f"{'path':10} {'clips differ':>13} {'burstF':>7} {'tokdiff':>8} "
          f"{'diffF':>6} {'@burst':>7} {'chance':>7} {'wEdit':>6} {'WER':>7}")
    for pname, _ in paths:
        t = totals[pname]
        print(f"{pname:10} {t['clips_text_differs']:>9}/{len(result['clips'])} "
              f"{t['burst_frames']:>7} {t['tokens_diff']:>8} {t['diff_frames']:>6} "
              f"{t['diff_frames_at_burst']:>7} {t['expected_at_burst']:>7.2f} "
              f"{t['edits_vs_ref']:>6} "
              f"{100.0*t['edits_vs_truth_text']/max(t['words'],1):>6.2f}%")
        result["paths"][pname] = t
    print()
    print("  burstF = burst frames in the encoder output;  diffF = frames carrying a differing")
    print("  token;  @burst = how many of those sit on a burst;  chance = how many would if the")
    print("  differing tokens were placed at random over the frames the decoder emits at.")

    # The f32-truth path's own WER, as the control: how much of the error is the MODEL, not us.
    ed = nw = 0
    for r in result["clips"]:
        e, w = wer_edits(normalize(refs_key(refs, r["clip"])), r["ref_text"])
        ed += e
        nw += w
    print(f"{'f32-truth':12} {'(control)':>17} {'-':>10} {'-':>9} {'-':>18} "
          f"{100.0*ed/max(nw,1):>12.2f}%")
    result["f32_truth_wer"] = 100.0 * ed / max(nw, 1)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")


def refs_key(refs, stem):
    for k in refs:
        if os.path.splitext(k)[0] == stem:
            return refs[k]
    return ""


if __name__ == "__main__":
    sys.exit(main())
