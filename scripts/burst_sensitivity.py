#!/usr/bin/env python3
"""How big does an encoder error burst have to be before the TRANSCRIPT changes?

The observational study (scripts/burst_to_transcript.py) compares the device encoder's real bursts
against f32 truth on 17 clips. That is only 17 samples of a rare event, so a null result there is
weak. This script asks the same question as a CONTROLLED experiment with as many samples as we want,
and it needs no device at all: take the f32 encoder output, INJECT a burst of a chosen magnitude at
a chosen frame, decode, and see whether the transcript moved.

The output is a sensitivity curve -- P(transcript changes | burst of magnitude m over L frames) --
which is what actually prices the precision thread. Combined with the observed burst distribution it
predicts how much transcript damage the real defect causes, and that prediction can be checked
against the 17-clip observation.

GENERATOR VALIDATION IS NOT OPTIONAL and runs by default (--validate-only to stop after it). A
synthetic generator whose distribution is not what the comment claims has produced a wrong ratio in
this project before, so the injector prints the ACHIEVED per-frame relative error (min/max/mean),
the sign balance of the injected delta, and confirms non-target frames are untouched bit-for-bit.

Usage (onnx-asr venv, CPU only):
  ~/npuvox-asr-bench/.venv/bin/python scripts/burst_sensitivity.py --ref /tmp/enc_ref
  ... --mags 0.2,0.4,0.6,0.8 --lengths 1,3,5 --positions 4
"""
import argparse
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


def inject(enc, frames, mag, rng):
    """Return a copy of `enc` with per-frame relative error exactly `mag` at `frames`.

    The perturbation is isotropic Gaussian rescaled so ||delta_t|| == mag * ||enc_t||, which makes
    the injected magnitude equal the statistic the parity gate measures -- so a curve indexed by
    `mag` is directly comparable to a measured per-frame rel-L2.
    """
    out = enc.copy()
    for t in frames:
        d = rng.standard_normal(enc.shape[1]).astype(np.float32)
        n = np.linalg.norm(d)
        if n == 0:
            continue
        out[t] = enc[t] + d * (mag * np.linalg.norm(enc[t]) / n)
    return out


def validate_injector(enc, rng):
    """Prove the generator does what the docstring says, before any ratio from it is believed."""
    print("=== injector validation ===")
    ok = True
    T = enc.shape[0]
    for mag in (0.15, 0.5, 1.0):
        frames = [T // 4, T // 2, 3 * T // 4]
        x = inject(enc, frames, mag, rng)
        e = np.linalg.norm(x - enc, axis=-1) / np.maximum(np.linalg.norm(enc, axis=-1), 1e-12)
        hit = e[frames]
        others = np.delete(e, frames)
        delta = (x - enc)[frames]
        pos = float((delta > 0).mean())
        amax = float(np.abs(x - enc).max())
        print(f"  mag={mag:<5} achieved per-frame rel-err min/mean/max = "
              f"{hit.min():.6f}/{hit.mean():.6f}/{hit.max():.6f}   "
              f"untouched frames max rel-err = {others.max():.3e}   "
              f"delta sign balance (+) = {pos:.3f}   |delta|max = {amax:.4f}")
        if not np.allclose(hit, mag, rtol=1e-5, atol=1e-6):
            print(f"  FAIL: achieved magnitude != requested {mag}")
            ok = False
        if others.max() != 0.0:
            print("  FAIL: non-target frames were modified")
            ok = False
        if not (0.4 < pos < 0.6):
            print(f"  FAIL: delta is not sign-balanced ({pos:.3f})")
            ok = False
    print(f"  injector: {'VALID' if ok else 'INVALID'}")
    return ok


def decode_text(asr, enc):
    eo = enc[None, :, :].astype(np.float32)
    lens = np.array([enc.shape[0]], np.int64)
    toks = []
    for tok, ts, lp in asr._decoding(eo, lens):
        toks = [int(x) for x in tok]
    return normalize(asr._decode_tokens(toks, None, None).text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--mags", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--lengths", default="1,3,5")
    ap.add_argument("--positions", type=int, default=4, help="random burst positions per clip")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    mags = [float(x) for x in args.mags.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    rng = np.random.default_rng(args.seed)

    names = sorted(os.path.basename(f) for f in os.listdir(args.ref) if f.endswith(".npy"))
    encs = {n: np.load(os.path.join(args.ref, n)).astype(np.float32) for n in names}

    if not validate_injector(encs[names[0]], rng):
        print("generator invalid -- refusing to report a curve from it")
        sys.exit(2)
    if args.validate_only:
        return 0

    asr = onnx_asr.load_model(MODEL, providers=["CPUExecutionProvider"]).asr
    base_text = {n: decode_text(asr, encs[n]) for n in names}

    print()
    print("=== transcript sensitivity to an injected burst ===")
    print(f"{'len':>4} {'mag':>5} {'trials':>7} {'changed':>8} {'P(change)':>10} {'wEdits':>7}")
    result = {"mags": mags, "lengths": lengths, "positions": args.positions, "rows": []}
    for L in lengths:
        for mag in mags:
            n_tr = n_ch = n_ed = 0
            for n in names:
                enc = encs[n]
                T = enc.shape[0]
                if T <= L + 2:
                    continue
                for p in rng.choice(T - L, size=min(args.positions, T - L), replace=False):
                    frames = list(range(int(p), int(p) + L))
                    txt = decode_text(asr, inject(enc, frames, mag, rng))
                    n_tr += 1
                    if txt != base_text[n]:
                        n_ch += 1
                        r, h = base_text[n].split(), txt.split()
                        prev = list(range(len(h) + 1))
                        for i, rw in enumerate(r, 1):
                            cur = [i]
                            for j, hw in enumerate(h, 1):
                                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                                               prev[j - 1] + (0 if rw == hw else 1)))
                            prev = cur
                        n_ed += prev[-1]
            p_ch = n_ch / max(n_tr, 1)
            print(f"{L:>4} {mag:>5.2f} {n_tr:>7} {n_ch:>8} {p_ch:>10.3f} {n_ed:>7}")
            result["rows"].append({"length": L, "mag": mag, "trials": n_tr,
                                   "changed": n_ch, "p_change": p_ch, "word_edits": n_ed})

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
