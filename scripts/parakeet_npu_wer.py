#!/usr/bin/env python3
"""Phase 4 bridge — WER of the Parakeet encoder via the encoder-swap method (same as
scripts/int8_wer_eval.py for GigaAM): onnx-asr does the 128-mel frontend + TDT decode on CPU,
and a swapped encoder output (our Rust NPU or host encoder, dumped as .npy) sits in the middle.

Two modes:
  dump-mels <out_dir>   : write per-clip mel features [128,T] (.npy) for the Rust encoder.
  decode-wer <enc_dir>  : load per-clip encoded [T',1024] (.npy), TDT-decode via onnx-asr, print WER.

Also `--cpu-oracle` recomputes the pure onnx-asr CPU WER (encoder included) as a control.
Usage: ~/npuvox-asr-bench/.venv/bin/python scripts/parakeet_npu_wer.py <mode> <dir> [--cpu-oracle]
"""
import argparse, glob, json, os, re, sys, time, unicodedata, wave
import numpy as np
import onnx_asr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# wer_clips (refs.json + wavs) are gitignored; they live in the main worktree's artifacts.
CLIPS = os.environ.get("WER_CLIPS") or next(
    (d for d in [os.path.join(REPO, "artifacts", "wer_clips"),
                 "$REPO/artifacts/wer_clips"]
     if os.path.isfile(os.path.join(d, "refs.json"))),
    os.path.join(REPO, "artifacts", "wer_clips"))
MODEL = "nemo-parakeet-tdt-0.6b-v3"

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE); _WS = re.compile(r"\s+")
def normalize(t):
    t = unicodedata.normalize("NFC", t or "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r: return (0.0 if not h else 1.0)
    prev = list(range(len(h)+1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if rw==hw else 1)))
        prev = cur
    return prev[-1]/len(r)
def read_wav(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes()); ch = w.getnchannels()
    x = np.frombuffer(raw, np.int16).astype(np.float32)/32768.0
    return x.reshape(-1, ch).mean(1) if ch > 1 else x

def agg(rows):
    out = {}
    for lab, langs in (("RU", {"ru"}), ("EN", {"en"}), ("ALL", {"ru", "en"})):
        sub = [r for r in rows if r["lang"] in langs]
        if sub: out[lab] = {"n": len(sub), "wer": sum(r["wer"] for r in sub)/len(sub)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump-mels", "decode-wer"])
    ap.add_argument("dir")
    ap.add_argument("--cpu-oracle", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    asr = onnx_asr.load_model(MODEL, providers=["CPUExecutionProvider"]).asr
    refs = json.load(open(os.path.join(CLIPS, "refs.json"), encoding="utf-8"))
    names = sorted(refs.keys())

    if a.mode == "dump-mels":
        for n in names:
            wavp = os.path.join(CLIPS, n)
            wav = read_wav(wavp)
            feats, flen = asr._preprocessor(wav[None, :].astype(np.float32), np.array([wav.shape[0]], np.int64))
            np.save(os.path.join(a.dir, f"{os.path.splitext(n)[0]}.npy"), feats[0].astype(np.float32))  # [128,T]
        print(f"wrote {len(names)} mel .npy to {a.dir}")
        return 0

    # decode-wer: load encoded [T',1024], decode, WER
    rows = []
    for n in names:
        stem = os.path.splitext(n)[0]
        encp = os.path.join(a.dir, f"{stem}.npy")
        if not os.path.isfile(encp):
            print(f"[skip] {stem} (no encoded)", file=sys.stderr); continue
        enc = np.load(encp).astype(np.float32)        # [T', 1024]
        eo = enc[None, :, :]                            # [1, T', 1024]
        lens = np.array([enc.shape[0]], np.int64)
        ids = []
        for tok, ts, lp in asr._decoding(eo, lens):
            ids = [int(x) for x in tok]
        hyp = normalize(asr._decode_tokens(ids, None, None).text)
        rows.append({"name": stem, "lang": stem.split("_")[0], "wer": wer(normalize(refs[n]), hyp), "hyp": hyp})
    A = agg(rows)
    print(f"\n=== Parakeet (swapped encoder from {a.dir}) WER ===")
    for lab in ("RU", "EN", "ALL"):
        if lab in A: print(f"  {lab:<4}(n={A[lab]['n']:>2})  {A[lab]['wer']*100:5.1f}%")

    if a.cpu_oracle:
        orows = []
        for n in names:
            hyp = normalize(asr_recognize(asr, os.path.join(CLIPS, n)))
            orows.append({"lang": n.split("_")[0], "wer": wer(normalize(refs[n]), hyp)})
        OA = agg(orows)
        print("\n=== CPU oracle (onnx-asr full pipeline) WER ===")
        for lab in ("RU", "EN", "ALL"):
            if lab in OA: print(f"  {lab:<4}(n={OA[lab]['n']:>2})  {OA[lab]['wer']*100:5.1f}%")
    return 0

def asr_recognize(asr, wavpath):
    wav = read_wav(wavpath)
    feats, flen = asr._preprocessor(wav[None, :].astype(np.float32), np.array([wav.shape[0]], np.int64))
    enc, lens = asr._encode(feats, flen)
    ids = []
    for tok, ts, lp in asr._decoding(enc, lens):
        ids = [int(x) for x in tok]
    return asr._decode_tokens(ids, None, None).text

if __name__ == "__main__":
    sys.exit(main())
