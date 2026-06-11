#!/usr/bin/env python3
"""int8-quantization WER eval for the GigaAM-v3 encoder (CPU only, NO NPU).

Builds a CPU pipeline that swaps ONLY the encoder: standalone encoder ONNX
([1,64,1600] mel + length -> encoded[1,768,400]) -> transpose -> onnx-asr RNNT
greedy decode (reusing model.asr._decoding + _decode_tokens). Runs every clip in
artifacts/wer_clips through each encoder variant and reports:
  - WER(variant vs reference)
  - WER(int8 vs fp32-encoder hypothesis)  <- isolates quantization error

Usage:
  ~/npuvox-asr-bench/.venv/bin/python scripts/int8_wer_eval.py [--variants fp32,int8_dynamic,int8_static]
"""
import argparse, glob, json, os, re, sys, time, unicodedata, wave
import numpy as np
import onnxruntime as ort
import onnx_asr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--istupakov--gigaam-v3-onnx/snapshots/*"))[0]
WIN = 1600
ENCODERS = {
    "fp32":         os.path.join(REPO, "models", "gigaam_v3_encoder_static.onnx"),
    "int8_dynamic": os.path.join(REPO, "models", "quant", "gigaam_v3_encoder_int8_dynamic.onnx"),
    "int8_static":  os.path.join(REPO, "models", "quant", "gigaam_v3_encoder_int8_static.onnx"),
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
def normalize(t):
    t = unicodedata.normalize("NFC", t or "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r: return (0.0 if not h else 1.0), 0
    prev = list(range(len(h)+1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if rw==hw else 1)))
        prev = cur
    return prev[-1]/len(r), len(r)

def read_wav(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes()); ch = w.getnchannels()
    x = np.frombuffer(raw, np.int16).astype(np.float32)/32768.0
    return x.reshape(-1, ch).mean(1) if ch > 1 else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=os.path.join(REPO, "artifacts", "wer_clips"))
    ap.add_argument("--variants", default="fp32,int8_dynamic")
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v.strip()]

    ASR = onnx_asr.load_model("gigaam-v3-rnnt", path=SNAP).asr
    sessions = {}
    for v in variants:
        p = ENCODERS[v]
        if not os.path.isfile(p):
            print(f"[warn] {v} encoder missing ({p}); skipping", file=sys.stderr); continue
        sessions[v] = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    variants = [v for v in variants if v in sessions]

    refs = json.load(open(os.path.join(a.clips, "refs.json"), encoding="utf-8"))
    names = sorted(refs.keys())

    def encode_decode(sess, feats):
        T = min(feats.shape[2], WIN)
        buf = np.zeros((1, 64, WIN), np.float32); buf[0, :, :T] = feats[0, :, :T]
        t0 = time.time()
        enc, enc_len = sess.run(None, {"audio_signal": buf, "length": np.array([T], np.int64)})
        dt = time.time()-t0
        v = int(enc_len[0])
        eo = enc[0].T[None, :v, :].astype(np.float32)
        ids = []
        for tok, ts, lp in ASR._decoding(eo, np.array([v], np.int64)):
            ids = [int(x) for x in tok]
        return normalize(ASR._decode_tokens(ids, None, None).text), dt

    rows = []
    enc_times = {v: [] for v in variants}
    for name in names:
        wavp = os.path.join(a.clips, name)
        if not os.path.isfile(wavp): continue
        wav = read_wav(wavp)
        feats, _ = ASR._preprocessor(wav[None, :].astype(np.float32), np.array([wav.shape[0]], np.int64))
        ref = normalize(refs[name])
        hyps = {}
        for v in variants:
            hyps[v], dt = encode_decode(sessions[v], feats)
            enc_times[v].append(dt)
        rows.append({"name": name, "lang": name.split("_")[0], "ref": ref, "hyps": hyps})
        print(f"[done] {name}", file=sys.stderr)

    # build result tables
    result = {"variants": variants, "per_clip": [], "agg": {}}
    for r in rows:
        rc = {"name": r["name"], "lang": r["lang"], "n_ref": len(r["ref"].split())}
        for v in variants:
            rc[f"{v}_vs_ref"] = wer(r["ref"], r["hyps"][v])[0]
        if "fp32" in variants:
            for v in variants:
                if v == "fp32": continue
                rc[f"{v}_vs_fp32"] = wer(r["hyps"]["fp32"], r["hyps"][v])[0]
        rc["hyps"] = r["hyps"]
        result["per_clip"].append(rc)

    def agg(sub, key):
        vals = [r[key] for r in sub if r.get(key) is not None]
        return sum(vals)/len(vals) if vals else None
    for label, langs in (("RU", {"ru"}), ("EN", {"en"}), ("ALL", {"ru", "en"})):
        sub = [r for r in result["per_clip"] if r["lang"] in langs]
        if not sub: continue
        d = {"n": len(sub)}
        for v in variants:
            d[f"{v}_vs_ref"] = agg(sub, f"{v}_vs_ref")
            if v != "fp32" and "fp32" in variants:
                d[f"{v}_vs_fp32"] = agg(sub, f"{v}_vs_fp32")
        result["agg"][label] = d
    result["enc_time_mean_s"] = {v: (sum(enc_times[v])/len(enc_times[v]) if enc_times[v] else None) for v in variants}

    # print
    def fmt(x): return "  -  " if x is None else f"{x*100:5.1f}%"
    print("\n=== per-clip WER vs reference ===")
    hdr = f"{'clip':<10}{'refW':>5}  " + "  ".join(f"{v:>13}" for v in variants)
    print(hdr); print("-"*len(hdr))
    for r in result["per_clip"]:
        print(f"{r['name']:<10}{r['n_ref']:>5}  " + "  ".join(fmt(r[f'{v}_vs_ref']) for v in variants))
    print("\n=== aggregate WER vs reference ===")
    for lbl in ("RU", "EN", "ALL"):
        if lbl not in result["agg"]: continue
        d = result["agg"][lbl]
        print(f"{lbl:<5}(n={d['n']})  " + "  ".join(f"{v}={fmt(d[f'{v}_vs_ref'])}" for v in variants))
    if "fp32" in variants and len(variants) > 1:
        print("\n=== int8 vs fp32-encoder divergence (quant error) ===")
        for lbl in ("RU", "EN", "ALL"):
            if lbl not in result["agg"]: continue
            d = result["agg"][lbl]
            cells = [f"{v}={fmt(d.get(f'{v}_vs_fp32'))}" for v in variants if v != "fp32"]
            print(f"{lbl:<5}(n={d['n']})  " + "  ".join(cells))
    print("\n=== mean encoder ONNX run time (s) ===")
    for v in variants: print(f"  {v:<14} {result['enc_time_mean_s'][v]:.3f}s")

    json.dump(result, open(os.path.join(REPO, "artifacts", "wer_clips", "int8_eval_results.json"), "w"),
              ensure_ascii=False, indent=2)
    return 0

if __name__ == "__main__":
    sys.exit(main())
