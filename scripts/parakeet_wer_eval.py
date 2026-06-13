#!/usr/bin/env python3
"""Phase-0 CPU oracle: Parakeet-tdt-0.6b-v3 (multilingual RU+EN) WER on the wer_clips set.

This is the "the engine can finally do English" measurement. It runs the cached
istupakov/parakeet-tdt-0.6b-v3-onnx model entirely on CPU via onnx-asr (pure
onnxruntime decode, no NPU) over the same 13 RU + 4 EN FLEURS clips + refs.json
that int8_wer_eval.py used for GigaAM, so the numbers are directly comparable to
the GigaAM baseline (RU ~11% / EN 100%).

WER normalization + Levenshtein are copied verbatim from scripts/int8_wer_eval.py
so RU numbers are apples-to-apples with the GigaAM eval.

Usage:
  ~/npuvox-asr-bench/.venv/bin/python scripts/parakeet_wer_eval.py
"""
import argparse, json, os, re, sys, time, unicodedata
import onnx_asr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "nemo-parakeet-tdt-0.6b-v3"

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=os.path.join(REPO, "artifacts", "wer_clips"))
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "wer_clips", "parakeet_eval_results.json"))
    a = ap.parse_args()

    print(f"[load] {MODEL} (CPU)", file=sys.stderr)
    model = onnx_asr.load_model(MODEL, providers=["CPUExecutionProvider"])

    refs = json.load(open(os.path.join(a.clips, "refs.json"), encoding="utf-8"))
    names = sorted(refs.keys())

    rows, times = [], []
    for name in names:
        wavp = os.path.join(a.clips, name)
        if not os.path.isfile(wavp):
            print(f"[skip] {name} (missing)", file=sys.stderr); continue
        t0 = time.time()
        hyp = model.recognize(wavp, sample_rate=16000)
        dt = time.time() - t0
        times.append(dt)
        ref = normalize(refs[name])
        hyp_n = normalize(hyp if isinstance(hyp, str) else hyp[0])
        w, nref = wer(ref, hyp_n)
        rows.append({"name": name, "lang": name.split("_")[0], "n_ref": nref,
                     "wer": w, "ref": ref, "hyp": hyp_n, "raw_hyp": hyp, "t_s": dt})
        print(f"[done] {name}  WER={w*100:5.1f}%  ({dt:.2f}s)", file=sys.stderr)

    def agg(langs):
        sub = [r for r in rows if r["lang"] in langs]
        if not sub: return None
        # micro-average: total edits / total ref words (matches int8 eval's per-clip mean? no -> macro)
        macro = sum(r["wer"] for r in sub) / len(sub)
        return {"n": len(sub), "wer_macro": macro}

    result = {"model": MODEL, "per_clip": rows,
              "agg": {"RU": agg({"ru"}), "EN": agg({"en"}), "ALL": agg({"ru", "en"})},
              "t_mean_s": sum(times)/len(times) if times else None}

    print("\n=== per-clip WER vs reference ===")
    print(f"{'clip':<10}{'refW':>5}  {'WER':>7}  {'t(s)':>6}")
    print("-"*32)
    for r in rows:
        print(f"{r['name']:<10}{r['n_ref']:>5}  {r['wer']*100:6.1f}%  {r['t_s']:6.2f}")
    print("\n=== aggregate WER (macro avg over clips) ===")
    for lbl in ("RU", "EN", "ALL"):
        d = result["agg"][lbl]
        if d: print(f"{lbl:<5}(n={d['n']:>2})  WER = {d['wer_macro']*100:5.1f}%")
    print(f"\nmean recognize() wall time: {result['t_mean_s']:.2f}s/clip (CPU)")

    json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n[wrote] {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
