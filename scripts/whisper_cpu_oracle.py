#!/usr/bin/env python3
"""whisper-small full-CPU WER oracle over the 17 FLEURS clips. Reference for P3."""
import json, re, unicodedata, os, numpy as np, soundfile as sf
from pathlib import Path
from transformers import pipeline
CLIPS = Path("artifacts/wer_clips"); MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-small")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE); _WS = re.compile(r"\s+")
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
asr = pipeline("automatic-speech-recognition", model=MODEL, device=-1)
refs = json.load(open(CLIPS/"refs.json")); out = {}
for name in sorted(refs):
    wav, sr = sf.read(CLIPS/name)
    if wav.ndim > 1:            # take channel 0 if stereo
        wav = wav[:, 0]
    wav = np.asarray(wav, dtype=np.float32)
    lang = "russian" if name.startswith("ru") else "english"
    hyp = asr({"raw": wav, "sampling_rate": sr},
              generate_kwargs={"language": lang, "task": "transcribe"})["text"]
    w, n = wer(normalize(refs[name]), normalize(hyp))
    out[name] = {"hyp": hyp, "wer": w, "nref": n}
    print(f"{name}: WER={w:.3f}")
json.dump(out, open(CLIPS/"whisper_small_oracle.json","w"), indent=2, ensure_ascii=False)
for split in ("ru","en"):
    ws = [v["wer"] for k,v in out.items() if k.startswith(split)]
    print(f"{split} mean WER = {sum(ws)/len(ws):.3f}")
