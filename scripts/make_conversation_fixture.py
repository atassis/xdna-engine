#!/usr/bin/env python3
"""Build a LONG two-speaker conversation, for testing the clustering stage.

The short overlap fixture fits inside one 10 s segmentation window, and with a single window
clustering is degenerate: the segmentation's local speaker slots already are the answer, and
clustering only exists to link local speakers ACROSS windows. Measured on that fixture, pyannote's
own VBx returns a single cluster and the two speakers survive anyway -- so it cannot tell a working
clusterer from a broken one.

This alternates two voices (each reused so the speaker identity is real, not merely a different
file) over ~45 s with some overlap, giving tens of windows and a genuine cross-window linking
problem.

Run: python3 scripts/make_conversation_fixture.py
"""
import json, os, wave
import numpy as np

OUT = "artifacts/pyannote/fixtures"
os.makedirs(OUT, exist_ok=True)
SR = 16000
A_SRC, A_SPEECH = "artifacts/wer_clips/en_01.wav", (1.07, 5.57)
B_SRC, B_SPEECH = "artifacts/wer_clips/ru_02.wav", (1.83, 6.04)

def read(p):
    with wave.open(p) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2), p
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

def speech_of(x, lo, hi):
    return x[int(lo * SR):int(hi * SR)]

a = speech_of(read(A_SRC), *A_SPEECH)
b = speech_of(read(B_SRC), *B_SPEECH)
b *= float(np.sqrt((a**2).mean())) / max(float(np.sqrt((b**2).mean())), 1e-6)   # match loudness

# turns: (speaker, start_s). Two turns deliberately overlap the previous one.
turns = [("A", 0.5), ("B", 5.4), ("A", 11.0), ("B", 15.2), ("A", 21.5),
         ("B", 25.4), ("A", 31.0), ("B", 35.0), ("A", 40.5)]
total = int((turns[-1][1] + len(a) / SR + 1.0) * SR)
mix = np.zeros(total, dtype=np.float32)
truth = []
for spk, t0 in turns:
    src = a if spk == "A" else b
    i = int(t0 * SR)
    mix[i:i + len(src)] += src
    truth.append({"speaker": spk, "start_s": round(t0, 3), "end_s": round(t0 + len(src) / SR, 3)})

peak = np.abs(mix).max()
if peak > 32767.0:
    mix *= 32767.0 / peak
wav_path = f"{OUT}/conversation_2spk.wav"
with wave.open(wav_path, "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(mix.astype(np.int16).tobytes())

meta = {"sample_rate": SR, "duration_s": round(total / SR, 3),
        "sources": {"A": A_SRC, "B": B_SRC}, "expect_speakers": 2, "turns": truth}
with open(f"{OUT}/conversation_2spk.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"{wav_path}  {meta['duration_s']}s, {len(turns)} turns, 2 speakers")
for t in truth: print(f"  {t['speaker']} {t['start_s']:>6.2f} - {t['end_s']:>6.2f}")
