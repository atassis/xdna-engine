#!/usr/bin/env python3
"""Build a two-speaker fixture with genuinely OVERLAPPING SPEECH, for the diarization parity gate.

Every shipped clip is single-speaker, and a single-speaker clip cannot tell a correct
masked-pooling implementation from one that ignores the mask: the two agree everywhere except where
speakers overlap. So the gate needs a clip that actually contains overlap.

The obvious construction -- overlap the two FILES -- does not work, and a first version of this
script made exactly that mistake. Both clips carry leading and trailing silence, so overlapping the
files by 2 s produced ~0.03 s of overlapping SPEECH, and pyannote itself reported overlap_s = 0.0.
This aligns on the measured SPEECH bounds instead, and matches the two speakers' loudness over
their speech so neither is masked by the other.

    [ A alone ][ A + B overlapping ][ B alone ]

Run: python3 scripts/make_overlap_fixture.py
"""
import json, os, wave
import numpy as np

OUT = "artifacts/pyannote/fixtures"
os.makedirs(OUT, exist_ok=True)
SR = 16000

# Speech bounds MEASURED with `npu diarize` on each clip, not guessed from file length.
A = {"path": "artifacts/wer_clips/en_02.wav", "speech": (0.71, 7.16)}
B = {"path": "artifacts/wer_clips/ru_02.wav", "speech": (1.83, 6.04)}
B_SPEECH_STARTS_AT = 4.0        # where B's speech should begin, inside A's speech

def read(path):
    with wave.open(path) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2), path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

def rms(x, lo, hi):
    seg = x[int(lo * SR): int(hi * SR)]
    return float(np.sqrt((seg ** 2).mean())) if len(seg) else 1.0

a, b = read(A["path"]), read(B["path"])
# Match loudness over the SPEECH regions, so the quieter speaker is not simply buried.
b *= rms(a, *A["speech"]) / max(rms(b, *B["speech"]), 1e-6)

delta = B_SPEECH_STARTS_AT - B["speech"][0]          # shift applied to the whole B clip
off = int(round(delta * SR))
total = max(len(a), off + len(b))
mix = np.zeros(total, dtype=np.float32)
mix[: len(a)] += a
mix[off : off + len(b)] += b

peak = np.abs(mix).max()
if peak > 32767.0:
    mix *= 32767.0 / peak                            # scale once, preserving the A:B balance
pcm = mix.astype(np.int16)

a_speech = A["speech"]
b_speech = (B["speech"][0] + delta, B["speech"][1] + delta)
ov = (max(a_speech[0], b_speech[0]), min(a_speech[1], b_speech[1]))

wav_path = f"{OUT}/overlap_2spk.wav"
with wave.open(wav_path, "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())

truth = {
    "sample_rate": SR,
    "duration_s": round(total / SR, 3),
    "sources": {"A": A["path"], "B": B["path"]},
    "speaker_A_speech": [round(x, 3) for x in a_speech],
    "speaker_B_speech": [round(x, 3) for x in b_speech],
    "overlap_speech": [round(x, 3) for x in ov],
    "overlap_s": round(max(0.0, ov[1] - ov[0]), 3),
    "expect_speakers": 2,
    "note": "speech bounds measured with `npu diarize` per source clip, not file extents",
}
with open(f"{OUT}/overlap_2spk.json", "w") as f:
    json.dump(truth, f, indent=2)
print(json.dumps(truth, indent=2))
print(f"\nwrote {wav_path}")
