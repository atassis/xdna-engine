#!/usr/bin/env python3
"""Dump pyannote's own outputs for a clip, as the oracle for verify_pyannote.

Writes artifacts/pyannote/ref/<stem>/:
  segmentation.npy  [n_windows, n_frames, 7] raw powerset logits
  reference.rttm    the shipped pipeline's final answer
  summary.json      speaker count, per-speaker total speech, overlap seconds

Parity is asserted at the SEGMENTATION boundary first: a clustering mismatch and an embedding
mismatch look identical in the final RTTM, so comparing only the RTTM cannot tell you which stage
is wrong.

Run: HF_TOKEN=... .venv-pyannote/bin/python scripts/verify_pyannote_reference.py <clip.wav>
"""
import json, os, sys
import numpy as np, torch, torchaudio
from pyannote.audio import Model, Pipeline

wav_path = sys.argv[1]
stem = os.path.splitext(os.path.basename(wav_path))[0]
out = f"artifacts/pyannote/ref/{stem}"
os.makedirs(out, exist_ok=True)
token = os.environ["HF_TOKEN"]

wav, sr = torchaudio.load(wav_path)
assert sr == 16000 and wav.shape[0] == 1, f"need 16k mono, got {sr} Hz {wav.shape[0]}ch"

# --- stage 1: raw segmentation logits, windowed exactly as the pipeline windows them -----------
seg = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token).eval()
dur = int(seg.specifications.duration * sr)
hop = int(0.1 * seg.specifications.duration * sr)     # SpeakerDiarization.__init__ default
starts = list(range(0, max(1, wav.shape[1] - dur + 1), hop)) or [0]
batch = torch.stack([
    torch.nn.functional.pad(wav[0, s:s + dur], (0, max(0, dur - len(wav[0, s:s + dur]))))
    for s in starts]).unsqueeze(1)
with torch.no_grad():
    logits = seg(batch).numpy().astype(np.float32)
np.save(f"{out}/segmentation.npy", logits)
print("segmentation:", logits.shape, f"({len(starts)} windows, hop {hop/sr:.2f}s)")

# --- stage 2: the shipped pipeline's final answer ----------------------------------------------
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
ann = pipe(wav_path)
with open(f"{out}/reference.rttm", "w") as f:
    ann.write_rttm(f)

spk = {}
for seg_, _, label in ann.itertracks(yield_label=True):
    spk.setdefault(label, 0.0)
    spk[label] += seg_.duration
overlap_s = 0.0
tl = list(ann.itertracks(yield_label=True))
for i in range(len(tl)):
    for j in range(i + 1, len(tl)):
        if tl[i][2] == tl[j][2]:
            continue
        a, b = tl[i][0], tl[j][0]
        o = min(a.end, b.end) - max(a.start, b.start)
        if o > 0:
            overlap_s += o

summary = {
    "clip": wav_path,
    "n_speakers": len(spk),
    "speech_s_per_speaker": {k: round(v, 3) for k, v in sorted(spk.items())},
    "overlap_s": round(overlap_s, 3),
    "segments": [
        {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": l}
        for s, _, l in sorted(tl, key=lambda t: t[0].start)],
}
with open(f"{out}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
