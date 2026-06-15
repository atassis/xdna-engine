#!/usr/bin/env python3
"""Part 1 (P3): dump whisper-small log-mel features for all 17 FLEURS WER clips.

For each clip in artifacts/wer_clips/*.wav, WhisperProcessor pads to a fixed 3000
frames -> input_features [1,80,3000]. Saved to artifacts/whisper-small/mels/<name>.npy.
The NPU encoder bin then consumes these (squeezed to [80,3000])."""
import numpy as np, soundfile as sf
from pathlib import Path
from transformers import WhisperProcessor

CLIPS = Path("artifacts/wer_clips")
OUT = Path("artifacts/whisper-small/mels")
OUT.mkdir(parents=True, exist_ok=True)

proc = WhisperProcessor.from_pretrained("openai/whisper-small")

for wavp in sorted(CLIPS.glob("*.wav")):
    wav, sr = sf.read(wavp)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = np.asarray(wav, dtype=np.float32)
    feats = proc(wav, sampling_rate=16000).input_features  # [1,80,3000]
    feats = np.asarray(feats, dtype=np.float32)
    assert feats.shape == (1, 80, 3000), f"{wavp.name}: {feats.shape}"
    np.save(OUT / f"{wavp.stem}.npy", feats)
    print(f"{wavp.stem}: {feats.shape}  sr={sr}")

print(f"\nwrote {len(list(OUT.glob('*.npy')))} mels -> {OUT}")
