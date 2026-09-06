#!/usr/bin/env python3
"""Part 1 (P3): dump whisper log-mel features for a clip set.

For each clip in --clips/*.wav, WhisperProcessor pads to a fixed 3000 frames ->
input_features [1,n_mels,3000] (80 for whisper-small, 128 for whisper-turbo, read off
the checkpoint). Saved to artifacts/<model>/mels_<clipset>/<name>.npy. The NPU encoder
bin then consumes these (squeezed to [n_mels,3000]).

Usage: python scripts/whisper_dump_mels.py [--model whisper-small|whisper-turbo]
                                            [--clips artifacts/wer_clips]
"""
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import WhisperProcessor

MODEL_HF = {
    "whisper-small": "openai/whisper-small",
    "whisper-turbo": "openai/whisper-large-v3-turbo",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="whisper-small", choices=sorted(MODEL_HF))
    ap.add_argument("--clips", default="artifacts/wer_clips")
    args = ap.parse_args()

    clips = Path(args.clips)
    out = Path("artifacts") / args.model / f"mels_{clips.name}"
    out.mkdir(parents=True, exist_ok=True)

    proc = WhisperProcessor.from_pretrained(MODEL_HF[args.model])
    n_mels = proc.feature_extractor.feature_size

    n = 0
    for wavp in sorted(clips.glob("*.wav")):
        wav, sr = sf.read(wavp)
        if wav.ndim > 1:
            wav = wav[:, 0]
        wav = np.asarray(wav, dtype=np.float32)
        feats = proc(wav, sampling_rate=16000).input_features  # [1,n_mels,3000]
        feats = np.asarray(feats, dtype=np.float32)
        assert feats.shape == (1, n_mels, 3000), f"{wavp.name}: {feats.shape}"
        np.save(out / f"{wavp.stem}.npy", feats)
        n += 1

    print(f"wrote {n} mels ({n_mels}-bin) -> {out}")


if __name__ == "__main__":
    main()
