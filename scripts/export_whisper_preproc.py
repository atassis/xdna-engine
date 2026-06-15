#!/usr/bin/env python
"""Export a Whisper log-mel front-end to ONNX (Plan A5).

waveform[1, N] (f32, 16 kHz mono, already /32768 to [-1,1]) -> input_features[1, 80, 3000].

The torch module mirrors `transformers.WhisperFeatureExtractor`:
  - pad / truncate the waveform to n_samples = 480000 (30 s)
  - STFT: n_fft=400, hop=160, Hann window, center-padded (reflect) -> [201, 3000] frames
  - power spectrogram |STFT|^2, drop the last frame (HF uses [..., :-1])
  - apply the 80-bin mel filterbank from `WhisperFeatureExtractor.mel_filters`
  - log10, then clamp to (log_max - 8.0), then normalize (log_spec + 4.0) / 4.0

Verifies against `WhisperProcessor(...).input_features` on en_01.wav (rel target < 5e-2).

Usage:  python scripts/export_whisper_preproc.py
Writes: artifacts/whisper-small/preprocessor.onnx
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "artifacts" / "whisper-small" / "onnx"
OUT = ROOT / "artifacts" / "whisper-small" / "preprocessor.onnx"

N_FFT = 400
HOP = 160
N_MELS = 80
N_SAMPLES = 480000      # 30 s @ 16 kHz
N_FRAMES = 3000


class WhisperPreproc(nn.Module):
    """Static log-mel front-end matched to WhisperFeatureExtractor."""

    def __init__(self, mel_filters: np.ndarray):
        super().__init__()
        n_freqs = N_FFT // 2 + 1                 # 201
        # Hann window (periodic, like torch.hann_window default)
        window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float32)  # [400]
        # Explicit DFT matrices so the whole thing is plain conv/matmul (torch.stft does not
        # trace cleanly through the legacy ONNX exporter). cos/sin basis: [n_freqs, n_fft].
        k = torch.arange(n_freqs).unsqueeze(1)   # [201,1]
        nidx = torch.arange(N_FFT).unsqueeze(0)  # [1,400]
        ang = 2.0 * np.pi * k * nidx / N_FFT     # [201,400]
        # conv1d weight layout: [out_channels=n_freqs, in_channels=1, kernel=n_fft]
        cos_w = (torch.cos(ang) * window).unsqueeze(1).float()   # [201,1,400]
        sin_w = (-torch.sin(ang) * window).unsqueeze(1).float()  # [201,1,400]
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)
        # mel_filters from HF: shape [n_freqs=201, n_mels=80]; we want [80, 201] to matmul
        mf = torch.from_numpy(np.ascontiguousarray(mel_filters.T)).float()  # [80, 201]
        self.register_buffer("mel_filters", mf)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, N_SAMPLES] (caller pads/truncates to exactly 480000)
        # center=True reflect pad by n_fft//2 on each side
        wav = waveform.unsqueeze(0)              # [1,1,N_SAMPLES]
        wav = torch.nn.functional.pad(wav, (N_FFT // 2, N_FFT // 2), mode="reflect")
        # framed DFT via conv1d, stride = hop -> [1, n_freqs, n_frames+1]
        real = torch.nn.functional.conv1d(wav, self.cos_w, stride=HOP)[0]  # [201, 3001]
        imag = torch.nn.functional.conv1d(wav, self.sin_w, stride=HOP)[0]  # [201, 3001]
        power = real * real + imag * imag        # |STFT|^2
        magnitudes = power[:, :-1]               # [201, 3000]  (HF drops last frame)
        mel_spec = self.mel_filters @ magnitudes  # [80, 3000]
        log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        # clamp to within 8 dB of the per-utterance max, then normalize
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.unsqueeze(0)             # [1, 80, 3000]


def main():
    from transformers import WhisperProcessor

    proc = WhisperProcessor.from_pretrained(str(PROC_DIR))
    fe = proc.feature_extractor
    mel_filters = np.asarray(fe.mel_filters)    # [201, 80]
    print(f"mel_filters shape {mel_filters.shape}")

    module = WhisperPreproc(mel_filters).eval()

    dummy = torch.zeros(1, N_SAMPLES, dtype=torch.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (dummy,),
        str(OUT),
        input_names=["waveform"],
        output_names=["input_features"],
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {OUT}")

    # ---- verify against the real processor on en_01.wav ----
    import wave

    wav_path = ROOT / "artifacts" / "wer_clips" / "en_01.wav"
    with wave.open(str(wav_path), "rb") as w:
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    assert sr == 16000, f"expected 16k, got {sr}"

    ref = proc(pcm, sampling_rate=16000, return_tensors="np").input_features  # [1,80,3000]

    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    # caller-side pad/truncate to exactly N_SAMPLES (the ONNX graph is fixed-shape)
    padded = np.zeros(N_SAMPLES, dtype=np.float32)
    m = min(len(pcm), N_SAMPLES)
    padded[:m] = pcm[:m]
    got = sess.run(["input_features"], {"waveform": padded[None, :]})[0]

    print("ref shape", ref.shape, "got shape", got.shape)
    denom = np.linalg.norm(ref)
    rel = np.linalg.norm(got - ref) / max(denom, 1e-8)
    print(f"rel error vs WhisperProcessor.input_features: {rel:.4e}")
    if rel < 5e-2:
        print("OK: rel < 5e-2")
    else:
        print("WARN: rel >= 5e-2 (verify end-to-end decode is sane)", file=sys.stderr)


if __name__ == "__main__":
    main()
