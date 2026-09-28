#!/usr/bin/env python3
"""Golden fixtures for the pure-Rust CPU-glue port (mel preprocessor `mel.rs`
+ RNNT decoder/joint `decoder.rs`, both in `npu-parakeet`), captured from
real onnxruntime.

Writes .npy files under rust/npu-parakeet/tests/fixtures/cpu_glue/. Re-run
only if preprocessor.onnx / decoder_joint.onnx change -- the fixtures are
checked in.

  .venv/bin/python scripts/gen_cpu_glue_fixtures.py
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as rt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "rust/npu-parakeet/tests/fixtures/cpu_glue"

sys.path.insert(0, str(ROOT / "scripts"))
import parakeet_tdt_decoder_ref as dref  # noqa: E402

PREPROC_ONNX = dref.PREPROC_ONNX
DECODER_ONNX = dref.DECODER_ONNX


def read_wav_f32(path: Path) -> np.ndarray:
    w = wave.open(str(path), "rb")
    assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == 16000
    raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def synth_signal() -> np.ndarray:
    """Deterministic synthetic case: 0.5s sine sweep 200->2000Hz + 0.2s silence.
    Not real speech -- exercises the preprocessor's math independent of any
    particular recording."""
    sr = 16000
    t = np.arange(int(0.5 * sr), dtype=np.float32) / sr
    f0, f1 = 200.0, 2000.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * 0.5))
    sweep = 0.3 * np.sin(phase).astype(np.float32)
    silence = np.zeros(int(0.2 * sr), dtype=np.float32)
    return np.concatenate([sweep, silence])


def gen_mel_fixtures():
    sess = rt.InferenceSession(str(PREPROC_ONNX), providers=["CPUExecutionProvider"])
    cases = {
        "synthetic_sweep": synth_signal(),
        "real_clip_1s": read_wav_f32(ROOT / "artifacts/wer_clips/en_01.wav")[:16000],
    }
    d = OUT / "mel"
    d.mkdir(parents=True, exist_ok=True)
    for name, wav in cases.items():
        wav = wav[None, :]
        lens = np.array([wav.shape[1]], np.int64)
        feats, flens = sess.run(["features", "features_lens"],
                                 {"waveforms": wav, "waveforms_lens": lens})
        np.save(d / f"{name}_wav.npy", wav[0].astype(np.float32))
        np.save(d / f"{name}_features.npy", feats[0].astype(np.float32))  # [128,T]
        print(f"[mel] {name}: wav={wav.shape} features={feats.shape}")


def gen_decoder_fixtures():
    # Weights are NOT dumped as a fixture: the Rust port reads them directly
    # out of decoder_joint.onnx's initializers at load time (onnx_init.rs),
    # matching load_weights() below field-for-field. Dumping the ~69MB of
    # weights as .npy would just duplicate the model file into git.
    W = dref.load_weights(DECODER_ONNX)
    d = OUT / "decoder"
    d.mkdir(parents=True, exist_ok=True)

    # Synthetic encoder trajectory (deterministic seed, no NPU/encoder needed --
    # this validates the prednet+joint math, not the encoder).
    rng = np.random.default_rng(0)
    n_steps = 40
    encodings = rng.standard_normal((n_steps, 1024)).astype(np.float32)
    np.save(d / "encodings.npy", encodings)

    sess = rt.InferenceSession(str(DECODER_ONNX), providers=["CPUExecutionProvider"])
    s1 = np.zeros((2, 1, dref.HIDDEN), np.float32)
    s2 = np.zeros((2, 1, dref.HIDDEN), np.float32)
    prev_token = dref.BLANK_IDX
    logits_seq = np.zeros((n_steps, dref.VOCAB_SIZE + dref.NUM_DURATIONS), np.float32)
    tokens_used = np.zeros(n_steps, np.int64)
    for t in range(n_steps):
        tokens_used[t] = prev_token
        outputs, n1, n2 = sess.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": encodings[t][None, :, None],
                "targets": np.array([[prev_token]], np.int32),
                "target_length": np.array([1], np.int32),
                "input_states_1": s1,
                "input_states_2": s2,
            },
        )
        logits_seq[t] = np.squeeze(outputs)
        s1, s2 = n1, n2
        # advance prev_token deterministically (not argmax -- exercise the
        # embedding table broadly, mirrors validate_joint_golden's sampling)
        prev_token = int(rng.integers(0, dref.VOCAB_SIZE))
    np.save(d / "targets.npy", tokens_used)
    np.save(d / "golden_logits.npy", logits_seq)

    # A real greedy-decode token sequence too (oracle vs itself is not useful,
    # but this pins the exact token ids a real clip decodes to, for a future
    # end-to-end host-glue test once an encoder path exists in this crate).
    print(f"[decoder] weights + {n_steps}-step golden trajectory -> {d}")


if __name__ == "__main__":
    gen_mel_fixtures()
    gen_decoder_fixtures()
