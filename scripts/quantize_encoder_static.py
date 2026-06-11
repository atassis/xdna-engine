#!/usr/bin/env python3
"""Static int8 quantization of the GigaAM-v3 encoder with mel-feature calibration.

Calibration set = mel features ([1,64,1600], padded) from a handful of RU clips,
produced by onnx-asr's own preprocessor (so the distribution matches inference).
"""
import glob, os, sys, wave
import numpy as np
import onnx_asr
from onnxruntime.quantization import (CalibrationDataReader, QuantType,
                                      QuantFormat, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--istupakov--gigaam-v3-onnx/snapshots/*"))[0]
WIN = 1600
CALIB_CLIPS = ["ru_01.wav", "ru_03.wav", "ru_05.wav", "ru_07.wav",
               "ru_09.wav", "ru_11.wav", "ru_13.wav", "en_01.wav"]

def read_wav(p):
    with wave.open(p, "rb") as w:
        raw = w.readframes(w.getnframes()); ch = w.getnchannels()
    x = np.frombuffer(raw, np.int16).astype(np.float32)/32768.0
    return x.reshape(-1, ch).mean(1) if ch > 1 else x

class MelReader(CalibrationDataReader):
    def __init__(self, ASR, clips):
        self.data = []
        for c in clips:
            wav = read_wav(os.path.join(REPO, "artifacts", "wer_clips", c))
            feats, _ = ASR._preprocessor(wav[None, :].astype(np.float32),
                                         np.array([wav.shape[0]], np.int64))
            T = min(feats.shape[2], WIN)
            buf = np.zeros((1, 64, WIN), np.float32); buf[0, :, :T] = feats[0, :, :T]
            self.data.append({"audio_signal": buf, "length": np.array([T], np.int64)})
        self.it = iter(self.data)
    def get_next(self):
        return next(self.it, None)

def main():
    src = os.path.join(REPO, "models", "gigaam_v3_encoder_static.onnx")
    pre = os.path.join(REPO, "models", "quant", "_enc_preproc.onnx")
    dst = os.path.join(REPO, "models", "quant", "gigaam_v3_encoder_int8_static.onnx")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    print("[static] pre-processing (shape inference)...", file=sys.stderr)
    quant_pre_process(src, pre, skip_symbolic_shape=False)

    ASR = onnx_asr.load_model("gigaam-v3-rnnt", path=SNAP).asr
    reader = MelReader(ASR, CALIB_CLIPS)
    print(f"[static] calibrating on {len(reader.data)} clips, QDQ int8...", file=sys.stderr)
    quantize_static(
        pre, dst, reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,
    )
    print("[static] done ->", dst, round(os.path.getsize(dst)/1e6, 1), "MB", file=sys.stderr)

if __name__ == "__main__":
    sys.exit(main())
