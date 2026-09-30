#!/usr/bin/env python3
"""Per-clip host ONNX reference for a Whisper decode gate: encoder hidden + greedy token ids.

Writes <out>/<clip>.npz with `enc` [1500, d] f32 (encoder_model.onnx) and `ids` (prompt + greedy
tokens from decoder_model/decoder_with_past, the shipped host decode). The device side
(designs/decode_fused/verify_whisper_rail.py) runs in the IRON venv, which has no onnxruntime.

  .venv-export/bin/python scripts/whisper_host_decode_ref.py --model whisper-small \
      --clips artifacts/wer_clips --out $XDNA_CACHE/whisper-rail/ref
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_wer_from_hidden import greedy_decode_kv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="whisper-small")
    ap.add_argument("--clips", default="artifacts/wer_clips")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    onnx_dir = Path("artifacts") / a.model / "onnx"
    proc = WhisperProcessor.from_pretrained(str(onnx_dir))
    tok = proc.tokenizer
    ids_of = tok.convert_tokens_to_ids
    so, cpu = ort.SessionOptions(), ["CPUExecutionProvider"]
    enc_sess = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"), so, providers=cpu)
    dec = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"), so, providers=cpu)
    dec_past = ort.InferenceSession(str(onnx_dir / "decoder_with_past_model.onnx"), so,
                                    providers=cpu)
    n_layers = (len(dec.get_outputs()) - 1) // 4
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = json.load(open(Path(a.clips) / "refs.json", encoding="utf-8"))
    for name in sorted(refs):
        wav, sr = sf.read(Path(a.clips) / name)
        wav = np.asarray(wav[:, 0] if wav.ndim > 1 else wav, np.float32)
        feats = proc.feature_extractor(wav, sampling_rate=sr, return_tensors="np").input_features
        enc = enc_sess.run(None, {"input_features": feats.astype(np.float32)})[0][0]
        lang = ids_of("<|ru|>" if name.startswith("ru") else "<|en|>")
        start = [ids_of("<|startoftranscript|>"), lang, ids_of("<|transcribe|>"),
                 ids_of("<|notimestamps|>")]
        ids = greedy_decode_kv(dec, dec_past, n_layers, enc[None], start,
                               ids_of("<|endoftext|>"))
        np.savez(out / f"{Path(name).stem}.npz", enc=enc.astype(np.float32),
                 ids=np.asarray(ids, np.int64), n_prompt=len(start))
        print(f"{name}: {len(ids) - len(start)} tokens: {tok.decode(ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
