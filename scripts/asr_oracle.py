#!/usr/bin/env python3
"""Ground-truth oracle + model export for the GigaAM-v3 RNNT ASR pipeline (Task 2).

Runs the reference onnx-asr pipeline on a real wav and dumps everything the Rust service
needs to (a) be built against and (b) be validated:
  - copies the 3 ONNX graphs (mel preprocessor, RNNT decoder, RNNT joint) + vocab into
    artifacts/asr/  (these are what the Rust `ort`-based service loads)
  - dumps reference tensors into artifacts/asr_ref/:  waveform, features [1,64,T],
    encoder_out [1,T',768] (+len), the decoded token ids, and the ground-truth text.

The Rust decode is validated by feeding it ref_encoder_out and matching ref_token_ids; the
full Rust pipeline (our NPU encoder) is scored by WER vs ref_text.

Run with the venv that has onnx_asr (py3.12):
  ~/npuvox-asr-bench/.venv/bin/python scripts/asr_oracle.py [--wav PATH]
"""
import argparse, glob, os, shutil, sys, wave
import numpy as np

HUB = os.path.expanduser("~/.cache/huggingface/hub")
SNAP = glob.glob(f"{HUB}/models--istupakov--gigaam-v3-onnx/snapshots/*")[0]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASR = os.path.join(REPO, "artifacts", "asr")
REF = os.path.join(REPO, "artifacts", "asr_ref")


def read_wav_16k(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.expanduser(
        "~/voxd/tests/fixtures/sample-ru-en.wav"))
    a = ap.parse_args()

    import onnx_asr
    os.makedirs(ASR, exist_ok=True)
    os.makedirs(REF, exist_ok=True)

    # 1. export the model files the Rust service loads
    pre = glob.glob(os.path.dirname(onnx_asr.__file__) + "/preprocessors/data/gigaam_v3.onnx")[0]
    shutil.copy(pre, os.path.join(ASR, "preprocessor.onnx"))
    for src, dst in [("v3_rnnt_decoder.onnx", "decoder.onnx"),
                     ("v3_rnnt_joint.onnx", "joint.onnx"),
                     ("v3_vocab.txt", "vocab.txt")]:
        shutil.copy(os.path.join(SNAP, src), os.path.join(ASR, dst))
    print(f"[export] copied preprocessor/decoder/joint/vocab -> {ASR}")

    # 2. load reference model + run the pipeline, capturing intermediates
    model = onnx_asr.load_model("gigaam-v3-rnnt", path=SNAP)
    wave_f = read_wav_16k(a.wav).astype(np.float32)
    n = wave_f.shape[0]
    print(f"[wav] {a.wav}  {n} samples = {n/16000:.2f} s")

    waveforms = wave_f[None, :]
    wav_lens = np.array([n], np.int64)
    features, feat_lens = model.asr._preprocessor(waveforms, wav_lens)   # [1,64,T]
    enc_out, enc_lens = model.asr._encode(features, feat_lens)            # [1,T',768], int64
    ids = None
    for tok_ids, ts, lp in model.asr._decoding(enc_out, enc_lens):
        ids = list(int(x) for x in tok_ids)
    text = model.asr._decode_tokens(ids, None, None).text

    # also the straight public transcription (sanity)
    pub = model.recognize(wave_f)
    print(f"[oracle] internal text : {text!r}")
    print(f"[oracle] public  text : {pub!r}")

    # 3. dump reference tensors
    np.save(f"{REF}/waveform.npy", wave_f)
    np.save(f"{REF}/features.npy", features.astype(np.float32))
    np.save(f"{REF}/features_lens.npy", feat_lens.astype(np.int64))
    np.save(f"{REF}/encoder_out.npy", enc_out.astype(np.float32))
    np.save(f"{REF}/encoder_out_lens.npy", enc_lens.astype(np.int64))
    np.save(f"{REF}/token_ids.npy", np.array(ids, np.int64))
    open(f"{REF}/text.txt", "w").write(text)
    print(f"[ref] features {features.shape} enc_out {enc_out.shape} enc_len {int(enc_lens[0])} "
          f"tokens {len(ids)} -> {REF}")
    print(f"[ref] token_ids = {ids}")


if __name__ == "__main__":
    sys.exit(main())
