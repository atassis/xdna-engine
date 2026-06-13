#!/usr/bin/env python3
"""Find the CPU↔NPU encoder crossover vs audio length. The NPU's per-dispatch overhead is fixed
per op, so it amortizes as the sequence (post-÷8 T') grows. This builds mels at increasing lengths
(T' ≤ 512, the static window), times the onnx-asr CPU encoder on each, and dumps the mels for the
Rust NPU encoder (parakeet_encode_npu) to time. Compare the two tables to see where NPU overtakes CPU.

Usage:
  dump:  parakeet_crossover.py dump <mel_dir>      # build mels + print CPU encoder times
"""
import json, os, sys, time, wave
import numpy as np
import onnx_asr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = next((d for d in [os.path.join(REPO, "artifacts", "wer_clips"),
              "$REPO/artifacts/wer_clips"]
             if os.path.isfile(os.path.join(d, "refs.json"))), None)

def read_wav(p):
    with wave.open(p, "rb") as w:
        raw = w.readframes(w.getnframes()); ch = w.getnchannels()
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(1) if ch > 1 else x

def main():
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/parakeet_xover"
    os.makedirs(out_dir, exist_ok=True)
    asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=["CPUExecutionProvider"]).asr
    names = sorted(os.path.splitext(n)[0] for n in os.listdir(CLIPS) if n.endswith(".wav"))
    allwav = np.concatenate([read_wav(os.path.join(CLIPS, f"{n}.wav")) for n in names])  # ~100s pool

    targets_s = [6, 12, 20, 28, 33]  # seconds (T' grows ~12.5/s; keep T' <= 512 -> <=~40s)
    print(f"{'audio_s':>8} {'T_mel':>6} {'Tprime':>7} {'CPU_enc_ms':>11}")
    for secs in targets_s:
        n_samp = int(secs * 16000)
        wav = allwav[:n_samp] if n_samp <= len(allwav) else np.tile(allwav, 2)[:n_samp]
        feats, flen = asr._preprocessor(wav[None, :].astype(np.float32), np.array([wav.shape[0]], np.int64))
        # time CPU encoder (min of 3)
        ts = []
        for _ in range(3):
            t0 = time.time(); enc, lens = asr._encode(feats, flen); ts.append(time.time() - t0)
        tprime = int(lens[0])
        if tprime > 512:
            print(f"{secs:>8} {feats.shape[2]:>6} {tprime:>7}  (skip: T'>512 window)")
            continue
        np.save(os.path.join(out_dir, f"len{tprime}.npy"), feats[0].astype(np.float32))
        print(f"{secs:>8} {feats.shape[2]:>6} {tprime:>7} {min(ts)*1000:>11.0f}")
    print(f"\n[dumped mels to {out_dir}] now run the NPU encoder on it:")
    print(f"  flock /tmp/xdna2-npu.flock -c './rust/target/release/parakeet_encode_npu {out_dir} /tmp/xover_enc'")

if __name__ == "__main__":
    main()
