#!/usr/bin/env python3
"""Encode a Whisper mel clip set through one CPU encoder VARIANT, for the
big-narrow-format-vs-small-bf16 WER experiment (lane-c-whisper-q).

Variants:
  fp32          onnxruntime CPU, artifacts/<model>/onnx/encoder_model.onnx (unmodified export).
  int8          onnxruntime CPU, artifacts/<model>/onnx_quant/encoder_model_int8.onnx (real ORT
                dynamic per-channel int8 PTQ, produced by whisper_encoder_int8_quantize.py). This
                actually EXECUTES int8xint8->int32 matmuls on CPU -- not a simulation.
  sim_int4_gN   PyTorch (HF WhisperForConditionalGeneration.encoder) with every Linear's weight
                fake-quantized: symmetric per-output-row, per-group-of-N-along-in_features,
                scale = max|W|/7 (4-bit signed, no zero-point -- same rule as the shipped
                int8xint4 dequant brick's int8 sibling, ctx2.rs quant_scale/quant_i8, just at 4
                bits), then DEQUANTIZED BACK TO FP32 before the forward pass. This reproduces the
                exact numerical effect of quantize->dequant-before-matmul (the brick's own
                correctness gate says the dequant step reproduces its input bit-for-bit; the only
                lossy step is the quant round-trip, which is what this simulates) WITHOUT a real
                int4 CPU or NPU kernel, which does not exist for this shape. NOT a hardware run:
                no on-chip rounding beyond the modelled quant/dequant, CPU fp32 matmul throughout.
  sim_int8_gN   Same PyTorch fake-quant path at int8 (scale = max|W|/127), group size N (0 = one
                scale per whole row, matching ctx2.rs's per-column weight scale). Cross-check
                against the real ORT `int8` variant above -- if they roughly agree, the simulation
                methodology is validated.
  sim_bf16      PyTorch, every Linear's weight rounded to bf16 (round-to-nearest-even, IEEE bf16
                mantissa truncation via `.to(torch.bfloat16)`) then cast back to fp32 before the
                forward pass -- a real, exact bf16 round-trip, not an approximation. This is the
                CPU stand-in for the blocked turbo@bf16-NPU arm and, here, for whisper-small@bf16
                run on CPU instead of the (queued-out, single-tenant) NPU -- same harness, same
                200-clip set, so it is directly comparable to the other sim_*/onnx arms. NOTE: the
                real NPU bf16 path may round differently (this project's own doctrine records the
                on-chip bfp16 path inheriting FLOOR rounding, not round-to-nearest) -- this is the
                textbook-bf16 quality estimate, not a bit-exact stand-in for the device.

Usage:
  python scripts/whisper_encode_cpu_variant.py --model whisper-turbo --variant fp32 \
      --mels artifacts/whisper-turbo/mels_wer_clips_large --out artifacts/whisper-turbo/enc_fp32_large
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_HF = {
    "whisper-small": "openai/whisper-small",
    "whisper-turbo": "openai/whisper-large-v3-turbo",
}


def onnx_variant_path(model, variant):
    if variant == "fp32":
        return Path("artifacts") / model / "onnx" / "encoder_model.onnx"
    if variant == "int8":
        return Path("artifacts") / model / "onnx_quant" / "encoder_model_int8.onnx"
    raise ValueError(variant)


def run_onnx(args, mel_files):
    import onnxruntime as ort

    p = onnx_variant_path(args.model, args.variant)
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    times = {}
    for f in mel_files:
        mel = np.load(f).astype(np.float32)  # [1,n_mels,3000]
        t0 = time.time()
        out = sess.run(["last_hidden_state"], {"input_features": mel})[0]  # [1,1500,d]
        dt = time.time() - t0
        stem = f.stem
        np.save(Path(args.out) / f"{stem}.npy", out[0])  # [1500,d]
        times[stem] = dt
        print(f"[{args.variant}] {stem}  {dt:.3f}s", flush=True)
    return times, str(p)


def quantize_dequantize_row(w, bits, group):
    """w: 2D torch tensor [out,in]. Per-row (out channel), per-group-of-`group`-along-`in`
    symmetric quant/dequant. group<=0 means one group per row (the whole `in` axis)."""
    import torch

    qmax = (1 << (bits - 1)) - 1  # 127 for int8, 7 for int4
    out_f, in_f = w.shape
    g = in_f if group <= 0 else group
    n_groups = (in_f + g - 1) // g
    wq = w.clone()
    for gi in range(n_groups):
        lo, hi = gi * g, min((gi + 1) * g, in_f)
        chunk = w[:, lo:hi]
        scale = chunk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
        q = torch.clamp(torch.round(chunk / scale), -qmax - 1, qmax)
        wq[:, lo:hi] = q * scale
    return wq


def run_sim(args, mel_files):
    import torch
    from transformers import WhisperForConditionalGeneration

    is_bf16 = args.variant == "sim_bf16"
    if not is_bf16:
        bits = 4 if args.variant.startswith("sim_int4") else 8
        group = int(args.variant.rsplit("_g", 1)[1])
    else:
        bits = group = None

    t0 = time.time()
    m = WhisperForConditionalGeneration.from_pretrained(MODEL_HF[args.model], dtype=torch.float32)
    enc = m.model.encoder.eval()
    n_lin = 0
    with torch.no_grad():
        for name, mod in enc.named_modules():
            if isinstance(mod, torch.nn.Linear):
                if is_bf16:
                    mod.weight.copy_(mod.weight.to(torch.bfloat16).to(torch.float32))
                else:
                    mod.weight.copy_(quantize_dequantize_row(mod.weight, bits, group))
                n_lin += 1
    print(f"[{args.variant}] quantized {n_lin} Linear layers in {time.time()-t0:.1f}s", flush=True)

    times = {}
    bs = args.batch
    with torch.no_grad():
        for i in range(0, len(mel_files), bs):
            batch = mel_files[i : i + bs]
            mels = np.stack([np.load(f).astype(np.float32)[0] for f in batch])  # [B,n_mels,3000]
            x = torch.from_numpy(mels)
            t0 = time.time()
            out = enc(x).last_hidden_state  # [B,1500,d]
            dt = time.time() - t0
            per = dt / len(batch)
            out_np = out.numpy()
            for j, f in enumerate(batch):
                stem = f.stem
                np.save(Path(args.out) / f"{stem}.npy", out_np[j])
                times[stem] = per
            print(f"[{args.variant}] batch {i}-{i+len(batch)-1}  {dt:.2f}s ({per:.3f}s/clip)", flush=True)
    src = "pytorch-sim bf16 round-to-nearest-even" if is_bf16 else f"pytorch-sim bits={bits} group={group}"
    return times, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_HF))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--mels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None,
                     help="cap CPU threads (this project's half-the-cores rule); applies to "
                          "both torch and onnxruntime intra-op parallelism")
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    mel_files = sorted(Path(args.mels).glob("*.npy"))
    if args.limit:
        mel_files = mel_files[: args.limit]
    print(f"{len(mel_files)} mel files from {args.mels}")

    if args.variant in ("fp32", "int8"):
        times, src = run_onnx(args, mel_files)
    elif args.variant.startswith("sim_int4") or args.variant.startswith("sim_int8") or args.variant == "sim_bf16":
        times, src = run_sim(args, mel_files)
    else:
        raise SystemExit(f"unknown variant {args.variant!r}")

    meta = {
        "model": args.model,
        "variant": args.variant,
        "source": src,
        "n": len(times),
        "mean_s": sum(times.values()) / len(times) if times else None,
        "times": times,
    }
    json.dump(meta, open(Path(args.out) / "_meta.json", "w"), indent=2)
    print(f"wrote {len(times)} hidden states -> {args.out}  mean={meta['mean_s']}")


if __name__ == "__main__":
    main()
