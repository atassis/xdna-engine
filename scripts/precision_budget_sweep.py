#!/usr/bin/env python3
"""CPU/simulator half of the per-op precision budget sweep (task mixed-precision-budget-sweep).

Runs the verified NumPy Parakeet encoder (parakeet_ref_encoder.py, block-for-block ONNX rel=3.1e-05)
over the real 17-clip WER corpus (artifacts/wer_mels) once per candidate arm, quantize-dequantizing
ONE op's activation stream (or the encoder weights) to a candidate format at every occurrence, and
scores each arm against the all-f32 pass with the same four-statistic gate as encoder_parity.py
(mean / worst-frame / worst-burst are computed here; new-burst needs a device-measured SHIPPED
baseline per-frame array, which this CPU harness does not have -- it is left as the on-device gate).

Candidate ops, in the order the task specifies (movement-floor ops first):
  resadd    the residual stream x after each of the 4 adds/block (bf16, int8-per-channel)
  affcast   the LayerNorm output immediately before each GEMM it feeds (bf16, int8-per-channel)
  fc1fc2    the silu(fc1) hidden activation, [T,4096] -- the largest single intermediate (bf16, int8)
  bfp16gemm the feed_forward1 GEMM operands (activation AND weight) quantized to block-fp16 (ebs8)
  wint8     every big matmul weight tensor, quantized int8 per-output-channel (scored on RSS, not ms)

Usage:
  scripts/precision_budget_sweep.py                      # full sweep, prints the per-op table
  scripts/precision_budget_sweep.py --json /tmp/sweep.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encoder_parity as ep          # frame_rel_err / clip_rel_l2 / worst_window / aggregate
import parakeet_ref_encoder as ref   # verified f32 block/mhsa/conv_module/subsample + weight loader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEL_DIR = os.path.join(REPO, "artifacts", "wer_mels")

# Weight tensors that actually ride a matmul tile (what a real int8/bfp16 kernel would touch).
# Excludes biases, LayerNorm gain/bias, pos_bias_u/v (tiny, kept f32) and the depthwise conv
# weight (per-channel [D,1,9], not a systolic-tile operand).
BIG_WEIGHTS = [
    "conv.pointwise_conv1.weight", "conv.pointwise_conv2.weight",
    "feed_forward1.linear1.weight", "feed_forward1.linear2.weight",
    "feed_forward2.linear1.weight", "feed_forward2.linear2.weight",
    "self_attn.linear_q.weight", "self_attn.linear_k.weight", "self_attn.linear_v.weight",
    "self_attn.linear_out.weight", "self_attn.linear_pos.weight",
]


# ---------------------------------------------------------------------------------------------
# Quantize-dequantize simulators. Each returns an f32 array of the SAME shape as its input, i.e.
# "if this tensor had been stored/computed in format X" rather than a packed representation --
# bytes are costed separately (bytes only need the shape+dtype, no device).
# ---------------------------------------------------------------------------------------------

def to_bf16(x):
    """Round-to-nearest-even truncation to 7 mantissa bits (the IEEE bf16 grid)."""
    u = np.asarray(x, np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + bias) & 0xFFFF0000).view(np.float32)


def to_int8_per_axis(x, axis):
    """Symmetric int8, one scale per index along `axis` (e.g. per output channel of a weight,
    or per feature of an activation) -- the granularity the KB's int8-cross-k-wer-* notes require;
    per-tensor is known to fail and is not offered here."""
    x64 = np.asarray(x, np.float64)
    other = tuple(a for a in range(x64.ndim) if a != axis)
    amax = np.max(np.abs(x64), axis=other, keepdims=True)
    scale = np.maximum(amax, 1e-12) / 127.0
    q = np.clip(np.round(x64 / scale), -127, 127)
    return (q * scale).astype(np.float32)


def to_bfp16(x, block=8, mantissa_bits=8):
    """Approximate AIE2P bfp16 (ebs8): groups of `block` elements along the last axis share one
    power-of-two exponent; each element keeps a `mantissa_bits`-bit signed integer mantissa
    relative to it. This is a NUMERIC MODEL of the documented ebs8 layout (op-precision-landscape-
    aie2p.md), not a device-verified rounding mode (the real hardware uses `conv_even`) -- flagged
    in the sweep output, resolved by the on-device gate."""
    x64 = np.asarray(x, np.float64)
    d = x64.shape[-1]
    pad = (-d) % block
    xp = np.pad(x64, [(0, 0)] * (x64.ndim - 1) + [(0, pad)]) if pad else x64
    shape = xp.shape[:-1] + (xp.shape[-1] // block, block)
    xb = xp.reshape(shape)
    amax = np.maximum(np.max(np.abs(xb), axis=-1, keepdims=True), 1e-30)
    qmax = 2 ** (mantissa_bits - 1) - 1
    # ceil, not floor: the shared exponent must be large enough that amax/scale <= qmax, or the
    # block's own largest element clips (floor(log2(amax)) under-scales whenever amax isn't
    # exactly a power of two -- caught by a unit check: it was truncating 1.304 -> 1.000).
    shared_exp = np.ceil(np.log2(amax))
    scale = np.maximum((2.0 ** shared_exp) / qmax, 1e-38)
    q = np.clip(np.round(xb / scale), -qmax - 1, qmax)
    out = (q * scale).reshape(xp.shape)
    return (out[..., :d] if pad else out).astype(np.float32)


FORMATS = {
    "f32": (lambda x: x, 4),
    "bf16": (to_bf16, 2),
    "int8": (lambda x: to_int8_per_axis(x, axis=-1), 1),
    "bfp16": (to_bfp16, 1.125),  # 8-bit mantissa/elem + 1 shared exponent byte per 8 elements
}


# ---------------------------------------------------------------------------------------------
# Swept forward pass -- a copy of parakeet_ref_encoder.block() with quantize-dequantize hooks at
# the five candidate points. Everything NOT under test stays exactly the verified f32 reference.
# ---------------------------------------------------------------------------------------------

def block_swept(x, blk, pos_enc, op, fmt):
    qfn = FORMATS[fmt][0] if fmt else (lambda t: t)
    q_res = qfn if op == "resadd" else (lambda t: t)
    q_aff = qfn if op == "affcast" else (lambda t: t)
    q_mid = qfn if op == "fc1fc2" else (lambda t: t)
    bfp16_gemm = (op == "bfp16gemm")

    def W(b, name):
        w = ref.W(b, name)
        return to_int8_per_axis(w, axis=0) if (op == "wint8" and name in BIG_WEIGHTS) else w

    def ln(gname, bname, t):
        return q_aff(ref.layernorm(t, W(blk, gname), W(blk, bname)))

    h = ln("norm_feed_forward1.weight", "norm_feed_forward1.bias", x)
    w1 = W(blk, "feed_forward1.linear1.weight")
    h_in, w1_in = (to_bfp16(h), to_bfp16(w1)) if bfp16_gemm else (h, w1)
    mid = q_mid(ref.silu(h_in @ w1_in))
    x = q_res(x + 0.5 * (mid @ W(blk, "feed_forward1.linear2.weight")))

    a_in = ln("norm_self_att.weight", "norm_self_att.bias", x)
    x = q_res(x + ref.mhsa(a_in, blk, pos_enc))

    c_in = ln("norm_conv.weight", "norm_conv.bias", x)
    x = q_res(x + ref.conv_module(c_in, blk))

    h2 = ln("norm_feed_forward2.weight", "norm_feed_forward2.bias", x)
    w1b = W(blk, "feed_forward2.linear1.weight")
    mid2 = q_mid(ref.silu(h2 @ w1b))
    x = q_res(x + 0.5 * (mid2 @ W(blk, "feed_forward2.linear2.weight")))

    return ref.layernorm(x, W(blk, "norm_out.weight"), W(blk, "norm_out.bias"))


def flatten_hcw(x):
    """The subsample flatten order parakeet_ref_encoder.subsample_flatten() picked as the winner
    against ONNX ground truth (gate2, rel=2.9e-07); hardcoded here since the other 16 clips have
    no independent ONNX target to re-select it against, and the order is a fixed reshape
    convention, not data-dependent."""
    C, Hh, Wf = x.shape
    out_w, out_b = ref.PE("out.weight"), ref.PE("out.bias")
    flat = np.transpose(x, (1, 0, 2)).reshape(Hh, C * Wf)
    return flat @ out_w + out_b


def subsample_clip(mel):
    """mel: [128, T_mel] -> ([T, 1024] block_in, pos_enc). Arm-invariant (no candidate op touches
    the subsample stem), so callers compute this ONCE per clip and reuse it across every arm --
    the naive nested-loop conv2d in parakeet_ref_encoder.py is the harness's dominant cost."""
    x = flatten_hcw(ref.subsample(mel[None]))
    return x, ref.rel_pos_encoding(x.shape[0], ref.D)


def encode_from(x0, pos_enc, op=None, fmt=None):
    """x0: [T, 1024] block_in -> [T, 1024] full-24-block encode, f32 truth if op is None."""
    x = x0
    for blk in range(ref.NB):
        x = block_swept(x, blk, pos_enc, op, fmt)
    return x


# ---------------------------------------------------------------------------------------------
# Bytes/clip -- static from shape, no device needed.
# ---------------------------------------------------------------------------------------------

OCCURRENCES_PER_BLOCK = {"resadd": 4, "affcast": 4, "fc1fc2": 2, "bfp16gemm": 1}


def bytes_per_clip(op, fmt, mel_ts):
    """Total bytes the op's tensor(s) occupy across one clip's full 24-block pass, at `fmt`,
    averaged over the corpus's real per-clip lengths (post-subsample T = len(x) from encode_clip,
    approximated here as T_mel // 8, matching the dw-striding /8 subsample -- exact enough for a
    bytes budget, not used for any rel-L2 number)."""
    itemsize = FORMATS[fmt][1]
    per_occ = {
        "resadd": lambda t: t * ref.D * itemsize,
        "affcast": lambda t: t * ref.D * itemsize,
        "fc1fc2": lambda t: t * ref.DFF * itemsize,
        "bfp16gemm": lambda t: t * ref.D * itemsize + ref.D * ref.DFF * itemsize,  # act + weight
    }[op]
    n = OCCURRENCES_PER_BLOCK[op]
    total = sum(n * per_occ(t // 8) for t in mel_ts) * ref.NB
    return total / len(mel_ts)


def wint8_weight_bytes():
    f32_total = int8_total = 0
    for name in BIG_WEIGHTS:
        n = ref.W(0, name).size  # same shape every block
        f32_total += n * 4
        int8_total += n * 1
    return f32_total * ref.NB, int8_total * ref.NB


# ---------------------------------------------------------------------------------------------

ARMS = [
    ("resadd", "bf16"), ("resadd", "int8"),
    ("affcast", "bf16"), ("affcast", "int8"),
    ("fc1fc2", "bf16"), ("fc1fc2", "int8"),
    ("bfp16gemm", "bfp16"),
    ("wint8", "int8"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst-window", type=int, default=5)
    ap.add_argument("--json", metavar="PATH", default=None)
    ap.add_argument("--clips", type=int, default=None, help="cap clip count (debug)")
    args = ap.parse_args()

    mel_files = sorted(glob.glob(os.path.join(MEL_DIR, "*.npy")))
    if args.clips:
        mel_files = mel_files[: args.clips]
    mels = {os.path.splitext(os.path.basename(f))[0]: np.load(f) for f in mel_files}
    mel_ts = [m.shape[1] for m in mels.values()]

    print(f"clips={len(mels)}  mel-frame range={min(mel_ts)}-{max(mel_ts)}", flush=True)
    stem = {name: subsample_clip(mel) for name, mel in mels.items()}  # {name: (x0, pos_enc)}
    truth = {name: encode_from(x0, pe) for name, (x0, pe) in stem.items()}
    post_t = {name: t.shape[0] for name, t in truth.items()}
    print(f"post-subsample T range={min(post_t.values())}-{max(post_t.values())}  "
          f"D={ref.D}  DFF={ref.DFF}\n", flush=True)

    header = f"{'op':10} {'fmt':6} {'bytes/clip':>12} {'vs f32':>8} {'mean':>8} {'worst-F':>9} {'worst-B':>9}  l1-risk"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    results = []
    f32_bytes_cache = {}
    for op, fmt in ARMS:
        cand = {name: encode_from(x0, pe, op, fmt) for name, (x0, pe) in stem.items()}
        rows = []
        for name in mels:
            e = ep.frame_rel_err(truth[name], cand[name])
            bv, bi = ep.worst_window(e, args.burst_window)
            rows.append({"clip": name, "mean": ep.clip_rel_l2(truth[name], cand[name]),
                         "worst_frame": float(e.max()), "worst_frame_idx": int(e.argmax()),
                         "worst_burst": bv, "worst_burst_idx": bi})
        agg = ep.aggregate(rows)
        agg["worst_burst_at"] = agg["worst_burst_at"].format(w=args.burst_window)

        if op == "wint8":
            b_f32, b_fmt = wint8_weight_bytes()
        else:
            if op not in f32_bytes_cache:
                f32_bytes_cache[op] = bytes_per_clip(op, "f32", mel_ts)
            b_f32 = f32_bytes_cache[op]
            b_fmt = bytes_per_clip(op, fmt, mel_ts)

        l1_risk = "YES (in-place accum, see separate-output-buffer-does-not-fit-so-in-place-is-forced)" \
            if op in ("fc1fc2", "bfp16gemm") else "no (elementwise, not a GEMM accumulator output)"

        print(f"{op:10} {fmt:6} {b_fmt:12,.0f} {b_fmt/b_f32:7.1%} "
              f"{agg['mean']:8.4f} {agg['worst_frame']:9.4f} {agg['worst_burst']:9.4f}  {l1_risk}",
              flush=True)

        results.append({"op": op, "fmt": fmt, "bytes_f32": b_f32, "bytes_fmt": b_fmt,
                        "bytes_ratio": b_fmt / b_f32, "aggregate": agg, "per_clip": rows,
                        "l1_risk": l1_risk})

    print("\nnew-burst: NOT computable on CPU -- it is candidate-minus-SHIPPED-baseline per-frame\n"
          "  delta, and this harness's baseline is f32 truth (zero error by construction). Needs\n"
          "  a device-measured baseline path's per-frame array; this is the on-device gate.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"clips": len(mels), "results": results}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
