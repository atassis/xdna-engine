#!/usr/bin/env python3
"""Host numpy reference for the WHOLE fish-audio DAC codec decoder, gated against the s2.cpp oracle.

    codec_latent.bin  ->  decoder  ->  codec_audio.bin

This is the parity harness the on-NPU port is measured against (spec G2). It exists so a mismatch
can be localised: the quantizer half (RVQ lookup, post_module transformer, quantizer upsamples) is
taken from the oracle as `codec_latent`, and only the decoder half is reimplemented here. Run it
first -- if this does not reproduce codec_audio.bin, no device result can be trusted, because the
graph or a weight layout is misunderstood before any kernel runs.

Graph (s2.cpp/src/s2_codec.cpp:499-524):

    x = causal_conv_1d(model.0.conv, z, stride=1, dilation=1)          1024 -> 1536, k 7
    for i, stride in enumerate(decoder_rates):                          [8, 8, 4, 2]
        x = snake(x, model.{i+1}.block.0.alpha)
        x = conv_transpose_1d(model.{i+1}.block.1.conv, stride, crop_right=stride)
        x = residual_unit(model.{i+1}.block.2, dilation 1)
        x = residual_unit(model.{i+1}.block.3, dilation 3)
        x = residual_unit(model.{i+1}.block.4, dilation 9)
    x = snake(x, model.5.alpha)
    x = causal_conv_1d(model.6.conv)                                     96 -> 1, k 7
    audio = tanh(x)

THE TWO CONVOLUTIONS TAKE OPPOSITE WEIGHT LAYOUTS, which is the single most dangerous detail here:
ggml_conv_1d wants numpy [c_out, c_in, k] and ggml_conv_transpose_1d wants numpy [c_in, c_out, k].
Every residual-unit weight is square, so swapping them is invisible to a shape check and shows up
only as a wrong waveform. See route_b_kernels/bricks/conv-1d/golden.py.

Everything is vectorised: the per-element reference forms in the brick goldens are the definition,
but at 112640 output samples they are unusably slow, so the loops here are over k only.

    python3 scripts/codec_decoder_ref.py <dump-dir> [--gguf <path>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402

DEFAULT_GGUF = codec_paths.gguf()
PREFIX = "c.decoder"
DECODER_RATES = [8, 8, 4, 2]


def load_dump(d, name):
    shape = [int(v) for v in (d / f"{name}.shape").read_text().split()]
    flat = np.fromfile(d / f"{name}.bin", dtype=np.float32)
    return flat, shape


def snake(x, alpha):
    """y = x + sin^2(alpha*x)/alpha, alpha per channel. f64 internally: alpha*x reaches ~10 periods
    at these magnitudes and the squared sine is where a f32 fold loses its digits."""
    a = np.asarray(alpha, np.float64).reshape(-1, 1)
    s = np.sin(a * x.astype(np.float64))
    return (x + (s * s) / a).astype(np.float32)


def conv_1d_causal(x, w, bias, dilation=1):
    """x [c_in, t], w [c_out, c_in, k] (ggml_conv_1d layout), bias [c_out] -> [c_out, t]."""
    c_out, c_in, k = w.shape
    t = x.shape[1]
    y = np.zeros((c_out, t), np.float32)
    for j in range(k):
        shift = (k - 1 - j) * dilation
        if shift < t:
            y[:, shift:] += w[:, :, j] @ x[:, :t - shift]
    return y + np.asarray(bias, np.float32).reshape(-1, 1)


def conv_transpose_1d(x, w, bias, stride, crop_right):
    """x [c_in, t], w [c_in, c_out, k] (ggml_conv_transpose_1d layout, the TRANSPOSE of the above),
    bias [c_out] -> [c_out, t*stride] after cropping."""
    c_in, c_out, k = w.shape
    t = x.shape[1]
    out_len = (t - 1) * stride + k
    y = np.zeros((c_out, out_len), np.float32)
    # [k, c_out, t]: every output channel's contribution from every tap, before the scatter.
    contrib = np.einsum("ioj,it->jot", w, x, optimize=True).astype(np.float32)
    for j in range(k):
        y[:, j::stride][:, :t] += contrib[j]
    y += np.asarray(bias, np.float32).reshape(-1, 1)
    return y[:, :out_len - crop_right] if crop_right else y


def residual_unit(x, W, prefix, dilation):
    a0 = W(f"{prefix}.block.0.alpha").reshape(-1)
    w1 = W(f"{prefix}.block.1.conv.weight")
    b1 = W(f"{prefix}.block.1.conv.bias").reshape(-1)
    a2 = W(f"{prefix}.block.2.alpha").reshape(-1)
    w3 = W(f"{prefix}.block.3.conv.weight")
    b3 = W(f"{prefix}.block.3.conv.bias").reshape(-1)
    y = conv_1d_causal(snake(x, a0), w1, b1, dilation)
    y = conv_1d_causal(snake(y, a2), w3, b3, 1)
    return (x + y).astype(np.float32)


def decode(z, W, verbose=False):
    def note(msg):
        if verbose:
            print(f"    {msg}", flush=True)

    x = conv_1d_causal(z, W(f"{PREFIX}.model.0.conv.weight"),
                       W(f"{PREFIX}.model.0.conv.bias").reshape(-1), 1)
    note(f"input conv -> {x.shape}")

    for i, stride in enumerate(DECODER_RATES):
        p = f"{PREFIX}.model.{i + 1}"
        x = snake(x, W(f"{p}.block.0.alpha").reshape(-1))
        x = conv_transpose_1d(x, W(f"{p}.block.1.conv.weight"),
                              W(f"{p}.block.1.conv.bias").reshape(-1), stride, crop_right=stride)
        note(f"stage {i + 1} upsample x{stride} -> {x.shape}")
        for unit, dil in ((2, 1), (3, 3), (4, 9)):
            x = residual_unit(x, W, f"{p}.block.{unit}", dil)
        note(f"stage {i + 1} residual units -> {x.shape}")

    last = len(DECODER_RATES) + 1
    x = snake(x, W(f"{PREFIX}.model.{last}.alpha").reshape(-1))
    w_tail = W(f"{PREFIX}.model.{last + 1}.conv.weight")
    if w_tail.ndim == 2:                       # ne=[7, 96] -> a single output channel
        w_tail = w_tail.reshape(1, *w_tail.shape)
    x = conv_1d_causal(x, w_tail, W(f"{PREFIX}.model.{last + 1}.conv.bias").reshape(-1), 1)
    note(f"tail conv -> {x.shape}")
    return np.tanh(x.astype(np.float64)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--gate", type=float, default=3e-2)
    args = ap.parse_args()

    cache = {}

    def W(name):
        if name not in cache:
            cache[name] = gx.load(args.gguf, name).astype(np.float32)
        return cache[name]

    zf, zshape = load_dump(args.dump_dir, "codec_latent")
    af, ashape = load_dump(args.dump_dir, "codec_audio")
    # The dump is written in ggml ne order (ne[0] fastest), so [latent_dim, frames] on disk is
    # frames-major once reshaped C-order -- transpose to the [channel, time] the graph works in.
    z = zf.reshape(zshape[1], zshape[0]).T.copy()
    ref_audio = af.reshape(-1)
    print(f"latent {z.shape}  oracle audio {ref_audio.shape}  "
          f"(expect {z.shape[1]} * {np.prod(DECODER_RATES)} = {z.shape[1] * np.prod(DECODER_RATES)})")

    got = decode(z, W, verbose=True).reshape(-1)
    print(f"decoded {got.shape}")
    assert got.shape == ref_audio.shape, f"length mismatch: {got.shape} vs {ref_audio.shape}"

    num = np.linalg.norm((got - ref_audio).astype(np.float64))
    den = np.linalg.norm(ref_audio.astype(np.float64))
    rl2 = float(num / den)
    print(f"  rel-L2 vs codec_audio.bin : {rl2:.6e}   (gate {args.gate:.1e})")
    print(f"  max abs err               : {float(np.max(np.abs(got - ref_audio))):.6e}")
    print(f"  oracle range              : [{ref_audio.min():+.4f}, {ref_audio.max():+.4f}]")
    print(f"  ref range                 : [{got.min():+.4f}, {got.max():+.4f}]")
    if rl2 > args.gate:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
