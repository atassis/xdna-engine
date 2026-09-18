#!/usr/bin/env python3
"""Host references for conv_1d_bf16.cc, the bf16 arm of the conv-1d brick.

Two goldens, same split verify_upscaler_bf16_conv2d.py already uses for its bf16 conv2d gate,
because they answer different questions:

  conv_1d_causal_ref (golden.py, f32 throughout)  -- the codec's TRUE reference. rel-L2 of the
      device bf16 kernel against THIS is "what does the format cost in accuracy" -- the number
      the residual-unit gate (3e-2) actually cares about.

  conv_1d_causal_bf16_model (below) -- quantizes x/w/bias to bf16 BEFORE the reduction and rounds
      the output to bf16 after, accumulating in float64 in between (never bf16-accumulated -- see
      kb/bf16-norm-numerics-and-accumulation-guards). rel-L2 of the device kernel against THIS
      isolates DEVICE correctness (wrong tile layout, wrong dilation math, a real device bug) from
      bf16's own, expected, unavoidable quantization loss: a device bug shows up here even when the
      format's own accuracy gate still passes against the f32 truth.

Rounding convention: ml_dtypes casts to bfloat16 round-to-nearest-even, which is what this kernel's
`aie::set_rounding(conv_even)` call intends (see conv_1d_bf16.cc's header for why that intent is
NOT the same as a confirmed device behaviour for this exact conversion -- gemm_bf16xbfp16.cc
measured the identical accfloat->bfloat16 narrow as rounding-mode-INERT on device). So this model is
a best-effort simulation of the INTENDED numerics, not a claim about what the hardware measurably
does; the device gate script is what actually answers that.

Usage: python3 golden_bf16.py
"""
import numpy as np
import ml_dtypes

BF16 = ml_dtypes.bfloat16


def _to_bf16(a):
    return np.asarray(a, np.float32).astype(BF16)


def conv_1d_causal_bf16_model(x, w, bias, dilation=1):
    """x: [c_in, t] f32. w: [c_out, c_in, k] f32. bias: [c_out] f32. Returns [c_out, t] f32
    (the bf16-quantized OUTPUT widened back to f32 for comparison, same convention the device
    kernel's own out tile uses -- bf16 storage, read back and widened by the harness).

    Vectorised over k only (matmul per tap), matching codec_decoder_ref.py's own conv_1d_causal --
    the per-(ci,j) outer-product form golden.py uses is the definition but is unusably slow at
    whole-chain lengths (t up to 135168 in host_bf16_codec_sim.py); this is the same identity."""
    xb = _to_bf16(x).astype(np.float64)   # widen once quantized; accumulate wide, never in bf16
    wb = _to_bf16(w).astype(np.float64)
    bb = _to_bf16(bias).astype(np.float64)
    c_out, c_in, k = wb.shape
    c_in_x, t = xb.shape
    assert c_in == c_in_x, f"channel mismatch: x has {c_in_x}, w has {c_in}"
    y = np.zeros((c_out, t), dtype=np.float64)
    for j in range(k):
        shift = (k - 1 - j) * dilation
        if shift < t:
            y[:, shift:] += wb[:, :, j] @ xb[:, :t - shift]
    y += bb.reshape(-1, 1)
    return _to_bf16(y.astype(np.float32)).astype(np.float32)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import conv_1d_causal_ref

    rng = np.random.default_rng(0)
    c_in, c_out, k, t = 3, 4, 7, 32
    x = rng.standard_normal((c_in, t)).astype(np.float32)
    w = (rng.standard_normal((c_out, c_in, k)).astype(np.float32) * 0.3)
    b = (rng.standard_normal(c_out).astype(np.float32) * 0.1)

    for d in (1, 3, 9):
        y_f32 = conv_1d_causal_ref(x, w, b, d)
        y_bf16 = conv_1d_causal_bf16_model(x, w, b, d)
        assert y_bf16.shape == y_f32.shape
        rl2 = rel_l2(y_bf16, y_f32)
        # Sanity band: bf16 has ~3 decimal digits (8-bit mantissa incl. implicit bit), so a random
        # small conv should land in the 1e-3..1e-1 range, not near-zero (would mean quantization
        # was silently skipped) and not >>1 (would mean a sign/shape bug, not quantization noise).
        assert 1e-4 < rl2 < 5e-1, f"dilation {d}: bf16 model rel-L2 {rl2:.3e} outside sanity band"
        print(f"dilation {d}: bf16-model vs f32-truth rel-L2 {rl2:.3e}")
    print("conv_1d golden_bf16 OK")
