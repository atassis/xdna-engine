#!/usr/bin/env python3
"""Host numpy golden for route_b_kernels/bricks/conv-1d/conv_1d.cc.

CAUSAL dilated 1-D convolution as the fish-audio DAC codec decoder uses it inside every residual
unit (s2.cpp/src/s2_codec.cpp:224-235, called from build_residual_unit at :381-393):

    causal_conv_1d(w, bias, x, stride=1, dilation=d)

s2.cpp builds it as left-pad-then-valid-conv: kernel_size = (k-1)*d + 1, pad = kernel_size - stride,
and with stride 1 that is exactly (k-1)*d zeros on the LEFT and none that affect the output on the
right, so out_len == in_len and output t depends only on inputs <= t. Written here in the equivalent
direct form, which is what a kernel can evaluate without materialising the padded copy:

    y[co, t] = bias[co] + sum_ci sum_j w[co, ci, j] * x[ci, t - (k-1-j)*d]     (x[.., <0] = 0)

LAYOUT IS [c_out, c_in, k], WHICH IS NOT WHAT conv_transpose_1d USES. ggml_conv_1d reshapes its
kernel as "[OC, IC, K] => [OC, IC * K]" (ggml/src/ggml.c:4495), so ne=[k, c_in, c_out] and the
numpy C-order view gguf_extract returns is [c_out, c_in, k]. ggml_conv_transpose_1d takes
ne=[k, c_out, c_in] instead, i.e. numpy [c_in, c_out, k] -- the convention the conv-transpose-1d
brick documents. The two are transposes of each other and every residual-unit weight in this codec
is SQUARE (96x96), so getting it backwards is invisible to a shape check and to any self-consistent
gate. Confirmed against the one non-square conv_1d in the decoder: c.decoder.model.0.conv.weight is
ne=[7, 1024, 1536] and maps the 1024-wide transformer output to the 1536 channels stage 1 consumes,
which only parses as [k, c_in, c_out].

Usage: python3 golden.py
"""
import numpy as np


def conv_1d_causal_ref(x, w, bias, dilation=1):
    """x: [c_in, t] f32. w: [c_out, c_in, k] f32. bias: [c_out] f32. Returns [c_out, t]."""
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    c_in, t = x.shape
    c_out, c_in_w, k = w.shape
    assert c_in == c_in_w, f"channel mismatch: x has {c_in}, w has {c_in_w}"
    y = np.zeros((c_out, t), dtype=np.float64)
    for ci in range(c_in):
        for j in range(k):
            shift = (k - 1 - j) * dilation
            if shift == 0:
                y += np.outer(w[:, ci, j], x[ci])
            elif shift < t:
                y[:, shift:] += np.outer(w[:, ci, j], x[ci, :t - shift])
    y += np.asarray(bias, dtype=np.float64).reshape(-1, 1)
    return y.astype(np.float32)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    c_in, c_out, k, t = 3, 4, 7, 32
    x = rng.standard_normal((c_in, t)).astype(np.float32)
    w = (rng.standard_normal((c_out, c_in, k)).astype(np.float32) * 0.3)
    b = (rng.standard_normal(c_out).astype(np.float32) * 0.1)

    for d in (1, 3, 9):
        y = conv_1d_causal_ref(x, w, b, d)
        assert y.shape == (c_out, t), f"causal conv must preserve length, got {y.shape}"

        # Cross-check against the literal pad-then-valid form s2.cpp builds, which is the actual
        # reference semantics -- a direct-form off-by-one would otherwise be invisible.
        pad = (k - 1) * d
        xp = np.concatenate([np.zeros((c_in, pad), np.float32), x], axis=1)
        y2 = np.zeros((c_out, t), np.float64)
        for ci in range(c_in):
            for j in range(k):
                y2 += np.outer(w[:, ci, j], xp[ci, j * d:j * d + t])
        y2 += b.astype(np.float64).reshape(-1, 1)
        assert rel_l2(y, y2.astype(np.float32)) < 1e-6, f"direct != pad-then-valid at dilation {d}"

        # CAUSALITY, the property the whole construction exists for: perturbing input t must never
        # change any output before t. A shape assert cannot catch a sign error in the shift.
        probe = t // 2
        x2 = x.copy()
        x2[:, probe] += 5.0
        yp = conv_1d_causal_ref(x2, w, b, d)
        assert np.allclose(yp[:, :probe], y[:, :probe], atol=1e-5), f"not causal at dilation {d}"
        assert not np.allclose(yp[:, probe], y[:, probe]), "perturbation had no effect at all"
        print(f"dilation {d}: shape {y.shape}, matches pad-then-valid, causal")
    print("conv_1d golden OK")
