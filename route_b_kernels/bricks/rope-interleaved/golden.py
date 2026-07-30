#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Host reference (numpy) for the rope-interleaved brick (route_b_kernels/bricks/rope-interleaved).

CORRECTS route_b_kernels/bricks/rope-lut, which implements split-half (GPT-NeoX/Llama) rotation.
The S2 model's `ggml_rope_ext(..., mode=0)` (GGML_ROPE_TYPE_NORMAL, ggml.h:250) is the OTHER
convention: ADJACENT-PAIR rotation, dims (0,1),(2,3),(4,5)... each get one (cos,sin), NOT
(i, i+D/2). Call sites: s2.cpp/src/s2_codec.cpp:338 (codec quantizer transformer) and
s2.cpp/src/s2_model.cpp:984 (slow AR transformer). Both conventions are self-consistent rotations
-- every gate that only checks against itself passes either way -- so `rope_interleaved_ref` below
is NOT trusted as a fresh derivation. It is cross-checked in __main__ against BOTH existing,
independently-written oracle formulations already in this repo:
  - scripts/s2_ar_ref.py::rope_interleaved       x: (n_tokens, n_heads, head_dim)
  - scripts/codec_quantizer_ref.py::_rope_normal x: (n_head, head_dim, T)
Measured (see __main__): the two oracles agree to rel-L2 ~4e-8 (float32 rounding noise, not a
convention difference) and `rope_interleaved_ref` here reproduces both to the same order.

theta_scale note: s2_ar_ref computes inv_freq via theta_scale = base**(-2/n_dims) then
theta_scale**arange(half); codec_quantizer_ref computes inv_freq via base**(-2i/head_dim) directly.
These are the same function of i (confirmed numerically: max abs diff ~1e-16, i.e. float64
rounding only) -- `build_inv_freq` below follows s2_ar_ref's form since that is what most closely
matches this brick's per-row [M, D] kernel ABI.

DEVICE DESIGN CHOICE (see rope_interleaved.cc header for the full writeup): unlike rope-lut, this
brick does NOT compute sin/cos on device at all. `build_cossin_resident` below performs that math
on the HOST in float64 and hands the device a plain float32 [M, ROT] resident table -- so there is
no on-device transcendental, no gather-LUT, and no LUT-quantization error to model. The "device
model" of this op is therefore just: round the INPUT to bf16 (what the device actually loads),
apply `rope_interleaved_ref` in float64, and let the device's own bf16 output rounding contribute
the remaining (sub-ulp, ~4e-3) error -- see verify_rope_interleaved.py.
"""
import numpy as np

# --------------------------------------------------------------------------------------------
# inv_freq / resident cos-sin table (host-side folding, like rope-lut's inv_freq and
# norm_gemv_prologue's gamma folding -- these are fixed given (n_dims, base) and do not depend on
# the runtime position, so they are computed once and handed to the device as a constant).
# --------------------------------------------------------------------------------------------

def build_inv_freq(n_dims: int, base: float = 10000.0) -> np.ndarray:
    """inv_freq[i] = base^(-2i/n_dims), i in [0, n_dims/2). Matches s2_ar_ref.rope_interleaved's
    theta_scale = base**(-2/n_dims); inv_freq = theta_scale**arange(half) -- confirmed numerically
    identical (max abs diff ~1e-16) to codec_quantizer_ref._rope_normal's direct
    base**(-2i/head_dim) form (see __main__)."""
    assert n_dims % 2 == 0
    half = n_dims // 2
    theta_scale = base ** (-2.0 / n_dims)
    return theta_scale ** np.arange(half, dtype=np.float64)


def build_cossin_resident(positions: np.ndarray, n_dims: int, base: float = 10000.0) -> np.ndarray:
    """positions: [M] -> cossin: [M, n_dims] float32, per row [cos(0..half-1) | sin(0..half-1)].

    This IS the resident buffer the device kernel receives (rope_interleaved.cc's `cossin` param):
    plain host-computed trig, no device-side sin/cos at all. Packing cos then sin (not
    interleaved) keeps the kernel's per-chunk load a single contiguous aie::load_v from each half,
    same convention as rope-lut packing pos+inv_freq into one buffer.
    """
    assert n_dims % 2 == 0
    half = n_dims // 2
    inv_freq = build_inv_freq(n_dims, base)                      # [half]
    theta = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None, :]  # [M, half]
    cossin = np.empty((len(positions), n_dims), dtype=np.float32)
    cossin[:, :half] = np.cos(theta).astype(np.float32)
    cossin[:, half:] = np.sin(theta).astype(np.float32)
    return cossin


# --------------------------------------------------------------------------------------------
# Reference rotation, row-wise ABI matching the kernel: x is [M, D], one row per (token, head)
# already flattened by the caller (a head is D=head_dim wide; M rows can mix tokens and heads
# freely as long as `positions[row]` is that row's token position -- same convention rope-lut
# uses for its `pos[]` input).
# --------------------------------------------------------------------------------------------

def rope_interleaved_ref(x: np.ndarray, positions: np.ndarray, n_dims: int,
                          base: float = 10000.0) -> np.ndarray:
    """x: [M, D] (D >= n_dims). ADJACENT-PAIR rotation: pairs (x[2i], x[2i+1]) for
    i in [0, n_dims/2) each rotate by theta[m,i] = positions[m] * inv_freq[i]:
        out[2i]   = x[2i]*cos(theta) - x[2i+1]*sin(theta)
        out[2i+1] = x[2i]*sin(theta) + x[2i+1]*cos(theta)
    dims [n_dims, D) pass through unchanged (partial rotary). All math in float64.

    Cross-checked against the two existing repo oracles in __main__ -- see module docstring.
    """
    m, d = x.shape
    assert n_dims <= d and n_dims % 2 == 0
    half = n_dims // 2
    inv_freq = build_inv_freq(n_dims, base)                       # [half]
    theta = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None, :]  # [M, half]
    cos = np.cos(theta)
    sin = np.sin(theta)

    xf = x.astype(np.float64)
    x0 = xf[:, 0:n_dims:2]   # pair-first:  x[2i] (a VIEW into xf, not a copy)
    x1 = xf[:, 1:n_dims:2]   # pair-second: x[2i+1] (also a view)
    # `out` must NOT alias xf: x0/x1 are views, so writing the even slice of xf in place before
    # the odd-slice line reads x0 again would silently read POST-rotation values for the second
    # line (measured: rel_l2 0.30 vs the oracle, a real bug caught only by the oracle cross-check
    # below -- this is precisely the self-consistent-but-wrong failure class the task warns about).
    out = xf.copy()
    out[:, 0:n_dims:2] = x0 * cos - x1 * sin
    out[:, 1:n_dims:2] = x0 * sin + x1 * cos
    # out[:, n_dims:] left unchanged (partial rotary pass-through)
    return out


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float(np.linalg.norm(a - b))


def _load_module(path, name):
    """importlib-load a sibling script by path, registering it in sys.modules FIRST -- required
    because s2_ar_ref.py uses @dataclass, whose machinery looks the defining class up via
    sys.modules[cls.__module__]; without pre-registration that lookup hits None and raises
    (measured: AttributeError: 'NoneType' object has no attribute '__dict__')."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    # CPU-ONLY oracle cross-check: rope_interleaved_ref must reproduce BOTH existing oracle
    # formulations, not just look plausible on its own -- this is the exact failure class the
    # task brief warns about (rope-lut shipped the wrong convention and every self-consistent
    # gate passed anyway).
    from pathlib import Path

    HERE = Path(__file__).resolve()
    REPO = HERE.parents[3]  # .../rope-interleaved -> bricks -> route_b_kernels -> repo root
    scripts = REPO / "scripts"
    ar_ref = _load_module(scripts / "s2_ar_ref.py", "s2_ar_ref")
    cq_ref = _load_module(scripts / "codec_quantizer_ref.py", "codec_quantizer_ref")

    rng = np.random.default_rng(0)
    n_tokens, n_heads, head_dim = 8, 3, 64  # head_dim=64 matches RVQ_HEAD_DIM (codec quantizer)
    x = rng.standard_normal((n_tokens, n_heads, head_dim)).astype(np.float32)
    positions = np.arange(n_tokens)
    base = 10000.0

    out_ar = ar_ref.rope_interleaved(x, positions, head_dim, base)          # (n_tokens,n_heads,D)

    x_cq = np.transpose(x, (1, 2, 0))                                       # (n_heads,D,n_tokens)
    out_cq = cq_ref._rope_normal(x_cq, positions.astype(np.float64), base)
    out_cq = np.transpose(out_cq, (2, 0, 1))                                # (n_tokens,n_heads,D)

    diff_oracles = rel_l2(out_ar, out_cq)
    print(f"[oracle cross-check] s2_ar_ref.rope_interleaved vs "
          f"codec_quantizer_ref._rope_normal  rel_l2={diff_oracles:.3e} "
          f"max_abs={np.abs(out_ar.astype(np.float64) - out_cq.astype(np.float64)).max():.3e}")
    assert diff_oracles < 1e-6, "the two existing oracles disagree -- STOP, do not trust either"

    # this brick's own reference, row-wise ABI: flatten (n_tokens,n_heads,D) -> (n_tokens*n_heads,D)
    x_rows = x.reshape(n_tokens * n_heads, head_dim)
    pos_rows = np.repeat(positions, n_heads)  # every head at a token shares that token's position
    out_mine = rope_interleaved_ref(x_rows, pos_rows, head_dim, base).reshape(n_tokens, n_heads, head_dim)

    diff_vs_ar = rel_l2(out_mine, out_ar)
    diff_vs_cq = rel_l2(out_mine, out_cq)
    print(f"[golden cross-check] rope_interleaved_ref vs s2_ar_ref            rel_l2={diff_vs_ar:.3e}")
    print(f"[golden cross-check] rope_interleaved_ref vs codec_quantizer_ref  rel_l2={diff_vs_cq:.3e}")
    assert diff_vs_ar < 1e-6 and diff_vs_cq < 1e-6, "golden disagrees with an oracle -- do not ship"

    # inv_freq formulation identity (theta_scale**i vs base**(-2i/head_dim)) claimed in the header
    half = head_dim // 2
    inv1 = build_inv_freq(head_dim, base)
    i = np.arange(half, dtype=np.float64)
    inv2 = base ** (-(2.0 * i) / head_dim)
    print(f"[inv_freq identity] theta_scale**i vs base**(-2i/D)  max_abs_diff={np.abs(inv1 - inv2).max():.3e}")
    assert np.abs(inv1 - inv2).max() < 1e-12

    # partial-rotary pass-through sanity (n_dims < D): dims [n_dims, D) must be untouched
    ROT2 = 32
    out2 = rope_interleaved_ref(x_rows, pos_rows, ROT2, base)
    assert np.allclose(out2[:, ROT2:], x_rows[:, ROT2:]), "pass-through dims must be untouched"
    print("partial-rotary pass-through check: OK")

    # resident cos/sin table shape + a spot-check that applying it by hand reproduces the ref
    cossin = build_cossin_resident(pos_rows, head_dim, base)
    assert cossin.shape == (n_tokens * n_heads, head_dim)
    cos_tab, sin_tab = cossin[:, :half], cossin[:, half:]
    x0 = x_rows[:, 0:head_dim:2].astype(np.float64)
    x1 = x_rows[:, 1:head_dim:2].astype(np.float64)
    manual = np.empty_like(x_rows, dtype=np.float64)
    manual[:, 0:head_dim:2] = x0 * cos_tab - x1 * sin_tab
    manual[:, 1:head_dim:2] = x0 * sin_tab + x1 * cos_tab
    diff_manual = rel_l2(manual, out_mine)
    print(f"[cossin resident spot-check] rel_l2={diff_manual:.3e}")
    assert diff_manual < 1e-6

    print("PASS: rope_interleaved_ref matches both existing oracles; resident-table path verified.")
