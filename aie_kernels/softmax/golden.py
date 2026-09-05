#!/usr/bin/env python3
"""Host numpy golden reference for the softmax AIE2P brick.

Matches softmax.cc exactly (max-subtract, exp, normalize; see softmax.cc's header for the exp2
route -- this golden uses ordinary numpy exp, it does not model the on-device polynomial):
    row_max = max(x)                    over the reduction axis, per row
    e       = exp(x - row_max)
    out     = e / sum(e)

Contract: input is a resident [tile, D] stream -- shape (rows, cols), softmax reduces over the
LAST axis (cols) per row. The brick is axis-agnostic about what "cols" represents in the caller's
real tensor (e.g. one (head, query) column's T_k keys) -- same division of labor as rmsnorm/
layernorm's D-axis contract.

Oracle: scripts/codec_quantizer_ref.py::_softmax_over_keys (~line 357):
    s = scores.astype(np.float64); s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s); return e / e.sum(axis=1, keepdims=True)
`_softmax_over_keys` reduces over axis=1 of a [n_head, T_k, T_q] tensor (keys). Its caller,
rvq_transformer (~line 366), adds an additive mask (0 / -1e9, from _causal_window_mask ~line 323)
BEFORE calling it -- -1e9-masked entries are the normal input this brick must handle cleanly, not
an edge case (see test case (ii) below).

Cross-check: scripts/s2_ar_ref.py::causal_attention (~line 582-596) is a SECOND, independent
formulation of the same normalization (mask = -inf instead of -1e9, reduces over axis=-1 of a
[n_head, T_q, T_k] tensor instead of axis=1 of [n_head, T_k, T_q]). Both source files are heavy
imports (codec_quantizer_ref.py pulls in codec_paths/gguf_extract/codec_decoder_ref at module
scope, s2_ar_ref.py pulls in codec_paths, and both may expect real GGUF files on disk) -- rather
than import them here and risk a fragile/irrelevant dependency chain for a pure-math cross-check,
the two normalizations are transcribed VERBATIM below (cited by line) and checked against each
other and against softmax_ref on identical data. See __main__ for the actual agreement numbers.

Later DEVICE pass: run softmax_f32 on-device over the same (rows, cols) input and compare against
`softmax_ref` below via rel-L2. Gate: rel_l2 < 3e-2 (brick catalog convention).
"""
import numpy as np


def softmax_ref(x: np.ndarray) -> np.ndarray:
    """x: (rows, cols) f32/f64. Softmax over the LAST axis (cols), per row. f64 internally for
    stability (matches _softmax_over_keys's own astype(np.float64)), f32 out."""
    x = np.asarray(x, dtype=np.float64)
    s = x - x.max(axis=-1, keepdims=True)
    e = np.exp(s)
    out = e / e.sum(axis=-1, keepdims=True)
    return out.astype(np.float32)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den != 0.0 else float(num)


def _oracle_softmax_over_keys(scores: np.ndarray) -> np.ndarray:
    """Verbatim transcription of scripts/codec_quantizer_ref.py::_softmax_over_keys (~line 357).
    scores: [n_head, T_k, T_q], reduces over axis=1 (keys)."""
    s = scores.astype(np.float64)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def _oracle_causal_attention_softmax(scores: np.ndarray) -> np.ndarray:
    """Verbatim transcription of the softmax step inside scripts/s2_ar_ref.py::causal_attention
    (~line 592-594; the mask-add on line 591 is assumed already applied by the caller here, same
    as _oracle_softmax_over_keys above). scores: [n_head, T_q, T_k], reduces over axis=-1 (keys)."""
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols = 8, 64  # cols=64 matches the brick's verified per-call row width (see softmax.cc)

    # ---- the four gate test cases (must match verify_softmax.py's construction) ----
    x_random = (rng.standard_normal((rows, cols)).astype(np.float32) * 4.0)

    x_masked = x_random.copy()
    mask = rng.random((rows, cols)) < 0.6
    mask[:, 0] = False  # keep >=1 unmasked entry per row (the real causal-mask invariant)
    x_masked[mask] = -1e9

    x_large_pos = (rng.standard_normal((rows, cols)).astype(np.float32) * 5.0 + 80.0)  # naive exp overflows here

    x_uniform = np.full((rows, cols), 3.0, dtype=np.float32)

    cases = {
        "random": x_random,
        "masked(-1e9)": x_masked,
        "large_positive": x_large_pos,
        "uniform": x_uniform,
    }

    print(f"rows={rows} cols={cols}")
    for name, x in cases.items():
        out = softmax_ref(x)
        row_sums = out.sum(axis=-1)
        finite = np.all(np.isfinite(out))
        print(f"[{name:16s}] sum(out) per row (expect 1.0): "
              f"min={row_sums.min():.6f} max={row_sums.max():.6f}  all_finite={finite}")
        assert finite, f"{name}: non-finite output"
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"{name}: row sums != 1"

    # uniform row must be exactly 1/cols
    out_u = softmax_ref(x_uniform)
    assert np.allclose(out_u, 1.0 / cols, atol=1e-6), "uniform row not exactly 1/n"
    print(f"[uniform] value (expect {1.0/cols:.6f}): {out_u[0, 0]:.6f}  OK")

    # large-positive row: a naive exp(x) (no max-subtract) would overflow (exp(80+) in f32 is inf);
    # confirm our max-subtracted ref stays finite and a NAIVE implementation would NOT.
    with np.errstate(over="ignore"):
        naive_overflow = not np.all(np.isfinite(np.exp(x_large_pos.astype(np.float32))))
    print(f"[large_positive] naive exp(x) (no max-subtract) overflows: {naive_overflow}")
    assert naive_overflow, "test case (iii) does not actually stress the max-subtract"

    # masked row: every masked position's probability must be ~0 relative to the unmasked ones
    out_m = softmax_ref(x_masked)
    masked_mass = out_m[mask].sum()
    print(f"[masked(-1e9)] total probability mass on masked entries: {masked_mass:.3e}")
    assert masked_mass < 1e-8, "masked entries carry non-negligible probability"

    # ---- cross-check softmax_ref against BOTH independent oracle formulations ----
    print()
    print("cross-check vs both oracle formulations (random + masked cases):")
    for name in ("random", "masked(-1e9)"):
        x = cases[name]
        ref = softmax_ref(x)

        # _softmax_over_keys: [n_head=1, T_k=cols, T_q=rows], reduce axis=1 -> transpose in/out
        oracle_a = _oracle_softmax_over_keys(x.T[None, :, :])[0].T

        # causal_attention: [n_head=1, T_q=rows, T_k=cols], reduce axis=-1 -> direct, no transpose
        oracle_b = _oracle_causal_attention_softmax(x[None, :, :])[0]

        d_a = rel_l2(oracle_a, ref)
        d_b = rel_l2(oracle_b, ref)
        d_ab = rel_l2(oracle_a, oracle_b)
        print(f"  [{name:16s}] ref vs _softmax_over_keys: {d_a:.3e}   "
              f"ref vs causal_attention: {d_b:.3e}   over-keys vs causal_attention: {d_ab:.3e}")
        # threshold is float64 summation-order noise (transpose changes reduction order), not a
        # real algorithmic disagreement -- both oracles and softmax_ref agree to ~1e-8, ~9 decimal
        # digits, i.e. float64 noise floor, not the 3e-2 device gate this brick actually needs.
        assert d_a < 1e-6, f"{name}: disagreement vs _softmax_over_keys ({d_a:.3e})"
        assert d_b < 1e-6, f"{name}: disagreement vs causal_attention ({d_b:.3e})"

    # informational only (not a gate case): confirm the causal_attention-style literal -inf mask
    # (as opposed to _softmax_over_keys's -1e9) produces the SAME result as softmax_ref given the
    # -1e9 masked input -- i.e. the two masking conventions are numerically interchangeable once
    # normalized, not just "close".
    x_inf = x_random.copy()
    x_inf[mask] = -np.inf
    out_inf = _oracle_causal_attention_softmax(x_inf[None, :, :])[0]
    out_1e9 = softmax_ref(x_masked)
    d_inf = rel_l2(out_inf, out_1e9)
    print(f"\n  -inf mask vs -1e9 mask (same unmasked values, informational): rel_l2={d_inf:.3e}")

    print("\ngolden self-check OK")
