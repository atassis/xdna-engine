#!/usr/bin/env python3
"""Is a ONE-PASS f32 LayerNorm variance accurate enough to replace the two-pass one?

This decides a dataflow question, not just a numerics one. `ln_2pass.cc` computes

    mean = sum(x)/N ;  var = sum((x-mean)^2)/N     (two-pass, exact-mean-centered)

The centered form needs the GLOBAL mean before any element can be centered. Distributed across the
8 array columns that forces three steps: reduce partial sums, BROADCAST the mean back to every
column, then reduce partial centered squares. That mid-reduction broadcast is what collides with the
compute tile's 2-of-2 input DMA channels.

The one-pass form

    var = sum(x^2)/N - mean^2                       (one-pass, uncentered)

needs a single reduction and no broadcast, so it would dissolve the whole constraint. `ln_2pass.cc`
rejects it for catastrophic cancellation -- but that rejection was measured on a bf16 sum. This asks
whether it survives an f32 sum on the encoder's REAL activations.

Cancellation is governed by mean^2/var: the subtraction loses about log10(mean^2/var) digits, so the
answer is a property of the data, not of the algorithm. That ratio is measured here rather than
assumed.

    python3 scripts/ln_onepass_vs_twopass.py
"""
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-5


def ln_ref(x):
    """float64 two-pass -- ground truth."""
    x = x.astype(np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS), mean, var


def ln_two_pass_f32(x):
    """What ln_2pass.cc does: f32 data, f32 accumulate, exact-mean-centered."""
    x = x.astype(np.float32)
    mean = (x.astype(np.float32).sum(axis=-1, keepdims=True, dtype=np.float32)
            / np.float32(x.shape[-1]))
    d = (x - mean).astype(np.float32)
    var = ((d * d).sum(axis=-1, keepdims=True, dtype=np.float32)
           / np.float32(x.shape[-1]))
    return (x - mean) / np.sqrt(var + np.float32(EPS)), mean, var


def ln_one_pass_f32(x):
    """The candidate: var = E[x^2] - mean^2, single reduction, no mid-broadcast."""
    x = x.astype(np.float32)
    n = np.float32(x.shape[-1])
    s = x.sum(axis=-1, keepdims=True, dtype=np.float32)
    s2 = (x * x).sum(axis=-1, keepdims=True, dtype=np.float32)
    mean = s / n
    var = s2 / n - mean * mean
    var = np.maximum(var, np.float32(0.0))  # uncentered form can go negative
    return (x - mean) / np.sqrt(var + np.float32(EPS)), mean, var


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def main():
    files = sorted(glob.glob(os.path.join(REPO, "artifacts", "refs", "*.npy")))
    rows = []
    for f in files:
        a = np.load(f)
        if a.ndim < 2 or a.shape[-1] not in (768, 1024, 3072, 4096):
            continue
        x = np.squeeze(a)
        if x.ndim != 2:
            continue
        ref, mean, var = ln_ref(x)
        two, _, var2 = ln_two_pass_f32(x)
        one, _, var1 = ln_one_pass_f32(x)

        # The cancellation driver, measured per row then summarised.
        ratio = (mean[..., 0] ** 2) / np.maximum(var[..., 0], 1e-30)
        neg = int((var1[..., 0] <= 0).sum())

        rows.append({
            "name": os.path.basename(f)[:-4], "D": x.shape[-1], "T": x.shape[0],
            "two": rel(ref, two), "one": rel(ref, one),
            "var_two": rel(var[..., 0], var2[..., 0]),
            "var_one": rel(var[..., 0], var1[..., 0]),
            "ratio_med": float(np.median(ratio)), "ratio_max": float(ratio.max()),
            "neg": neg,
        })

    if not rows:
        print("no usable activation dumps found")
        sys.exit(2)

    print(f"{'tensor':<16}{'D':>5}{'  two-pass':>12}{'  one-pass':>12}"
          f"{'  var 2p':>11}{'  var 1p':>11}{'  mean^2/var med':>17}{'  max':>10}{'  neg':>6}")
    for r in rows:
        print(f"{r['name']:<16}{r['D']:>5}{r['two']:>12.3e}{r['one']:>12.3e}"
              f"{r['var_two']:>11.2e}{r['var_one']:>11.2e}"
              f"{r['ratio_med']:>17.3g}{r['ratio_max']:>10.3g}{r['neg']:>6}")

    worst_one = max(r["one"] for r in rows)
    worst_two = max(r["two"] for r in rows)
    worst_ratio = max(r["ratio_max"] for r in rows)
    total_neg = sum(r["neg"] for r in rows)
    print()
    print(f"worst rel-L2 vs float64: two-pass {worst_two:.3e}   one-pass {worst_one:.3e}"
          f"   ({worst_one / max(worst_two, 1e-30):.1f}x)")
    print(f"worst mean^2/var over all rows: {worst_ratio:.3g}  "
          f"(~{np.log10(max(worst_ratio, 1)):.1f} decimal digits lost in the subtraction)")
    print(f"rows where one-pass produced a NEGATIVE variance: {total_neg}")
    print()
    # The gate that matters: the shipped device encoder sits at 0.089 rel-L2 vs fp32 truth, and the
    # measured transcript cliff is a per-frame rel-err of 1.0. An algorithm change is irrelevant to
    # accuracy if it is orders below the noise the device already has.
    print(f"for scale: the shipped device encoder is 8.9e-02 from fp32 truth, and the measured")
    print(f"transcript-sensitivity cliff is 1.0 per-frame rel-err.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
