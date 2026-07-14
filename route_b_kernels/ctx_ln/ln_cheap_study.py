#!/usr/bin/env python3
# ln_cheap_study.py -- numpy STUDY for the "cheap on-chip LayerNorm prologue" research (feat/r1-ln-cheap).
#
# QUESTION: is there a THIRD path for a pre-matmul LayerNorm that is CHEAP on-chip (~one A-stream,
# like the SiLU epilogue), after TWO prior on-chip attempts failed on device:
#   (1) TWO-PASS on-chip reduction (mm_ln_prologue.cc): the full-K A row ([64,1024] bf16 = 128 KB)
#       does NOT fit L1 (64 KB), so A is re-streamed from L3 and the 2-pass costs ~9 ms/dispatch
#       (NPU bucket 98 ms -> ~900 ms). DEAD by L1 capacity.
#   (2) EPILOGUE-CORRECTION (algebraic, single pass): compute raw x@W' in bf16, then correct by
#       -mean*colsum(W') and scale by inv_std. DEAD by catastrophic cancellation at high |mean|.
#
# The THIRD path tested here: HOST computes the CHEAP per-row stats [T,2] = (mean, inv_std) (the same
# reduction the host LayerNorm already does), and the NPU does ONLY a SINGLE-PASS affine-normalize of
# the A-tile fused before the matmul (one mul-add per element) -- NO on-chip reduction, NO second
# A-stream. The stats are delivered IN-BAND on the existing A DMA channel (the compute tile has only
# 2 input channels, both used by A and B -- no room for a 3rd), so the delivery precision of `mean`
# matters. We measure several delivery encodings.
#
# This script:
#   (A) reproduces the epilogue-correction cancellation to CONFIRM it is dead;
#   (B) validates the single-pass prologue-apply numerics under different `mean` delivery encodings
#       (f32, bf16, double-bf16) and different regimes of |mean|/std, vs the true f32 LayerNorm;
#   (C) tests the K-augmentation variant (subtract mean inside the f32 mmul accumulator).
#
# Pure numpy; NO device, NO aiecc. Run: python3 ln_cheap_study.py
import numpy as np

np.random.seed(0)

# ---- bf16 emulation (round-to-nearest-even, 8-bit exponent, 7-bit mantissa) ----
def to_bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # round-to-nearest-even on the low 16 bits
    rounding_bias = 0x00007FFF + ((u >> 16) & 1)
    u = u + rounding_bias
    u = u & 0xFFFF0000
    return u.view(np.float32)


# ---- reference: true f32 two-pass centered LayerNorm, NORMALIZE-ONLY (affine folded elsewhere) ----
def ln_ref_f32(x, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return (x - mu) * inv, mu.squeeze(1), inv.squeeze(1)


def rel_err(approx, ref):
    return np.linalg.norm(approx.ravel() - ref.ravel()) / (np.linalg.norm(ref.ravel()) + 1e-30)


def make_inputs(T, K, mean_shift=0.0, scale=1.0):
    """Row activations with a controllable per-row mean bias. mean_shift multiplies a per-row
    random offset (in units of the per-row std) so we can sweep |mean|/std."""
    x = np.random.randn(T, K).astype(np.float32) * scale
    if mean_shift != 0.0:
        # per-row DC offset of magnitude ~ mean_shift * std
        x = x + (np.random.randn(T, 1).astype(np.float32) * mean_shift * scale)
    return x


def mu_over_std(x):
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return np.abs(mu) / (sd + 1e-30)


# =====================================================================================
# (A) EPILOGUE-CORRECTION: compute raw x@W' in bf16, correct by -mean*colsum(W'), *inv_std.
#     out[i,:] = inv_i * ( bf16(x_i) @ bf16(W)  -  mean_i * colsum(bf16(W)) )
#     The two terms x@W and mean*colsum(W) are both LARGE and nearly equal -> cancellation.
# =====================================================================================
def epilogue_correction(x, W, eps=1e-5):
    xb = to_bf16(x)
    Wb = to_bf16(W)
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    # raw product, rounded to bf16 on output (the C bucket is bf16) -- this is the killer:
    raw = to_bf16(xb @ Wb)                      # [T,N] ~ large
    colsum = to_bf16(Wb.sum(axis=0, keepdims=True))  # [1,N]
    corr = to_bf16(mu * colsum)                 # [T,N] ~ large, nearly equal to raw's mean part
    out = inv * (raw - corr)                    # subtract two large bf16 quantities -> cancellation
    return out


# =====================================================================================
# (B) PROLOGUE-APPLY: host gives per-row (mu, inv); NPU normalizes the A-tile in one pass, in f32,
#     stores bf16 (the matmul input dtype), THEN the matmul consumes it. We emulate ONLY the
#     normalize+bf16-store here (the matmul that follows is the ordinary bf16 GEMM, unchanged).
#     encoding = how (mu, inv) are DELIVERED in-band on the bf16 A stream.
# =====================================================================================
def prologue_apply(x, mu, inv, encoding="f32"):
    xb = to_bf16(x)  # A arrives bf16 (host packed the RAW row -- NO host normalize)
    if encoding == "f32":
        mu_d, inv_d = mu.astype(np.float32), inv.astype(np.float32)
    elif encoding == "bf16":
        mu_d, inv_d = to_bf16(mu), to_bf16(inv)
    elif encoding == "double_bf16":
        # mu ~ mu_hi + mu_lo (two bf16, "double-bf16" / two-sum) recovers ~14 mantissa bits.
        mu_hi = to_bf16(mu)
        mu_lo = to_bf16(mu - mu_hi)
        mu_d = (mu_hi.astype(np.float32) + mu_lo.astype(np.float32))
        inv_d = to_bf16(inv)  # inv multiplies (x-mu)~O(std); 0.4% err on inv -> 0.4% out err, fine
    else:
        raise ValueError(encoding)
    # on-chip math is f32 (upcast bf16 x), store bf16:
    y = (xb.astype(np.float32) - mu_d[:, None]) * inv_d[:, None]
    return to_bf16(y)


# =====================================================================================
# (C) K-AUGMENTATION: subtract mean INSIDE the f32 mmul accumulator (like bias-via-K-aug).
#     A_aug = [x | mu]  (one extra k-col = per-row mean, bf16)
#     W_aug = [W ; -colsum(W)] (one extra k-row = -sum_k W[k,:], bf16)
#     A_aug @ W_aug = x@W - mu*colsum(W) = (x-mu)@W  -- accumulated in f32, THEN *inv (epilogue).
#     Contrast with (A): the subtraction happens in the f32 accumulator, never rounded to bf16 first.
# =====================================================================================
def kaug_centered_matmul(x, W, mu, inv):
    xb = to_bf16(x)
    Wb = to_bf16(W)
    mu_b = to_bf16(mu)
    colsum = to_bf16(Wb.sum(axis=0))          # bf16 extra weight row (-colsum)
    A_aug = np.concatenate([xb, mu_b[:, None]], axis=1)          # [T, K+1]
    W_aug = np.concatenate([Wb, -colsum[None, :]], axis=0)       # [K+1, N]
    acc = A_aug.astype(np.float32) @ W_aug.astype(np.float32)    # f32 accumulate (native mmul acc)
    return acc * inv[:, None]                                    # per-row inv epilogue scale


def matmul_true_ln(x, W, ynorm_ref):
    # reference: (true f32 LN of x) @ bf16 W, output kept f32 -> what the fused op SHOULD produce
    Wb = to_bf16(W)
    return to_bf16(ynorm_ref) @ Wb.astype(np.float32)


# =====================================================================================
# RUN
# =====================================================================================
def banner(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


T, K, N = 64, 1024, 512  # one resident A-tile [m=64, KRES=1024]; N cols of weight
regimes = [
    ("benign  |mean|/std~0", 0.0),
    ("mild    |mean|/std~1", 1.0),
    ("high    |mean|/std~4", 4.0),
    ("extreme |mean|/std~16", 16.0),
]

banner("(A) EPILOGUE-CORRECTION -- confirm it is DEAD (rel err of the FUSED matmul output)")
print(f"{'regime':28s} {'meanmu/std':>10s} {'rel_err(epilogue-corr)':>24s}")
for name, ms in regimes:
    x = make_inputs(T, K, mean_shift=ms)
    W = (np.random.randn(K, N).astype(np.float32) * (1.0 / np.sqrt(K)))
    ynorm_ref, mu, inv = ln_ref_f32(x)
    ref_out = matmul_true_ln(x, W, ynorm_ref)
    ep = epilogue_correction(x, W)
    print(f"{name:28s} {mu_over_std(x).mean():10.2f} {rel_err(ep, ref_out):24.4e}")

banner("(B) PROLOGUE-APPLY -- normalize-only rel err vs TRUE f32 LN, by mean-delivery encoding")
print("    (measures the normalized A-tile itself; bf16-store floor ~= 4e-3 is the matmul input dtype)")
print(f"{'regime':28s} {'meanmu/std':>10s} {'f32':>11s} {'double_bf16':>13s} {'bf16':>11s}")
for name, ms in regimes:
    x = make_inputs(T, K, mean_shift=ms)
    ynorm_ref, mu, inv = ln_ref_f32(x)
    e_f32 = rel_err(prologue_apply(x, mu, inv, "f32"), ynorm_ref)
    e_dbf = rel_err(prologue_apply(x, mu, inv, "double_bf16"), ynorm_ref)
    e_bf = rel_err(prologue_apply(x, mu, inv, "bf16"), ynorm_ref)
    print(f"{name:28s} {mu_over_std(x).mean():10.2f} {e_f32:11.3e} {e_dbf:13.3e} {e_bf:11.3e}")

banner("(B') PROLOGUE-APPLY -> then bf16 matmul: rel err of the FUSED output vs true-LN-then-matmul")
print(f"{'regime':28s} {'meanmu/std':>10s} {'f32':>11s} {'double_bf16':>13s} {'bf16':>11s}")
for name, ms in regimes:
    x = make_inputs(T, K, mean_shift=ms)
    W = (np.random.randn(K, N).astype(np.float32) * (1.0 / np.sqrt(K)))
    ynorm_ref, mu, inv = ln_ref_f32(x)
    ref_out = matmul_true_ln(x, W, ynorm_ref)
    Wb = to_bf16(W).astype(np.float32)
    for tag in ("f32", "double_bf16", "bf16"):
        a_norm = prologue_apply(x, mu, inv, tag).astype(np.float32)
        globals()[f"_o_{tag}"] = rel_err(a_norm @ Wb, ref_out)
    print(f"{name:28s} {mu_over_std(x).mean():10.2f} {_o_f32:11.3e} {_o_double_bf16:13.3e} {_o_bf16:11.3e}")

banner("(C) K-AUGMENTATION (subtract mean in the f32 accumulator; mean delivered bf16)")
print(f"{'regime':28s} {'meanmu/std':>10s} {'rel_err(kaug fused out)':>24s}")
for name, ms in regimes:
    x = make_inputs(T, K, mean_shift=ms)
    W = (np.random.randn(K, N).astype(np.float32) * (1.0 / np.sqrt(K)))
    ynorm_ref, mu, inv = ln_ref_f32(x)
    ref_out = matmul_true_ln(x, W, ynorm_ref)
    ka = kaug_centered_matmul(x, W, mu, inv)
    print(f"{name:28s} {mu_over_std(x).mean():10.2f} {rel_err(ka, ref_out):24.4e}")

banner("(A2) EPILOGUE-CORRECTION with a DC-BIASED weight (colsum LARGE) -- the true catastrophe")
print("    real weights often have a per-column DC bias -> colsum(W) is O(K*wscale), not O(sqrt(K)),")
print("    so mean*colsum swamps x@W and the bf16 subtraction cancels catastrophically.")
print(f"{'regime':28s} {'meanmu/std':>10s} {'epi-corr':>11s} {'prologue-f32':>13s} {'kaug-bf16':>11s}")
for name, ms in regimes:
    x = make_inputs(T, K, mean_shift=ms)
    # DC-biased weight: every column has a +0.5 offset -> colsum(W)[j] ~ 0.5*K = 512 (huge)
    W = (np.random.randn(K, N).astype(np.float32) * (1.0 / np.sqrt(K))) + 0.5
    ynorm_ref, mu, inv = ln_ref_f32(x)
    ref_out = matmul_true_ln(x, W, ynorm_ref)
    ep = rel_err(epilogue_correction(x, W), ref_out)
    Wb = to_bf16(W).astype(np.float32)
    pr = rel_err(prologue_apply(x, mu, inv, "f32").astype(np.float32) @ Wb, ref_out)
    ka = rel_err(kaug_centered_matmul(x, W, mu, inv), ref_out)
    print(f"{name:28s} {mu_over_std(x).mean():10.2f} {ep:11.3e} {pr:13.3e} {ka:11.3e}")

print("\nDONE. Interpretation in ln_cheap_verdict.md.")
