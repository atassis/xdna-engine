#!/usr/bin/env python3
"""Golden for the RESIDENT-INTERMEDIATE Whisper-encoder FFN draft (ctx2 FfnMm2::forward_resident).

Validates two things on the FFN node  f = gelu(ln2 @ W1 + b1) @ W2 + b2  (Whisper-small shapes,
d_model=768, d_ff=3072, one PAD_M=512 row-tile):

  (A) GATE  rel-L2(resident_emulation, f32_host_reference) <= 0.08
      -- the resident path's only numerically-relevant change vs an idealized f32 FFN is the bf16
         truncation of the [M,3072] intermediate (already true of the shipped non-resident NPU path)
         plus the on-chip bf16 GEMMs; this must stay well inside the engine's WER-safe rel gate.

  (B) IDENTITY  resident_emulation == nonresident_emulation  (bitwise on the f32 result)
      -- the resident reorganization (hold the activated intermediate in ONE bf16 buffer, fc2 reads
         bf16 column-slices) consumes a BYTE-IDENTICAL intermediate to the non-resident path
         (fc2 there re-converts the same activated f32 -> bf16), so it introduces ZERO numerical
         change. This is the core correctness claim of the draft.

No NPU required (CPU-only emulation of the kernel's bf16 matmul + on-chip GELU + K-split accumulate).
"""
import numpy as np

D_MODEL, D_FF, M = 768, 3072, 512
KA = 768
KSPLIT = D_FF // KA  # 4


def bf16(x):
    """Round f32 -> bf16 (round-to-nearest-even on the top 16 bits), back to f32. Matches the
    Rust npu_xrt f32_to_bf16_bits / pack_f32_to_bf16 used for every device activation/weight BO."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    rounding = ((u >> 16) & 1) + np.uint64(0x7FFF)
    u = (u + rounding) >> 16 << 16
    return u.astype(np.uint32).view(np.float32)


def gelu(x):
    # Whisper uses the exact (erf) GELU; the host reference is gelu() in npu_asr_host.
    from math import sqrt
    return 0.5 * x * (1.0 + np.vectorize(_erf)(x / sqrt(2.0))).astype(np.float32)


def _erf(v):
    # std erf via math (vectorized above); keeps numpy-only deps.
    import math
    return math.erf(v)


def f32_reference(ln2, W1, b1, W2, b2):
    """The host f32 FFN node (the reference)."""
    h = gelu(ln2 @ W1 + b1)
    return (h @ W2 + b2).astype(np.float32)


def kernel_matmul(a_f32, w_f32):
    """Emulate one ctx2 bf16 GEMM: bf16-round both operands, accumulate in f32 (the device matmul
    accumulates in f32; inputs are the bf16 BOs)."""
    return (bf16(a_f32) @ bf16(w_f32)).astype(np.float32)


def nonresident_emulation(ln2, W1, b1, W2, b2):
    """The shipped path: fc1 (on-chip GELU, modal) -> f32 intermediate -> fc2 re-converts each
    768-col slice to bf16, K-split accumulate in f32, +b2 once."""
    z = kernel_matmul(ln2, W1) + b1            # fc1 raw (bias K-aug'd on-chip)
    h = gelu(z).astype(np.float32)             # on-chip GELU (modal mode=2), f32 out
    acc = np.zeros((ln2.shape[0], D_MODEL), dtype=np.float32)
    for i in range(KSPLIT):
        hk = h[:, i * KA:(i + 1) * KA]         # fc2 re-converts this f32 slice -> bf16 in the BO
        acc += kernel_matmul(hk, W2[i * KA:(i + 1) * KA, :])
    return (acc + b2).astype(np.float32)


def resident_emulation(ln2, W1, b1, W2, b2):
    """The resident draft: fc1's activated output is packed ONCE to a [M,3072] bf16 buffer (the
    resident intermediate); fc2 reads bf16 column-slices of it directly (no re-conversion)."""
    z = kernel_matmul(ln2, W1) + b1
    h = gelu(z).astype(np.float32)
    inter = bf16(h)                            # packed ONCE into the resident bf16 buffer
    acc = np.zeros((ln2.shape[0], D_MODEL), dtype=np.float32)
    for i in range(KSPLIT):
        hk = inter[:, i * KA:(i + 1) * KA]     # already bf16 -> fed straight to the BO (just bf16())
        acc += (hk @ bf16(W2[i * KA:(i + 1) * KA, :])).astype(np.float32)
    return (acc + b2).astype(np.float32)


def rel_l2(a, b):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))


def main():
    rng = np.random.default_rng(0)
    # Whisper-scale activations/weights (LN output ~unit, weights ~1/sqrt(K)).
    ln2 = rng.standard_normal((M, D_MODEL)).astype(np.float32)
    W1 = (rng.standard_normal((D_MODEL, D_FF)) / np.sqrt(D_MODEL)).astype(np.float32)
    b1 = (rng.standard_normal(D_FF) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((D_FF, D_MODEL)) / np.sqrt(D_FF)).astype(np.float32)
    b2 = (rng.standard_normal(D_MODEL) * 0.1).astype(np.float32)

    ref = f32_reference(ln2, W1, b1, W2, b2)
    nonres = nonresident_emulation(ln2, W1, b1, W2, b2)
    res = resident_emulation(ln2, W1, b1, W2, b2)

    gate = rel_l2(res, ref)
    identity = rel_l2(res, nonres)
    bitwise = np.array_equal(res, nonres)

    print(f"(A) GATE     rel-L2(resident, f32_reference)      = {gate:.5f}   (gate <= 0.08)")
    print(f"(B) IDENTITY rel-L2(resident, nonresident)        = {identity:.3e}")
    print(f"    BITWISE  resident == nonresident              = {bitwise}")
    print(f"    aux      rel-L2(nonresident, f32_reference)   = {rel_l2(nonres, ref):.5f}")

    ok = gate <= 0.08 and bitwise
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
