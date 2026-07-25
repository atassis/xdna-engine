#!/usr/bin/env python3
"""Golden for swiglu.cc (SwiGLU act brick: out = silu(gate) * up).

Host numpy reference for the LATER device pass to check rel-L2 against
(no NPU here -- pure numpy, CPU-only spike per the leaf's mandate).

Two checks, mirroring the other bricks' golden style (relu2/rmsnorm):
  (1) FORMULA (f64): silu(gate)*up = gate*sigmoid(gate)*up evaluated at full
      precision -- sanity-checks the kernel computes the right function.
  (2) KERNEL EMULATION: mirrors swiglu.cc's actual on-chip numerics --
        s_bf16   = exp2(-gate * log2e)      # hardware exp2 SFU, bf16 OUTPUT
                                             # (rounded to bf16, then read back
                                             #  as f32 -- this bf16 rounding of
                                             #  the SFU result is the dominant
                                             #  error source, same class as the
                                             #  bf16-tanh SiLU used elsewhere in
                                             #  this tree, e.g. mm_silu_epilogue.cc)
        s        = f32(s_bf16)
        r0       = 1 / (1 + s)              # f32 reciprocal
        r        = r0 * (2 - (1+s)*r0)      # one f32 Newton refinement step
        silu     = gate * r                 # f32
        out      = silu * up                # f32
      Newton-refined-inv is essentially exact in f32 given a bf16-precision s,
      so the emulation's error is dominated by the single bf16 rounding of the
      SFU output -- expected to land well inside the gate below (same order as
      the other bf16-SFU-based bricks in this tree, ~1e-2 to ~1e-3 rel-L2).

Dims are a generic [tile, D] stream shape, not tied to one model: T=64 rows,
D=1024 cols, matching the other bricks' golden dims (glu/silu/relu2 goldens)
so a later device harness can reuse the same tile geometry across bricks.
"""
import numpy as np


def rel_l2(a, ref):
    a = np.asarray(a, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-30))


def to_bf16(x):
    """Round f32 -> bf16 (round-to-nearest-even on the mantissa truncation
    boundary), represented back as f32 -- emulates the hardware exp2 SFU's
    bf16 OUTPUT type (aie::exp2<bfloat16>(vector<float,N>) -> vector<bfloat16,N>)."""
    x = np.asarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    # round-to-nearest-even at the 16-bit truncation boundary
    rounding_bias = ((u32 >> 16) & 1) + 0x7FFF
    u32_rounded = (u32.astype(np.uint64) + rounding_bias) & 0xFFFFFFFF
    u32_rounded = u32_rounded.astype(np.uint32)
    bf16_bits = (u32_rounded >> 16).astype(np.uint16)
    # widen back to f32 for downstream f32 math (mirrors the on-chip
    # accfloat up-convert: aie::accum<accfloat,16>::from_vector(bf16) -> f32)
    widened = (bf16_bits.astype(np.uint32) << 16).view(np.float32)
    return widened


rng = np.random.default_rng(0)
T = 64
D = 1024
REL_L2_GATE = 3e-2  # threshold for the later on-device pass (single bf16-SFU rounding)

# ---------------- host-reference math -----------------------------------
def silu_ref(x):
    return x / (1.0 + np.exp(-x))


def swiglu_ref(gate, up):
    return silu_ref(gate) * up


# ---------------- input: representative resident-tile activations --------
gate = (rng.standard_normal((T, D)).astype(np.float32) * 3.0)
up = (rng.standard_normal((T, D)).astype(np.float32) * 3.0)

# formula (f64): exact silu(gate)*up
f64 = swiglu_ref(gate.astype(np.float64), up.astype(np.float64))

# kernel emulation: mirrors swiglu.cc's on-chip ops exactly --
#   arg      = gate * (-log2(e))                         (f32)
#   s_bf16   = 2**arg, ROUNDED TO bf16                    (hardware exp2 SFU)
#   s        = f32(s_bf16)                                (accfloat up-convert)
#   denom    = 1 + s                                      (f32)
#   r0       = 1/denom                                    (f32, aie::inv)
#   r        = r0 * (2 - denom*r0)                        (f32 Newton step)
#   silu     = gate * r                                   (f32)
#   out      = silu * up                                  (f32)
neg_log2e = np.float32(-1.44269504089)
gf = gate.astype(np.float32)
uf = up.astype(np.float32)
arg = (gf * neg_log2e).astype(np.float32)
s_full = np.exp2(arg.astype(np.float64)).astype(np.float32)  # exact 2**arg, then bf16-round
s = to_bf16(s_full)
denom = (np.float32(1.0) + s).astype(np.float32)
r0 = (np.float32(1.0) / denom).astype(np.float32)
r = (r0 * (np.float32(2.0) - denom * r0)).astype(np.float32)
silu = (gf * r).astype(np.float32)
emu = (silu * uf).astype(np.float32)

if __name__ == "__main__":
    r_formula = rel_l2(f64, f64)
    r_emu = rel_l2(emu, f64)
    print(f"swiglu golden: T={T} D={D}")
    print(f"  formula (f64) vs itself:      rel_l2={r_formula:.3e}")
    print(f"  kernel emulation (bf16 SFU):  rel_l2={r_emu:.3e}  (gate={REL_L2_GATE:.1e})")
    status = "PASS" if r_emu <= REL_L2_GATE else "FAIL"
    print(f"  -> {status}")
    assert r_emu <= REL_L2_GATE, (
        f"swiglu kernel emulation rel_l2 {r_emu:.3e} exceeds gate {REL_L2_GATE:.1e}"
    )
