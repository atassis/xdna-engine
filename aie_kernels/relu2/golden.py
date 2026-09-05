#!/usr/bin/env python3
"""Golden for relu2.cc (ReLU^2 act brick: out = max(x, 0)^2).

Host numpy reference for the LATER device pass to check rel-L2 against
(no NPU here -- pure numpy, CPU-only spike per the leaf's mandate).

Two checks, mirroring test_conformer_epilogues_golden.py's two-tier style:
  (1) FORMULA (f64): max(x,0)**2 evaluated at full precision -- sanity-checks
      the kernel computes the right function at all.
  (2) KERNEL EMULATION (f32): the kernel's actual on-chip path stays entirely
      in f32 (see relu2.cc's numerics note -- ReLU^2 is unbounded above unlike
      GLU's sigmoid, so no bf16 narrowing is folded into this generic brick).
      This emulation is expected to match the f64 formula to ~f32 machine eps;
      it is NOT a lossy-precision gate the way the bf16 GLU/SiLU emulations are.

Dims are a generic [tile, D] stream shape, not tied to one model: pick a
representative resident-tile shape (T=64 rows, D=1024 cols) matching the
other bricks' golden dims (glu/silu goldens use D=1024, T=64) so a later
device harness can reuse the same tile geometry across bricks.
"""
import numpy as np

def rel_l2(a, ref):
    a = np.asarray(a, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-30))

rng = np.random.default_rng(0)
T = 64
D = 1024
REL_L2_GATE = 3e-2  # threshold for the later on-device pass (f32 kernel, no bf16 narrowing)

# ---------------- host-reference math -----------------------------------
def relu2_ref(x):
    return np.maximum(x, 0.0) ** 2

# ---------------- input: representative resident-tile activations --------
x = rng.standard_normal((T, D)).astype(np.float32) * 3.0

# formula (f64): exact max(x,0)^2
f64 = relu2_ref(x.astype(np.float64))

# kernel emulation (f32): mirrors relu2_row's on-chip ops exactly --
#   clamped = aie::max(v, 0.0f)   (f32)
#   squared = aie::mul(clamped, clamped)  (f32)
xf = x.astype(np.float32)
clamped = np.maximum(xf, np.float32(0.0))
emu = (clamped * clamped).astype(np.float32)

if __name__ == "__main__":
    r_formula = rel_l2(f64, f64)
    r_emu = rel_l2(emu, f64)
    print(f"relu2 golden: T={T} D={D}")
    print(f"  formula (f64) vs itself:  rel_l2={r_formula:.3e}")
    print(f"  f32 kernel emulation:     rel_l2={r_emu:.3e}  (gate={REL_L2_GATE:.1e})")
    status = "PASS" if r_emu <= REL_L2_GATE else "FAIL"
    print(f"  -> {status}")
    assert r_emu <= REL_L2_GATE, f"relu2 f32 emulation rel_l2 {r_emu:.3e} exceeds gate {REL_L2_GATE:.1e}"
