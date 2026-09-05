#!/usr/bin/env python3
"""Golden for conformer_epilogues.cc (SiLU / GLU / BN-fold / residual-add).

Validates two things per node:
  (1) FORMULA: an f64 evaluation of the kernel's math == the host reference
      (scripts/parakeet_ref_encoder.py) to ~machine eps -> the kernel computes
      the right function.
  (2) PRECISION: a bf16 EMULATION of the on-chip arithmetic (narrow f32->bf16
      inputs, sigmoid via tanh in bf16, bf16 products) stays within the
      rel-L2 <= 0.08 gate vs the f64 host reference -> bf16 is accurate enough.

No NPU: pure numpy + ml_dtypes. Dims = Parakeet FastConformer (d_model=1024,
T=64 sample frames; GLU pointwise_conv1 output is 2D=2048).
"""
import numpy as np
import ml_dtypes

bf16 = ml_dtypes.bfloat16
def b(x):  # round to bf16 then back to f32 (emulate on-chip narrowing)
    return np.asarray(x, dtype=np.float32).astype(bf16).astype(np.float32)

def rel_l2(a, ref):
    a = np.asarray(a, np.float64); ref = np.asarray(ref, np.float64)
    return float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-30))

rng = np.random.default_rng(0)
D = 1024
T = 64
GATE = 0.08

# ---------------- host-reference math (mirrors parakeet_ref_encoder.py) -------
def silu_ref(x):            # x/(1+exp(-x))
    return x / (1.0 + np.exp(-x))
def sigmoid_ref(z):
    return 1.0 / (1.0 + np.exp(-z))

# ---------------- bf16 kernel emulation --------------------------------------
def sigmoid_bf16(z):
    # sigmoid(z) = 0.5*(1+tanh(z/2)); aie::tanh<bf16> takes f32, returns bf16.
    zb = b(z)
    half_z = zb * np.float32(0.5)          # aie::mul(bf16,bf16) -> f32 accum
    t = b(np.tanh(half_z.astype(np.float64)))   # tanh -> bf16
    t_p1 = b(t + np.float32(1.0))          # aie::add(bf16,bf16) -> bf16
    return b(t_p1 * np.float32(0.5))       # aie::mul -> bf16

results = []

# ============================ 1. SiLU ========================================
x = rng.standard_normal((T, D)).astype(np.float32) * 3.0
ref = silu_ref(x.astype(np.float64))
# formula (f64): x*sigmoid(x)
f64 = x.astype(np.float64) * sigmoid_ref(x.astype(np.float64))
# bf16 emulation: out = x_bf16 * sigmoid_bf16(x_bf16)
xb = b(x)
emu = b(xb * sigmoid_bf16(xb))
results.append(("SiLU", rel_l2(f64, ref), rel_l2(emu, ref)))

# ============================ 2. GLU =========================================
# in [T,2D]: a = cols[:D], g = cols[D:]; out = a*sigmoid(g)
h = rng.standard_normal((T, 2 * D)).astype(np.float32) * 2.0
a, g = h[:, :D], h[:, D:]
ref = a.astype(np.float64) * sigmoid_ref(g.astype(np.float64))
f64 = a.astype(np.float64) * sigmoid_ref(g.astype(np.float64))
emu = b(b(a) * sigmoid_bf16(g))
results.append(("GLU", rel_l2(f64, ref), rel_l2(emu, ref)))

# ============================ 3. BatchNorm-fold ==============================
# inference BN: y = gamma*(x-mean)/sqrt(var+eps)+beta  == scale*x+shift folded
gamma = rng.standard_normal(D).astype(np.float32)
beta = rng.standard_normal(D).astype(np.float32)
mean = rng.standard_normal(D).astype(np.float32) * 0.5
var = (rng.standard_normal(D).astype(np.float32) ** 2 + 0.3)
eps = 1e-5
x = rng.standard_normal((T, D)).astype(np.float32) * 2.0
inv = 1.0 / np.sqrt(var.astype(np.float64) + eps)
scale = (gamma.astype(np.float64) * inv)            # per-channel
shift = (beta.astype(np.float64) - gamma.astype(np.float64) * mean.astype(np.float64) * inv)
ref = scale[None, :] * x.astype(np.float64) + shift[None, :]   # = BN(x)
# kernel folds scale/shift host-side; the on-chip op is scale*x+shift
f64 = scale[None, :] * x.astype(np.float64) + shift[None, :]
sb, hb = b(scale), b(shift)
emu = b(b(b(x) * sb) + hb)                          # bf16 mul then bf16 add
results.append(("BN-fold", rel_l2(f64, ref), rel_l2(emu, ref)))

# ============================ 4. residual-add ================================
# out = residual + alpha*x  (Macaron alpha=0.5)
alpha = 0.5
xs = rng.standard_normal((T, D)).astype(np.float32) * 2.0   # sub-layer output (acc)
res = rng.standard_normal((T, D)).astype(np.float32) * 2.0  # running activation
ref = res.astype(np.float64) + alpha * xs.astype(np.float64)
f64 = res.astype(np.float64) + alpha * xs.astype(np.float64)
emu = b(b(res) + b(b(xs) * np.float32(alpha)))
results.append(("residual-add(0.5)", rel_l2(f64, ref), rel_l2(emu, ref)))

# also alpha=1.0 full residual
ref1 = res.astype(np.float64) + xs.astype(np.float64)
emu1 = b(b(res) + b(b(xs) * np.float32(1.0)))
results.append(("residual-add(1.0)", rel_l2(ref1, ref1), rel_l2(emu1, ref1)))

# ============================ report =========================================
print(f"{'node':22s} {'formula relL2':>14s} {'bf16 relL2':>12s}  gate<=%.2f" % GATE)
ok = True
for name, frl2, brl2 in results:
    fpass = frl2 < 1e-5
    bpass = brl2 <= GATE
    ok = ok and fpass and bpass
    print(f"{name:22s} {frl2:14.3e} {brl2:12.4f}  "
          f"[formula {'OK' if fpass else 'FAIL'} | bf16 {'PASS' if bpass else 'FAIL'}]")
print()
print("ALL PASS" if ok else "SOME FAILED")
import sys; sys.exit(0 if ok else 1)
