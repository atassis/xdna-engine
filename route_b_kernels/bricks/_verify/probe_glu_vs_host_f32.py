#!/usr/bin/env python3
"""Measure the device GLU against a true-f32 host GLU on identical f32 input.

Closes the attribution by measurement rather than by reading the kernel. Reading glu.cc predicts that
the whole device-vs-host difference is ONE bf16 rounding of the sigmoid value:

    glu.cc keeps `a` un-rounded f32, keeps the tanh argument g/2 in f32, and does the final a*sigmoid
    multiply in f32; only `aie::tanh<bfloat16>` narrows. A sigmoid is bounded in [0,1], so that is
    ~2^-8 relative, and it multiplies straight into the output.

PREDICTION UNDER TEST: device-vs-host rel-L2 lands near 1 bf16 eps (3.906e-03), and -- the sharper
half -- the device output should match a HOST MODEL that reproduces the same bf16 sigmoid far better
than it matches a true-f32 GLU. If the bf16-sigmoid model explains the difference, that gap is large.

Sizes chosen against the CoreTile limits this run already hit: 2 in / 2 out DMA channels (so one input
buffer, one output), dma_bd <= 16383 words, and double-buffered objectFIFOs in a 64 KB L1.

Run from the repo root, single-tenant (npu-vox stopped), under scripts/npu_lock.sh.
"""
import pathlib
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, "route_b_kernels/bricks/_verify")
from bricklib import GEN, iron, _build_oneshot  # noqa: E402

# Sized against the two CoreTile limits this run has already hit twice:
#   dma_bd <= 16383 words -> T*2*D <= 16383   (T=32,D=256 gives exactly 16384: off by one)
#   double-buffered objectFIFOs in 64 KB L1 -> 2*(T*2*D*4) + 2*(T*D*4) = 24*T*D bytes
# T=8, D=256: 4096 words in, 49152 B doubled. Both clear.
T, D = 8, 256
COLS = D

SHIM = r"""
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "%(GLU)s"

// in[T*2D] f32 (value half then gate half PER ROW, as conv pw1 emits), out[T*D] f32
extern "C" void glu_probe(float *restrict in, float *restrict out) {
  for (int t = 0; t < %(T)d; ++t)
    glu_row(in + t * 2 * %(D)d, out + t * %(D)d, %(D)d);
}
""" % {"T": T, "D": D,
       # absolute: the shim compiles from ~/.npu/cache, so a repo-relative include cannot resolve
       "GLU": str(pathlib.Path("mlir-aie/aie_kernels/aie2p/glu.cc").resolve())}


def host_glu_f32(h):
    """True f32 GLU: a * sigmoid(g), the reference the encoder's host path computes."""
    a, g = h[:, :D], h[:, D:]
    return (a * (1.0 / (1.0 + np.exp(-g.astype(np.float64))))).astype(np.float32)


def host_glu_bf16sigmoid(h):
    """Sigmoid VALUE rounded to bf16 -- my first (too loose) model of the kernel."""
    a, g = h[:, :D], h[:, D:]
    sig = 1.0 / (1.0 + np.exp(-g.astype(np.float64)))
    sig_b = sig.astype(ml_dtypes.bfloat16).astype(np.float64)
    return (a * sig_b).astype(np.float32)


def host_glu_kernel_exact(h):
    """The kernel's EXACT sequence: tanh(g/2) -> bf16, then (+1) and (*0.5) also in bf16."""
    a, g = h[:, :D], h[:, D:]
    bf = ml_dtypes.bfloat16
    half_g = (g.astype(np.float32) * np.float32(0.5))
    th = np.tanh(half_g.astype(np.float64)).astype(bf)          # tanh OUTPUT narrowed
    p1 = (th.astype(np.float64) + 1.0).astype(bf)               # add in bf16
    sig = (p1.astype(np.float64) * 0.5).astype(bf)              # mul in bf16
    return (a * sig.astype(np.float64)).astype(np.float32)


def host_glu_tanh_approx(h, bits):
    """Is the residual a TANH APPROXIMATION error rather than output rounding? Model a tanh
    accurate to `bits` mantissa bits, keeping everything else f32."""
    a, g = h[:, :D], h[:, D:]
    th = np.tanh((g.astype(np.float64) * 0.5))
    q = 2.0 ** -bits
    th_q = np.round(th / q) * q                                  # quantise the tanh VALUE
    sig = 0.5 * (1.0 + th_q)
    return (a * sig).astype(np.float32)


def rel(x, y):
    return float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30))


def main():
    rng = np.random.default_rng(20260802)
    shim = GEN / "glu_probe_shim.cc"
    shim.write_text(SHIM)
    design = _build_oneshot("glu_probe", shim, [T * 2 * D], T * D,
                            [np.float32], np.float32, [])

    print(f"{'case':>5}{'vs f32':>11}{'vs bf16sig':>13}{'vs KERNEL-EXACT':>17}{'vs tanh@6b':>12}")
    for case in range(4):
        h = (rng.standard_normal((T, 2 * D)) * 1.5).astype(np.float32)
        it = iron.tensor(np.ascontiguousarray(h.reshape(-1)), dtype=np.float32, device="npu")
        ot = iron.zeros((T * D,), dtype=np.float32, device="npu")
        design(it, ot)
        dev = ot.numpy().reshape(T, D)

        e_f32 = rel(host_glu_f32(h), dev)
        e_bf16 = rel(host_glu_bf16sigmoid(h), dev)
        e_exact = rel(host_glu_kernel_exact(h), dev)
        e_apx = rel(host_glu_tanh_approx(h, 6), dev)
        print(f"{case:>5}{e_f32:>11.3e}{e_bf16:>13.3e}{e_exact:>17.3e}{e_apx:>12.3e}")

    print(f"\nbf16 eps = {2**-8:.3e}")
    print("A model that MATCHES the device drives its column toward zero.")
    print("If KERNEL-EXACT is no better than vs-f32, the residual is not output rounding at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
