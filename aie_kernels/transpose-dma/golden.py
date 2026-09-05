#!/usr/bin/env python3
"""Golden for transpose_dma.cc (transpose-dma brick, movement group).

Host numpy reference for the LATER device pass to check rel-L2 against
(no NPU here -- pure numpy, CPU-only spike per the leaf's mandate).

This brick is PURE DATA MOVEMENT (no arithmetic) -- on real hardware the
relayout is expressed as dma_bd stride/wrap dimension registers and costs
the shim/mem-tile DMA its ordinary bandwidth, not extra core cycles. The
CPU kernel (transpose_dma.cc) and this golden are therefore BIT-EXACT
copies of each other's addressing math; there is no precision loss to
budget for (unlike an activation/quantization brick), so the rel-L2 gate
here is a correctness tripwire, not a numerics-tolerance knob.

Two checks:
  (1) 2-D TRANSPOSE instance -- transpose_dma_2d's exact case, matching
      transpose_tile.cc's [mb, nb] -> [nb, mb] contract bit-for-bit, so this
      brick is a verified drop-in for that compute-tile fallback.
  (2) GENERIC 4-D STRIDED instance -- a block-scatter-style relayout that a
      plain 2-D transpose kernel could NOT express, exercising the same
      code path (transpose_dma's up-to-4-dim stride model) a real dma_bd
      would use for e.g. tiled/interleaved layouts, not just transpose.

Dims: representative resident-tile shape T=64 rows, D=1024 cols (matches
glu/silu/relu2 goldens in sibling bricks) for the 2-D case; a smaller
explicit shape for the 4-D case so the nested-stride math is legible.
"""
import numpy as np


def rel_l2(a, ref):
    a = np.asarray(a, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-30))


REL_L2_GATE = 3e-2  # correctness tripwire for the later on-device BD pass
                     # (pure permutation -> exact match expected, i.e. 0.0;
                     # kept as a rel-L2 gate for a uniform verify harness
                     # across bricks, not because any loss is anticipated)

rng = np.random.default_rng(0)


# ---------------- kernel emulation: mirrors transpose_dma.cc exactly -----
def transpose_dma_emulate(x_flat, d0, d1, d2, d3, is_, os_, out_size):
    """Element-for-element emulation of the C++ 4-nested-loop kernel body.

    x_flat: 1-D array of input elements (addressed via `is_` strides).
    is_ = (is0, is1, is2, is3), os_ = (os0, os1, os2, os3) -- element strides.
    Returns a 1-D output array of length out_size (addressed via `os_`).
    """
    is0, is1, is2, is3 = is_
    os0, os1, os2, os3 = os_
    out = np.zeros(out_size, dtype=x_flat.dtype)
    for i0 in range(d0):
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    in_addr = i0 * is0 + i1 * is1 + i2 * is2 + i3 * is3
                    out_addr = i0 * os0 + i1 * os1 + i2 * os2 + i3 * os3
                    out[out_addr] = x_flat[in_addr]
    return out


# ---------------- (1) 2-D transpose instance ------------------------------
T, D = 64, 1024  # [mb=T, nb=D] resident tile, matches sibling-brick dims
mb, nb = T, D
x2d = rng.standard_normal((mb, nb)).astype(np.float32)
x2d_flat = x2d.reshape(-1)

ref_transpose = x2d.T  # host reference: numpy transpose

emu_flat = transpose_dma_emulate(
    x2d_flat, mb, nb, 1, 1,
    is_=(nb, 1, 0, 0), os_=(1, mb, 0, 0),
    out_size=mb * nb,
)
emu_2d = emu_flat.reshape(nb, mb)  # out shape is [nb, mb] per the transpose contract

r_2d = rel_l2(emu_2d, ref_transpose)


# ---------------- (2) generic 4-D strided (block-scatter) instance -------
# A [2, 3, 4] logical tensor block-scattered into a differently-strided
# output layout -- exercises the 4-dim path a plain 2-D transpose can't.
s0, s1, s2 = 2, 3, 4
x4 = rng.standard_normal((s0, s1, s2)).astype(np.float32)
x4_flat = x4.reshape(-1)

# input strides = C-contiguous strides of (s0, s1, s2)
is4 = (s1 * s2, s2, 1, 0)
# output: permute to (s1, s2, s0) layout (a relayout no 2-D transpose covers).
# out is C-contiguous (s1, s2, s0); addr(i1,i2,i0) = i1*(s2*s0) + i2*s0 + i0,
# so as a function of the ITERATION order (i0,i1,i2): os0=1, os1=s2*s0, os2=s0.
os4 = (1, s2 * s0, s0, 0)
out_size4 = s0 * s1 * s2

ref_permute = np.transpose(x4, (1, 2, 0))  # numpy reference permutation
emu4_flat = transpose_dma_emulate(
    x4_flat, s0, s1, s2, 1, is_=is4, os_=os4, out_size=out_size4
)
emu4 = emu4_flat.reshape(s1, s2, s0)

r_4d = rel_l2(emu4, ref_permute)


if __name__ == "__main__":
    print(f"transpose_dma golden: 2-D case mb={mb} nb={nb}")
    print(f"  2-D transpose vs numpy .T:      rel_l2={r_2d:.3e}  (gate={REL_L2_GATE:.1e})")
    print(f"transpose_dma golden: 4-D case shape={(s0, s1, s2)} -> perm(1,2,0)")
    print(f"  4-D strided vs numpy transpose: rel_l2={r_4d:.3e}  (gate={REL_L2_GATE:.1e})")
    status = "PASS" if (r_2d <= REL_L2_GATE and r_4d <= REL_L2_GATE) else "FAIL"
    print(f"  -> {status}")
    assert r_2d <= REL_L2_GATE, f"2-D transpose rel_l2 {r_2d:.3e} exceeds gate {REL_L2_GATE:.1e}"
    assert r_4d <= REL_L2_GATE, f"4-D strided rel_l2 {r_4d:.3e} exceeds gate {REL_L2_GATE:.1e}"
