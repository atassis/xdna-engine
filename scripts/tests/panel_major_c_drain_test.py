#!/usr/bin/env python3
"""CPU gate for the modal GEMM's panel-major C drain (--c-panel-width).

The panel-major drain folds the fc1->fc2 deinterleave into the GEMM's own output DMA:
instead of writing C row-major as [M, N] and then running a separate deinterleave+cast
dispatch to build [N/W, M, W], the shim writes that layout directly. That deletes a whole
dispatch -- and, because the K-panel packing lives in its own xclbin, the two hardware-context
transitions around it.

The re-stride is only correct if the drain TAP's group dimension coincides with the chunk
boundary, so this test proves the permutation element-for-element rather than eyeballing
strides. It is CPU-only (no NPU, no build): it inspects the generator's TAPs directly.

    python3 scripts/tests/panel_major_c_drain_test.py

Needs the IRON toolchain on PYTHONPATH -- run it under `source scripts/iron_env.sh`.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "route_b_kernels", "whole_array_fused"))

import whole_array_modal_iron as G  # noqa: E402

# The shipped Parakeet fc1: M=PAD_M, K=KRES, N=DFF, 8 columns. Both tile shapes are exercised:
# the C drain's OUTER size is the tile-block group the runtime loop drains at a time, which
# equals M//(m*n_aie_rows) at m=64 but not at m=32 -- a re-stride that assumes the former
# silently mis-permutes C at the latter, so the gate has to see both.
M, K, N = 512, 1024, 4096
k = 32
n = 128  # n_aie_cols*n must equal the panel width; at 8 columns that pins n to 128
N_AIE_COLS = 8
CHUNK_W = 1024  # == KRES == n_aie_cols*n
TILE_MS = (64, 32)

failures = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        failures.append(msg)


def taps(c_panel_width, m):
    _, _, C = G.my_matmul(
        "npu2", M, K, N, m, k, n, N_AIE_COLS, 0, True, True, 0,
        generate_taps=True, c_panel_width=c_panel_width,
    )
    return list(C)


print("panel-major C drain gate")
for m in TILE_MS:
    print(f"\n--- tile {m}x{k}x{n}, {N_AIE_COLS} cols ---")
    row_major = taps(0, m)
    panel_major = taps(CHUNK_W, m)

    # 1. The re-stride must not change the rank, the sizes or the walk order -- those are what
    #    the C objectFIFO feeds the shim, and they are the reason this is an instruction-stream-
    #    only change that leaves the array program (and so the xclbin) alone.
    check(len(row_major) == len(panel_major), "same number of C drains")
    check(
        all(list(a.sizes) == list(b.sizes) for a, b in zip(row_major, panel_major)),
        "same TAP sizes (same walk order off the fifo)",
    )
    check(
        all(len(b.sizes) <= 4 for b in panel_major),
        "panel-major TAPs stay within the shim BD's 4 dimensions",
    )
    check(
        max(max(b.strides) for b in panel_major)
        <= max(max(a.strides) for a in row_major),
        "panel-major strides do not exceed the row-major ones already in production",
    )

    # 2. The permutation itself, element for element: the k-th datum the fifo hands the shim
    #    must land where a separate packing command would have put it: out[c,i,j] = C[i, c*W+j].
    n_panels = N // CHUNK_W
    bad = 0
    for d, (a, b) in enumerate(zip(row_major, panel_major)):
        for src, dst in zip(a.access_generator(), b.access_generator()):
            i, j_full = divmod(int(src), N)
            want = (j_full // CHUNK_W) * M * CHUNK_W + i * CHUNK_W + (j_full % CHUNK_W)
            if int(dst) != want:
                bad += 1
                if bad <= 3:
                    print(f"       drain {d}: C[{i},{j_full}] -> {dst}, want {want}")
    check(bad == 0, f"every element lands at its packed address ({bad} mismatches)")

    # 3. The drains must still tile the destination exactly once -- no gap, no double write.
    cover = np.zeros(n_panels * M * CHUNK_W, dtype=np.int32)
    for b in panel_major:
        for idx in b.access_generator():
            cover[idx] += 1
    check(
        cover.min() == 1 and cover.max() == 1,
        "the drains cover the panel-major buffer exactly once",
    )

    # 4. End to end against a numpy K-panel packing, so the gate fails on a wrong permutation even
    #    if somebody rewrites the checks above.
    rng = np.random.default_rng(0)
    c_ref = np.asarray(rng.standard_normal((M, N)), dtype=np.float32)
    flat = c_ref.reshape(-1)
    out = np.zeros(n_panels * M * CHUNK_W, dtype=np.float32)
    for a, b in zip(row_major, panel_major):
        for src, dst in zip(a.access_generator(), b.access_generator()):
            out[dst] = flat[src]
    want = np.stack(
        [c_ref[:, c * CHUNK_W:(c + 1) * CHUNK_W] for c in range(n_panels)]
    ).reshape(-1)
    check(np.array_equal(out, want), "simulated drain == numpy K-panel packing of the same C")

# 5. The alignment requirement is load-bearing, so it must fail loud rather than mis-permute.
try:
    taps(CHUNK_W // 2, TILE_MS[0])
    check(False, "a panel width != n_aie_cols*n is rejected")
except AssertionError:
    check(True, "a panel width != n_aie_cols*n is rejected")

print()
if failures:
    print(f"GATE RED ({len(failures)} failed)")
    sys.exit(1)
print("GATE GREEN")
