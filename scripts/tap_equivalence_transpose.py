#!/usr/bin/env python3
"""Offline correctness test for coalescing the TRANSPOSE operator's per-batch DMA
unroll into ONE batched 4D BD (NO NPU). Same idea as tap_equivalence_check.py
(GEMV), for iron/operators/transpose/design.py.

The transpose op unrolls the L3<->L2 fill (taps_in_L3L2) and L1<->L3 drain
(taps_out_L1L3) per batch (a fresh task_group + wait per batch). For the DECODE's
configs (num_columns=num_channels=1, m=M, n=N -> single tile) the tile-grid dims
collapse to size 1, so each per-batch transfer is a CONTIGUOUS run of M*N at
offset batch*M*N (the within-matrix transpose is done by the kernel via the
batch-independent L2L1 TAP, not the L3 DMA). This proves the batched 4D BD
[1, num_batches, run_hi, run_lo] strides [0, M*N, run_lo, 1] enumerates the EXACT
same DRAM index sequence as the num_batches unrolled per-matrix transfers, for
both the in (fill) and out (drain) directions.
"""
import itertools, sys

def enum_tap(offset, sizes, strides):
    out = []
    for idx in itertools.product(*[range(s) for s in sizes]):
        out.append(offset + sum(i * st for i, st in zip(idx, strides)))
    return out

def split_run(n, lim=1023):
    for lo in range(min(lim, n), 0, -1):
        if n % lo == 0 and (n // lo) <= lim:
            return (n // lo, lo)
    raise ValueError(f"cannot split run {n} <= {lim}")

# Mirror transpose/design.py for the decode single-tile case (num_columns=num_channels=1,
# m=M, n=N, i=j=0). num_elements = M*N.
def unrolled_in(M, N, num_batches):
    ne = M * N
    return [enum_tap(batch * ne, [1, 1, M, N], [M * N, N, N, 1]) for batch in range(num_batches)]

def unrolled_out(M, N, num_batches):
    ne = M * N
    return [enum_tap(batch * ne, [1, 1, N, M], [M, N * M, M, 1]) for batch in range(num_batches)]

def coalesced(M, N, num_batches):
    ne = M * N
    rhi, rlo = split_run(ne)
    return enum_tap(0, [1, num_batches, rhi, rlo], [0, ne, rlo, 1])

def check(name, unrolled_list, coalesced_seq):
    flat = [x for seq in unrolled_list for x in seq]
    ok = flat == coalesced_seq
    print(f"  {name}: {'IDENTICAL' if ok else 'MISMATCH'} "
          f"(unrolled {len(flat)} idx, batched {len(coalesced_seq)} idx)")
    if not ok:
        for j, (x, y) in enumerate(zip(flat, coalesced_seq)):
            if x != y:
                print(f"    first diff at pos {j}: unrolled={x} batched={y}"); break
    return ok

# Decode transpose configs (from the K=12 work-dir): M128_N64 and M64_N64, batch=B*H.
allok = True
for B, H in [(16, 12), (128, 12)]:
    nb = B * H
    print(f"Transpose TAP equivalence (B={B}, num_batches=B*H={nb}):")
    for M, N in [(128, 64), (64, 64)]:
        c = coalesced(M, N, nb)
        allok &= check(f"M{M}_N{N} in ", unrolled_in(M, N, nb), c)
        allok &= check(f"M{M}_N{N} out", unrolled_out(M, N, nb), c)
print("RESULT:", "ALL IDENTICAL — batched 4D TAP == per-batch transpose access (offline-equivalent)"
      if allok else "MISMATCH — fix before building")
sys.exit(0 if allok else 1)
