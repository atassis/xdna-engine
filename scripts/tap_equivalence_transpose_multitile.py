#!/usr/bin/env python3
"""Offline correctness derivation for the MULTI-TILE transpose coalesce (production S=448 / TP=1536).

Single-tile (scripts/tap_equivalence_transpose.py) works because the per-batch L3 transfer is a
contiguous M*N run. Production transposes (M*N > 8192 tile cap) are MULTI-TILE: m < M, so the
per-batch in/out TAPs have a grid dim. The IN is still contiguous (grid_row stride = m*N = inner
block → telescopes), so the IN coalesce is unchanged. The OUT is a transpose-SCATTER whose
enumeration order ([grid_row, n, m] per batch) is NOT contiguous and MUST be preserved, with batch
prepended as the OUTERMOST iteration (the drain consumes the FIFO in kernel-output order = batch
outer, then tiles).

AIE2p BD caps (empirical, from the GEMV bring-up): dim size ≤1023, ONE size-uncapped dim, the
ITERATION (outermost) dim ≤64, stride ≤2**20. Batch must be outermost (order) but the outermost dim
is ≤64-capped → a single BD can't hold num_batches>64. So the OUT is BATCH-CHUNKED: ceil(nb/64) BDs,
each [chunk≤64, grid_row, n, m] strides [num_elements, m, n*M, 1]. This script proves the chunked OUT
BDs concatenate to the EXACT per-batch unrolled OUT order, and the IN single coalesce still matches.
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

# ---- IN (fill): per-batch is [M//m, 1, m, n] strides [m*N, n, N, 1] (grid_row, 1, m, n). ----
def unrolled_in(M, N, m, n, nb):
    ne = M * N
    gr = M // m  # grid rows (N//n == 1 for decode since n==N==HD)
    return [enum_tap(b * ne, [gr, 1, m, n], [m * N, n, N, 1]) for b in range(nb)]

def coalesced_in(M, N, nb):  # contiguous M*N run, batch in the uncapped dim (single-tile layout)
    ne = M * N
    rhi, rlo = split_run(ne)
    return enum_tap(0, [1, nb, rhi, rlo], [0, ne, rlo, 1])

# ---- OUT (drain): per-batch is [M//m, 1, n, m] strides [m, n*M, M, 1] (grid_row, 1, n, m). ----
def unrolled_out(M, N, m, n, nb):
    ne = M * N
    gr = M // m
    return [enum_tap(b * ne, [gr, 1, n, m], [m, n * M, M, 1]) for b in range(nb)]

def chunked_out(M, N, m, n, nb, chunk=64):
    ne = M * N
    gr = M // m
    seqs = []
    for c0 in range(0, nb, chunk):
        cb = min(chunk, nb - c0)
        # [batch(<=64, OUTER), grid_row, n, m] strides [ne, m, M, 1] — batch outermost preserves order.
        # (grid_row stride = m, n stride = M, m stride = 1; the original size-1 dim's n*M stride is dropped.)
        seqs.append(enum_tap(c0 * ne, [cb, gr, n, m], [ne, m, M, 1]))
    return seqs

def flat(seqs):
    return [x for s in seqs for x in s]

def check(name, a_flat, b_flat):
    ok = a_flat == b_flat
    print(f"  {name}: {'IDENTICAL' if ok else 'MISMATCH'} ({len(a_flat)} vs {len(b_flat)} idx)")
    if not ok:
        for j, (x, y) in enumerate(zip(a_flat, b_flat)):
            if x != y:
                print(f"    first diff at pos {j}: unrolled={x} candidate={y}"); break
    return ok

# Production decode multi-tile configs. n == N == HD == 64 (so N//n==1, grid only on M).
# m = largest divisor of M with m*n <= 8192 (m <= 128). The generator's pick_tt picks this.
HD = 64
def pick_m(M, n=HD, cap=8192):
    for m in range(min(cap // n, M), 0, -1):
        if M % m == 0:
            return m
    raise ValueError(M)

# Order-correctness is scale-independent: we only need nb to cross the 64-batch chunk boundary.
# nb=192 (B=16, 3 chunks) exercises the chunk concatenation; the full nb=1536 is identical logic
# (more chunks) and ~150M indices, so we don't enumerate it. The cap on chunk-count (24 at B=128) is
# noted, not enumerated.
allok = True
for label, M in [("tr_s S=448", 448), ("tr_c TP=1536", 1536)]:
    n = HD
    m = pick_m(M)
    for nb in [192, 200]:  # 192 = B=16; 200 = non-multiple-of-64 (3 full + 1 partial chunk)
        gr = M // m
        nch = len(chunked_out(M, n, m, n, nb))
        print(f"{label}: M={M} N={n} m={m} (grid_rows={gr}, multi-tile={gr>1}) nb={nb} -> {nch} drain BDs")
        allok &= check("IN  coalesce", flat(unrolled_in(M, n, m, n, nb)), coalesced_in(M, n, nb))
        allok &= check("OUT chunked(64)",
                       flat(unrolled_out(M, n, m, n, nb)), flat(chunked_out(M, n, m, n, nb)))
print("RESULT:", "ALL IDENTICAL — multi-tile IN coalesce + batch-chunked OUT == per-batch access"
      if allok else "MISMATCH — design wrong, do NOT build")
sys.exit(0 if allok else 1)
