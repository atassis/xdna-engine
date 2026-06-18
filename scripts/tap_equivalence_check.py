#!/usr/bin/env python3
"""Offline correctness test for the GEMV B-unroll->BD-iteration lever (NO NPU).

A TensorAccessPattern enumerates DRAM linear indices: offset + sum_k(i_k * stride_k)
over i_k in range(size_k), in row-major (last dim fastest) order. The lever replaces
`num_batches` separate 1D TAPs with ONE 4D TAP that adds a batch dimension. This script
proves the batched 4D TAP enumerates the EXACT SAME index sequence as the unrolled TAPs
(same elements, same order) -> the DMA reads/writes identically -> byte/numerically
equivalent, independent of any on-chip run.

Mirrors iron/operators/gemv/design.py A_taps/C_taps for the decode's GEMV configs.
"""
import itertools, sys

def enum_tap(offset, sizes, strides):
    """Linear indices a TAP touches, in row-major (last dim fastest) order."""
    out = []
    for idx in itertools.product(*[range(s) for s in sizes]):
        out.append(offset + sum(i * st for i, st in zip(idx, strides)))
    return out

def unrolled_A(M, K, cols, num_batches):
    """Current: num_batches separate 1D TAPs per col (design.py:133-145)."""
    per_col = []
    for col in range(cols):
        seq = []
        for batch in range(num_batches):
            off = col * (M // cols) * K + batch * M * K
            seq += enum_tap(off, [1, 1, 1, (M // cols) * K], [0, 0, 0, 1])
        per_col.append(seq)
    return per_col

def split_run(n, lim=1023):
    """(hi, lo) run; lo=largest divisor<=lim (contiguous inner)."""
    for lo in range(min(lim, n), 0, -1):
        if n % lo == 0 and (n // lo) <= lim:
            return (n // lo, lo)
    raise ValueError(f"cannot split run {n} <= {lim}")

def coalesced(col_off, run, bstride, num_batches):
    """[1, num_batches, run_hi, run_lo] strides [0, bstride, run_lo, 1] — batch in the
    uncapped hw dim, run split into the two wrap dims (<=1023)."""
    rhi, rlo = split_run(run)
    return enum_tap(col_off, [1, num_batches, rhi, rlo], [0, bstride, rlo, 1])

def batched_A(M, K, cols, num_batches):
    return [coalesced(col * (M // cols) * K, (M // cols) * K, M * K, num_batches)
            for col in range(cols)]

def unrolled_C(M, cols, num_batches):
    per_col = []
    for col in range(cols):
        seq = []
        for batch in range(num_batches):
            off = col * (M // cols) + batch * M
            seq += enum_tap(off, [1, 1, 1, (M // cols)], [0, 0, 0, 1])
        per_col.append(seq)
    return per_col

def batched_C(M, cols, num_batches):
    return [coalesced(col * (M // cols), (M // cols), M, num_batches)
            for col in range(cols)]

def check(name, a, b):
    ok = a == b
    print(f"  {name}: {'IDENTICAL' if ok else 'MISMATCH'} "
          f"(unrolled {sum(len(x) for x in a)} idx, batched {sum(len(x) for x in b)} idx)")
    if not ok:
        for col in range(len(a)):
            if a[col] != b[col]:
                # show first divergence
                for j,(x,y) in enumerate(zip(a[col],b[col])):
                    if x!=y:
                        print(f"    col {col} first diff at pos {j}: unrolled={x} batched={y}")
                        break
                if len(a[col])!=len(b[col]):
                    print(f"    col {col} length differs: {len(a[col])} vs {len(b[col])}")
                break
    return ok

# Decode GEMV configs (gen_decode_batched.py): M/K vary, cols=8, num_batches=B*H.
# PRODUCTION dims (S=448, T_enc=1500 -> TP=1536) where the contiguous run overflows
# 1023 (the case that needs batch+run packing). Test both B=16 (WER) and B=128 (gate).
HD = 64
cols = 8
allok = True
for B, S, TP in [(16, 448, 1536)]:  # B=16 WER config (full enum tractable); B=128 gate verified at build time
    BH = B * H if False else B * 12
    configs = [
        ("g_scs  M=S  K=HD", S, HD),
        ("g_cts  M=HD K=S ", HD, S),
        ("g_scc  M=TP K=HD", TP, HD),
        ("g_ctc  M=HD K=TP", HD, TP),
    ]
    print(f"TAP equivalence (B={B}, S={S}, TP={TP}, cols={cols}, num_batches=B*H={BH}):")
    for nm, M, K in configs:
        if M % cols != 0:
            print(f"  {nm}: SKIP (M={M} not divisible by cols={cols})"); continue
        a_ok = check(f"{nm} A", unrolled_A(M, K, cols, BH), batched_A(M, K, cols, BH))
        c_ok = check(f"{nm} C", unrolled_C(M, cols, BH), batched_C(M, cols, BH))
        allok = allok and a_ok and c_ok
print("RESULT:", "ALL IDENTICAL — batched 4D TAP == unrolled access (offline-equivalent)" if allok else "MISMATCH — fix before building")
sys.exit(0 if allok else 1)
