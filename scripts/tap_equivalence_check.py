#!/usr/bin/env python3
"""Offline correctness for the default-on GEMV batch-coalesce (NO NPU, NO toolchain).

Proves the coalesced single BD enumerates the EXACT same DRAM index sequence as the
stock per-batch unrolled BDs, for the decode GEMV configs AND adversarial fallback
configs, and that num_batches==1 collapses to the stock single BD (byte-identical).
Imports the packer that is mirrored into iron/operators/gemv/design.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemv_coalesce_packer import (
    split_run, can_coalesce, coalesced_tap, unrolled_taps, enum_tap, verifier_legal,
)

fails = []


def seq_unrolled(col_off, run, bstride, num_batches, total):
    out = []
    for off, sz, st in unrolled_taps(col_off, run, bstride, num_batches):
        out += enum_tap(off, sz, st, total)
    return out


def seq_coalesced(col_off, run, bstride, num_batches, total):
    off, sz, st = coalesced_tap(col_off, run, bstride, num_batches)
    return enum_tap(off, sz, st, total)


def check_equiv(name, M, K, cols, num_batches):
    """A run=(M//cols)*K bstride=M*K ; C run=(M//cols) bstride=M. Per column."""
    total_A = num_batches * M * K
    total_C = num_batches * M
    ok = True
    for tag, run, bstride, col_step, total in [
        ("A", (M // cols) * K, M * K, (M // cols) * K, total_A),
        ("C", (M // cols), M, (M // cols), total_C),
    ]:
        if not can_coalesce(run, bstride, num_batches):
            print(f"  {name} {tag}: SKIP-coalesce (config falls back to unroll)")
            continue
        tag_ok = True
        for col in range(cols):
            col_off = col * col_step
            u = seq_unrolled(col_off, run, bstride, num_batches, total)
            c = seq_coalesced(col_off, run, bstride, num_batches, total)
            if u != c:
                tag_ok = False
                fails.append(f"{name} {tag} col{col}")
        ok = ok and tag_ok
        print(f"  {name} {tag}: {'IDENTICAL' if tag_ok else 'MISMATCH'} "
              f"(cols={cols}, num_batches={num_batches}, run={run})")
    return ok


# 1. Decode GEMV configs (B=16 WER config; num_batches=B*H=192 > 63 exercises uncapped D2).
HD, cols, H = 64, 8, 12
B, S, TP = 16, 448, 1536
BH = B * H
print(f"== Decode configs (B={B}, num_batches={BH}, cols={cols}) ==")
for nm, M, K in [("g_scs M=S K=HD", S, HD), ("g_cts M=HD K=S", HD, S),
                 ("g_scc M=TP K=HD", TP, HD), ("g_ctc M=HD K=TP", HD, TP)]:
    if M % cols:
        print(f"  {nm}: SKIP (M={M} % cols={cols})"); continue
    check_equiv(nm, M, K, cols, BH)

# 2. num_batches==1 -> packer must NOT coalesce; unrolled == stock single BD.
print("== num_batches==1 byte-identity ==")
for M, K in [(448, 64), (64, 1536)]:
    run, bstride = (M // cols) * K, M * K
    assert not can_coalesce(run, bstride, 1), "num_batches=1 must not coalesce"
    taps = unrolled_taps(0, run, bstride, 1)
    assert taps == [(0, [1, 1, 1, run], [0, 0, 0, 1])], f"stock single BD expected, got {taps}"
    print(f"  M={M} K={K}: stock single BD (sizes=[1,1,1,{run}]) OK")

# 3. num_batches > 63 (uncapped D2 path) small synthetic, full enumeration.
print("== num_batches>63 (uncapped D2) ==")
check_equiv("synth nb=100", 256, 128, 8, 100)

# 4. Fallback: bstride > 2**20 -> must NOT coalesce.
print("== fallback: bstride > 2**20 ==")
big_M, big_K = 4096, 512  # bstride_A = M*K = 2,097,152 > 1,048,575
assert not can_coalesce((big_M // cols) * big_K, big_M * big_K, 64), "must fall back"
print(f"  bstride={big_M*big_K} > {(1<<20)-1}: falls back to unroll OK")

# 5. Fallback: run with no two-dim split <=1023 -> must NOT coalesce.
print("== fallback: unsplittable run ==")
assert split_run(1031) is None, "prime run > 1023 must be unsplittable"  # 1031 is prime
print("  prime run 1031 unsplittable: falls back to unroll OK")

# 6. B=128 legality WITHOUT full enumeration (num_batches=1536).
print("== B=128 verifier legality (no enum) ==")
for nm, M, K in [("g_scc M=TP K=HD", TP, HD), ("g_ctc M=HD K=TP", HD, TP)]:
    run, bstride = (M // cols) * K, M * K
    if not can_coalesce(run, bstride, 128 * H):
        print(f"  {nm}: SKIP"); continue
    _, sz, st = coalesced_tap(0, run, bstride, 128 * H)
    legal, msg = verifier_legal(sz, st)
    if not legal:
        fails.append(f"{nm} B=128 illegal: {msg}")
    print(f"  {nm}: sizes={sz} strides={st} -> {'LEGAL' if legal else 'ILLEGAL: '+msg}")

# 7. Granularity: run_lo is always even (4-byte aligned), and an even-split run that
#    would otherwise pick an odd inner factor stays access-equivalent.
print("== granularity (even run_lo) ==")
for run in [1026, 3584, 12288, 56, 2050]:
    s = split_run(run)
    if s is not None:
        assert (
            s[0] * s[1] == run and s[1] % 2 == 0 and s[0] <= 1023 and s[1] <= 1023
        ), f"bad split {run}->{s}"
run, bstride, nb = 1026, 2052, 3  # largest divisor of 1026 (513) is odd -> needs even split
assert can_coalesce(run, bstride, nb), "1026 should coalesce via an even split"
total = nb * bstride * 2
assert seq_unrolled(0, run, bstride, nb, total) == seq_coalesced(
    0, run, bstride, nb, total
), "even-split coalesced != unrolled for run=1026"
print(f"  run=1026 -> split {split_run(1026)} (even), access-equivalent OK")
assert split_run(2 * 1031) is None, "2*prime run has no even split -> fallback"
assert not can_coalesce(1024, 1025, 4), "odd bstride must fall back"
print("  2*prime run + odd bstride -> fallback OK")

print("RESULT:", "ALL PASS" if not fails else f"FAIL: {fails}")
sys.exit(0 if not fails else 1)
