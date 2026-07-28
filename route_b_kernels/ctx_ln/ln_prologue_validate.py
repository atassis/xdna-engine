#!/usr/bin/env python3
"""Bit-exact validation of the LN-prologue's L1 addressing (RECREATED 2026-07-28).

`mm_ln_prologue.cc` reduces per-row over the contraction dim K, but the A block it reads has already
been re-ordered by the A_l2l1 objectFIFO into `aie::mmul`-BLOCKED layout. A per-row reduction is NOT
layout-independent the way the elementwise SiLU epilogue is, so the kernel walks the tile in the same
tiled coordinates the microkernel consumes, using

    off(i, c) = (i/r)*(r*k) + (i%r)*s + (c/s)*(r*s) + (c%s)

and reads each row as `k/s` chunks of `s` contiguous elements, base `(i/r)*(r*k) + (i%r)*s`, stride
`r*s` between chunks. The kernel's header cites a validation of that claim; the script was lost, so
this recreates it.

The check is exact, not statistical: walk the generator's real `dims_to_stream` to build the true
destination-to-source map the DMA produces, then assert the kernel's closed-form offset lands on the
right logical element for every (i, c). A stride bug shows up as a mismatch, not as a small error.

    python3 route_b_kernels/ctx_ln/ln_prologue_validate.py

CPU only. No device, no toolchain, no build.
"""
import itertools
import sys

# The A path's dims_to_stream, verbatim from whole_array_modal_iron.py:
#     [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
# Shipped Parakeet fc1 tile plus the m=32 variant the bf16-out build needs, and the r/s the
# mmul<r,s,t> microkernel uses.
CONFIGS = [
    dict(name="shipped fc1  m=64 k=32", m=64, k=32, r=8, s=8),
    dict(name="bf16-out     m=32 k=32", m=32, k=32, r=8, s=8),
    dict(name="wide-k       m=64 k=64", m=64, k=64, r=8, s=8),
    dict(name="r=4 variant  m=64 k=32", m=64, k=32, r=4, s=8),
]


def dma_dest_to_src(m, k, r, s):
    """Destination index -> source (row-major) offset, from the 4-D DMA access pattern.

    dims_to_stream is (size, stride) pairs applied to the SOURCE; the DMA emits those elements
    contiguously into the destination. So enumerating the nest in order gives dest index d = the
    position in that enumeration, holding source offset sum(idx_j * stride_j).
    """
    dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    sizes = [d[0] for d in dims]
    strides = [d[1] for d in dims]
    out = []
    for idx in itertools.product(*[range(n) for n in sizes]):
        out.append(sum(i * st for i, st in zip(idx, strides)))
    return out


def kernel_offset(i, c, k, r, s):
    """The closed form mm_ln_prologue.cc actually computes."""
    return (i // r) * (r * k) + (i % r) * s + (c // s) * (r * s) + (c % s)


def check(cfg):
    m, k, r, s = cfg["m"], cfg["k"], cfg["r"], cfg["s"]
    if m % r or k % s:
        return None, "tile not divisible by the mmul sub-tile"

    dest_to_src = dma_dest_to_src(m, k, r, s)
    if len(dest_to_src) != m * k:
        return False, f"DMA walk covers {len(dest_to_src)} of {m*k} elements"
    if sorted(dest_to_src) != list(range(m * k)):
        return False, "DMA walk is not a permutation (elements dropped or duplicated)"

    # 1. the kernel's offset for (i,c) must hold logical element (i,c)
    bad = []
    for i in range(m):
        for c in range(k):
            off = kernel_offset(i, c, k, r, s)
            if not (0 <= off < m * k):
                bad.append((i, c, off, "out of range"))
            elif dest_to_src[off] != i * k + c:
                bad.append((i, c, off, f"holds {dest_to_src[off]}, want {i*k+c}"))
    if bad:
        return False, f"{len(bad)} mismatches, first: {bad[0]}"

    # 2. the kernel's offsets must be a bijection over the tile
    offs = [kernel_offset(i, c, k, r, s) for i in range(m) for c in range(k)]
    if sorted(offs) != list(range(m * k)):
        return False, "kernel offsets are not a bijection over the tile"

    # 3. the contiguity claim: each row is k/s chunks of s CONTIGUOUS elements,
    #    base (i/r)*(r*k)+(i%r)*s, stride r*s between chunks. This is what makes the
    #    aie::load_v<S> in the kernel legal.
    for i in range(m):
        base = (i // r) * (r * k) + (i % r) * s
        for cj in range(k // s):
            want = [base + cj * (r * s) + e for e in range(s)]
            got = [kernel_offset(i, cj * s + e, k, r, s) for e in range(s)]
            if got != want:
                return False, f"row {i} chunk {cj} not contiguous: {got} vs {want}"
    return True, f"{m*k} elements, bijection, {k//s} contiguous {s}-wide chunks per row"


def main():
    print("LN-prologue L1 addressing validation (exact, element-for-element)")
    print(f"{'config':<26} {'result':<7} detail")
    ok_all = True
    for cfg in CONFIGS:
        ok, detail = check(cfg)
        if ok is None:
            print(f"{cfg['name']:<26} {'SKIP':<7} {detail}")
            continue
        ok_all &= ok
        print(f"{cfg['name']:<26} {'PASS' if ok else 'FAIL':<7} {detail}")

    # Negative controls: a PASS above proves nothing unless the check demonstrably FAILS on a wrong
    # formula. Each mutation is asserted to actually differ from the correct one first -- the obvious
    # candidate (r*s -> s*s) is DEGENERATE at the shipped r == s == 8 and silently proves nothing.
    m, k, r, s = 64, 32, 8, 8
    dest_to_src = dma_dest_to_src(m, k, r, s)
    coords = [(i, c) for i in range(m) for c in range(k)]
    correct = [kernel_offset(i, c, k, r, s) for i, c in coords]

    mutations = {
        "chunk stride drops r": lambda i, c: (i // r) * (r * k) + (i % r) * s + (c // s) * s + (c % s),
        "row base uses r not s": lambda i, c: (i // r) * (r * k) + (i % r) * r + (c // s) * (r * s) + (c % s),
        "row-major (no blocking)": lambda i, c: i * k + c,
        "degenerate r*s -> s*s": lambda i, c: (i // r) * (r * k) + (i % r) * s + (c // s) * (s * s) + (c % s),
    }
    print()
    all_caught = True
    for name, fn in mutations.items():
        mut = [fn(i, c) for i, c in coords]
        if mut == correct:
            print(f"{('  ctl: ' + name):<26} {'SKIP':<7} degenerate at r={r},s={s} -- proves nothing")
            continue
        caught = (sorted(mut) != list(range(m * k))) or any(
            dest_to_src[o] != i * k + c for o, (i, c) in zip(mut, coords) if 0 <= o < m * k)
        all_caught &= caught
        print(f"{('  ctl: ' + name):<26} {'PASS' if caught else 'FAIL':<7} "
              f"{'rejected as expected' if caught else 'NOT CAUGHT -- instrument is blind here'}")
    ok_all &= all_caught

    print()
    print("VALIDATION", "GREEN" if ok_all else "RED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
