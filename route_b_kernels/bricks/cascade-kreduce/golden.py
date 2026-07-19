#!/usr/bin/env python3
"""Golden (host, CPU-only, no NPU) reference for the cascade-kreduce brick.

Brick: fused cascade-accumulator K-REDUCTION across N_CASCADE adjacent cores.
Each core c in [0, N_CASCADE) holds a K-chunk slice A_c[M,K_chunk] of the
contraction and this core's B_c[K_chunk,N] slice, computes a bf16 PARTIAL
product (device: aie::mmul<8,8,8,bfloat16,bfloat16,accauto> systolic tile;
host golden: plain matmul), and the partials are summed down the cascade:

    out[M,N] = r[M,N] + sum_{c=0}^{N_CASCADE-1} (A_c @ B_c)

matching the HEAD/MIDDLE/TAIL role split in cascade_kreduce.cc:
  HEAD   (c=0):              running = partial_0 + r
  MIDDLE (0 < c < last):     running = running + partial_c
  TAIL   (c=last):           out     = running + partial_last

r is an optional residual/bias injected at the HEAD (all-zero if the op-type
instance being scheduled has none to fuse in -- matches matvec_cascade_add.py's
"R injected at HEAD" convention studied from route_b_kernels/cascade_ffn/
STRUCTURE.md section A.2).

Precision note: this is a bf16 brick (bf16 in, fp32-accumulate per-core via
the mmul systolic tile, bf16 partial out, bf16 running-sum across cascade
hops) -- NOT exact like gemm-int8's int64 reference. Truncating each hop
to bf16 (as the device's cascade_kreduce_{head,middle,tail}_bf16_n entries
do) accumulates rounding error across N_CASCADE hops; this golden reproduces
that SAME bf16-per-hop truncation (not a single fp32 running sum) so the
later device rel-L2 check is apples-to-apples against the intended kernel
behavior, not an idealized fp32 reduction.
"""
import numpy as np


def _to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16, represented as f32 (truncated
    mantissa), matching the device's set_rounding(conv_even) convention used
    throughout this project's bf16 kernels (see cascade_ffn/mv_bf16_gelu.cc
    and the project's banked "WER-path bf16 casts must set_rounding(conv_even)"
    rule)."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    rounded = (bits.astype(np.uint64) + rounding_bias) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def cascade_kreduce_partial(a_c: np.ndarray, b_c: np.ndarray) -> np.ndarray:
    """One core's PARTIAL: bf16 A_c[M,K_chunk] @ bf16 B_c[K_chunk,N] ->
    bf16 partial[M,N] (fp32-accumulate internally, matching aie::mmul's
    accauto accumulator, truncate to bf16 on the way out -- matches the
    device's cascade_kreduce_partial_tile() single writeback per tile)."""
    assert a_c.shape[1] == b_c.shape[0]
    M, K_chunk = a_c.shape
    _, N = b_c.shape
    assert M % 8 == 0 and K_chunk % 8 == 0 and N % 8 == 0, \
        "cascade-kreduce brick requires M,K_chunk,N multiples of 8 (native mmul<8,8,8> tile)"
    a32 = _to_bf16(a_c).astype(np.float32)
    b32 = _to_bf16(b_c).astype(np.float32)
    acc = a32 @ b32  # fp32 accumulate, mirrors the mmul accauto accumulator
    return _to_bf16(acc)


def cascade_kreduce_ref(a_chunks: list, b_chunks: list, r: np.ndarray) -> np.ndarray:
    """Full cascade: N_CASCADE per-core partials combined HEAD -> MIDDLE* ->
    TAIL, each hop truncated to bf16 (matches the device's bf16 running-sum
    combine, NOT an idealized single fp32 reduction).

    a_chunks: list of N_CASCADE arrays, a_chunks[c] is [M, K_chunk_c]
    b_chunks: list of N_CASCADE arrays, b_chunks[c] is [K_chunk_c, N]
    r: residual/bias [M,N] injected at HEAD (pass np.zeros(...) if none)
    """
    n_cascade = len(a_chunks)
    assert n_cascade == len(b_chunks) and n_cascade >= 2, \
        "cascade-kreduce requires >= 2 cascade cores (HEAD + TAIL minimum)"

    partials = [cascade_kreduce_partial(a_chunks[c], b_chunks[c])
                for c in range(n_cascade)]

    # HEAD (c=0): running = partial_0 + r
    running = _to_bf16(partials[0].astype(np.float32) + _to_bf16(r).astype(np.float32))
    # MIDDLE (0 < c < last): running = running + partial_c
    for c in range(1, n_cascade - 1):
        running = _to_bf16(running.astype(np.float32) + partials[c].astype(np.float32))
    # TAIL (c=last): out = running + partial_last
    out = _to_bf16(running.astype(np.float32) + partials[n_cascade - 1].astype(np.float32))
    return out


def rel_l2(device: np.ndarray, ref: np.ndarray) -> float:
    device = device.astype(np.float64)
    ref = ref.astype(np.float64)
    num = np.linalg.norm(device - ref)
    den = np.linalg.norm(ref)
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Shape matching the 8x64x8 device instance (cascade_kreduce_partial_8x64x8):
    # M=8 (one mmul row-tile), K_chunk=64 per core, N=8, N_CASCADE=8 cores
    # (the Whisper-FFN-fc2-style 8-core K-reduction from the model kernel:
    # STRUCTURE.md B.5, FF=3072=8x384 folded down to this brick's minimal
    # legal tile for a CPU-only smoke golden).
    M, K_CHUNK, N, N_CASCADE = 8, 64, 8, 8

    a_chunks = [rng.standard_normal((M, K_CHUNK), dtype=np.float32) * 0.1
                for _ in range(N_CASCADE)]
    b_chunks = [rng.standard_normal((K_CHUNK, N), dtype=np.float32) * 0.1
                for _ in range(N_CASCADE)]
    r = rng.standard_normal((M, N), dtype=np.float32) * 0.01  # residual/bias at HEAD

    out = cascade_kreduce_ref(a_chunks, b_chunks, r)

    # Idealized fp32 reduction (no per-hop bf16 truncation) as a sanity upper
    # bound -- the two should be close but not identical, quantifying the
    # bf16-per-hop truncation error this golden is deliberately modeling.
    ideal = r.astype(np.float64)
    for c in range(N_CASCADE):
        ideal = ideal + (a_chunks[c].astype(np.float64) @ b_chunks[c].astype(np.float64))

    print(f"cascade-kreduce golden: M={M} K_CHUNK={K_CHUNK} N={N} N_CASCADE={N_CASCADE}")
    print(f"out (bf16-per-hop) vs ideal fp32-reduction rel-L2 = {rel_l2(out, ideal):.4e}")
    print("Later NPU pass: feed the SAME a_chunks/b_chunks/r into the device "
          "kernel (cascade_kreduce_partial_8x64x8 per core + "
          "cascade_kreduce_{head,middle,tail}_bf16_n chained per the MLIR "
          "npu_cascade wiring), compare device `out` against this script's "
          "`out` (bf16-per-hop reference, NOT `ideal`) at rel-L2 <= 3e-2.")
