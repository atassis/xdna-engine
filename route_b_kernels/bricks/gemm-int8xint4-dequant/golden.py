#!/usr/bin/env python3
"""Golden (host, CPU-only, no NPU) reference for the gemm-int8xint4-dequant brick.

Brick: the FUSED int8xint4 GEMM + post-mmul group-dequant epilogue --
  C[M,N] (bf16) = sum over K-groups g of  scale[g,n] * (A[M,Kg] @ Bq[Kg,N])
symmetric AWQ int4 (HAS_ZP=0): B is int4, group-quantized along K with group
size G, scale[g,n] is the per-group per-output-column f32 scale ([K/G, N]).

Because scale is constant WITHIN a group, the kernel's grouped structure
("accumulate int32 within a group, x scale, sum groups in f32") is algebraically
identical to dequantizing the weight first and doing one matmul:
    W[k,n] = Bq[k,n] * scale[k // G, n]     (symmetric)
    C[m,n] = sum_k A[m,k] * W[k,n]           (f32 accumulate)
    C_bf16 = round_to_bf16(C)                (conv_even, the kernel's epilogue)
This script computes that reference. Layout note (same split the gemm-int8xint4
golden documents): this math is plain row-major; the int4 nibble PACK and the
device TILE-BLOCKED order are DMA concerns handled by the device-verify harness,
not this reference. The device kernel outputs bf16 tiles; the harness un-tiles
and gates rel-L2 vs `ref_bf16` below.

int8 x int4 -> int32 is exact; the ONLY inexactness is the single bf16 narrow of
the final f32 tile, so this is a bf16-grade brick (rel-L2 gate ~2e-2, like the
other bf16-output bricks -- NOT the near-0 of the pure-int gemm-int8xint4).
"""
import numpy as np

GROUP_DEFAULT = 64


def pack_int4(x: np.ndarray) -> np.ndarray:
    """Pack signed int4 (as int8, [-8,7]) into bytes: 2 lanes/byte, low nibble =
    even flat index, high nibble = odd flat index (aie_api sub-byte convention).
    Identical to the gemm-int8xint4 sibling's pack so the device path is shared.
    """
    assert x.dtype == np.int8
    flat = x.reshape(-1)
    assert flat.size % 2 == 0, "int4 packing requires an even element count"
    lo = (flat[0::2].astype(np.uint8) & 0x0F)
    hi = (flat[1::2].astype(np.uint8) & 0x0F)
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray, count: int) -> np.ndarray:
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8)
    hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
    out = np.empty(count, dtype=np.int8)
    out[0::2] = lo[: (count + 1) // 2]
    out[1::2] = hi[: count // 2]
    return out


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round f32 -> bf16 (round-to-nearest-even at the 16-bit mantissa boundary),
    widened back to f32 -- emulates the kernel's conv_even bf16 store. Shared
    idiom with the swiglu / geglu bf16 bricks in this tree."""
    x = np.asarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    rounding_bias = ((u32 >> 16) & 1) + 0x7FFF
    u32_rounded = ((u32.astype(np.uint64) + rounding_bias) & 0xFFFFFFFF).astype(np.uint32)
    bf16_bits = (u32_rounded >> 16).astype(np.uint16)
    return (bf16_bits.astype(np.uint32) << 16).view(np.float32)


def dequant_gemm_ref(a: np.ndarray, b: np.ndarray, scale: np.ndarray,
                     group: int):
    """Reference: a int8 [M,K], b int4-as-int8 [K,N] (unpacked logical values),
    scale f32 [K/G, N]. Returns (ref_f32, ref_bf16). Symmetric (no zero-point)."""
    assert a.dtype == np.int8 and b.dtype == np.int8
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % 4 == 0 and K % 16 == 0 and N % 16 == 0, \
        "native mmul_8_4 tile requires M%4==0, K%16==0, N%16==0"
    assert group % 16 == 0 and K % group == 0, "G%16==0 and K%G==0"
    assert scale.shape == (K // group, N), f"scale must be [K/G,N]={(K//group, N)}"
    assert np.all(b >= -7) and np.all(b <= 7), "int4 range [-8,7]; golden uses [-7,7]"
    # broadcast the per-group scale down to per-(k,n): row k uses group k//G.
    scale_full = np.repeat(scale, group, axis=0)          # [K, N]
    w = b.astype(np.float32) * scale_full                 # dequantized weight [K,N]
    ref_f32 = a.astype(np.float32) @ w                    # [M,N] f32
    return ref_f32, to_bf16(ref_f32)


def rel_l2(ref: np.ndarray, got: np.ndarray) -> float:
    ref64 = ref.astype(np.float64).ravel()
    got64 = got.astype(np.float64).ravel()
    return float(np.linalg.norm(got64 - ref64) / (np.linalg.norm(ref64) + 1e-12))


def run_case(name: str, M: int, K: int, N: int, group: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, size=(M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, size=(K, N), dtype=np.int64).astype(np.int8)
    # AWQ-ish scales: small positive per-group per-column magnitudes.
    scale = rng.uniform(0.005, 0.05, size=(K // group, N)).astype(np.float32)

    # self-check the device-facing int4 packing is lossless.
    b_rt = unpack_int4(pack_int4(b), b.size).reshape(K, N)
    assert np.array_equal(b, b_rt), "int4 pack/unpack round-trip mismatch"

    ref_f32, ref_bf16 = dequant_gemm_ref(a, b, scale, group)
    # rel-L2 the bf16-narrowed reference against the exact f32 -- this is the
    # format-rounding floor the device (which also outputs bf16) rides on; the
    # on-NPU pass replaces ref_bf16 on one side with the un-tiled device output.
    r = rel_l2(ref_f32, ref_bf16)
    print(f"  {name:20s} [{M},{K}]x[{K},{N}] G={group:<4d} int8xint4->bf16  "
          f"rel-L2(bf16-vs-f32)={r:.2e}  |C|range=[{ref_f32.min():.1f},{ref_f32.max():.1f}]")
    return r


def main():
    print("gemm-int8xint4-dequant brick golden (CPU-only, no NPU) -- fused "
          "int8xint4 GEMM + group-dequant epilogue, bf16 out\n")
    cases = [
        ("M8K64N64_g64",       8,   64,   64,   64),
        ("M64K128N128_g64",    64,  128,  128,  64),
        ("M4K1024N4096_g128",  4,   1024, 4096, 128),
    ]
    worst = 0.0
    for (nm, M, K, N, G) in cases:
        r = run_case(nm, M, K, N, G, seed=M * 1000 + K * 3 + N + G)
        worst = max(worst, r)
    print(f"\nworst bf16-vs-f32 rel-L2 across cases: {worst:.2e}")
    print("rel-L2 gate for the on-device verify pass: <= 2e-2")
    print("(int8xint4->int32 is exact; the only loss is the single bf16 narrow of")
    print(" the f32 result tile -- a device rel-L2 materially above this floor")
    print(" indicates a real kernel bug or a host/device layout mismatch, not a")
    print(" format tradeoff. Symmetric group-quant only -- no zero-point.)")


if __name__ == "__main__":
    main()
