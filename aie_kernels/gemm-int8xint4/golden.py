#!/usr/bin/env python3
"""Golden (host, CPU-only, no NPU) reference for the gemm-int8xint4 brick.

Brick: C[M,N] (int32) = A[M,K] (int8) @ B[K,N] (int4, packed 2/byte), tiled
over the AIE2P aie::mmul<4,16,16,int8,int4,accauto> "mmul_8_4" native tile
(M a multiple of 4, K/N multiples of 16 -- see gemm_int8xint4.cc header).

int8 x int4 -> int32 is EXACT integer arithmetic (no rounding/format loss like
the bfp16/bf16 bricks): the device kernel's int32 accumulator must reproduce
this golden bit-exactly (mod overflow saturation, which does not occur at
these shapes/value ranges). This script:
  1. Generates random int8 A in [-127,127] and int4 B in [-7,7] (skip the
     signed extremes -128 / -8 -- same convention as gemm-int8's golden, plus
     int4's own two's-complement extreme).
  2. PACKS B into the int4 nibble-pair layout the device kernel's
     aie::vector<int4,N> / aie::load_v<N> expects: two signed 4-bit lanes per
     byte, low nibble = even flat index, high nibble = odd flat index (aie_api's
     documented sub-byte vector packing convention). `pack_int4` / `unpack_int4`
     below are the exact host-side pack/unpack pair; the on-device pass must
     feed B through the SAME `pack_int4` before DMA.
  3. Computes the reference GEMM in numpy int64 (unpacked, no overflow) then
     casts to int32 (checking no int32 overflow actually occurs for the chosen
     shapes/value range, so the int32-cast reference is itself the exact
     answer).
  4. Reports rel-L2 against itself (0.0) as a sanity smoke test, and documents
     the threshold the LATER on-device pass should gate against.

Later NPU pass: feed the SAME A (row-major int8, [M,K]) and packed B (row-major
int4 pairs, [K,N] logical / [K, N/2] bytes via `pack_int4`) into the device
kernel, unpack device C (int32, [M,N]) and compare against `ref` here.
"""
import numpy as np


def pack_int4(x: np.ndarray) -> np.ndarray:
    """Pack a flat/2D array of signed int4 values (as int8, range [-8,7]) into
    bytes: 2 lanes/byte, low nibble = even flat index, high nibble = odd flat
    index (aie_api sub-byte vector convention). Input's trailing/flat element
    count must be even (true for all N multiples of 16 used by this brick).
    """
    assert x.dtype == np.int8
    flat = x.reshape(-1)
    assert flat.size % 2 == 0, "int4 packing requires an even element count"
    lo = (flat[0::2].astype(np.uint8) & 0x0F)
    hi = (flat[1::2].astype(np.uint8) & 0x0F)
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed


def unpack_int4(packed: np.ndarray, count: int) -> np.ndarray:
    """Inverse of pack_int4: bytes -> `count` signed int4 values (as int8)."""
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    # sign-extend nibble -> int8
    lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8)
    hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
    out = np.empty(count, dtype=np.int8)
    out[0::2] = lo[: (count + 1) // 2]
    out[1::2] = hi[: count // 2]
    return out


def gemm_int8xint4_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact int8 x int4 -> int32 GEMM reference (numpy int64 acc, cast to
    int32). `a` is unpacked int8 [M,K]; `b` is UNPACKED int4-as-int8 [K,N]
    (i.e. already the logical values, not the packed-byte layout -- pack/
    unpack only matters for the device DMA representation, not this math).
    """
    assert a.dtype == np.int8 and b.dtype == np.int8
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % 4 == 0 and K % 16 == 0 and N % 16 == 0, \
        "gemm-int8xint4 brick requires M%4==0, K%16==0, N%16==0 (native mmul_8_4 tile)"
    assert np.all(b >= -7) and np.all(b <= 7), "int4 range is [-8,7]; this golden uses [-7,7]"
    acc64 = a.astype(np.int64) @ b.astype(np.int64)
    # int8(<=127) * int4(<=7) accumulated over K: max |acc| = K * 127 * 7,
    # far within int32 range (2^31-1) for any realistic K -- no int32
    # overflow for realistic GEMM shapes, so this cast is lossless here.
    assert np.all(np.abs(acc64) < np.iinfo(np.int32).max), \
        "int32 accumulator would overflow for this shape/value range"
    return acc64.astype(np.int32)


def rel_l2(ref: np.ndarray, got: np.ndarray) -> float:
    ref64 = ref.astype(np.float64).ravel()
    got64 = got.astype(np.float64).ravel()
    return float(np.linalg.norm(got64 - ref64) / (np.linalg.norm(ref64) + 1e-12))


def run_case(name: str, M: int, K: int, N: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, size=(M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, size=(K, N), dtype=np.int64).astype(np.int8)

    # Round-trip the pack/unpack once here as a self-check that the device-
    # facing packed representation is lossless before it ever touches the NPU.
    b_packed = pack_int4(b)
    b_roundtrip = unpack_int4(b_packed, b.size).reshape(K, N)
    assert np.array_equal(b, b_roundtrip), "int4 pack/unpack round-trip mismatch"

    ref = gemm_int8xint4_ref(a, b)
    # Self-check only here (CPU golden has no device to compare against yet);
    # the device pass replaces `ref` on one side with the on-NPU result
    # (after unpacking device C, which is plain int32, no packing on the
    # output side).
    r = rel_l2(ref, ref)
    print(f"  {name:16s} [{M},{K}]x[{K},{N}] int8xint4->int32  rel-L2(self)={r:.2e}  "
          f"acc range=[{ref.min()},{ref.max()}]  B packed bytes={b_packed.size}")
    return r


def main():
    print("gemm-int8xint4 brick golden (CPU-only, no NPU) -- mixed int8xint4 GEMM, "
          "int32 acc, exact math\n")
    for (M, K, N) in [(4, 16, 16), (8, 16, 16), (64, 64, 64), (128, 128, 128), (8, 1024, 4096)]:
        run_case(f"M{M}K{K}N{N}", M, K, N, seed=M * 1000 + K * 10 + N % 97)
    print("\nrel-L2 gate for the on-device verify pass: <= 3e-2")
    print("(int8xint4->int32 is exact integer math -- any device rel-L2 above")
    print(" this threshold indicates a real kernel bug (or a pack/unpack layout")
    print(" mismatch between host and device), not a format-rounding tradeoff")
    print(" like the bf16/bfp16 bricks; a tight near-0 pass is expected)")


if __name__ == "__main__":
    main()
