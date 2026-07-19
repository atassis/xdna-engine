#!/usr/bin/env python3
"""Golden for the gemv-int8xint4 brick (decode M=1 matvec, int8 activation x
int4 weight, mmul_8_4 at the native M=4 tile). CPU-only numpy reference; no
NPU. This is the host reference the LATER device-validation pass checks
on-device output against (rel-L2).

Shape contract (matches gemv_int8xint4.cc):
  a: [M_NATIVE, K] int8   -- row 0 = real decode token, rows 1..M_NATIVE-1 = 0
  b: [K, N]        int4-range int8 storage -- symmetric per-tensor weight
  c: [M_NATIVE, N] int32  -- only row 0 is meaningful

NOTE (flagged for the device leaf, see the kernel file's NOTE comment): this
golden computes the STRAIGHT logical row-major matvec (int32 accumulate of
int8 x int4-range values). It does NOT attempt to reproduce aie2p's
`mac_4x16_16x16_conf` internal B/C sub-tile permutation -- that hardware
packing is not re-derivable from the aie_api headers alone (no in-repo
example calls mmul_8_4 directly). The device pass must confirm/adapt the
B-repack and C-untile before diffing device output against this golden;
until then this is the MATH reference, not a byte-exact layout reference.
"""
import numpy as np

M_NATIVE = 4     # native mmul_8_4 M tile (decode M=1 padded up to this)
K_NATIVE = 16
N_NATIVE = 16

INT4_MIN, INT4_MAX = -8, 7   # signed int4 range for the weight operand


def quantize_int4_symmetric(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int4 quantization: returns (int4-range int8 codes, scale)."""
    absmax = np.max(np.abs(w))
    absmax = 1.0 if absmax == 0 else absmax
    scale = absmax / INT4_MAX
    q = np.clip(np.round(w / scale), INT4_MIN, INT4_MAX).astype(np.int8)
    return q, scale


def gemv_int8xint4_ref(a_row: np.ndarray, w_q: np.ndarray) -> np.ndarray:
    """Reference matvec: a_row [K] int8, w_q [K,N] int4-range int8 -> c [N] int32.

    Mirrors what the kernel computes for the real row (row 0); rows 1..M-1 of
    the kernel's padded A tile are zero and their C rows are discardable.
    """
    assert a_row.ndim == 1 and w_q.ndim == 2 and a_row.shape[0] == w_q.shape[0]
    return (a_row.astype(np.int32) @ w_q.astype(np.int32)).astype(np.int32)


def build_padded_kernel_inputs(a_row: np.ndarray, w_q: np.ndarray):
    """Build the exact [M_NATIVE,K] / tiled-[K,N] buffers the .cc kernel expects
    (row 0 = real token, rows 1..M_NATIVE-1 = zero; weight tiled K_NATIVE x N_NATIVE
    sub-blocks in k-major, n-minor order -- see kernel NOTE on B sub-tile layout)."""
    K = a_row.shape[0]
    N = w_q.shape[1]
    assert K % K_NATIVE == 0 and N % N_NATIVE == 0

    a_tile = np.zeros((M_NATIVE, K), dtype=np.int8)
    a_tile[0, :] = a_row

    kt_n = K // K_NATIVE
    nt_n = N // N_NATIVE
    b_tiled = np.zeros((kt_n, nt_n, K_NATIVE * N_NATIVE), dtype=np.int8)
    for kt in range(kt_n):
        for nt in range(nt_n):
            block = w_q[kt * K_NATIVE:(kt + 1) * K_NATIVE, nt * N_NATIVE:(nt + 1) * N_NATIVE]
            b_tiled[kt, nt, :] = block.reshape(-1)  # row-major within the sub-block
    return a_tile, b_tiled


def relL2(ref: np.ndarray, got: np.ndarray) -> float:
    return float(np.linalg.norm((got - ref).astype(np.float64).ravel())
                 / (np.linalg.norm(ref.astype(np.float64).ravel()) + 1e-12))


def run_case(name: str, K: int, N: int, seed: int, threshold: float = 3e-2):
    rng = np.random.default_rng(seed)
    a_row = rng.integers(-127, 128, size=K, dtype=np.int64).astype(np.int8)  # int8 activation
    w_f = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    w_q, scale = quantize_int4_symmetric(w_f)

    ref_int32 = gemv_int8xint4_ref(a_row, w_q)
    a_tile, b_tiled = build_padded_kernel_inputs(a_row, w_q)

    # Self-check: re-derive the same int32 result from the padded/tiled buffers
    # by undoing the tiling (this is the CPU-side sanity check the .cc kernel's
    # untiled math must match once device sub-tile layout is confirmed).
    kt_n, nt_n = K // K_NATIVE, N // N_NATIVE
    w_untiled = np.zeros((K, N), dtype=np.int8)
    for kt in range(kt_n):
        for nt in range(nt_n):
            w_untiled[kt * K_NATIVE:(kt + 1) * K_NATIVE, nt * N_NATIVE:(nt + 1) * N_NATIVE] = (
                b_tiled[kt, nt, :].reshape(K_NATIVE, N_NATIVE))
    got_int32 = gemv_int8xint4_ref(a_tile[0], w_untiled)

    r = relL2(ref_int32, got_int32)
    print(f"  {name:16s} K={K:4d} N={N:4d}  weight-scale={scale:.5f}  "
          f"rel-L2(self-check)={r:.2e}  {'PASS' if r <= threshold else 'FAIL'} (<={threshold})")
    return r


def main():
    print("gemv-int8xint4 golden (CPU-only, host reference for the device rel-L2 check)")
    print("M_NATIVE=%d K_NATIVE=%d N_NATIVE=%d\n" % (M_NATIVE, K_NATIVE, N_NATIVE))
    worst = 0.0
    # Gemma-270M-ish decode shapes: hidden=640, ffn intermediate slices, lm-head slice.
    worst = max(worst, run_case("ffn.down_proj", 2048, 640, seed=1))
    worst = max(worst, run_case("ffn.gate_proj", 640, 2048, seed=2))
    worst = max(worst, run_case("attn.o_proj", 640, 640, seed=3))
    worst = max(worst, run_case("lm_head.slice", 640, 256, seed=4))
    print(f"\n worst rel-L2 (packing self-check) = {worst:.2e}")
    print(" NOTE: this validates the golden's OWN tiling round-trip, not the real")
    print(" aie2p mac_4x16_16x16_conf sub-tile permutation -- that must be confirmed")
    print(" on-device (or from AMD lowering docs) before diffing real kernel output.")


if __name__ == "__main__":
    main()
