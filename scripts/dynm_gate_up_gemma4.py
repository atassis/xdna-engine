#!/usr/bin/env python3
"""Spike: dyn-seq whole-array GEMM (#3368) adapted to gemma4-12b gate_up geometry.

K=3840 (compile-time), N=15360, M runtime in {1,7,64,100,256,1000,2048}. bf16
in/out with AIE2P bf16-emulated-via-bfp16 mmul (our prefill's path). Tile dims
m=32,k=128,n=64 (K and N divide exactly; prefill's own tile is 32x128x64 at 8
columns for gate_up -- this vendored example hardcodes n_aie_cols in {1,2,4},
so we run at n_aie_cols=4, the closest legal value, and note the gap).

Reuses the upstream generator unmodified, imported by path from the pinned
mlir-aie worktree (13286f4d5ec, wt-mlir-aie-pin) -- never edited.

Usage:
  dynm_gate_up_gemma4.py correctness         # step 2: CPU ref check per M
  dynm_gate_up_gemma4.py static-vs-dynamic    # step 3: alternated ABBA timing
  dynm_gate_up_gemma4.py txn-build            # step 3b: host TXN build time
"""
import argparse
import statistics
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

import os
MLIR_AIE_PIN = os.environ.get("MLIR_AIE_PIN", os.path.join(os.path.dirname(__file__), "..", "..", "wt-mlir-aie-pin"))
sys.path.insert(0, f"{MLIR_AIE_PIN}/test/npu-xrt/matmul_whole_array_dynamic")

import aie.iron as iron  # noqa: E402
from whole_array_dynamic import whole_array_dynamic, N_AIE_ROWS  # noqa: E402

K = 3840
N = 15360
M_TILE, K_TILE, N_TILE = 32, 128, 64
N_AIE_COLS = 4  # example's hardcoded max; real gate_up prefill uses 8 columns
M_VALUES = [1, 7, 64, 100, 256, 1000, 2048]

M_MAX = 2048  # covers every M below once padded to the tile grid
M_PAD_UNIT = M_TILE * N_AIE_ROWS  # 128


def pad_m(m):
    return ((m + M_PAD_UNIT - 1) // M_PAD_UNIT) * M_PAD_UNIT


def _design(*, static_m=None):
    kwargs = dict(
        A_elements=M_MAX * K,
        B_elements=K * N,
        C_elements=M_MAX * N,
        K=K,
        m=M_TILE,
        k=K_TILE,
        n=N_TILE,
        n_aie_cols=N_AIE_COLS,
        dtype_in_str="bf16",
        dtype_out_str="bf16",
    )
    design = whole_array_dynamic.specialize(**kwargs)
    return design


def _run(design, m_real, *, static=False, rng=None):
    m_pad = pad_m(m_real)
    rng = rng or np.random.default_rng(1726250518)
    a = np.zeros((M_MAX * K,), dtype=bfloat16)
    b = np.zeros((K * N,), dtype=bfloat16)
    a_real = rng.uniform(-1, 1, size=(m_real * K,)).astype(bfloat16)
    b_real = rng.uniform(-1, 1, size=(K * N,)).astype(bfloat16)
    a[: m_real * K] = a_real
    # b is reused whole (K*N fixed); pad rows of a beyond m_real are left zero
    b[: K * N] = b_real

    A = iron.tensor(a, dtype=bfloat16, device="npu")
    B = iron.tensor(b, dtype=bfloat16, device="npu")
    C = iron.zeros((M_MAX * N,), dtype=bfloat16, device="npu")
    if static:
        design.specialize(M=m_pad, N=N)(A, B, C)
    else:
        design(A, B, C, M=m_pad, N=N)
    c = C.numpy()[: m_pad * N].reshape(m_pad, N).copy()
    return c[:m_real], a_real.reshape(m_real, K), b_real.reshape(K, N)


def cmd_correctness():
    # A per-element rel-err (|a-e|/max(|e|,eps)) blows up on the many
    # near-zero elements a random bf16 GEMM produces (denominator ~1e-3,
    # bf16 rounding noise ~1e-1 absolute -> rel err ~100). Use per-row L2
    # relative error instead, which is the standard GEMM correctness metric
    # and is insensitive to individual near-zero outputs.
    design = _design()
    print(f"{'M':>6} {'M_pad':>7} {'max_row_rel_l2':>16} {'result':>8}")
    worst = 0.0
    for m in M_VALUES:
        actual, a_real, b_real = _run(design, m)
        expected = (a_real.astype(np.float32) @ b_real.astype(np.float32)).astype(np.float32)
        actual_f32 = actual.astype(np.float32)
        row_err = np.linalg.norm(actual_f32 - expected, axis=1)
        row_norm = np.linalg.norm(expected, axis=1)
        rel_l2 = row_err / np.maximum(row_norm, 1e-6)
        max_rel = float(rel_l2.max())
        worst = max(worst, max_rel)
        ok = "PASS" if max_rel < 0.06 else "FAIL"  # bfp16-emulated bf16 mmul noise floor
        print(f"{m:>6} {pad_m(m):>7} {max_rel:>16.6f} {ok:>8}")
    print(f"worst per-row rel-L2 over all M: {worst:.6f}")


def cmd_static_vs_dynamic():
    dyn_design = _design()
    # Build two STATIC designs, specialized at M=64 and M=256 (each M padded
    # to the tile grid: 64->128, 256->256 already aligned).
    static_designs = {}
    for m in (64, 256):
        m_pad = pad_m(m)
        kwargs = dict(
            A_elements=m_pad * K,
            B_elements=K * N,
            C_elements=m_pad * N,
            K=K,
            m=M_TILE,
            k=K_TILE,
            n=N_TILE,
            n_aie_cols=N_AIE_COLS,
            dtype_in_str="bf16",
            dtype_out_str="bf16",
        )
        static_designs[m] = whole_array_dynamic.specialize(**kwargs).specialize(M=m_pad, N=N)

    rng = np.random.default_rng(7)
    for m in (64, 256):
        m_pad = pad_m(m)
        print(f"\n=== M={m} (padded {m_pad}) : static vs dynamic, ABBA x3+warmup ===")
        a = rng.uniform(-1, 1, size=(m_pad * K,)).astype(bfloat16)
        b = rng.uniform(-1, 1, size=(K * N,)).astype(bfloat16)
        A_s = iron.tensor(a.copy(), dtype=bfloat16, device="npu")
        B_s = iron.tensor(b.copy(), dtype=bfloat16, device="npu")
        C_s = iron.zeros((m_pad * N,), dtype=bfloat16, device="npu")
        A_d = iron.tensor(np.zeros((M_MAX * K,), dtype=bfloat16), dtype=bfloat16, device="npu")
        B_d = iron.tensor(np.zeros((K * N,), dtype=bfloat16), dtype=bfloat16, device="npu")
        C_d = iron.zeros((M_MAX * N,), dtype=bfloat16, device="npu")
        A_d_np = A_d.numpy()
        A_d_np[: m_pad * K] = a
        B_d_np = B_d.numpy()
        B_d_np[: K * N] = b

        static_kernel = static_designs[m]
        dyn_kernel = dyn_design

        def run_static():
            static_kernel(A_s, B_s, C_s)

        def run_dynamic():
            dyn_kernel(A_d, B_d, C_d, M=m_pad, N=N)

        # warmup both
        run_static()
        run_dynamic()

        n_rounds = 3
        static_times, dynamic_times = [], []
        for r in range(n_rounds):
            for label, fn, bucket in (("A", run_static, static_times), ("B", run_dynamic, dynamic_times)):
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                bucket.append((t1 - t0) * 1e3)
            for label, fn, bucket in (("B", run_dynamic, dynamic_times), ("A", run_static, static_times)):
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                bucket.append((t1 - t0) * 1e3)

        def stat(xs):
            return (statistics.median(xs), min(xs), max(xs), max(xs) - min(xs))

        s_med, s_min, s_max, s_spread = stat(static_times)
        d_med, d_min, d_max, d_spread = stat(dynamic_times)
        print(f"  static : n={len(static_times)} median={s_med:.4f}ms min={s_min:.4f} max={s_max:.4f} spread={s_spread:.4f}")
        print(f"  dynamic: n={len(dynamic_times)} median={d_med:.4f}ms min={d_min:.4f} max={d_max:.4f} spread={d_spread:.4f}")
        delta = d_med - s_med
        print(f"  delta (dynamic - static) median: {delta:+.4f} ms ({100*delta/s_med:+.2f}%)")


def cmd_txn_build():
    """Isolate host-side TXN build time per call from device time.

    Use the design's dispatch object directly: iron's jit call path builds the
    txn (generate_txn_main_sequence) then dispatches. We approximate the split
    by timing back-to-back calls at the SAME shape (steady dispatch cost is
    amortized/cached identically each call for the static path, so the delta
    vs the dynamic path's per-call rebuild approximates the host build cost).
    """
    dyn_design = _design()
    m = 256
    m_pad = pad_m(m)
    rng = np.random.default_rng(3)
    A = iron.tensor(rng.uniform(-1, 1, size=(M_MAX * K,)).astype(bfloat16), dtype=bfloat16, device="npu")
    B = iron.tensor(rng.uniform(-1, 1, size=(K * N,)).astype(bfloat16), dtype=bfloat16, device="npu")
    C = iron.zeros((M_MAX * N,), dtype=bfloat16, device="npu")

    dyn_design(A, B, C, M=m_pad, N=N)  # warmup + compile
    N_ITERS = 50
    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        dyn_design(A, B, C, M=m_pad, N=N)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    print(f"end-to-end dynamic call (host build + dispatch), n={N_ITERS}:")
    print(f"  median={statistics.median(times):.2f}us min={min(times):.2f}us max={max(times):.2f}us")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["correctness", "static-vs-dynamic", "txn-build"])
    args = p.parse_args()
    {"correctness": cmd_correctness, "static-vs-dynamic": cmd_static_vs_dynamic, "txn-build": cmd_txn_build}[args.cmd]()
