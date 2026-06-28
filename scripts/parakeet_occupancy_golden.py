#!/usr/bin/env python3
"""Parakeet encoder per-op occupancy: CPU golden + roofline (NO NPU).

This is the CPU half of the Phase-0 measure-first occupancy harness (brick #8,
the measure-first gate; spec 2026-06-28-parakeet-tdt-full-npu-brick-honoring.md).
It runs WITHOUT the device and produces two things the on-NPU harness consumes:

  1. GOLDEN: for each distinct encoder GEMM shape that the phase-3 resident kernel
     dispatches (M=512 padded, K=1024, N in {1024,2048,4096}; src/npu.rs), save
     A,B (bf16) and the f32-accumulate reference C = A@B. The on-device harness
     (scripts/parakeet_occupancy_harness.py) loads these, feeds A,B to the FULL
     resident xclbin, and gates the returned C at rel-L2 <= 0.08 -- proving the
     timing twin is dispatching a CORRECT GEMM (so the A/B compute-fraction is
     meaningful), and asserts the STUB returns ~0 (compute actually elided).

  2. ROOFLINE (analysis-only, assumptions labelled): an independent, device-free
     estimate of compute-lower-bound vs DMA-lower-bound per shape, so the ranked
     occupancy table has a first-principles prior BEFORE the NPU run. The device
     A/B measurement is the real gate; this just brackets expectations.

Why these 3 shapes cover the whole encoder: the zero-switch resident design runs
EVERY encoder matmul on one K=1024 whole_array kernel, selecting N by instruction
stream (src/npu.rs). K=4096 ops (ff.l2, subsample-out) are K-split into 4x N=1024
partials -> same per-dispatch occupancy as N=1024. So {1024,2048,4096} = all of it.

Run (CPU, no NPU):
  .venv-iron/bin/python scripts/parakeet_occupancy_golden.py
  # or any python with numpy + ml_dtypes
Outputs: artifacts/parakeet/occupancy/golden_{M}x{K}x{N}.npz + roofline.json
"""
import json
import os

import numpy as np

try:
    from ml_dtypes import bfloat16
except Exception:  # pragma: no cover
    bfloat16 = None

# --- resident-kernel dispatch shape (src/npu.rs: PAD_M=512, KRES=1024) ---
M = 512
K = 1024
N_SHAPES = [1024, 2048, 4096]

# Logical encoder ops -> (N, count per 24-layer block). K=4096 ops are 4x N=1024
# partials (K-split). This annotates the ranked table with what each shape drives.
OP_MAP = {
    1024: [
        ("self_attn.linear_q/k/v", 3),
        ("self_attn.linear_out", 1),
        ("self_attn.linear_pos", 1),
        ("conv.pointwise_conv2", 1),
        ("ff1.linear2 (4x N=1024 K-split)", 4),
        ("ff2.linear2 (4x N=1024 K-split)", 4),
    ],
    2048: [("conv.pointwise_conv1", 1)],
    4096: [("ff1.linear1", 1), ("ff2.linear1", 1)],
}

# --- machine constants (aie2p-brick-catalog.md; ASSUMPTIONS labelled) ---
N_CORES = 32  # whole_array compute grid = 8 cols x 4 rows (core_tiles rows 2-5)
PEAK_MAC = {  # MAC/cyc/core by format (catalog LAYER 1)
    "bf16_emul": 128,   # emulated bf16 FMA+shuffle (native 32x32x32 tile)
    "bfp16_true": 512,  # TRUE systolic bfp16 (fast 64x32x128 tile, BFP16_IREE)
}
FREQ_HZ = float(os.environ.get("AIE_FREQ_HZ", "1.25e9"))  # ~1.25 GHz (assumption)
L3_BW = float(os.environ.get("L3_BW_GBPS", "120")) * 1e9   # LPDDR ~120 GB/s (physics wall)
ON_CHIP_BW = float(os.environ.get("ONCHIP_BW_GBPS", "800")) * 1e9  # L2<->L1 ~800 GB/s


def roofline(m, k, n):
    macs = m * k * n
    a_bytes = m * k * 2       # A bf16 in
    b_bytes = k * n * 2       # B bf16 weight (resident-cached after 1st call)
    c_bytes = m * n * 4       # C f32 out
    out = {"M": m, "K": k, "N": n, "macs": macs, "flops": 2 * macs,
           "bytes_A_in": a_bytes, "bytes_B_weight": b_bytes, "bytes_C_out": c_bytes}
    for fmt, peak in PEAK_MAC.items():
        comp_us = macs / (peak * N_CORES * FREQ_HZ) * 1e6
        out[f"compute_lb_us_{fmt}"] = round(comp_us, 3)
    # DMA lower bounds: warm = weights resident (A in + C out over L3); cold = + B.
    warm = (a_bytes + c_bytes) / L3_BW * 1e6
    cold = (a_bytes + b_bytes + c_bytes) / L3_BW * 1e6
    out["dma_lb_us_warm_L3"] = round(warm, 3)
    out["dma_lb_us_cold_L3"] = round(cold, 3)
    # arithmetic intensity (MAC/byte, warm) and the device balance points.
    out["arith_intensity_warm"] = round(macs / (a_bytes + c_bytes), 2)
    # First-principles verdict: is the fast-tile compute LB above the warm DMA LB?
    out["roofline_bound_fast_warm"] = (
        "compute" if out["compute_lb_us_bfp16_true"] > warm else "movement")
    out["roofline_bound_native_warm"] = (
        "compute" if out["compute_lb_us_bf16_emul"] > warm else "movement")
    return out


def main():
    outdir = "artifacts/parakeet/occupancy"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(0)
    roof = {"_assumptions": {
        "freq_hz": FREQ_HZ, "n_cores": N_CORES, "peak_mac_per_cyc_per_core": PEAK_MAC,
        "L3_bw_bytes_s": L3_BW, "onchip_bw_bytes_s": ON_CHIP_BW,
        "note": "compute LB and DMA LB are first-principles brackets; the on-device "
                "A/B latency diff (parakeet_occupancy_harness.py) is the real gate."},
        "shapes": {}, "op_map": OP_MAP}

    for n in N_SHAPES:
        r = roofline(M, K, n)
        roof["shapes"][f"{M}x{K}x{n}"] = r
        if bfloat16 is not None:
            A = rng.uniform(-1, 1, size=(M, K)).astype(bfloat16)
            B = rng.uniform(-1, 1, size=(K, n)).astype(bfloat16)
            ref = A.astype(np.float32) @ B.astype(np.float32)
            np.savez(os.path.join(outdir, f"golden_{M}x{K}x{n}.npz"),
                     A=A.view(np.uint16), B=B.view(np.uint16), ref=ref)
        print(f"[{M}x{K}x{n}] macs={r['macs']:.2e} "
              f"compute_lb(fast)={r['compute_lb_us_bfp16_true']}us "
              f"compute_lb(native)={r['compute_lb_us_bf16_emul']}us "
              f"dma_lb(warm)={r['dma_lb_us_warm_L3']}us "
              f"AI={r['arith_intensity_warm']} "
              f"-> fast:{r['roofline_bound_fast_warm']} native:{r['roofline_bound_native_warm']}")

    with open(os.path.join(outdir, "roofline.json"), "w") as f:
        json.dump(roof, f, indent=2)
    print(f"\nWrote goldens + {outdir}/roofline.json "
          f"({'with' if bfloat16 is not None else 'NO ml_dtypes -> roofline only,'} bf16 goldens)")


if __name__ == "__main__":
    main()
