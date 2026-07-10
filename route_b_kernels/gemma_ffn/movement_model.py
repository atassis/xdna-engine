#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analytical LPDDR-movement model for the Gemma FFN sub-block: resident-fused vs non-resident baseline.

Predicts the r1 movement gate (Task 5) from first principles, BEFORE the ELF exists, at real Gemma
shapes. Grounded in the audit's measured DMA constants:
  - fixed dispatch floor  c0 = 91 us   (byte-independent; per [[encoder-dma-occupancy]])
  - effective LPDDR rate  ~57 GB/s  ->  17.6 us/MB   (measured, ~half the 120 GB/s datasheet)
  - bf16 = 2 bytes/elem everywhere (activations, weights, output)

The r1 thesis: a resident fused block keeps the gate/up/GeGLU/down intermediates ON-CHIP, so the ONLY
LPDDR traffic is weights + block input + block output. The non-resident baseline round-trips every
intermediate to LPDDR and pays one dispatch floor per op. Weights (3*I*D) are streamed by BOTH arms
(not resident even in prefill, 64KB L1) -- so they are the shared floor; the WIN is the eliminated
intermediate bytes + eliminated dispatches. Reported: dispatches-eliminated (primary, each 91us) and
intermediate-bytes-eliminated (secondary), with the 91us floor netted so we never sell a byte cut as
bandwidth.
"""
C0_US = 91.0            # fixed dispatch floor, microseconds (byte-independent)
US_PER_MB = 17.6        # DMA time per MB at measured ~57 GB/s effective
BYTES = 2               # bf16

MODELS = {  # (D=d_model, I=ffn intermediate, gated) -- Gemma from captured oracles; Parakeet = genericity check
    "gemma3-270m": (640, 2048, True),      # gated GeGLU
    "gemma4-e2b":  (1536, 12288, True),    # gated GeGLU (real I=12288, not config 6144)
    "parakeet-enc": (1024, 4096, False),   # ungated conformer FFN (fc1->gelu->fc2) -- SAME primitive, diff shape
}


def mb(nbytes):
    return nbytes / 1e6


def dma_us(nbytes):
    return mb(nbytes) * US_PER_MB


def analyze(D, I, M, res_dispatch, base_dispatch, gated=True):
    """M = prefill rows (M>=8 regime). gated: Gemma GeGLU (gate+up+down, 3 weights, 3 intermediates);
    ungated: conformer fc1->gelu->fc2 (2 weights, 1 intermediate)."""
    n_weight = 3 if gated else 2             # gated: gate+up+down; ungated: fc1+fc2
    W = n_weight * (I * D) * BYTES           # weights (shared floor, both arms)
    x_io = 2 * (M * D) * BYTES               # block input read + output write (both arms)

    # resident-fused: weights + block I/O only; intermediates stay on-chip
    res_bytes = W + x_io
    res_us = res_dispatch * C0_US + dma_us(res_bytes)

    # non-resident baseline: same weights + block I/O, PLUS every intermediate round-tripped (w+r).
    # gated: normed(M*D) + gate(M*I) + up(M*I) + h(M*I);  ungated: normed(M*D) + fc1(M*I).
    n_mi = 3 if gated else 1                  # # of M*I intermediates round-tripped
    inter = (2 * (M * D) + n_mi * 2 * (M * I)) * BYTES
    base_bytes = W + x_io + inter
    base_us = base_dispatch * C0_US + dma_us(base_bytes)

    return {
        "weights_MB": mb(W), "inter_MB": mb(inter),
        "res_us": res_us, "base_us": base_us,
        "disp_saved": base_dispatch - res_dispatch,
        "disp_saved_us": (base_dispatch - res_dispatch) * C0_US,
        "bytes_saved_us": dma_us(inter),
        "total_saved_us": base_us - res_us,
        "speedup": base_us / res_us,
    }


if __name__ == "__main__":
    # dispatch counts: resident = 1 fused (ideal) or n_chunks if I forces multi-dispatch; baseline =
    # pre_norm + gate + up + geglu + down = 5 ops. We show resident=1 (ideal) and a chunked variant.
    print(f"# constants: dispatch floor {C0_US}us, {US_PER_MB}us/MB (~57 GB/s), bf16")
    print(f"{'model':13s} {'M':>4s} {'wt_MB':>7s} {'int_MB':>7s} {'res_us':>8s} {'base_us':>8s} "
          f"{'disp_save':>9s} {'byte_save':>9s} {'tot_save':>8s} {'speedup':>7s}")
    for name, (D, I, gated) in MODELS.items():
        base_disp = 5 if gated else 4   # gated: norm+gate+up+geglu+down; ungated: norm+fc1+fc2 (+narrow)
        for M in (8, 64, 512):
            r = analyze(D, I, M, res_dispatch=1, base_dispatch=base_disp, gated=gated)
            print(f"{name:13s} {M:>4d} {r['weights_MB']:>7.2f} {r['inter_MB']:>7.3f} "
                  f"{r['res_us']:>8.1f} {r['base_us']:>8.1f} "
                  f"{r['disp_saved_us']:>8.1f}u {r['bytes_saved_us']:>8.1f}u {r['total_saved_us']:>7.1f}u "
                  f"{r['speedup']:>6.2f}x")
    print("\n# read: 'disp_save' = dispatch-floor us eliminated (primary r1 lever); 'byte_save' = "
          "intermediate LPDDR us eliminated (grows with M). weights are the shared floor both arms pay.")
