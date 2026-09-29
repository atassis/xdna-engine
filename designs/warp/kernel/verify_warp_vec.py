#!/usr/bin/env python3
"""Device-verify warp_kernel_vec.cc (vectorized: bf16 data, Q8.7 fixed-point flow), C=3
and C=4. Gate on determinism + agreement with the bf16/Q8.7-quantized reference (proves
the KERNEL LOGIC, not the format choice); rel_l2 against the plain fp32 reference is
reported separately as a NOTE (the format's own accuracy cost, not a correctness bug --
per error-metrics-are-notes-not-gates)."""
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

HERE = Path(__file__).parent
WORKSPACE = HERE.parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE / "xdna-engine" / "aie_kernels" / "_test"))
import bricklib as bl  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from cpu_ref import quantize_flow_q87, warp_tile_halo_vec, warp_tile_halo  # noqa: E402

BF16 = ml_dtypes.bfloat16
TILE_W, TILE_H, HALO = 16, 4, 8
PAD_W, PAD_H = TILE_W + 2 * HALO, TILE_H + 2 * HALO

SHIM_BODY = """
extern "C" void warp_kernel_vec_shim(bfloat16 *in_padded, int16_t *flow, bfloat16 *out) {
    warp_kernel_vec(in_padded, flow, out);
}
"""


def run_one(ch):
    rng = np.random.default_rng(5 + ch)
    img_f32 = rng.standard_normal((PAD_H, PAD_W, ch)).astype(np.float32)
    img_bf16 = img_f32.astype(BF16)
    fx = (rng.standard_normal((TILE_H, TILE_W)) * (HALO * 0.6)).astype(np.float32)
    fy = (rng.standard_normal((TILE_H, TILE_W)) * (HALO * 0.6)).astype(np.float32)
    fxi, fyi = quantize_flow_q87(fx, fy)

    golden_quantized = warp_tile_halo_vec(img_bf16.astype(np.float64), fxi, fyi, HALO)
    golden_quantized_bf16 = golden_quantized.astype(BF16).astype(np.float64)
    golden_fp32 = warp_tile_halo(img_f32, fx, fy, HALO)  # true fp32 reference, no quant

    kernel_cc = str(HERE / "warp_kernel_vec.cc")
    compile_flags = [f"-DWARP_TILE_W={TILE_W}", f"-DWARP_TILE_H={TILE_H}",
                      f"-DWARP_HALO={HALO}", f"-DWARP_CH={ch}", "-Oz"]
    flow_packed = np.concatenate([fxi.reshape(-1), fyi.reshape(-1)])

    result = bl.verify_oneshot(
        name=f"warp_kernel_vec_c{ch}",
        brick_cc=kernel_cc,
        shim_body=SHIM_BODY,
        symbol="warp_kernel_vec_shim",
        inputs=[(img_bf16.reshape(-1), BF16), (flow_packed, np.int16)],
        out_numel=TILE_H * TILE_W * ch,
        out_shape=(TILE_H, TILE_W, ch),
        unpack=lambda d: d.astype(np.float64).reshape(TILE_H, TILE_W, ch),
        golden=golden_quantized_bf16,
        gate=1e-2,
        compile_flags=compile_flags,
        out_dt=BF16,
        stack_size=0x1000,
    )
    got = result["got"].reshape(TILE_H, TILE_W, ch)
    diff_quant = np.abs(got - golden_quantized_bf16)
    diff_fp32 = np.abs(got - golden_fp32)
    rel_fp32 = np.linalg.norm(diff_fp32) / np.linalg.norm(golden_fp32)
    print(f"[warp_vec C={ch}] vs bf16/Q8.7 golden: max={diff_quant.max():.3e} "
          f"mean={diff_quant.mean():.3e} (correctness gate)")
    print(f"[warp_vec C={ch}] vs fp32 reference (NOTE, format cost): rel_l2={rel_fp32:.3e} "
          f"max_abs={diff_fp32.max():.3e}")
    print(f"[warp_vec C={ch}] determinism={result['run2run']:.3e} status={result['status']}")
    return result["ok"]


if __name__ == "__main__":
    ok3 = run_one(3)
    ok4 = run_one(4)
    sys.exit(0 if (ok3 and ok4) else 1)
