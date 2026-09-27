#!/usr/bin/env python3
"""Device-verify warp_kernel.cc against cpu_ref.warp_tile_halo, for C=3 and C=4.

Tile/halo shrunk to fit a C=4 crop in one core's 64KB L1 at objectFIFO depth 2 (a C=4,
HALO=16, TILE=32x8 crop overflows -- "allocated buffers exceeded available memory").
Same kernel, same numerics -- see method-aie2p-device-test-build.md."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
WORKSPACE = HERE.parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE / "xdna-engine" / "aie_kernels" / "_test"))
import bricklib as bl  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from cpu_ref import warp_tile_halo  # noqa: E402

TILE_W, TILE_H, HALO = 16, 4, 8
PAD_W, PAD_H = TILE_W + 2 * HALO, TILE_H + 2 * HALO

SHIM_BODY = """
extern "C" void warp_kernel_shim(float *in_padded, float *flow, float *out) {
    warp_kernel(in_padded, flow, out);
}
"""


def run_one(ch):
    rng = np.random.default_rng(3 + ch)
    img = rng.standard_normal((PAD_H, PAD_W, ch)).astype(np.float32)
    # flow within +/- HALO so the tile-halo reference matches the kernel's own fallback
    # behaviour exactly (a flow exceeding HALO is a documented approximation, not a bug).
    fx = (rng.standard_normal((TILE_H, TILE_W)) * (HALO * 0.6)).astype(np.float32)
    fy = (rng.standard_normal((TILE_H, TILE_W)) * (HALO * 0.6)).astype(np.float32)
    golden = warp_tile_halo(img, fx, fy, HALO)
    flow = np.concatenate([fx.reshape(-1), fy.reshape(-1)])

    kernel_cc = str(HERE / "warp_kernel.cc")
    compile_flags = [f"-DWARP_TILE_W={TILE_W}", f"-DWARP_TILE_H={TILE_H}",
                      f"-DWARP_HALO={HALO}", f"-DWARP_CH={ch}", "-Oz"]

    result = bl.verify_oneshot(
        name=f"warp_kernel_c{ch}",
        brick_cc=kernel_cc,
        shim_body=SHIM_BODY,
        symbol="warp_kernel_shim",
        inputs=[(img.reshape(-1), np.float32), (flow, np.float32)],
        out_numel=TILE_H * TILE_W * ch,
        out_shape=(TILE_H, TILE_W, ch),
        unpack=lambda d: d.reshape(TILE_H, TILE_W, ch),
        golden=golden,
        gate=1e-5,
        compile_flags=compile_flags,
        out_dt=np.float32,
        stack_size=0x1000,
    )
    got = result["got"].reshape(TILE_H, TILE_W, ch)
    diff = np.abs(got - golden)
    print(f"[warp C={ch}] max_abs_diff={diff.max():.3e} mean={diff.mean():.3e} "
          f"rel_l2={result['rel_l2']:.3e} determinism={result['run2run']:.3e} "
          f"status={result['status']}")
    return result["ok"]


if __name__ == "__main__":
    ok3 = run_one(3)
    ok4 = run_one(4)
    sys.exit(0 if (ok3 and ok4) else 1)
