#!/usr/bin/env python3
"""Device-verify the vectorized fsr1_easu_vec kernel (EASU only -- see fsr1_kernel_vec.cc
header for why RCAS is a separate pass) against cpu_ref.py, as a note not a bit-exact gate:
aie::inv/aie::invsqrt replace FSR1's own bit-trick reciprocal approximations (a deliberate
brick choice -- aie2p has a real SFU, FSR1's bit tricks exist for GPUs that don't).
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
WORKSPACE = HERE.parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE / "xdna-engine" / "aie_kernels" / "_test"))
import bricklib as bl  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from cpu_ref import fsr_easu  # noqa: E402

IN_W = 16
IN_H = 6
OUT_W, OUT_H = IN_W * 3, IN_H * 3

SHIM_BODY = """
extern "C" void fsr1_easu_vec_shim(float *in_rgb, float *out_rgb) {
    fsr1_easu_vec(in_rgb, out_rgb);
}
"""


def main():
    rng = np.random.default_rng(7)
    img = (rng.integers(20, 235, size=(IN_H, IN_W, 3)).astype(np.float32) / 255.0)
    golden = fsr_easu(img, OUT_W, OUT_H)

    kernel_cc = str(HERE / "fsr1_kernel_vec.cc")
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={IN_H}", "-Oz"]

    t0 = time.time()
    result = bl.verify_oneshot(
        name="fsr1_easu_vec",
        brick_cc=kernel_cc,
        shim_body=SHIM_BODY,
        symbol="fsr1_easu_vec_shim",
        inputs=[(img.reshape(-1), np.float32)],
        out_numel=OUT_H * OUT_W * 3,
        out_shape=(OUT_H, OUT_W, 3),
        unpack=lambda d: d.reshape(OUT_H, OUT_W, 3),
        golden=golden,
        gate=5e-2,
        compile_flags=compile_flags,
        out_dt=np.float32,
        stack_size=0x4000,
    )
    build_s = time.time() - t0
    got = result["got"].reshape(OUT_H, OUT_W, 3)
    diff = np.abs(got - golden)
    nan_mask = np.isnan(golden).any(axis=-1)
    diff_clean = diff[~nan_mask]
    print(f"[fsr1_easu_vec] build+run wall time: {build_s:.2f}s")
    print(f"[fsr1_easu_vec] max_abs_diff={diff_clean.max():.3e} mean={diff_clean.mean():.3e} "
          f"vs cpu_ref.fsr_easu (note: aie::inv/invsqrt swap, expect a real but bounded gap)")
    print(f"[fsr1_easu_vec] determinism run2run={result['run2run']:.3e} status={result['status']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
