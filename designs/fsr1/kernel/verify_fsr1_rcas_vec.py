#!/usr/bin/env python3
"""Device-verify the vectorized fsr1_rcas_vec kernel (RCAS alone, same-size in->out) against
cpu_ref.py's fsr_rcas. Input is a random RGB image, not chained through EASU -- RCAS is defined
on any image, and random content exercises the mn4==mx4 hitMin/hitMax singularity harder than
EASU's smoothly-varying output does (see fsr1_kernel.cc's rcas_pixel comment).
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
from cpu_ref import fsr_rcas  # noqa: E402

# Reuses FSR1_IN_W/FSR1_IN_H as the RCAS operand's shape divisors: fsr1_rcas_vec loops over
# FSR1_OUT_W x FSR1_OUT_H (= 3x IN_W/IN_H), matching the EASU macros so both kernels build from
# the same -D pair.
IN_W = 16
IN_H = 6
W, H = IN_W * 3, IN_H * 3

SHIM_BODY = """
extern "C" void fsr1_rcas_vec_shim(float *in_rgb, float *out_rgb) {
    fsr1_rcas_vec(in_rgb, out_rgb);
}
"""


def main():
    rng = np.random.default_rng(11)
    img = rng.random(size=(H, W, 3)).astype(np.float32)
    golden = fsr_rcas(img, np.float32(0.2))

    kernel_cc = str(HERE / "fsr1_kernel_vec.cc")
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={IN_H}", "-Oz"]

    t0 = time.time()
    result = bl.verify_oneshot(
        name="fsr1_rcas_vec",
        brick_cc=kernel_cc,
        shim_body=SHIM_BODY,
        symbol="fsr1_rcas_vec_shim",
        inputs=[(img.reshape(-1), np.float32)],
        out_numel=H * W * 3,
        out_shape=(H, W, 3),
        unpack=lambda d: d.reshape(H, W, 3),
        golden=golden,
        gate=1e-2,  # note only, per error-metrics-are-notes-not-gates -- real gate is determinism
        compile_flags=compile_flags,
        out_dt=np.float32,
        stack_size=0x4000,
    )
    build_s = time.time() - t0
    got = result["got"].reshape(H, W, 3)
    diff = np.abs(got - golden)
    nan_mask = np.isnan(golden).any(axis=-1)
    diff_clean = diff[~nan_mask]
    rel_l2 = np.linalg.norm(diff_clean) / max(np.linalg.norm(golden[~nan_mask]), 1e-30)
    print(f"[fsr1_rcas_vec] build+run wall time: {build_s:.2f}s")
    print(f"[fsr1_rcas_vec] max_abs_diff={diff_clean.max():.3e} mean={diff_clean.mean():.3e} "
          f"rel_l2={rel_l2:.3e} (excluding {nan_mask.sum()} px on the known hitMin/hitMax "
          f"div-by-zero singularity)")
    print(f"[fsr1_rcas_vec] determinism run2run={result['run2run']:.3e} status={result['status']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
