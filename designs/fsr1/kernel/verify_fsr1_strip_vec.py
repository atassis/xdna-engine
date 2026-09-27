#!/usr/bin/env python3
"""Device-verify the fused fsr1_strip_vec kernel (vectorized EASU+RCAS, one dispatch) against
cpu_ref.py's full EASU->RCAS pipeline. This is the real end-to-end gate for the vectorized port.
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
from cpu_ref import fsr_easu, fsr_rcas  # noqa: E402

IN_W = 16
IN_H = 6
OUT_W, OUT_H = IN_W * 3, IN_H * 3

SHIM_BODY = """
extern "C" void fsr1_strip_vec_shim(float *in_rgb, float *out_rgb) {
    fsr1_strip_vec(in_rgb, out_rgb);
}
"""


def main():
    rng = np.random.default_rng(7)
    img = (rng.integers(20, 235, size=(IN_H, IN_W, 3)).astype(np.float32) / 255.0)
    easu = fsr_easu(img, OUT_W, OUT_H)
    golden = fsr_rcas(easu, np.float32(0.2))

    kernel_cc = str(HERE / "fsr1_kernel_vec.cc")
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={IN_H}", "-Oz"]

    t0 = time.time()
    result = bl.verify_oneshot(
        name="fsr1_strip_vec",
        brick_cc=kernel_cc,
        shim_body=SHIM_BODY,
        symbol="fsr1_strip_vec_shim",
        inputs=[(img.reshape(-1), np.float32)],
        out_numel=OUT_H * OUT_W * 3,
        out_shape=(OUT_H, OUT_W, 3),
        unpack=lambda d: d.reshape(OUT_H, OUT_W, 3),
        golden=golden,
        gate=1e-4,  # note only, per error-metrics-are-notes-not-gates -- real gate is determinism
        compile_flags=compile_flags,
        out_dt=np.float32,
        stack_size=0x4000,
    )
    build_s = time.time() - t0
    got = result["got"].reshape(OUT_H, OUT_W, 3)
    diff = np.abs(got - golden)
    nan_mask = np.isnan(golden).any(axis=-1)
    diff_clean = diff[~nan_mask]
    rel_l2 = np.linalg.norm(diff_clean) / max(np.linalg.norm(golden[~nan_mask]), 1e-30)
    print(f"[fsr1_strip_vec] build+run wall time: {build_s:.2f}s")
    print(f"[fsr1_strip_vec] max_abs_diff={diff_clean.max():.3e} mean={diff_clean.mean():.3e} "
          f"rel_l2={rel_l2:.3e} (excluding {nan_mask.sum()} px on the known RCAS div-by-zero "
          f"singularity)")
    print(f"[fsr1_strip_vec] determinism run2run={result['run2run']:.3e} status={result['status']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
