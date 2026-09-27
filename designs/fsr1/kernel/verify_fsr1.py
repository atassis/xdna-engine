#!/usr/bin/env python3
"""Device-verify the fsr1_strip kernel (EASU+RCAS, one resident 8x8 RGB crop -> 24x24)
against designs/fsr1/cpu_ref.py, using the aie_kernels/_test bricklib rail (read-only
import -- this script and the kernel it builds live entirely under designs/fsr1/).
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
WORKSPACE = HERE.parent.parent.parent.parent  # kernel/fsr1/designs/wt-npu-fsr1/<workspace>
sys.path.insert(0, str(WORKSPACE / "xdna-engine" / "aie_kernels" / "_test"))
import bricklib as bl  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from cpu_ref import fsr_easu, fsr_rcas  # noqa: E402

IN_W = IN_H = 8
OUT_W = OUT_H = IN_W * 3

SHIM_BODY = f"""
extern "C" void fsr1_strip_shim(float *in_rgb, float *out_rgb) {{
    fsr1_strip(in_rgb, out_rgb);
}}
"""


def main():
    rng = np.random.default_rng(7)
    img = (rng.integers(20, 235, size=(IN_H, IN_W, 3)).astype(np.float32) / 255.0)
    easu = fsr_easu(img, OUT_W, OUT_H)
    golden = fsr_rcas(easu, np.float32(0.2))

    kernel_cc = str(HERE / "fsr1_kernel.cc")
    # -Oz: at -O2 this scalar kernel's .text is ~18KB, over the aie2p 16KB program memory
    # (measured via llvm-size on a standalone Peano compile, see designs/fsr1/README.md);
    # -Oz brings it to ~12.5KB. All arithmetic here is software-emulated (__mulsf3/__divsf3
    # libcalls -- no aie_api vector float path used), so this is a correctness-first port,
    # not a brick-first one; see README for what that costs.
    compile_flags = [f"-DFSR1_IN_W={IN_W}", f"-DFSR1_IN_H={IN_H}", "-Oz"]

    t0 = time.time()
    result = bl.verify_oneshot(
        name="fsr1_strip",
        brick_cc=kernel_cc,
        shim_body=SHIM_BODY,
        symbol="fsr1_strip_shim",
        inputs=[(img.reshape(-1), np.float32)],
        out_numel=OUT_H * OUT_W * 3,
        out_shape=(OUT_H, OUT_W, 3),
        unpack=lambda d: d.reshape(OUT_H, OUT_W, 3),
        golden=golden,
        gate=1e-2,  # note only, per error-metrics-are-notes-not-gates -- real gate is determinism
        compile_flags=compile_flags,
        out_dt=np.float32,
        stack_size=0x2000,
    )
    build_s = time.time() - t0
    got = result["got"].reshape(OUT_H, OUT_W, 3)
    diff = np.abs(got - golden)
    nan_mask = np.isnan(golden).any(axis=-1)
    diff_clean = diff[~nan_mask]
    print(f"[fsr1_strip] build+run wall time: {build_s:.2f}s")
    print(f"[fsr1_strip] max_abs_diff={diff_clean.max():.3e} mean={diff_clean.mean():.3e} "
          f"(excluding {nan_mask.sum()} px on the known RCAS div-by-zero singularity)")
    print(f"[fsr1_strip] determinism run2run={result['run2run']:.3e} status={result['status']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
