#!/usr/bin/env python3
# Is a K=4096 a_panel GEMM correct when the resident drains bf16?
# (task mode-switched-multi-program-xclbin -- the isolated probe its `next:` prescribed)
#
# WHY IN ISOLATION. The fold+krtp composition fails encoder parity at rel-L2 0.9942, and a four-arm
# in-encoder bisect already exonerated the two consumers it could reach: the krtp ARTIFACT SET (same
# fold, collapse off, on the krtp artifacts = 0.0823 PASS) and the residual-add CONSUMER (unfused,
# FFN read back at its own boundary = still 0.9942). What survives is a statement about ONE dispatch:
# a K=4096 a_panel GEMM is correct with an f32 drain and wrong with a bf16 one. That statement does
# not need an encoder to test, and testing it inside one is how the last three suspects cost a run
# each -- confirm the instrument before escalating. One xclbin, one dispatch, a host reference.
#
# THE ARMS ARE A FACTORIAL, and the m=32 f32 cell is the one that had to be built for it. The banked
# artifacts vary dtype_out and tile-m TOGETHER (every f32 arm is m=64, every bf16 arm m=32), so a
# bf16-vs-f32 difference read off them is confounded with the tile. `512x4096x1024_32x32x128_8c_
# modalidkrtpapanel1024` holds m=32 and krtp+apanel fixed and moves ONLY the drain width, which is
# the comparison the verdict rests on:
#
#     suffix                                              m   dtype_out  krtp+apanel
#     512x4096x1024_64x32x128_8c_modalid                  64   f32        no
#     512x4096x1024_64x32x128_8c_modalidkrtpapanel1024    64   f32        yes
#     512x4096x1024_32x32x128_8c_modalidbf16out           32   bf16       no
#     512x4096x1024_32x32x128_8c_modalidkrtpapanel1024    32   f32        yes   <- built for this probe
#     512x4096x1024_32x32x128_8c_modalidbf16outkrtpapanel1024  32  bf16   yes   <- the suspect
#
# THE STREAM IS SELF-CONTAINED, which is what makes one dispatch a complete test: the krtp arms bake
# `rtp_write(@rtp_c_r, 1, 128)` for every core (128 = K/k = 4096/32, the full trip count) and
# `rtp[0]=0` for the identity epilogue, so nothing the encoder does per-dispatch is missing here.
# Verified by reading the generated MLIR, not assumed.
#
# TWO PATTERNS, because they fail differently and only one of them is a proof.
#   exact -- A is +-1 and B maps k -> (k+1) % N. Every value is a power of two, so bfp16's shared
#            exponent is lossless, the f32 accumulate sums integers, and a bf16 drain holds the
#            result EXACTLY: at K/N = 4 each output is a sum of four +-1 terms, an integer in
#            [-4, 4]. The gate is equality, not a tolerance, and it catches a wrong trip count
#            (contracting 1024 of 4096 drops three of the four blocks) as readily as a wrong value.
#   dense -- random bf16 A and B against an f64 host reference, gated on rel-L2. bfp16 rounding
#            makes this inexact by construction (~1e-2 is healthy), so it cannot prove correctness
#            -- it is here to catch a precision-class defect that +-1 data would mask, and to be the
#            same statistic the encoder parity gate reports.
#
# Run (NPU free -- use the device wrapper, which stops xdna-engine and npu-vox):
#   bash scripts/gemm_drain_width_isolated_device.sh
import argparse
import json
import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemm_command_accounting import design_operands, insts_bin, shape_of

from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor
import aie.utils as aie_utils

DTYPE = {"bf16": bfloat16, "f32": np.float32}

ARMS = [
    "512x4096x1024_64x32x128_8c_modalid",
    "512x4096x1024_64x32x128_8c_modalidkrtpapanel1024",
    "512x4096x1024_32x32x128_8c_modalidbf16out",
    "512x4096x1024_32x32x128_8c_modalidkrtpapanel1024",
    "512x4096x1024_32x32x128_8c_modalidbf16outkrtpapanel1024",
]


def load_arm(suffix, o):
    """Load one xclbin into its own hw_context; return the runtime handle and its output dtype."""
    xclbin = os.path.join(o.build_dir, f"final_{suffix}.xclbin")
    insts = os.path.join(o.build_dir, f"insts_{suffix}.txt")
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")
    operands = design_operands(o.build_dir, suffix)
    if not operands:
        sys.exit(f"{suffix}: could not read runtime_sequence operands")
    if len(operands) > 3:
        sys.exit(f"{suffix}: traced design -- this probe runs no-trace arms only")
    kern = NPUKernel(xclbin_path=xclbin, insts_path=insts_bin(insts, o.artifacts),
                     kernel_name=o.kernel)
    return aie_utils.DefaultNPURuntime.load(kern), operands[-1][1]


def a_as_the_stream_reads_it(flat, suffix, M, K):
    """A in the layout the stream contracts over.

    `--a-panel-width W` reads the SAME host bytes as [K/W, M, W], so a panel arm handed a row-major
    buffer contracts over a known PERMUTATION of A rather than over A. Feeding each arm its preferred
    layout instead would change the operand between arms, which is the one thing this comparison
    cannot afford -- so the permutation goes in the reference.
    """
    w = re.search(r"apanel(\d+)", suffix)
    if not w:
        return flat.reshape(M, K)
    w = int(w.group(1))
    j = np.arange(K)
    return flat[(j // w)[None, :] * M * w + np.arange(M)[:, None] * w + (j % w)[None, :]]


def reference(a, b, K, N):
    """Host C = A @ B, blocked the way B's period does it, in f64."""
    return sum(a[:, p * N:(p + 1) * N].astype(np.float64) @ b[p * N:(p + 1) * N].astype(np.float64)
               for p in range(K // N))


def main(o):
    os.makedirs(o.artifacts, exist_ok=True)
    rt = aie_utils.DefaultNPURuntime
    M, K, N = shape_of(ARMS[0], o)
    for s in o.arms:
        if shape_of(s, o) != (M, K, N):
            sys.exit(f"{s}: shape {shape_of(s, o)} != {(M, K, N)} -- arms must share one shape")
    if K % N:
        sys.exit(f"this probe's B construction needs K % N == 0, got K={K} N={N}")

    rng = np.random.default_rng(seed=42)
    patterns = {}
    # exact: +-1 A, single-cycle permutation B. Lossless end to end -- see the header.
    a_ex = np.where(rng.random((M, K)) < 0.5, -1.0, 1.0).astype(bfloat16)
    b_ex = np.zeros((K, N), dtype=bfloat16)
    b_ex[np.arange(K), (np.arange(K) + 1) % N] = 1.0
    patterns["exact"] = (a_ex, b_ex)
    # dense: the precision-class control. Small magnitudes keep the f32 accumulate well away from
    # saturation so a large rel-L2 is a defect and not a range artifact.
    patterns["dense"] = (rng.normal(0, 0.5, (M, K)).astype(bfloat16),
                         rng.normal(0, 0.5, (K, N)).astype(bfloat16))

    results = {}
    for suffix in o.arms:
        handle, out_dt = load_arm(suffix, o)
        results[suffix] = {"out_dtype": out_dt}
        for pname, (a, b) in patterns.items():
            At = tensor(a.reshape(M * K), dtype=bfloat16)
            Bt = tensor(b.reshape(K * N), dtype=bfloat16)
            # Poison the drain through the same host->device path A and B take: a dispatch that
            # never ran, or one whose k-loop wrote nothing, must not be able to return a plausible
            # buffer left over from the previous pattern. NaN survives no arithmetic, so a partial
            # write shows up as `nan_out` rather than as a small error.
            Ct = tensor(np.full(M * N, np.nan, dtype=DTYPE[out_dt]), dtype=DTYPE[out_dt])
            rt.run(handle, [At, Bt, Ct])
            got = np.asarray(Ct).reshape(M, N).astype(np.float64)
            ref = reference(a_as_the_stream_reads_it(
                np.asarray(At).reshape(M * K).astype(np.float64), suffix, M, K), b, K, N)

            bad = int(np.count_nonzero(got != ref))
            rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
            results[suffix][pname] = {
                "mismatches": bad, "elements": int(got.size), "rel_l2": rel,
                "max_abs_err": float(np.max(np.abs(got - ref))),
                "nan_out": int(np.count_nonzero(~np.isfinite(got))),
            }
            print(f"  {suffix:<58s} {pname:<6s} rel-L2 {rel:>9.4f}  "
                  f"bad {bad:>7d}/{got.size}  maxabs {np.max(np.abs(got - ref)):>9.4f}"
                  f"{'  NAN' if not np.all(np.isfinite(got)) else ''}", flush=True)

    print()
    # The exact pattern is the verdict: equality or a defect, nothing in between.
    failed = [s for s in o.arms if results[s]["exact"]["mismatches"]]
    for s in o.arms:
        e = results[s]["exact"]
        print(f"  {'FAIL' if e['mismatches'] else 'PASS'}  {s}  "
              f"(exact {e['mismatches']}/{e['elements']} bad, dense rel-L2 "
              f"{results[s]['dense']['rel_l2']:.4f})")
    with open(o.out, "w") as f:
        json.dump({"shape": [M, K, N], "results": results}, f, indent=2)
    print(f"\nwrote {o.out}")
    return 1 if len(failed) == len(o.arms) else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="mlir-aie/programming_examples/basic/"
                                          "matrix_multiplication/whole_array/build")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--arms", nargs="+", default=ARMS)
    # shape_of() prefers explicit flags and otherwise reads the suffix's own MxKxN token.
    p.add_argument("--M", type=int)
    p.add_argument("--K", type=int)
    p.add_argument("--N", type=int)
    p.add_argument("--out", default="artifacts/gemm_drain_width_isolated.json")
    sys.exit(main(p.parse_args()))
