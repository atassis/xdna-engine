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

import pyxrt
from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor
import aie.utils as aie_utils

DTYPE = {"bf16": bfloat16, "f32": np.float32}

# The fold's resident: fc1's own bf16-out panel build, and the program every modal stream
# borrows under PARAKEET_FOLD_FC1=1.
RESIDENT = "512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024krtp"

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


def cross_arms(o):
    """(host xclbin, stream) pairs -- run a stream on an array program it was not built for.

    THIS IS WHAT THE ENCODER ACTUALLY DOES and what the per-arm run above deliberately does not.
    `stream_k_ex` resolves an insts path only and binds the instruction BO to the ONE resident
    kernel, so under the fold the K=4096 collapse stream executes on fc1's N=4096 array program
    rather than on its own N=1024 build.

    ORDER IS AN INDEPENDENT VARIABLE, which is why the pairs are a command-line list rather than a
    constant. Measured 2026-08-23: the K=4096 stream FAILS on the resident when it follows a
    K=1024 dispatch and PASSES when it follows another K=4096 one, so a fixed list reproduces a
    sequence rather than a property. Pass `--cross-pairs host:stream ...` to run one explicitly;
    a bare stream name is paired with the fold resident.
    """
    if not o.cross_pairs:
        return [(RESIDENT, "512x1024x1024_32x32x128_8c_modalidbf16outkrtp"),
                (RESIDENT, "512x4096x1024_32x32x128_8c_modalidbf16outkrtp"),
                (RESIDENT, "512x4096x1024_32x32x128_8c_modalidbf16outkrtpapanel1024")]
    out = []
    for spec in o.cross_pairs:
        host, _, stream = spec.rpartition(":")
        out.append((host or RESIDENT, stream))
    return out


def run_cross(o, patterns, results):
    """Drive each (host, stream) pair and gate the stream's OWN closed form."""
    rt = aie_utils.DefaultNPURuntime
    # Load each distinct host ONCE and reuse it across the whole sequence. Re-loading per step
    # would put a context creation between consecutive dispatches, which is itself a state reset --
    # and order is the independent variable, so that confound has to go.
    loaded = {}
    for pos, (host, stream) in enumerate(cross_arms(o)):
        if host not in loaded:
            handle, _ = load_arm(host, o)
            kh = handle
            while not hasattr(kh, "kernel"):
                kh = kh._handle
            loaded[host] = kh
        kh = loaded[host]
        M, K, N = shape_of(stream, o)
        hM, hK, hN = shape_of(host, o)
        out_dt = design_operands(o.build_dir, stream)[-1][1]
        arr = np.frombuffer(open(os.path.join(o.build_dir, f"insts_{stream}.txt"), "rb").read(),
                            dtype=np.uint32)
        ibo = rt._tensor_class(arr, flags=pyxrt.bo.cacheable, group_id=kh.kernel.group_id(1),
                               xrt_device=rt._device).buffer_object()
        # Size every BO to the MAX of the two designs. The HOST program's baked tile-loop bound may
        # drive more traffic than the stream's own shape implies, and an under-sized BO would fault
        # or corrupt neighbouring memory instead of returning the wrong answer this probe is here
        # to see.
        # Position is part of the key: order is load-bearing, so the same pair at two
        # points in a sequence is two different observations, not one overwritten.
        key = f"[{pos}] {stream} ON {host}"
        results[key] = {"out_dtype": out_dt, "host": host, "stream": stream}
        # ONE dispatch per step when a single pattern is selected. Order is the independent
        # variable here, so an arm that silently issues two dispatches makes the sequence
        # ambiguous -- which is exactly how the first reading of this effect was misread.
        for pname in (("exact", "dense") if o.cross_pattern == "both" else (o.cross_pattern,)):
            a, b = patterns[pname](M, K, N)
            a_pad = np.zeros(max(M * K, hM * hK), dtype=bfloat16); a_pad[:M * K] = a.reshape(M * K)
            b_pad = np.zeros(max(K * N, hK * hN), dtype=bfloat16); b_pad[:K * N] = b.reshape(K * N)
            c_pad = np.full(max(M * N, hM * hN), np.nan, dtype=DTYPE[out_dt])
            At, Bt = tensor(a_pad, dtype=bfloat16), tensor(b_pad, dtype=bfloat16)
            Ct = tensor(c_pad, dtype=DTYPE[out_dt])
            for t in (At, Bt, Ct):
                t.to("npu")
            kh.kernel(3, ibo, arr.nbytes, At.buffer_object(), Bt.buffer_object(),
                      Ct.buffer_object()).wait()
            Ct.to("cpu")
            got = np.asarray(Ct)[:M * N].reshape(M, N).astype(np.float64)
            ref = reference(a_as_the_stream_reads_it(
                a.reshape(M * K).astype(np.float64), stream, M, K), b, K, N)
            bad = int(np.count_nonzero(got != ref))
            rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
            results[key][pname] = {
                "mismatches": bad, "elements": int(got.size), "rel_l2": rel,
                "max_abs_err": float(np.max(np.abs(got - ref))),
                "nan_out": int(np.count_nonzero(~np.isfinite(got))),
            }
            print(f"  {key:<58s} {pname:<6s} rel-L2 {rel:>9.4f}  bad {bad:>7d}/{got.size}  "
                  f"nan {int(np.count_nonzero(~np.isfinite(got))):>7d}", flush=True)


def main(o):
    os.makedirs(o.artifacts, exist_ok=True)
    rt = aie_utils.DefaultNPURuntime
    M, K, N = shape_of(ARMS[0], o)
    for s in o.arms:
        if shape_of(s, o) != (M, K, N):
            sys.exit(f"{s}: shape {shape_of(s, o)} != {(M, K, N)} -- arms must share one shape")
    if K % N:
        sys.exit(f"this probe's B construction needs K % N == 0, got K={K} N={N}")

    # Shape-parameterized, because the cross arms below run streams of a DIFFERENT shape than the
    # per-arm set. Seeded per call so a given (pattern, shape) is the same data everywhere.
    def exact(m, k, n):
        """+-1 A, single-cycle permutation B. Lossless end to end -- see the header."""
        r = np.random.default_rng(seed=42)
        a = np.where(r.random((m, k)) < 0.5, -1.0, 1.0).astype(bfloat16)
        b = np.zeros((k, n), dtype=bfloat16)
        b[np.arange(k), (np.arange(k) + 1) % n] = 1.0
        return a, b

    def dense(m, k, n):
        """The precision-class control. Small magnitudes keep the f32 accumulate well away from
        saturation, so a large rel-L2 is a defect and not a range artifact."""
        r = np.random.default_rng(seed=43)
        return (r.normal(0, 0.5, (m, k)).astype(bfloat16),
                r.normal(0, 0.5, (k, n)).astype(bfloat16))

    patterns = {"exact": exact, "dense": dense}

    results = {}
    for suffix in o.arms:
        handle, out_dt = load_arm(suffix, o)
        results[suffix] = {"out_dtype": out_dt}
        for pname, mk in patterns.items():
            a, b = mk(M, K, N)
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

    if o.cross:
        print("\n--- cross-program: the stream on an array program it was not built for ---")
        run_cross(o, patterns, results)

    print()
    # The exact pattern is the verdict: equality or a defect, nothing in between.
    for s, r in results.items():
        e = r.get("exact") or r["dense"]
        dense = f", dense rel-L2 {r['dense']['rel_l2']:.4f}" if "dense" in r else ""
        print(f"  {'FAIL' if e['mismatches'] else 'PASS'}  {s}  "
              f"({e['mismatches']}/{e['elements']} bad{dense})")
    with open(o.out, "w") as f:
        json.dump({"shape": [M, K, N], "results": results}, f, indent=2)
    print(f"\nwrote {o.out}")
    return 0


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
    p.add_argument("--cross-pattern", choices=("exact", "dense", "both"), default="both",
                   help="one dispatch per step unless 'both'")
    p.add_argument("--cross-pairs", nargs="+", default=None,
                   help="[host:]stream pairs to run in THIS ORDER; order is load-bearing")
    p.add_argument("--cross", action="store_true",
                   help="also run each stream on the fold resident's array program")
    p.add_argument("--out", default="artifacts/gemm_drain_width_isolated.json")
    sys.exit(main(p.parse_args()))
