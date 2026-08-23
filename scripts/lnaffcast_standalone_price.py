#!/usr/bin/env python3
# What does the SHIPPED standalone lnaffcast op cost, at the row count the encoder actually runs it?
# (task mode-switched-multi-program-xclbin -- the last item before the merge verdict)
#
# WHY THIS NUMBER DECIDES SOMETHING. Merging lnaffcast's context into the modal GEMM's is priced at
# -240.0 ms/clip of transition tax (brick-merge-ranking-on-the-post-composition-ledger), but that
# ledger prices only the BOUNDARIES. It says nothing about what the op costs once it is a mode, and
# the mode is a different program on a different topology: the standalone op runs 8 cores, one row
# per acquire, with real f32 operands; the mode borrows the GEMM's bf16 fifos across all 32 cores.
# So the merge's net is
#
#     -240.0 ms/clip  +  dispatches_per_clip x (mode - standalone)
#
# and the second term has never been measured. This measures its right half. The left half is
# already banked at 590.9 us (lnaffcast_1x_1x512.json, 615.7/566.1 forward/reversed).
#
# THE ROW COUNT IS NOT A CHOICE. npu.rs pads x to PAD_M = 512 and dispatches the one baked
# instruction stream, so the shipped op runs 512 rows on every clip regardless of T -- the encoder
# has exactly one lnaffcast row count and it is 512. That is also the only row count the standalone
# xclbin can run: sequence_length is baked into its taps.
#
# THE COMPARISON IS CROSS-XCLBIN AND CANNOT USE THE USUAL CONTROL. Every prior arm on this task
# rode a modal xclbin, so the GEMM stream ran as a within-group control
# (gemm-stream-controls-the-cross-xclbin-comparison). The standalone op's xclbin has no GEMM stream
# to run. The wrapper therefore BRACKETS this measurement with GEMM controls on either side in the
# same quiesced session: if the two brackets agree, the session did not drift across the load in
# between, which is the same evidence by a different route.
#
# BOTH SIDES ARE GATED ON ONE REFERENCE. The x, gamma and beta here are drawn with the same seeds
# and in the same order as lnaffcast_burst_isolation.py's operands(), so the host reference is
# byte-identical to the one the mode is gated against. A parity number from this script is directly
# comparable to the mode's rather than merely similar in kind.
#
# Run (NPU free -- use the device wrapper, which stops xdna-engine and npu-vox):
#   bash scripts/lnaffcast_standalone_device.sh
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemm_command_accounting import insts_bin

import pyxrt
from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor
import aie.utils as aie_utils

COLS = 1024      # the lnaffcast embedding dim (KRES)
ROWS = 512       # PAD_M -- the encoder's only lnaffcast row count, and this xclbin's baked one
EPS = 1e-5       # ln_affine_cast.cc's epsilon, and mm_mode_lnaffcast.cc's after it


def reference(rows):
    """x, gamma|beta and the host answer, drawn exactly as lnaffcast_burst_isolation.py draws them.

    Seeds and draw ORDER both matter: gamma and beta come off ONE rng in that sequence, so drawing
    them separately or swapping them would give a different reference and quietly make the two
    sides of the comparison ungated against each other.
    """
    rng = np.random.default_rng(seed=7)
    gamma = rng.normal(1.0, 0.1, COLS).astype(np.float32)
    beta = rng.normal(0.0, 0.1, COLS).astype(np.float32)
    x = np.random.default_rng(seed=11).standard_normal((rows, COLS)).astype(np.float32)

    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    ref = ((x - mean) / np.sqrt(var + EPS)) * gamma + beta
    gb = np.concatenate([gamma, beta]).astype(np.float32)
    return x, gb, ref.astype(bfloat16).astype(np.float64)


def parity(got, ref):
    d = got - ref
    return {
        "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(ref)),
        "max_abs_err": float(np.max(np.abs(d))),
        "exact": int(np.count_nonzero(got == ref)),
        "elements": int(ref.size),
        "nan_out": int(np.count_nonzero(~np.isfinite(got))),
    }


def main(o):
    os.makedirs(o.artifacts, exist_ok=True)
    rt = aie_utils.DefaultNPURuntime

    xclbin = os.path.join(o.ln_dir, f"final_{o.stem}.xclbin")
    insts = os.path.join(o.ln_dir, f"insts_{o.stem}.txt")
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")

    kern = NPUKernel(xclbin_path=xclbin,
                     insts_path=insts_bin(insts, o.artifacts),
                     kernel_name=o.kernel)
    kh = rt.load(kern)
    while not hasattr(kh, "kernel"):
        kh = kh._handle

    arr = np.frombuffer(open(insts, "rb").read(), dtype=np.uint32)
    ibo = rt._tensor_class(arr, flags=pyxrt.bo.cacheable,
                           group_id=kh.kernel.group_id(1),
                           xrt_device=rt._device).buffer_object()

    x, gb, ref = reference(o.rows)
    # out is poisoned with NaN so a dispatch that wrote nothing reads as nan rather than as zeros,
    # which would otherwise score as a merely-bad rel-L2 instead of as "the stream did not run".
    out = np.full(o.rows * COLS, np.nan, dtype=bfloat16)

    print(f"--- standalone {o.stem}: {o.rows} rows x {COLS} cols, {o.reps} reps per block "
          f"(rep 0 dropped as cold) ---")
    print(f"    x {x.nbytes / 1024:.0f} KB f32 in, gb {gb.nbytes} B, out {out.nbytes / 1024:.0f} KB bf16")

    results = {"blocks": {}}
    for bname in ("forward", "reversed"):
        Xt = tensor(x.reshape(-1).copy(), dtype=np.float32)
        Gt = tensor(gb.copy(), dtype=np.float32)
        Ot = tensor(out.copy(), dtype=bfloat16)
        for t in (Xt, Gt, Ot):
            t.to("npu")
        us = []
        for _ in range(o.reps):
            t0 = time.perf_counter_ns()
            kh.kernel(3, ibo, arr.nbytes, Xt.buffer_object(), Gt.buffer_object(),
                      Ot.buffer_object()).wait()
            us.append((time.perf_counter_ns() - t0) / 1000.0)
        Ot.to("cpu")
        got = np.asarray(Ot)[: o.rows * COLS].reshape(o.rows, COLS).astype(np.float64)
        steady = us[1:] or us
        rec = {"us": us, "median_us": statistics.median(steady), "parity": parity(got, ref)}
        results["blocks"][bname] = rec
        p = rec["parity"]
        print(f"  {bname:<10s} median {rec['median_us']:>8.1f} us  "
              f"reps {' '.join(f'{v:.0f}' for v in us)}  "
              f"rel-L2 {p['rel_l2']:>10.4e}  exact {p['exact']}/{p['elements']}"
              f"{'  NAN' if p['nan_out'] else ''}", flush=True)

    med = [results["blocks"][b]["median_us"] for b in ("forward", "reversed")]
    mean = statistics.mean(med)
    spread = abs(med[0] - med[1]) / mean
    results["mean_us"] = mean
    results["block_spread"] = spread
    print(f"\n  mean of blocks {mean:>8.1f} us   block spread {spread:.1%}")

    # This op is the thing being replaced, so it MUST be correct -- there is no control arm here
    # whose job is to fail.
    p_ = results["blocks"]["forward"]["parity"]
    ok = p_["rel_l2"] < o.rel_l2_gate and not p_["nan_out"]
    print(f"  parity {'PASS' if ok else 'FAIL'} "
          f"(rel-L2 {p_['rel_l2']:.4e} vs gate {o.rel_l2_gate:.1e})")

    with open(o.out, "w") as f:
        json.dump({"stem": o.stem, "ln_dir": o.ln_dir, "rows": o.rows, "cols": COLS,
                   "reps": o.reps, "results": results}, f, indent=2)
    print(f"\nwrote {o.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ln-dir", default="/mnt/data/models/xdna-artifacts/parakeet/ln",
                   help="the SHIPPED artifact dir -- price what the encoder loads, not a rebuild")
    p.add_argument("--stem", default=f"lnaffcast_{ROWS}x{COLS}")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--rows", type=int, default=ROWS)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--rel-l2-gate", type=float, default=1e-4)
    p.add_argument("--out", default="artifacts/lnaffcast_standalone.json")
    sys.exit(main(p.parse_args()))
