#!/usr/bin/env python3
# Is the lnaffcast mode slow because its shim taps burst at 16 B?
# (task mode-switched-multi-program-xclbin -- the isolating control its `next:` prescribed)
#
# WHAT THE CLAIM RESTS ON TODAY. The mode runs correct and slow: 2422 us for 256 rows of an
# elementwise op at group-rounds 4, against ~440 us for a whole 512x1024x1024 GEMM on the SAME
# xclbin in the same session. The emitted BDs say all three mode taps walk 8 contiguous bf16 and
# then jump 64, against the GEMM's 32 and 128 -- 16 B against 64/256 B -- and 4x is also the ratio
# of the wall clocks per byte. That is a hypothesis the numbers are CONSISTENT with, not a
# measurement of it: the two streams differ in more than burst length (different core body,
# different fill contract, different byte count), so nothing in that comparison isolates the burst.
# See lnaffcast-mode-taps-burst-at-16-bytes.
#
# THE CONTROL. `--lnaffcast-contig-taps c` re-strides the C tap to the contiguous walk for its own
# sizes and changes nothing else: same rank, same sizes, same offset, same span, same BD count,
# same core work, same byte count. Only the innermost contiguous run moves, 8 -> 16384 elements
# (16 B -> 32 KB). `abc` does the same to all three taps. Verified in the generated MLIR: every
# differing line between a control and the baseline is an `aie.dma_bd`, and only its strides.
#
# The controls read out GARBAGE by construction -- the real strides ARE the un-permute, so undoing
# them scrambles the output. That is why parity is reported here but gated only on the BASELINE:
# a control that passed parity would mean the re-stride did not take.
#
# ONE HARDWARE CONTEXT, THREE STREAMS. The array program is byte-identical across the arms (the
# taps live in the runtime sequence), so all three insts run on the baseline's loaded xclbin. That
# deletes the program-transition tax (~1.543 ms) from the comparison outright rather than
# subtracting it, which matters because the effect under test is of the same order.
#
# ORDER IS AN INDEPENDENT VARIABLE on this vehicle -- measured on this task, a stream's result
# depends on what ran before it -- so the whole arm sweep runs FORWARD and then REVERSED, and a
# verdict needs both blocks to agree.
#
# Run (NPU free -- use the device wrapper, which stops xdna-engine and npu-vox):
#   bash scripts/lnaffcast_burst_isolation_device.sh
import argparse
import json
import os
import re
import statistics
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemm_command_accounting import design_operands, insts_bin

import pyxrt
from aie.utils.npukernel import NPUKernel
from aie.utils.tensor_factory import tensor
import aie.utils as aie_utils

# The shipped resident's own shape, which is what the taps are derived for -- the generator
# refuses any other, so these are read-only here.
M, K, N = 512, 1024, 1024
COLS = 1024      # --mode-lnaffcast: the lnaffcast embedding dim
ROWS = 256       # x rows one dispatch covers, the default cap for these tensors
EPS = 1e-5       # mm_mode_lnaffcast.cc's epsilon, and ln_affine_cast.cc's before it

HOST = "512x1024x1024_32x32x128_8c_modalidbf16outkrtpkrllnaff1024rtp18g4"
ARMS = [HOST, HOST + "ctgc", HOST + "ctgabc"]

BD_RE = re.compile(
    r"aie\.dma_bd\((%\w+) : memref<\d+x\w+> offset = \d+ len = \d+ "
    r"sizes = \[([\d, ]+)\] strides = \[([\d, ]+)\]\)")


def innermost_run(sizes, strides):
    """Elements a BD walks contiguously before its first non-unit jump, folding trailing dims.

    This is the independent variable, so it is READ OFF the emitted MLIR rather than taken from
    the flag that was meant to set it -- a flag that silently did nothing would otherwise report
    as "burst length does not matter".
    """
    run = expect = 1
    for s, d in zip(reversed(sizes), reversed(strides)):
        if d != expect:
            break
        run *= s
        expect = run
    return run


def tap_bursts(build_dir, suffix):
    """{operand: (sizes, strides, innermost run)} for each distinct BD walk in a design."""
    path = os.path.join(build_dir, f"aie_{suffix}.mlir")
    if not os.path.exists(path):
        sys.exit(f"missing generated MLIR: {path}")
    out = {}
    for m in BD_RE.finditer(open(path).read()):
        sizes = [int(x) for x in m.group(2).split(",")]
        strides = [int(x) for x in m.group(3).split(",")]
        out.setdefault(m.group(1), (sizes, strides, innermost_run(sizes, strides)))
    return out


def operands(x):
    """A, B, C as the mode reads them, plus the host reference for y.

    x rides the A tensor as f32-in-bf16 and gb = [gamma | beta] rides B the same way -- both are
    the GEMM's own bf16 operand buffers, reinterpreted by the mode body. C is poisoned with NaN so
    a dispatch that wrote nothing reads as nan rather than as the previous arm's output.
    """
    rng = np.random.default_rng(seed=7)
    gamma = rng.normal(1.0, 0.1, COLS).astype(np.float32)
    beta = rng.normal(0.0, 0.1, COLS).astype(np.float32)

    a = np.zeros(M * K, dtype=bfloat16)
    a.view(np.float32)[: ROWS * COLS] = x.reshape(-1)
    b = np.zeros(K * N, dtype=bfloat16)
    b.view(np.float32)[:COLS] = gamma
    b.view(np.float32)[COLS: 2 * COLS] = beta
    c = np.full(M * N, np.nan, dtype=bfloat16)

    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    ref = ((x - mean) / np.sqrt(var + EPS)) * gamma + beta
    return a, b, c, ref.astype(bfloat16).astype(np.float64)


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

    xclbin = os.path.join(o.build_dir, f"final_{o.host}.xclbin")
    if not os.path.exists(xclbin):
        sys.exit(f"missing host xclbin: {xclbin}")
    kern = NPUKernel(xclbin_path=xclbin,
                     insts_path=insts_bin(os.path.join(o.build_dir, f"insts_{o.host}.txt"),
                                          o.artifacts),
                     kernel_name=o.kernel)
    kh = rt.load(kern)
    while not hasattr(kh, "kernel"):
        kh = kh._handle

    # Every arm's own instruction BO, bound to the ONE loaded kernel above.
    streams, bursts = {}, {}
    for suffix in o.arms:
        insts = os.path.join(o.build_dir, f"insts_{suffix}.txt")
        if not os.path.exists(insts):
            sys.exit(f"missing insts: {insts}")
        if len(design_operands(o.build_dir, suffix)) > 3:
            sys.exit(f"{suffix}: traced design -- this probe runs no-trace arms only")
        arr = np.frombuffer(open(insts, "rb").read(), dtype=np.uint32)
        streams[suffix] = (arr, rt._tensor_class(arr, flags=pyxrt.bo.cacheable,
                                                 group_id=kh.kernel.group_id(1),
                                                 xrt_device=rt._device).buffer_object())
        bursts[suffix] = tap_bursts(o.build_dir, suffix)

    print("--- taps, read off the emitted BDs (the independent variable) ---")
    for suffix in o.arms:
        runs = " ".join(f"{a}:{r}el({r * 2}B)" for a, (_s, _st, r) in sorted(bursts[suffix].items()))
        print(f"  {suffix.replace(o.host, '<host>'):<22s} {runs}")

    x = np.random.default_rng(seed=11).standard_normal((ROWS, COLS)).astype(np.float32)
    a, b, c, ref = operands(x)

    results = {s: {"blocks": {}, "taps": {k: {"sizes": v[0], "strides": v[1], "innermost_run": v[2]}
                                          for k, v in bursts[s].items()}} for s in o.arms}
    blocks = [("forward", list(o.arms)), ("reversed", list(reversed(o.arms)))]
    for bname, order in blocks:
        print(f"\n--- block '{bname}': {o.reps} reps per arm, rep 0 dropped as cold ---")
        for suffix in order:
            arr, ibo = streams[suffix]
            At, Bt = tensor(a.copy(), dtype=bfloat16), tensor(b.copy(), dtype=bfloat16)
            Ct = tensor(c.copy(), dtype=bfloat16)
            for t in (At, Bt, Ct):
                t.to("npu")
            us = []
            for _ in range(o.reps):
                t0 = time.perf_counter_ns()
                kh.kernel(3, ibo, arr.nbytes, At.buffer_object(), Bt.buffer_object(),
                          Ct.buffer_object()).wait()
                us.append((time.perf_counter_ns() - t0) / 1000.0)
            Ct.to("cpu")
            got = np.asarray(Ct)[: ROWS * COLS].reshape(ROWS, COLS).astype(np.float64)
            steady = us[1:] or us
            rec = {"us": us, "median_us": statistics.median(steady), "parity": parity(got, ref)}
            results[suffix]["blocks"][bname] = rec
            p = rec["parity"]
            print(f"  {suffix.replace(o.host, '<host>'):<22s} median {rec['median_us']:>8.1f} us  "
                  f"reps {' '.join(f'{v:.0f}' for v in us)}  "
                  f"rel-L2 {p['rel_l2']:>10.4e}  exact {p['exact']}/{p['elements']}"
                  f"{'  NAN' if p['nan_out'] else ''}", flush=True)

    print("\n--- verdict ---")
    base = results[o.host]
    base_med = {bn: base["blocks"][bn]["median_us"] for bn, _ in blocks}
    for suffix in o.arms:
        r = results[suffix]
        run = max(v[2] for v in bursts[suffix].values())
        line = []
        for bn, _ in blocks:
            med = r["blocks"][bn]["median_us"]
            line.append(f"{bn} {med:>8.1f} us ({med / base_med[bn] - 1:+6.1%})")
        spread = [r["blocks"][bn]["median_us"] for bn, _ in blocks]
        agree = abs(spread[0] - spread[1]) / statistics.mean(spread)
        print(f"  {suffix.replace(o.host, '<host>'):<22s} max innermost run {run:>6d} el  "
              f"{'  '.join(line)}   block spread {agree:.1%}")

    bp = base["blocks"]["forward"]["parity"]
    ok = bp["rel_l2"] < o.rel_l2_gate and not bp["nan_out"]
    print(f"\n  baseline parity {'PASS' if ok else 'FAIL'} "
          f"(rel-L2 {bp['rel_l2']:.4e} vs gate {o.rel_l2_gate:.1e})")
    for suffix in o.arms[1:]:
        cp = results[suffix]["blocks"]["forward"]["parity"]
        if cp["rel_l2"] < o.rel_l2_gate:
            print(f"  WARNING {suffix}: control PASSES parity -- the re-stride did not take, "
                  f"so its timing is not a control")
            ok = False

    with open(o.out, "w") as f:
        json.dump({"host": o.host, "shape": [M, K, N], "cols": COLS, "rows": ROWS,
                   "reps": o.reps, "results": results}, f, indent=2)
    print(f"\nwrote {o.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="mlir-aie/programming_examples/basic/"
                                          "matrix_multiplication/whole_array/build")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--kernel", default="MLIR_AIE")
    p.add_argument("--host", default=HOST, help="xclbin every arm's stream runs on")
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--rel-l2-gate", type=float, default=1e-4,
                   help="baseline gate; the mode measured 5.7361e-06 on first dispatch")
    p.add_argument("--out", default="artifacts/lnaffcast_burst_isolation.json")
    sys.exit(main(p.parse_args()))
