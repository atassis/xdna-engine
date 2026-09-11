#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TIER 1 of the rail's correctness gate: per-kernel/per-block numerics against a HIGH-PRECISION
reference, element-wise, over the full output.

WHY THIS EXISTS. The rail used to gate a new decode/prefill path on "produces byte-identical tokens
to the existing path". That gate was written when prefill WAS the M=1 decode path, so identity was
free. It stopped being satisfiable the moment the rail gained a second implementation of an op:
batched prefill projects through `iron/operators/gemm` (`aie_kernels/aie2p/mm.cc`) and decode
through `iron/operators/gemv` (`mv.cc`) -- two kernels, written separately, with independently
chosen operand formats, reduction widths and rounding modes. Measured, device against device, the
best achievable layer-0 KV agreement is 1.18 bf16 ULP on V and 2.66 on K. A wiring defect gives
zero or a permutation; it does not give 1.18 ULP. Identity was never the standard being failed.

WHAT REPLACES IT, and it is the ordinary standard in this domain rather than anything of ours:

  * rtol = 1.6e-2 is PyTorch/vLLM's canonical bf16 tolerance. It is the relative term, and it is
    what an element of typical magnitude is judged against.
  * atol is the ABSOLUTE floor, for elements small enough that `rtol*|ref|` vanishes. Those
    elements are not accurate to their own magnitude: they fell out of a reduction over terms of
    the tensor's typical magnitude, so what they are accurate to is one quantum at THAT scale.
    Hence the rule below -- `2**-8 * rms(ref)`, one bf16 quantum at the tensor's RMS magnitude --
    and it is computed BY THE GENERATOR, from the golden it just wrote, and carried in the
    artifact's own meta.json. Declared where the shape is picked (kernel-contract K007's habit),
    not passed on a command line where nothing records what it was.

THE REFERENCE MUST NOT BE bf16. The generators' original goldens round every intermediate to bf16
exactly where the device rounds, which makes them a model of the device rather than a measurement
of it: a device and a bf16 golden agree on precisely the errors this gate exists to catch. So each
generator now also emits `buffers/golden_f32/<tensor>.bin` -- the same dataflow on the same bf16
INPUTS the device was handed, with every intermediate kept in float32. The bf16 goldens are kept:
they are what the device-side rel-L2 probes still compare against, and the gap between the two is
itself the datapath's rounding cost.

SCOPE. This is a PER-KERNEL / PER-BLOCK gate. bf16 rounding compounds down a deep stack, so a
28-layer `xout` is not a Tier 1 subject -- its layer-0 KV slabs are, and the end-to-end question is
Tier 2's (top-k token-set inclusion, scripts/gate_token_set.py).

  # judge a dump that already exists (no device):
  python3 scripts/gate_numeric.py /mnt/data/xdna/scratch/prefill/mlp_m256 --dump <dumpdir>

  # the device half that produces <dumpdir> is a separate command -- see scripts/gate_llm.sh
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:                                    # numpy-only consumers (the unit tests)
    BF16 = None

# PyTorch/vLLM's canonical bf16 tolerance. Fixed here rather than per artifact: it is a property of
# the DTYPE, not of a shape, and letting each artifact carry its own would let a build lower the bar
# on itself. An artifact that records a different rtol is stale and is rejected, not honoured.
RTOL = 1.6e-2

# How much worse than a FAITHFUL bf16 host implementation of the same dataflow the device is
# allowed to be. Sized from a measurement on this rail, not picked: our GEMM's default operand
# format is bfp16 block float (one shared exponent per 8 elements), and against a float64 reference
# on the same bf16 inputs that costs 1.219e-2 where plain bf16 costs 4.332e-3 -- 2.8x. A margin of
# 4 admits that format with headroom and rejects anything materially worse. It cannot distinguish
# "narrower format than we thought" from "defect"; it is not supposed to. It fails and you look.
ATOL_MARGIN = 4.0

ATOL_RULE = ("ATOL_MARGIN * max(|bf16_golden - f32_ref| - rtol*|f32_ref|, 0): the widest "
             "element-wise miss a faithful bf16 HOST implementation of this exact dataflow already "
             "shows against the high-precision reference, times the margin. Derived per artifact "
             "because it is a property of the dataflow's cancellation, not of the dtype: a "
             "single-GEMM artifact gets a far tighter floor than a six-op block, and should.")


def rms(a):
    """Root-mean-square magnitude, in float64 so a large bf16 tensor cannot overflow the sum."""
    f = np.asarray(a, np.float64).ravel()
    return float(np.sqrt((f * f).mean())) if f.size else 0.0


def bf16_floor(ref, golden_bf16, rtol=RTOL):
    """What atol must still cover after `rtol*|ref|` has taken its share, at the bf16 floor.

    `golden_bf16` is the SAME dataflow narrowed to bf16 at every step the device narrows. It is not
    the device and it is not the truth: it is the best a bf16 datapath can do, so the residual it
    leaves is a floor no implementation beats, and an atol below it would fail a perfect device.
    """
    r = np.asarray(ref, np.float64).ravel()
    g = np.asarray(golden_bf16, np.float64).ravel()
    if g.shape != r.shape:
        raise ValueError(f"floor shape {g.shape} != reference shape {r.shape}")
    return float(np.maximum(np.abs(g - r) - rtol * np.abs(r), 0.0).max()) if r.size else 0.0


def atol_for(ref, golden_bf16, margin=ATOL_MARGIN):
    """The per-tensor absolute floor. Generators call this and write the result into meta.json."""
    return margin * bf16_floor(ref, golden_bf16)


def mean_rel_l1(got, ref):
    """mean(|got-ref|) / mean(|ref|).

    A whole-tensor relative error that, unlike rel-L2, is not dominated by the few largest
    elements -- which is why it is the number the reference implementations in this domain quote.
    Reported, never gated on: the gate is the element-wise isclose below.
    """
    g = np.asarray(got, np.float64).ravel()
    r = np.asarray(ref, np.float64).ravel()
    den = float(np.abs(r).mean())
    return float(np.abs(g - r).mean() / den) if den > 0 else float(np.abs(g - r).mean())


def check(got, ref, atol, rtol=RTOL):
    """The gate: `|got - ref| <= atol + rtol*|ref|` on EVERY element of the full output.

    Full output, not a sample and not a summary statistic: a single structurally wrong element --
    a transposed tile, a stale row, an off-by-one in a slab offset -- moves no aggregate metric
    far enough to fail, and is exactly the defect class this rail keeps hitting.
    """
    g = np.asarray(got, np.float64).ravel()
    r = np.asarray(ref, np.float64).ravel()
    if g.shape != r.shape:
        raise ValueError(f"shape mismatch: device {g.shape} vs reference {r.shape}")
    err = np.abs(g - r)
    tol = atol + rtol * np.abs(r)
    bad = err > tol
    n_bad = int(bad.sum())
    worst = int(np.argmax(err - tol)) if g.size else 0
    return {
        "n": int(g.size),
        "n_bad": n_bad,
        "frac_bad": n_bad / g.size if g.size else 0.0,
        "mean_rel_L1": mean_rel_l1(g, r),
        "rtol": rtol,
        "atol": float(atol),
        "max_abs_err": float(err.max()) if g.size else 0.0,
        "worst_index": worst,
        "worst_got": float(g[worst]) if g.size else 0.0,
        "worst_ref": float(r[worst]) if g.size else 0.0,
        "pass": n_bad == 0,
    }


# ------------------------------------------------------------------------------------------------
# Artifact plumbing. The gate block an artifact must carry, and the loud failures when it does not.
# ------------------------------------------------------------------------------------------------
_NP = {"float32": np.float32, "float64": np.float64}


def _dtype(name):
    if name in _NP:
        return _NP[name]
    if name in ("bfloat16", "bf16"):
        if BF16 is None:
            raise SystemExit("ERROR: ml_dtypes is not importable, so bf16 buffers cannot be read")
        return BF16
    raise SystemExit(f"ERROR: unknown dtype '{name}' in the artifact's gate block")


def _read(path, dtype, n_expected=None):
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: missing {path}")
    a = np.fromfile(path, dtype=dtype)
    if n_expected is not None and a.size != n_expected:
        raise SystemExit(f"ERROR: {path} holds {a.size} elements, the gate block declares "
                         f"{n_expected}; the dump and the artifact are not the same build")
    return np.asarray(a, np.float64)


def load_gate(art_dir):
    """The artifact's gate block, or a loud failure naming the fix.

    A MISSING golden must never read as a pass. An artifact built before this gate existed has no
    `gate` block at all, and the only correct response to that is to refuse to run.
    """
    meta_path = os.path.join(art_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"ERROR: no meta.json in {art_dir}")
    meta = json.load(open(meta_path))
    gate = meta.get("gate")
    if not gate or not gate.get("tensors"):
        raise SystemExit(
            f"ERROR: {meta_path} carries no `gate` block, so this artifact has no high-precision "
            f"reference and CANNOT be gated. Write one -- no aiecc, no device:\n"
            f"  bash scripts/gate_llm.sh --refresh-goldens {art_dir}")
    if abs(gate.get("rtol", RTOL) - RTOL) > 1e-12:
        raise SystemExit(f"ERROR: {meta_path} declares rtol={gate['rtol']}, this gate is {RTOL}. "
                         f"The artifact predates the current tolerance -- regenerate its goldens.")
    return meta, gate


def gate_artifact(art_dir, dump_dir, tensors=None, atol_override=None):
    """Run the Tier 1 check for every declared tensor. Returns [(name, result), ...]."""
    meta, gate = load_gate(art_dir)
    ref_dt = _dtype(gate.get("ref_dtype", "float32"))
    dev_dt = _dtype(gate.get("device_dtype", "bfloat16"))
    want = tensors or list(gate["tensors"])
    unknown = [t for t in want if t not in gate["tensors"]]
    if unknown:
        raise SystemExit(f"ERROR: {unknown} not declared in {art_dir}/meta.json gate.tensors "
                         f"(declared: {sorted(gate['tensors'])})")
    out = []
    for name in want:
        spec = gate["tensors"][name]
        n = int(np.prod(spec["shape"]))
        ref = _read(os.path.join(art_dir, spec["ref"]), ref_dt, n)
        got = _read(os.path.join(dump_dir, f"{name}.bin"), dev_dt, n)
        atol = float(spec["atol"]) if atol_override is None else float(atol_override)
        r = check(got, ref, atol)
        # The device's error over the bf16 floor's. NOT gated on -- our own doctrine keeps error
        # metrics as notes -- but it is the sensitive number: a datapath 2.6x less accurate than
        # another can sit well inside an element-wise tolerance and still show up here.
        r["floor_mean_rel_L1"] = float(spec.get("bf16_floor_mean_rel_L1", 0.0))
        r["vs_floor"] = (r["mean_rel_L1"] / r["floor_mean_rel_L1"]
                         if r["floor_mean_rel_L1"] > 0 else float("nan"))
        out.append((name, r))
    return out


def report(art_dir, results, atol_override=None, margin=None):
    """One line per tensor plus a verdict. Returns True if every tensor passed."""
    print(f"[tier1] {art_dir}"
          + (f"  (atol = {margin:g}x the measured bf16 floor)" if margin else ""))
    if atol_override is not None:
        print(f"[tier1] atol OVERRIDDEN to {atol_override:g} -- the artifact's own value is ignored")
    print(f"[tier1] {'tensor':<12} {'mean_rel_L1':>12} {'x bf16 floor':>13} {'rtol':>9} "
          f"{'atol':>11} {'bad/total':>18}  verdict")
    ok = True
    for name, r in results:
        ok &= r["pass"]
        print(f"[tier1] {name:<12} {r['mean_rel_L1']:>12.4e} {r['vs_floor']:>13.2f} "
              f"{r['rtol']:>9.3e} {r['atol']:>11.4e} {r['n_bad']:>8}/{r['n']:<9}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
        if not r["pass"]:
            print(f"[tier1]   worst element [{r['worst_index']}]: device {r['worst_got']:.6g} "
                  f"vs reference {r['worst_ref']:.6g}, |err| {r['max_abs_err']:.4e}")
    print("[tier1] 'x bf16 floor' is a NOTE, never the gate: how much worse than a faithful bf16 "
          "host\n[tier1] implementation of the same dataflow this datapath is. 1.0 is host "
          "quality.")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artifact", nargs="+", help="artifact dir(s) holding meta.json + goldens")
    ap.add_argument("--dump", required=True,
                    help="dir of device outputs, <tensor>.bin, written by the probe's "
                         "GATE_DUMP_DIR. One per artifact, in the same order, or one for all.")
    ap.add_argument("--tensors", default=None, help="comma-separated subset to check")
    ap.add_argument("--atol", type=float, default=None,
                    help="override every declared atol -- for sizing experiments, not for gating")
    ap.add_argument("--json", default=None, help="write the full result table here")
    a = ap.parse_args()

    dumps = a.dump.split(",")
    if len(dumps) not in (1, len(a.artifact)):
        raise SystemExit(f"ERROR: {len(dumps)} --dump dirs for {len(a.artifact)} artifacts")
    tensors = a.tensors.split(",") if a.tensors else None
    ok, blob = True, {}
    for i, art in enumerate(a.artifact):
        dump = dumps[i] if len(dumps) > 1 else dumps[0]
        res = gate_artifact(art, dump, tensors, a.atol)
        ok &= report(art, res, a.atol, load_gate(art)[1].get("atol_margin"))
        blob[art] = {n: r for n, r in res}
        print()
    if a.json:
        json.dump(blob, open(a.json, "w"), indent=1)
    print(f"[tier1] *** {'PASS' if ok else 'FAIL'} *** "
          f"({len(a.artifact)} artifact(s), rtol={RTOL:g}, atol per artifact)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
