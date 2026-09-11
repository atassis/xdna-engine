"""Run a precision PLAN through the host lab, as one paired arm against a bf16 control.

The point is that one plan drives both sides: `gen_llm_decode.py` builds the artifact from it and
this measures what it costs the model, so the arm whose perplexity was measured is the arm that
got built. Without that, a format sweep and a build are two hand-kept lists that drift.

WHAT THIS CAN AND CANNOT SEE -- read before quoting a verdict.

CAN, and this is the dominant term for a weight format: the information the format destroys in
the weights. A format only changes which numbers the weights hold, so a CPU forward pass with
those numbers sees exactly that loss. The instrument is anchored -- validate_formats.py holds the
symmetric path bit-identical to the shipped packer, and the lab reproduces both recorded device
deltas inside their CIs (MLP int4/g128 +11.24% against the device's +11.51%; lm-head +2.50%
against +2.63%).

CANNOT:
  * the `kv` site, AT ALL. A quantized KV cache perturbs an ACTIVATION written and re-read at
    run time; there is no weight to round here, so an arm that quantizes kv is measured as if it
    did not. `coverage()` reports this rather than letting the number be read as whole-plan.
  * the kernel's own arithmetic. `kernel_round` emulates the bf16 narrowing mv_quant.cc does per
    product and nothing else -- not accumulation order, not the rounding MODE the core register
    happens to hold (the documented default is floor, and it cost 1.3x accuracy on a real
    encoder, invisible to every gate in this tree).
  * trajectory drift. Every position is scored on the TRUE prefix, so compounding is invisible by
    construction. divergence.py is the instrument
    for that question and it is a different run.
  * latency, in either direction. A byte cut is not a time cut until a device A/B says so, and the
    standing objection to int8 is precisely that its dequant has no zero-overhead loop.

SENSITIVITY IS REPORTED, NOT ASSUMED. The verdict carries the smallest perplexity change this run
could have resolved, computed from the realized spread of the paired differences. A "no
significant damage" at 200 positions and one at 6000 are different claims and must not read alike.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import precision as P                                            # noqa: E402
import wq_formats as F                                           # noqa: E402
from wq_eval import TARGETS, load_model, baseline_bf16, paired, run, tokenize   # noqa: E402

QLAB = os.environ.get("QLAB_WORK", "/mnt/data/xdna/qlab")

# The plane's scale_kind vocabulary against wq_formats' spec fields. Symmetric `clip` is a
# host-side scale SEARCH the lab does not implement, so it is reported as uncovered rather than
# silently measured as absmax.
_SIM = {"absmax": {}, "zero_grid": {"zero_on_grid": True}, "free_min": {"zero_on_grid": False}}


def spec_to_sim(spec):
    """One plane Spec -> a wq_formats spec dict, or None if the lab cannot simulate it."""
    if not spec.quantized:
        return {"scheme": "bf16"}
    if spec.scale_kind not in _SIM:
        return None
    nbits = 4 if spec.dtype in ("int4", "int4a") else 8
    sim = {"scheme": "affine" if spec.affine else "sym", "nbits": nbits,
           "group": spec.group_size, "kernel_round": True}
    # The symmetric wire header is an f32 scale; the affine one is a bf16 scale and a bf16 min.
    sim["scale_dtype"] = "bf16" if spec.affine else "f32"
    sim.update(_SIM[spec.scale_kind])
    return sim


def coverage(plan):
    """Per site: (covered, why-not). A plan is only as measurable as its least measurable site."""
    out = {}
    for key, site in P.SITES.items():
        spec = plan.get(key, P.BF16_SPEC)
        if not spec.quantized:
            out[key] = (True, "bf16 -- nothing to measure")
        elif site.hostlab_class is None:
            out[key] = (False, f"{site.kind}: quantized at run time, not a weight this lab can "
                               "round")
        elif spec_to_sim(spec) is None:
            out[key] = (False, f"scale_kind={spec.scale_kind!r} is not simulated here")
        else:
            out[key] = (True, str(spec))
    return out


def uncovered_mb(plan):
    """MB/token the verdict does NOT account for."""
    return sum(P.SITES[k].mb_per_token for k, (ok, _) in coverage(plan).items() if not ok)


def sensitivity(d_nats, power=0.80, alpha=0.05):
    """Smallest perplexity change this run could have resolved, as a percentage.

    Computed from the REALIZED spread of the paired per-position differences, so it describes the
    run that happened rather than a planning assumption. Two-sided alpha, normal approximation
    (n is in the thousands): MDE = (z_a/2 + z_power) * sd / sqrt(n), read as a perplexity ratio.
    """
    n = len(d_nats)
    sd = float(np.std(d_nats, ddof=1))
    mde_nats = (1.959964 + 0.841621) * sd / np.sqrt(n)
    return {"n": n, "sd_nats": sd, "mde_nats": float(mde_nats),
            "mde_ppl_pct": float((np.exp(mde_nats) - 1) * 100),
            "power": power, "alpha": alpha}


def apply_plan(model, plan, pristine, touched_all):
    """Restore every touched weight, then quantize per site. Returns (n_params, mean_bits)."""
    for name, mod in touched_all:
        mod.weight.data = torch.from_numpy(pristine[name].astype(np.float32))
    npar = qbits = 0
    for key, (covered, _) in coverage(plan).items():
        spec = plan.get(key, P.BF16_SPEC)
        if not (covered and spec.quantized):
            continue
        sim = spec_to_sim(spec)
        bits = F.bits_per_element(sim["nbits"], sim["group"], sim["scheme"],
                                  sim.get("scale_dtype", "bf16"), "bf16")
        for name, mod in touched(model, [P.SITES[key].hostlab_class]):
            W = mod.weight.detach().numpy()
            mod.weight.data = torch.from_numpy(F.apply(W, sim))
            npar += W.size
            qbits += W.size * bits
    return npar, (qbits / npar if npar else 16.0)


def touched(model, classes):
    sfx = tuple(s for c in classes if c != "head" for s in TARGETS[c])
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(name.endswith(s) for s in sfx) or (name == "lm_head" and "head" in classes):
            yield name, mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="a preset name, JSON, or a path -- the SAME "
                                                  "value PRECISION takes")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--max-ppl-pct", type=float, default=None,
                    help="fail if the paired perplexity delta's LOWER CI bound exceeds this")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("QLAB_THREADS", "10")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    plan = (P.parse_plan(json.dumps(P.PRESETS[a.plan][0])) if a.plan in P.PRESETS
            else P.parse_plan(a.plan))
    cov = coverage(plan)
    print(P.describe(plan))
    print("\ncoverage of this eval:")
    for key, (ok, why) in sorted(cov.items(), key=lambda kv: -P.SITES[kv[0]].mb_per_token):
        print(f"  {'yes' if ok else 'NO ':3}  {key:7} {P.SITES[key].mb_per_token:8.2f} MB  {why}")
    un = uncovered_mb(plan)
    if un:
        print(f"  -> {un:.2f} MB/token ({100 * un / P.CENSUS_TOKEN_MB:.1f}%) is NOT in the "
              "verdict below")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    ids = tokenize(a.corpus, a.tokens, tok)

    t0 = time.time()
    model = load_model()
    baseline_bf16(model)
    all_cls = [s.hostlab_class for s in P.SITES.values() if s.hostlab_class]
    touched_all = list(touched(model, all_cls))
    pristine = {n: m.weight.detach().numpy().copy() for n, m in touched_all}

    r0 = run(model, ids)
    npar, bits = apply_plan(model, plan, pristine, touched_all)
    r1 = run(model, ids)

    pw = paired(r1["nll"], r0["nll"])
    sens = sensitivity(r1["nll"] - r0["nll"])
    res = {
        "plan": {k: str(v) for k, v in sorted(plan.items())},
        "corpus": os.path.basename(a.corpus), "tokens": a.tokens,
        "params_quantized": int(npar), "mean_bits": round(bits, 3),
        "projected_mb_per_token": round(P.token_mb(plan)["total"], 2),
        "uncovered_mb_per_token": round(un, 2),
        "control_ppl": float(np.exp(r0["nll"].mean())),
        "arm_ppl": float(np.exp(r1["nll"].mean())),
        "top1_control": float((r0["top1"] == r0["tgt"]).mean()),
        "top1_arm": float((r1["top1"] == r1["tgt"]).mean()),
        "top1_agree": float((r0["top1"] == r1["top1"]).mean()),
        "paired": pw, "sensitivity": sens, "secs": round(time.time() - t0, 1),
    }
    lo, hi = pw["ppl_pct_ci"]
    print(f"\nperplexity {res['control_ppl']:.4f} -> {res['arm_ppl']:.4f}  "
          f"{pw['ppl_pct']:+.2f}% [{lo:+.2f}, {hi:+.2f}] t={pw['t']:.2f}  "
          f"top1 agree {res['top1_agree']:.1%}")
    print(f"sensitivity: this run resolves a {sens['mde_ppl_pct']:.2f}% change at "
          f"{int(sens['power'] * 100)}% power (n={sens['n']}, sd={sens['sd_nats']:.4f} nats). "
          f"A null here means 'smaller than that', never 'zero'.")

    verdict = "PASS"
    if a.max_ppl_pct is not None:
        # Gate on the CI's lower bound: a wide interval whose bottom clears the bar has not
        # shown the arm is under it, and the sensitivity line says how wide is wide.
        if lo > a.max_ppl_pct:
            verdict = "FAIL"
        print(f"gate: lower CI bound {lo:+.2f}% against a {a.max_ppl_pct:+.2f}% bar -> {verdict}")
    res["verdict"] = verdict
    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        json.dump(res, open(a.out_json, "w"), indent=1)
        print(f"wrote {a.out_json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
