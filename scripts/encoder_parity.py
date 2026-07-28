#!/usr/bin/env python3
"""Numerical-parity gate for device encoder changes (replaces the chaotic 17-clip greedy WER for
validating numerically-equivalent changes).

The 17-clip greedy-decode WER is CHAOTIC at ~1e-5: perturbing the SHIPPED encoder by a meaningless
+-1e-5 per-element noise swings its WER across an ~8.2-9.2 band. So the greedy WER on a small clip set
CANNOT tell a correct device change from the shipped path. This gate instead measures each path's
distance from the TRUE f32 encoder and asks: is the candidate any further from truth than the shipped
baseline?

WHY THIS IS NOT JUST A MEAN. A whole-clip mean rel-L2 structurally cannot see a LOCALIZED error
burst. The shipped encoder carries contiguous runs of 1-5 frames at 15-79% per-frame relative error
on 16 of 17 clips while its whole-clip mean sits at 0.089 -- the mean averages them into invisibility.
Those bursts are also the part that MOVES under any perturbation, so a mean-only gate both hides a
standing defect and misprices every change that reshuffles it. This gate therefore reports and gates
on three statistics:

  mean         whole-clip Frobenius rel-L2 vs f32 truth, averaged over clips  (the old gate)
  worst-frame  max over all frames of per-frame rel-L2                        (a 1-frame spike)
  worst-burst  max over all windows of the mean per-frame rel-L2 in a         (a contiguous run)
               contiguous window of --burst-window frames (default 5)
  new-burst    max over all frames of (candidate - baseline) per-frame        (manufactured damage)
               rel-L2 -- i.e. how much error the worst-hurt frame GAINED

The first three are gated relative to the baseline, as the old mean was: PASS iff within
baseline * (1 + margin). `new-burst` is a DELTA, so it is gated against an absolute policy
threshold (--max-new-burst) instead. It exists because the relative checks have a ceiling problem:
the shipped baseline's worst frame is already 0.79, so a change could take a clean 0.06 frame to
0.78 and still be "within 15% of the worst frame". The delta check is what actually catches a
manufactured burst -- and it does: both encoder changes measured on 2026-07-28 take ru_02[77] from
0.068 to 0.49 and 0.67 respectively, a frame that is clean in the shipped path.

A change PASSES only if ALL FOUR pass. Any one failing is reported by name, with the clip and frame
index responsible, so a regression is localizable rather than merely detected.

Usage:
  # 1. capture the three encode dirs (17-clip mels):
  cargo run -p npu-probes --release --bin parakeet_encode_npu -- artifacts/wer_mels /tmp/enc_ref --cpu
  NPU_XCLBIN_ROOT=$PWD cargo run ... --bin parakeet_encode_npu -- artifacts/wer_mels /tmp/enc_ship
  PARAKEET_FUSED_BLOCK=1 NPU_XCLBIN_ROOT=$PWD cargo run ... -- artifacts/wer_mels /tmp/enc_cand
  # 2. gate:
  scripts/encoder_parity.py /tmp/enc_ref /tmp/enc_ship /tmp/enc_cand [--margin 0.15]

  # baseline-only survey (no candidate): what does this path's error profile look like?
  scripts/encoder_parity.py /tmp/enc_ref /tmp/enc_ship --survey

Self-test (no device, proves the gate can see what the mean cannot):
  scripts/tests/encoder_parity_sees_bursts_test.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def frame_rel_err(ref, x):
    """Per-frame relative L2 error. ref/x are [T, D] (leading singleton axes squeezed)."""
    ref = np.squeeze(ref)
    x = np.squeeze(x)
    if ref.ndim != 2 or x.ndim != 2:
        raise ValueError(f"expected [T, D] after squeeze, got {ref.shape} and {x.shape}")
    if ref.shape != x.shape:
        raise ValueError(f"shape mismatch: ref {ref.shape} vs {x.shape}")
    num = np.linalg.norm(ref - x, axis=-1)
    den = np.maximum(np.linalg.norm(ref, axis=-1), 1e-12)
    return (num / den).astype(np.float64)


def clip_rel_l2(ref, x):
    """Whole-clip Frobenius relative L2 -- the statistic the old gate used."""
    ref = np.squeeze(ref)
    x = np.squeeze(x)
    return float(np.linalg.norm(ref - x) / max(np.linalg.norm(ref), 1e-12))


def worst_window(e, w):
    """(value, start_index) of the contiguous w-frame window with the highest mean error.

    Clips shorter than w are scored on their whole length, so a short clip is never given an
    artificially low burst score by having no full window.
    """
    if len(e) == 0:
        return 0.0, 0
    w = min(w, len(e))
    csum = np.concatenate([[0.0], np.cumsum(e)])
    means = (csum[w:] - csum[:-w]) / w
    i = int(np.argmax(means))
    return float(means[i]), i


def profile(ref_dir, path_dir, names, burst_window):
    """Per-clip statistics of one path against f32 truth."""
    rows = []
    for n in names:
        ref = np.load(os.path.join(ref_dir, n))
        x = np.load(os.path.join(path_dir, n))
        e = frame_rel_err(ref, x)
        bv, bi = worst_window(e, burst_window)
        rows.append(
            {
                "clip": os.path.splitext(n)[0],
                "frames": int(len(e)),
                "mean": clip_rel_l2(ref, x),
                "worst_frame": float(e.max()),
                "worst_frame_idx": int(e.argmax()),
                "worst_burst": bv,
                "worst_burst_idx": bi,
                "p99_frame": float(np.percentile(e, 99)),
                "_per_frame": e,
            }
        )
    return rows


def aggregate(rows):
    """Corpus-level statistics. mean is averaged over clips (as the old gate did); the localized
    statistics are MAXIMA over clips -- averaging a worst-case would re-introduce the blind spot."""
    wf = max(rows, key=lambda r: r["worst_frame"])
    wb = max(rows, key=lambda r: r["worst_burst"])
    return {
        "mean": float(np.mean([r["mean"] for r in rows])),
        "worst_clip_mean": float(max(r["mean"] for r in rows)),
        "worst_clip_mean_at": max(rows, key=lambda r: r["mean"])["clip"],
        "worst_frame": wf["worst_frame"],
        "worst_frame_at": f"{wf['clip']}[{wf['worst_frame_idx']}]",
        "worst_burst": wb["worst_burst"],
        "worst_burst_at": f"{wb['clip']}[{wb['worst_burst_idx']}:+{{w}}]",
    }


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("dirs", nargs="+", help="<f32_ref_dir> <baseline_dir> [candidate_dir]")
    ap.add_argument("--margin", type=float, default=0.15, help="allowed relative growth, all stats")
    ap.add_argument("--margin-mean", type=float, default=None, help="override --margin for mean")
    ap.add_argument("--margin-worst-frame", type=float, default=None)
    ap.add_argument("--margin-worst-burst", type=float, default=None)
    ap.add_argument("--burst-window", type=int, default=5, help="frames per contiguous burst window")
    ap.add_argument("--burst-floor", type=float, default=0.15,
                    help="per-frame rel-err above which a frame counts as 'in a burst'")
    ap.add_argument("--cliff", type=float, default=1.0,
                    help="per-frame rel-err at which a burst starts changing the TRANSCRIPT. "
                         "MEASURED, not chosen: scripts/burst_sensitivity.py injects a burst of a "
                         "known magnitude into the f32 encoder output and decodes. P(transcript "
                         "changes) is flat at ~0.5%% from 0.15 to 0.8 and turns up at 1.0 "
                         "(1.7-3.8%%), 1.4 (4.6-18.5%%), 1.8 (17-46%%). The shipped encoder's worst "
                         "frame is 0.788, so this is the real accuracy budget -- about 1.3x of "
                         "headroom -- and it is what a speed change must not spend.")
    ap.add_argument("--max-new-burst", type=float, default=0.15,
                    help="absolute cap on how much per-frame rel-err any single frame may GAIN. "
                         "Policy knob, not a noise floor: the device is bit-deterministic, so a "
                         "same-config rerun measures exactly 0.0 here and any value above 0 is "
                         "real. Default matches --burst-floor: no frame may gain as much error "
                         "as it takes to call a frame bursty in the first place.")
    ap.add_argument("--survey", action="store_true", help="baseline-only error profile, no gate")
    ap.add_argument("--per-clip", action="store_true", help="print the per-clip table")
    ap.add_argument("--json", metavar="PATH", default=None, help="also write the full result as JSON")
    args = ap.parse_args()

    m_mean = args.margin if args.margin_mean is None else args.margin_mean
    m_wf = args.margin if args.margin_worst_frame is None else args.margin_worst_frame
    m_wb = args.margin if args.margin_worst_burst is None else args.margin_worst_burst

    if args.survey:
        if len(args.dirs) != 2:
            ap.error("--survey takes <f32_ref_dir> <path_dir>")
        ref_dir, base_dir = args.dirs
        cand_dir = None
    else:
        if len(args.dirs) != 3:
            ap.error("expected <f32_ref_dir> <baseline_dir> <candidate_dir> (or --survey with 2)")
        ref_dir, base_dir, cand_dir = args.dirs

    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(ref_dir, "*.npy")))
    if not names:
        print(f"no .npy in {ref_dir}")
        sys.exit(2)

    base = profile(ref_dir, base_dir, names, args.burst_window)
    agg_b = aggregate(base)
    w = args.burst_window

    def fmt_agg(tag, a):
        print(f"{tag:9} vs f32-truth : mean {a['mean']:.4f}   "
              f"worst-clip {a['worst_clip_mean']:.4f} ({a['worst_clip_mean_at']})   "
              f"worst-frame {a['worst_frame']:.4f} ({a['worst_frame_at']})   "
              f"worst-burst[{w}] {a['worst_burst']:.4f} ({a['worst_burst_at'].format(w=w)})")

    print(f"clips={len(names)}  frames={sum(r['frames'] for r in base)}  burst-window={w}")
    fmt_agg("baseline", agg_b)

    burst_clips = sum(1 for r in base if r["worst_frame"] >= args.burst_floor)
    burst_frames = sum(int((r["_per_frame"] >= args.burst_floor).sum()) for r in base)
    print(f"baseline  burst load     : {burst_clips}/{len(base)} clips carry a frame "
          f">= {args.burst_floor:.2f}; {burst_frames} such frames total")

    result = {"clips": len(names), "burst_window": w, "baseline": agg_b,
              "baseline_burst_clips": burst_clips, "baseline_burst_frames": burst_frames}

    if cand_dir is None:
        if args.per_clip:
            print()
            print(f"{'clip':8} {'T':>4} {'mean':>8} {'worstF':>8} {'@':>5} {'burst':>8} {'@':>5}")
            for r in sorted(base, key=lambda r: -r["worst_frame"]):
                print(f"{r['clip']:8} {r['frames']:4d} {r['mean']:8.4f} {r['worst_frame']:8.4f} "
                      f"{r['worst_frame_idx']:5d} {r['worst_burst']:8.4f} {r['worst_burst_idx']:5d}")
        if args.json:
            for r in base:
                r.pop("_per_frame", None)
            result["baseline_per_clip"] = base
            with open(args.json, "w") as f:
                json.dump(result, f, indent=2)
        sys.exit(0)

    cand = profile(ref_dir, cand_dir, names, args.burst_window)
    agg_c = aggregate(cand)
    fmt_agg("candidate", agg_c)

    cb = [clip_rel_l2(np.load(os.path.join(base_dir, n)), np.load(os.path.join(cand_dir, n)))
          for n in names]
    print(f"candidate vs baseline  : mean rel-L2 = {float(np.mean(cb)):.4f}")

    # Localized-regression detector: frames where the candidate is materially worse than the
    # baseline AND is itself in burst territory. This is what a mean can never show.
    regressed = []
    new_burst = 0.0
    new_burst_at = "-"
    n_new, n_repaired = 0, 0
    for rb, rc in zip(base, cand):
        eb, ec = rb["_per_frame"], rc["_per_frame"]
        bad = np.where((ec > eb * (1.0 + m_wf)) & (ec >= args.burst_floor))[0]
        for t in bad:
            regressed.append((rb["clip"], int(t), float(eb[t]), float(ec[t])))
        n_new += int(((eb < args.burst_floor) & (ec >= args.burst_floor)).sum())
        n_repaired += int(((eb >= args.burst_floor) & (ec < args.burst_floor)).sum())
        k = int(np.argmax(ec - eb))
        if float(ec[k] - eb[k]) > new_burst:
            new_burst = float(ec[k] - eb[k])
            new_burst_at = f"{rb['clip']}[{k}] {eb[k]:.3f}->{ec[k]:.3f}"
    regressed.sort(key=lambda x: -(x[3] - x[2]))
    print(f"localized regressions  : {len(regressed)} frames worse by >{m_wf:.0%} and >= "
          f"{args.burst_floor:.2f}" + (f"; worst: " + ", ".join(
              f"{c}[{t}] {a:.3f}->{b:.3f}" for c, t, a, b in regressed[:5]) if regressed else ""))
    print(f"burst-frame churn      : {n_new} frames crossed INTO burst territory, "
          f"{n_repaired} crossed out")

    checks = [
        ("mean", agg_b["mean"], agg_c["mean"], m_mean),
        ("worst-frame", agg_b["worst_frame"], agg_c["worst_frame"], m_wf),
        (f"worst-burst[{w}]", agg_b["worst_burst"], agg_c["worst_burst"], m_wb),
    ]
    print()
    ok_all = True
    for name, b, c, m in checks:
        thr = b * (1.0 + m)
        ok = c <= thr
        ok_all &= ok
        print(f"GATE {name:16} candidate {c:.4f} {'<=' if ok else '>':2} baseline*(1+{m:g}) "
              f"{thr:.4f}  -> {'PASS' if ok else 'FAIL'}")
    ok_nb = new_burst <= args.max_new_burst
    ok_all &= ok_nb
    print(f"GATE {'new-burst':16} candidate {new_burst:.4f} {'<=' if ok_nb else '>':2} "
          f"cap {args.max_new_burst:.4f}  -> {'PASS' if ok_nb else 'FAIL'}   at {new_burst_at}")
    checks.append(("new-burst", 0.0, new_burst, args.max_new_burst))

    # The ACCURACY gate, as opposed to the three STABILITY gates above. Everything else here is
    # relative to a baseline that is itself defective; this one is absolute and calibrated against
    # a measured transcript-sensitivity curve.
    n_cliff_b = sum(int((r["_per_frame"] >= args.cliff).sum()) for r in base)
    n_cliff_c = sum(int((r["_per_frame"] >= args.cliff).sum()) for r in cand)
    ok_cl = n_cliff_c <= n_cliff_b
    ok_all &= ok_cl
    print(f"GATE {'cliff':16} candidate {n_cliff_c} frames >= {args.cliff:.2f} "
          f"{'<=' if ok_cl else '>':2} baseline {n_cliff_b}  -> {'PASS' if ok_cl else 'FAIL'}")
    print(f"GATE {'OVERALL':16} -> {'PASS' if ok_all else 'FAIL'}")
    print(f"     (mean/worst-frame/worst-burst/new-burst are STABILITY checks against a baseline "
          f"that is itself\n      defective; `cliff` is the ACCURACY check and is absolute.)")
    checks.append(("cliff", n_cliff_b, n_cliff_c, 0.0))

    if args.per_clip:
        print()
        print(f"{'clip':8} {'T':>4} {'mean_b':>8} {'mean_c':>8} {'wF_b':>7} {'wF_c':>7} "
              f"{'burst_b':>8} {'burst_c':>8}")
        for rb, rc in sorted(zip(base, cand), key=lambda p: -(p[1]["worst_frame"] - p[0]["worst_frame"])):
            print(f"{rb['clip']:8} {rb['frames']:4d} {rb['mean']:8.4f} {rc['mean']:8.4f} "
                  f"{rb['worst_frame']:7.4f} {rc['worst_frame']:7.4f} "
                  f"{rb['worst_burst']:8.4f} {rc['worst_burst']:8.4f}")

    if args.json:
        for r in base + cand:
            r.pop("_per_frame", None)
        result.update({
            "candidate": agg_c,
            "checks": [{"name": n, "baseline": b, "candidate": c, "margin": m,
                        "pass": bool(c <= (b * (1 + m) if n != "new-burst" else m))}
                       for n, b, c, m in checks],
            "new_burst": new_burst, "new_burst_at": new_burst_at,
            "burst_frames_in": n_new, "burst_frames_out": n_repaired,
            "pass": bool(ok_all),
            "localized_regressions": [
                {"clip": c, "frame": t, "baseline": a, "candidate": v} for c, t, a, v in regressed],
            "baseline_per_clip": base,
            "candidate_per_clip": cand,
        })
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)

    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
