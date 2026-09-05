#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Timing harness for a fused decoder-LLM ELF -- ms/token, phase split, census, determinism.

Sibling to verify_llm_decode.py (the correctness gate). This file asserts NOTHING about token
identity and must never be imported by or merged into the gate; it drives the SAME graph via
build_graph so the numbers describe the artifact the parity gate already exercises, not a re-typed
copy that can drift from it.

  python designs/decode_fused/bench_llm_decode.py --spec qwen3-0.6b \
      --weights /path/to/artifacts-qwen3-0.6b/weights --out-json /tmp/bench.json

Measured with time.perf_counter() (monotonic), never wall clock:

  1. ms/token swept over n_past, mean + sd + 95% CI (normal approx) per point, plus a linear fit
     (slope, intercept) across the swept points.
  2. Per-token phase split -- host embed gather + `x` write, host RoPE row + write, scratchpad
     param write + sync, the input BO sync (x/rope_global -> device), the device dispatch itself
     (run.start()+wait(), read off SequenceCallable.last_elapsed so it excludes both syncs), the
     output BO sync (the 303872 B `logits` readback), the host bf16->f32 copy, and the host argmax.
  3. A dispatch/context census: an in-process counter around `_run` (every call is one
     run.start()+wait() = one dispatch), plus a window where the callable's hw_context is held open
     so `xrt-smi examine -r aie-partitions` can be pointed at it from another shell.
  4. Run-to-run determinism: N independent passes over one fixed prompt, KV cache reset to zero
     between passes (the weights dict already holds the zero-fill arrays gen_llm_decode.py builds),
     produced token sequences compared bit-for-bit.

NOTE on the n_past sweep (methodology, not a shortcut hidden from the reader): `kv_off`/`sm_mask`
are written directly for the target n_past WITHOUT replaying the tokens in between. Per
gen_llm_decode.py's own docstring the ELF is CONSTANT across tokens and every per-token difference
is carried by (x, rope_global, kv_off, sm_mask) -- so wall-clock cost is a pure function of those
values, not of how the KV cache at earlier offsets got there. This produces numerically meaningless
logits (most of the KV cache window is still zero-fill, not real history) but a real, honest wall-
clock number; the determinism check below instead does a REAL monotonic walk (pos 0..N-1) so at
least one part of this file exercises the actual autoregressive path.

Instrumentation is done by wrapping the callable's `_run`/`_sync_inputs`/`_sync_outputs` methods on
the INSTANCE this script owns (plain Python monkeypatching of an object we constructed) -- no
tracked file is edited to get these numbers.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports (new-mlir-aie port shim)
from gen_llm_decode import build_graph, report_artifact_freshness  # noqa: E402

BF16 = ml_dtypes.bfloat16


def rope_row(pos, head_dim, theta):
    """Copied verbatim from verify_llm_decode.py -- see that file for the layout note."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64)[:half] / head_dim))
    ang = pos * inv
    row = np.empty(head_dim, dtype=np.float32)
    row[0::2] = np.cos(ang)
    row[1::2] = np.sin(ang)
    return row.astype(BF16)


def now():
    return time.perf_counter()


def mean_sd_ci(xs):
    n = len(xs)
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    half = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": sd, "ci95": [m - half, m + half]}


def linear_fit(xs, ys):
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


PHASES = ["embed_write", "rope_write", "params", "sync_in", "dispatch_only", "sync_out",
          "logits_copy", "argmax"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--positions", type=int, nargs="+", default=[1, 16, 64, 256, 512, 1024, 2047],
                     help="n_past values to sweep (sm_mask = pos+1 <= max-seq)")
    ap.add_argument("--reps", type=int, default=30, help="timed reps per swept position")
    ap.add_argument("--warmup", type=int, default=5, help="untimed reps per position, discarded")
    ap.add_argument("--det-runs", type=int, default=5, help="independent determinism passes")
    ap.add_argument("--det-ref", default=None, help="greedy_ref.json/bf16_oracle.json for the "
                     "determinism prompt; falls back to a fixed literal prompt if omitted")
    ap.add_argument("--det-steps", type=int, default=8, help="free-running tokens per determinism pass")
    ap.add_argument("--hold-secs", type=float, default=8.0,
                     help="seconds to hold the hw_context open (idle) for an external census probe")
    ap.add_argument("--det-full-reset", action="store_true",
                     help="also zero every non-weight scratch buffer (q/k/v/sc/sw/vt/... -- "
                     "everything the sweep phase left dirty at S-wide extents), not just the KV "
                     "cache, before each determinism pass. Distinguishes a genuine hardware race "
                     "from stale-scratch carry-over across passes in THIS process.")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    report_artifact_freshness(a.weights)

    t0 = now()
    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    build_s = now() - t0
    NL, S = md["NL"], md["S"]
    HD, D, VOCAB = sp.head_dim, sp.d_model, sp.vocab
    print(f"[bench] build_graph: {build_s:.1f}s  ({sp.name}, {NL} layers, S={S}, vocab={VOCAB})",
          flush=True)
    try:
        print(f"[bench] mlir_artifact: {fused.artifacts[0].mlir_input.filename}", flush=True)
    except Exception as e:  # noqa: BLE001 -- diagnostic only, never fatal
        print(f"[bench] mlir_artifact: unavailable ({e})", flush=True)

    t0 = now()
    c = fused.get_callable()
    print(f"[bench] get_callable (hw_context + kernel load): {now() - t0:.3f}s", flush=True)
    params = c.params
    if params is None:
        raise SystemExit("[bench] no runtime parameters bound -- params.txt missing from the build")

    t0 = now()
    for name, arr in weights.items():
        buf = c.get_buffer(name)
        np.copyto(buf.data, np.asarray(arr, BF16).reshape(-1))
    print(f"[bench] weight load: {now() - t0:.1f}s ({len(weights)} buffers)", flush=True)

    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0

    xin = c.get_buffer("x")
    rope_buf = c.get_buffer("rope_global")
    out = c.get_buffer("logits")

    # ---- instrumentation: wrap THIS instance's methods, no tracked file touched ----
    dispatch_count = [0]
    orig_run = c._run

    def counted_run():
        dispatch_count[0] += 1
        orig_run()

    c._run = counted_run

    last_sync_in = [0.0]
    last_sync_out = [0.0]
    orig_sync_in = c._sync_inputs
    orig_sync_out = c._sync_outputs

    def timed_sync_in():
        t0_ = now()
        orig_sync_in()
        last_sync_in[0] = now() - t0_

    def timed_sync_out():
        t0_ = now()
        orig_sync_out()
        last_sync_out[0] = now() - t0_

    c._sync_inputs = timed_sync_in
    c._sync_outputs = timed_sync_out

    def one_token(pos, tok):
        t0_ = now()
        np.copyto(xin.data, np.asarray(embed[tok] * scale, BF16).reshape(-1))
        t1 = now()
        np.copyto(rope_buf.data, rope_row(pos, HD, sp.rope_theta_global).reshape(-1))
        t2 = now()
        params.write("kv_off", int(pos * HD))
        params.write("sm_mask", int(pos + 1))
        params.sync()
        t3 = now()
        c()
        t4 = now()
        lg = np.asarray(out.data[:VOCAB], dtype=np.float32)
        t5 = now()
        nxt = int(np.argmax(lg))
        t6 = now()
        ph = {
            "embed_write": t1 - t0_,
            "rope_write": t2 - t1,
            "params": t3 - t2,
            "sync_in": last_sync_in[0],
            "dispatch_only": c.last_elapsed,
            "sync_out": last_sync_out[0],
            "logits_copy": t5 - t4,
            "argmax": t6 - t5,
        }
        return ph, nxt, lg

    TOK = 100  # arbitrary valid token id; only wall-clock is under test in the sweep below

    # ---- census: one warm dispatch, then hold the hw_context open for an external probe ----
    one_token(0, TOK)
    print(f"[bench] CENSUS_HOLD_START dispatch_count={dispatch_count[0]} "
          f"pid={os.getpid()} hold_secs={a.hold_secs}", flush=True)
    time.sleep(a.hold_secs)
    print(f"[bench] CENSUS_HOLD_END dispatch_count={dispatch_count[0]}", flush=True)

    # ---- 1+2: ms/token sweep over n_past, with the full phase split at every point ----
    sweep = {}
    for pos in a.positions:
        if pos + 1 > S:
            print(f"[bench] SKIP pos={pos}: sm_mask={pos + 1} > S={S}", flush=True)
            continue
        for _ in range(a.warmup):
            one_token(pos, TOK)
        per_phase = {k: [] for k in PHASES}
        totals = []
        for _ in range(a.reps):
            ph, _, _ = one_token(pos, TOK)
            for k in PHASES:
                per_phase[k].append(ph[k])
            totals.append(sum(ph.values()))
        stats_total = mean_sd_ci([t * 1000 for t in totals])
        stats_phase = {k: mean_sd_ci([v * 1000 for v in vals]) for k, vals in per_phase.items()}
        sweep[pos] = {"total_ms": stats_total, "phases_ms": stats_phase}
        print(f"[bench] pos={pos:5d}  total {stats_total['mean']:7.3f} ms "
              f"(sd {stats_total['sd']:.3f}, n={stats_total['n']})  "
              f"dispatch_only {stats_phase['dispatch_only']['mean']:7.3f} ms  "
              f"sync_out {stats_phase['sync_out']['mean']:7.3f} ms", flush=True)

    xs = [p for p in sweep]
    ys = [sweep[p]["total_ms"]["mean"] for p in xs]
    slope, intercept = (linear_fit(xs, ys) if len(xs) >= 2 else (float("nan"), float("nan")))
    print(f"[bench] linear fit over sweep: total_ms = {slope:.6f} * n_past + {intercept:.4f}",
          flush=True)

    # ---- 4: run-to-run determinism, real monotonic walk over one fixed prompt ----
    if a.det_ref:
        ref = json.load(open(a.det_ref))
        prompt_ids = ref["prompt_ids"]
    else:
        prompt_ids = [785, 6722, 315, 9625, 374]  # "The capital of France is" (Qwen3 tokenizer)

    cache_names = md["cache_names"]

    def reset_kv():
        for name in cache_names:
            buf = c.get_buffer(name)
            np.copyto(buf.data, weights[name])

    # Every scratch buffer the runlist ever writes that is NOT a loaded weight -- q/k/v/sc/sw/vt/
    # cx/a/g/u/gh/d/hn/hf/x1 per layer. These are declared by SIZE only (bufsz in
    # gen_llm_decode.py), so unlike kc/vc there is no zero-fill array to copy from; a fresh
    # np.zeros per buffer is built once and reused.
    extra_scratch = {}
    if a.det_full_reset:
        for name, (buf_type, _off, length) in fused.subbuffer_layout.items():
            if buf_type == "scratch" and name not in weights:
                extra_scratch[name] = np.zeros(length // 2, BF16)  # bf16 element count

    def reset_all():
        reset_kv()
        for name, zeros in extra_scratch.items():
            np.copyto(c.get_buffer(name).data, zeros)

    det_sequences = []
    for run_idx in range(a.det_runs):
        reset_all() if a.det_full_reset else reset_kv()
        fed = list(prompt_ids)
        produced = []
        tok = fed[0]
        for pos in range(len(fed) + a.det_steps - 1):
            _, nxt, _ = one_token(pos, tok)
            if pos + 1 < len(fed):
                tok = fed[pos + 1]
            else:
                produced.append(nxt)
                tok = nxt
            if len(produced) >= a.det_steps:
                break
        det_sequences.append(produced)
        print(f"[bench] determinism run {run_idx}: {produced}", flush=True)

    ref_seq = det_sequences[0]
    match_count = sum(1 for s in det_sequences if s == ref_seq)
    mismatches = []
    for i, seq in enumerate(det_sequences[1:], start=1):
        if seq != ref_seq:
            for step, (a_tok, b_tok) in enumerate(zip(ref_seq, seq)):
                if a_tok != b_tok:
                    prev_ref = ref_seq[step - 1] if step > 0 else None
                    mismatches.append({
                        "run": i, "step": step, "run0_token": a_tok, "run_i_token": b_tok,
                        "run_i_equals_prev_step_of_run0": (b_tok == prev_ref),
                    })
    print(f"[bench] determinism: {match_count}/{a.det_runs} runs bit-identical to run 0", flush=True)
    if mismatches:
        print(f"[bench] determinism MISMATCHES: {mismatches}", flush=True)

    result = {
        "spec": sp.name, "layers": NL, "max_seq": S, "vocab": VOCAB, "d_model": D, "head_dim": HD,
        "build_s": build_s,
        "dispatch_count_total": dispatch_count[0],
        "sweep": {str(k): v for k, v in sweep.items()},
        "linear_fit": {"slope_ms_per_pos": slope, "intercept_ms": intercept},
        "determinism": {
            "runs": a.det_runs, "steps": a.det_steps, "prompt_ids": prompt_ids,
            "sequences": det_sequences, "match_count": match_count, "mismatches": mismatches,
        },
    }
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[bench] wrote {a.out_json}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
