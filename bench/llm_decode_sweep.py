#!/usr/bin/env python3
"""Decompose an OpenAI-API LLM server's e2e latency into per-token overhead + achieved bandwidth.

At M=1 decode the per-token cost is a line in the weight bytes streamed per token:

    ITL(model) ~= overhead_ms + footprint_GB / BW_eff

so sweeping ONE model family at ONE quantization (weight bytes the only moving part)
and regressing steady-state inter-token latency against footprint recovers both terms
from black-box timings alone: the slope is 1/BW_eff, the intercept is the fixed
per-token dispatch overhead. Neither needs any visibility into the server's kernels.

Two sweeps:
  decode  -- ITL vs model footprint  -> (overhead_ms, BW_eff GB/s)
  prefill -- TTFT vs prompt length   -> (setup_ms, prefill tok/s)

The server is single-tenant on the NPU, so models run strictly sequentially: serve,
warm, measure, stop. Pin --pmode (default `performance`); leaving it unset lets the
governor move under the sweep and the fit degrades to noise.

Usage:
    bench/llm_decode_sweep.py --models qwen3:0.6b,qwen3:1.7b,qwen3:4b,qwen3:8b \
        --out bench/results/flm-decode-sweep.json
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.energy import EnergyMeter, readable
from bench.sysmetrics import CpuSampler, ProcCpu, peak_rss_kb

CATALOG = "/usr/share/flm/model_list.json"
PORT = 11439
PROMPT = "Count upward from one, one number per line, and do not stop early."


def footprints():
    """tag:size -> (footprint GB, quantization) from the shipped catalog."""
    d = json.load(open(CATALOG))["models"]
    out = {}
    for tag, sizes in d.items():
        if not isinstance(sizes, dict):
            continue
        for sz, m in sizes.items():
            if isinstance(m, dict) and "details" in m:
                out[f"{tag}:{sz}"] = (m.get("footprint"), m["details"].get("quantization_level"))
    return out


class Server:
    """`flm serve` lifecycle. Single-tenant: exactly one of these alive at a time."""

    def __init__(self, model, port=PORT, pmode="performance", ctx=2048):
        self.model, self.port, self.pmode, self.ctx = model, port, pmode, ctx
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["flm", "serve", self.model, "--pmode", self.pmode,
             "--port", str(self.port), "-c", str(self.ctx)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{self.port}/v1/models"
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"flm serve {self.model} exited rc={self.proc.returncode}")
            try:
                if requests.get(url, timeout=2).ok:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"flm serve {self.model} never became ready")

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        time.sleep(2)  # let the NPU context tear down before the next tenant

    def peak_rss_kb(self):
        try:
            return peak_rss_kb(self.proc.pid)
        except (FileNotFoundError, ProcessLookupError):
            return 0


def stream(port, model, prompt, max_tokens):
    """One streamed completion. Returns (ttft_s, [inter-token deltas], n_tokens)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    stamps = []
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json=body, stream=True, timeout=900)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload.strip() == b"[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        # Count chunks carrying generated text; FLM emits a role-only preamble
        # chunk that is not a token. Thinking models (qwen3:1.7b/4b/8b) stream
        # their chain-of-thought as `reasoning_content` -- same decode cost per
        # token, so it counts too, else those models look like they emit nothing.
        if delta.get("content") or delta.get("reasoning_content"):
            stamps.append(time.perf_counter())
    if not stamps:
        raise RuntimeError("no token chunks received")
    ttft = stamps[0] - t0
    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    return ttft, itl, len(stamps)


def idle_watts(seconds=3.0):
    """Whole-SoC idle draw, for subtraction.

    RAPL here is PACKAGE-wide, so raw J/token bills the whole SoC's idle floor to
    the model. Measure the floor with the server up but not generating, and report
    both raw and net -- at ~8 W idle and ~10 ms/token the floor alone is ~0.09
    J/token, which swamps a small model's real cost if left in.
    """
    if not readable():
        return None
    with EnergyMeter() as em:
        time.sleep(seconds)
    return em.watts


def decode_point(model, port, max_tokens, repeats, warmups, pid=None, idle_w=None):
    """Steady-state ITL for one model. Median-of-run-medians + spread.

    Also charges the server process's own CPU time to the tokens it produced:
    host_cpu_ms_per_token is the dynamic counterpart to reading the host source
    -- it says how much CPU work the runtime really does per decode step.
    """
    for _ in range(warmups):
        stream(port, model, PROMPT, 16)
    runs = []
    cpu = ProcCpu(pid) if pid else _Null()
    em = EnergyMeter() if readable() else _Null()
    with em, CpuSampler() as cs, cpu:
        for _ in range(repeats):
            ttft, itl, n = stream(port, model, PROMPT, max_tokens)
            if len(itl) < 8:
                raise RuntimeError(f"{model}: only {n} tokens, too few for a stable median")
            runs.append({"ttft_ms": ttft * 1e3,
                         "itl_median_ms": statistics.median(itl) * 1e3,
                         "itl_mean_ms": statistics.fmean(itl) * 1e3,
                         "n_tokens": n})
    med = statistics.median(r["itl_median_ms"] for r in runs)
    tot_tok = sum(r["n_tokens"] for r in runs)
    has_e = isinstance(em, EnergyMeter)
    has_c = isinstance(cpu, ProcCpu)
    return {
        "runs": runs,
        "itl_median_ms": med,
        "itl_spread_ms": max(r["itl_median_ms"] for r in runs) - min(r["itl_median_ms"] for r in runs),
        "ttft_median_ms": statistics.median(r["ttft_ms"] for r in runs),
        "tok_per_s": 1e3 / med,
        "joules_total": em.joules if has_e else None,
        "j_per_token": (em.joules / tot_tok) if (has_e and tot_tok) else None,
        # idle-subtracted: the marginal energy of generating, with the SoC floor removed
        "j_per_token_net": ((em.joules - idle_w * em.t) / tot_tok)
                           if (has_e and tot_tok and idle_w is not None) else None,
        "idle_watts": idle_w,
        "mean_watts": em.watts if has_e else None,
        "cpu_idle_frac": cs.idle_frac,
        "host_cpu_ms_total": cpu.ms if has_c else None,
        "host_cpu_ms_per_token": (cpu.ms / tot_tok) if (has_c and tot_tok) else None,
        "host_cpu_frac_of_itl": (cpu.ms / tot_tok / med) if (has_c and tot_tok and med) else None,
    }


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def prefill_point(model, port, prompt_words, repeats):
    """TTFT at one prompt length. Prompt is filler words -> ~1 token/word."""
    prompt = " ".join(["data"] * prompt_words) + "\nReply with one word."
    for _ in range(2):
        stream(port, model, prompt, 4)
    ttfts = []
    for _ in range(repeats):
        ttft, _, _ = stream(port, model, prompt, 4)
        ttfts.append(ttft * 1e3)
    return {"prompt_words": prompt_words, "ttft_median_ms": statistics.median(ttfts),
            "ttft_all_ms": ttfts}


def ols(xs, ys):
    """Least-squares y = a + b*x, plus R^2. Plain code -- no scipy dependency."""
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"intercept": a, "slope": b, "r2": 1 - ss_res / ss_tot if ss_tot else 1.0, "n": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="comma-separated flm tags, one family/quant")
    p.add_argument("--prefill-model", default=None, help="model for the TTFT-vs-prompt-length sweep")
    p.add_argument("--prefill-lens", default="16,64,256,1024")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--pmode", default="performance")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if not readable():
        print("WARN: RAPL not readable -- energy fields will be absent", file=sys.stderr)

    fp = footprints()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    quants = {fp.get(m, (None, None))[1] for m in models}
    if len(quants) > 1:
        print(f"WARN: mixed quantization {quants} -- the bytes/param term is not constant, "
              f"the fit will not mean what it should", file=sys.stderr)

    out = {"pmode": args.pmode, "max_tokens": args.max_tokens, "repeats": args.repeats,
           "flm_version": subprocess.run(["flm", "version"], capture_output=True, text=True).stdout.strip(),
           "decode": {}, "prefill": {}}

    for m in models:
        gb, q = fp.get(m, (None, None))
        print(f"[decode] {m}  footprint={gb} GB  q={q}", file=sys.stderr)
        with Server(m, args.port, args.pmode) as s:
            # idle floor measured with THIS model loaded but not generating, so the
            # baseline includes its resident footprint rather than a bare-machine floor
            iw = idle_watts()
            pt = decode_point(m, args.port, args.max_tokens, args.repeats, args.warmups,
                              pid=s.proc.pid, idle_w=iw)
            pt.update({"footprint_gb": gb, "quantization": q, "peak_rss_kb": s.peak_rss_kb()})
        out["decode"][m] = pt
        hc = pt.get("host_cpu_ms_per_token")
        jn = pt.get("j_per_token_net")
        line = f"          ITL {pt['itl_median_ms']:.2f} ms  ({pt['tok_per_s']:.1f} tok/s)  spread {pt['itl_spread_ms']:.2f} ms"
        if hc:
            line += f"  hostCPU {hc:.2f} ms/tok ({pt['host_cpu_frac_of_itl']*100:.0f}%)"
        if jn is not None:
            line += f"  {pt['j_per_token']:.3f} J/tok raw / {jn:.3f} net (idle {pt['idle_watts']:.1f} W)"
        print(line, file=sys.stderr)

    pts = [(v["footprint_gb"], v["itl_median_ms"]) for v in out["decode"].values() if v["footprint_gb"]]
    if len(pts) >= 2:
        fit = ols([x for x, _ in pts], [y for _, y in pts])
        # slope is ms per GB -> GB/s ; intercept is the fixed per-token cost
        out["fit_decode"] = {
            "overhead_ms_per_token": fit["intercept"],
            "slope_ms_per_gb": fit["slope"],
            "bw_eff_gb_s": 1e3 / fit["slope"] if fit["slope"] > 0 else None,
            "r2": fit["r2"], "n_points": fit["n"],
        }

    pm = args.prefill_model or models[0]
    lens = [int(x) for x in args.prefill_lens.split(",")]
    print(f"[prefill] {pm} lens={lens}", file=sys.stderr)
    with Server(pm, args.port, args.pmode):
        out["prefill"][pm] = [prefill_point(pm, args.port, L, args.repeats) for L in lens]
    ppts = [(d["prompt_words"], d["ttft_median_ms"]) for d in out["prefill"][pm]]
    if len(ppts) >= 2:
        fit = ols([x for x, _ in ppts], [y for _, y in ppts])
        out["fit_prefill"] = {
            "setup_ms": fit["intercept"],
            "ms_per_prompt_token": fit["slope"],
            "prefill_tok_per_s": 1e3 / fit["slope"] if fit["slope"] > 0 else None,
            "r2": fit["r2"], "n_points": fit["n"],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out.get("fit_decode", {}), indent=2))
    print(json.dumps(out.get("fit_prefill", {}), indent=2))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
