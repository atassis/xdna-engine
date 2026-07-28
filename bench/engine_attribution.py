#!/usr/bin/env python3
"""Which engine actually runs prefill, and which runs decode?

A bandwidth-bound decode looks the SAME from end-to-end timings whether it runs on
the NPU, the iGPU or the CPU -- all three stream Q4 weights out of the same LPDDR, so
"inter-token latency is linear in model bytes" does not by itself attribute the work.
Some Ryzen AI stacks genuinely do split it (prefill on NPU, decode on iGPU), so the
attribution has to be measured, not inferred.

This samples, at ~50 Hz across one streamed completion:
  - iGPU busy%      (/sys/class/drm/cardN/device/gpu_busy_percent, amdgpu)
  - VCN busy%       (video engine, same path -- sanity control, should stay 0)
  - server CPU%     (utime+stime delta of the serving process, all threads)
and splits the samples at the first-token timestamp into a PREFILL window and a
DECODE window.

Reading:
  decode on iGPU  -> iGPU busy high during the decode window
  decode on CPU   -> server CPU% >= ~100% (a full core) through the decode window
  decode on NPU   -> both low; the host is blocked/idle waiting on the accelerator

Usage:
    bench/engine_attribution.py --model qwen3:4b --port 11439 --gen-tokens 128
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.sysmetrics import proc_cpu_ms

DRM = "/sys/class/drm"


def find_igpu():
    """The amdgpu render node exposing gpu_busy_percent."""
    for c in sorted(Path(DRM).glob("card*/device/gpu_busy_percent")):
        return c.parent
    return None


def _read_int(p):
    try:
        return int(Path(p).read_text().strip())
    except (OSError, ValueError):
        return None


class Sampler(threading.Thread):
    """Background sampler; one row per tick."""

    def __init__(self, pid, dev, hz=50):
        super().__init__(daemon=True)
        self.pid, self.dev, self.dt = pid, dev, 1.0 / hz
        self.rows = []
        self.stop = threading.Event()

    def run(self):
        prev_cpu, prev_t = proc_cpu_ms(self.pid), time.perf_counter()
        while not self.stop.is_set():
            time.sleep(self.dt)
            now = time.perf_counter()
            try:
                cpu = proc_cpu_ms(self.pid)
            except (FileNotFoundError, ProcessLookupError, IndexError):
                break
            wall_ms = (now - prev_t) * 1e3
            self.rows.append({
                "t": now,
                # percent of ONE core: 100 = one core saturated
                "cpu_pct": (cpu - prev_cpu) / wall_ms * 100 if wall_ms else 0.0,
                "gpu_pct": _read_int(self.dev / "gpu_busy_percent"),
                "vcn_pct": _read_int(self.dev / "vcn_busy_percent"),
            })
            prev_cpu, prev_t = cpu, now


def stream(port, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True}
    stamps = []
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json=body, stream=True, timeout=1800)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        p = line[6:]
        if p.strip() == b"[DONE]":
            break
        try:
            ch = json.loads(p)
        except json.JSONDecodeError:
            continue
        d = (ch.get("choices") or [{}])[0].get("delta", {})
        if d.get("content") or d.get("reasoning_content"):
            stamps.append(time.perf_counter())
    return t0, stamps


def summarize(rows, lo, hi, label):
    w = [r for r in rows if lo <= r["t"] <= hi]
    if not w:
        return {"window": label, "n": 0}
    def stat(k):
        vs = [r[k] for r in w if r[k] is not None]
        if not vs:
            return None
        return {"mean": statistics.fmean(vs), "max": max(vs),
                "p90": sorted(vs)[int(len(vs) * 0.9)]}
    return {"window": label, "n": len(w), "seconds": hi - lo,
            "cpu_pct": stat("cpu_pct"), "gpu_pct": stat("gpu_pct"), "vcn_pct": stat("vcn_pct")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=11439)
    ap.add_argument("--pmode", default="performance")
    ap.add_argument("--prompt-words", type=int, default=1500,
                    help="long prompt so the prefill phase is resolvable")
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = find_igpu()
    if dev is None:
        sys.exit("no amdgpu gpu_busy_percent found -- cannot attribute iGPU work")
    print(f"[attr] iGPU counter: {dev}/gpu_busy_percent", file=sys.stderr)

    proc = subprocess.Popen(
        ["flm", "serve", a.model, "--pmode", a.pmode, "--port", str(a.port), "-c", str(a.ctx)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{a.port}/v1/models"
        for _ in range(900):
            if proc.poll() is not None:
                sys.exit(f"flm serve exited rc={proc.returncode}")
            try:
                if requests.get(url, timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)

        prompt = ("Summarize the following. " + " ".join(["data"] * a.prompt_words)
                  + "\nNow write a long detailed explanation.")
        stream(a.port, a.model, "hi", 8)          # warm

        # idle baseline just before the timed request
        base = Sampler(proc.pid, dev)
        base.start(); time.sleep(2.0); base.stop.set(); base.join(timeout=2)
        idle = summarize(base.rows, base.rows[0]["t"], base.rows[-1]["t"], "idle") if base.rows else {}

        s = Sampler(proc.pid, dev)
        s.start()
        t0, stamps = stream(a.port, a.model, prompt, a.gen_tokens)
        s.stop.set(); s.join(timeout=2)
        if len(stamps) < 8:
            sys.exit(f"only {len(stamps)} tokens -- not enough to resolve a decode window")

        ttft_end = stamps[0]
        out = {
            "model": a.model, "pmode": a.pmode, "n_tokens": len(stamps),
            "ttft_s": ttft_end - t0,
            "itl_median_ms": statistics.median(
                (b - x) * 1e3 for x, b in zip(stamps, stamps[1:])),
            "idle": idle,
            "prefill": summarize(s.rows, t0, ttft_end, "prefill"),
            # skip the first token: it straddles the boundary
            "decode": summarize(s.rows, stamps[1], stamps[-1], "decode"),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)

    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    for w in ("idle", "prefill", "decode"):
        d = out.get(w) or {}
        if d.get("n"):
            print(f"{w:8s} n={d['n']:4d} {d['seconds']:6.2f}s  "
                  f"cpu {d['cpu_pct']['mean']:6.1f}% (max {d['cpu_pct']['max']:.0f})  "
                  f"iGPU {d['gpu_pct']['mean']:5.1f}% (max {d['gpu_pct']['max']:.0f})",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
