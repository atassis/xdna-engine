#!/usr/bin/env python3
"""Benchmark orchestrator: head-to-head ASR comparison on the (single-tenant) NPU.

Runs each backend SEQUENTIALLY — the NPU holds only one tenant at a time, so we
stop every other NPU user, run one backend to completion, stop it, then run the
next. Per clip we measure latency, WER (vs refs.json), RAPL package energy,
peak RSS, and CPU-idle fraction. Aggregates EN/RU mean WER, median latency,
mean J/clip, peak RAM, mean CPU-idle%.

Usage (from worktree root, with $VENV):
    $VENV bench/compare.py --backends ours,flm --scenario scenarios/asr-whisper-small.toml

CRITICAL: stop voxd.service before running; this script restarts it at the end.
"""
import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import requests

# Reusable bench pieces (import, don't reimplement).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.energy import EnergyMeter, readable
from bench.sysmetrics import CpuSampler, peak_rss_kb
from bench.backends import FLM, ours

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "bench" / "results"
LIB = os.path.expanduser("~/.local/lib/xdna-engine")
OUR_PORT = 11435

# --- WER (mirrors scripts/whisper_cpu_oracle.py: same normalize + edit-distance) ---
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(t):
    t = unicodedata.normalize("NFC", t or "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return (0.0 if not h else 1.0), 0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if rw == hw else 1)))
        prev = cur
    return prev[-1] / len(r), len(r)


# --- service / process control ---------------------------------------------
def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def stop_all_npu():
    """Free the NPU: stop every known NPU tenant. Single-tenant device."""
    for svc in ("xdna-engine.service", "npu-vox.service", "voxd.service", "flm-asr.service"):
        sh("systemctl", "--user", "stop", svc)
    # kill any stray server we may have left behind
    sh("pkill", "-f", "release/npu serve")
    sh("pkill", "-f", "flm serve")
    time.sleep(2)


def child_pids(pid):
    """All descendants of pid (so we catch RSS in subprocesses, e.g. FLM workers)."""
    out = sh("pgrep", "-P", str(pid)).stdout.split()
    kids = [int(p) for p in out if p.isdigit()]
    return kids + [g for k in kids for g in child_pids(k)]


def tree_rss_kb(pid):
    """Peak RSS (VmHWM) summed over pid + all its descendants."""
    if not pid:
        return 0
    total = peak_rss_kb(pid)
    for k in child_pids(pid):
        total += peak_rss_kb(k)
    return total


def poll_ready(url_base, timeout=40):
    """Wait until the server is listening and serving.

    Both backends bind their port only once the model is loaded, so *any* HTTP
    response (even 404) means the server is up. We probe `/v1/models` first (FLM
    answers 200 there; its `/health` returns 404), then fall back to `/health`
    (our engine answers 200 there). A connection error = not up yet.
    """
    base = url_base.rsplit("/v1/", 1)[0]
    probes = [base + "/v1/models", base + "/health"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        for p in probes:
            try:
                r = requests.get(p, timeout=2)
                # Got an HTTP response at all -> the server socket is live.
                if r.status_code < 500:
                    return True
            except requests.RequestException:
                pass
        time.sleep(1)
    return False


# --- per-backend runners ----------------------------------------------------
def run_ours(scenario, clips, refs, model_name="parakeet"):
    """Serve our engine via the `npu serve` CLI (the shipped entry point).

    `scenario` here is the engine.toml config path -- the CLI takes a config that
    names the resident models, not a bare scenario file.
    """
    stop_all_npu()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    binpath = str(ROOT / "rust" / "target" / "release" / "npu")
    if not Path(binpath).exists():
        binpath = os.path.expanduser("~/.local/bin/npu")
    proc = subprocess.Popen(
        [binpath, "serve", "--config", scenario, "--port", str(OUR_PORT)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    backend = ours(OUR_PORT, model_name)
    try:
        if not poll_ready(backend.url, timeout=120):
            raise RuntimeError("ours `npu serve` did not become ready")
        rows = measure(backend, clips, refs, pid=proc.pid)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        sh("pkill", "-f", "release/npu serve")
        time.sleep(1)
    return rows


def flm_pid():
    r = sh("systemctl", "--user", "show", "-p", "MainPID", "flm-asr.service")
    for line in r.stdout.splitlines():
        if line.startswith("MainPID="):
            v = line.split("=", 1)[1].strip()
            if v.isdigit() and int(v) > 0:
                return int(v)
    # Fallback: locate the running `flm serve` process directly.
    out = sh("pgrep", "-f", "flm serve").stdout.split()
    for p in out:
        if p.isdigit():
            return int(p)
    return None


def run_flm(clips, refs, companion="qwen3:0.6b", pmode="performance"):
    """Serve FLM's Whisper directly.

    `flm serve whisper-v3:turbo` auto-pulls an LLM, so we serve an already-installed
    small LLM with `-a 1` (load ASR alongside) instead -- the documented way in. The
    companion must be version-compatible with the installed flm or serve refuses.
    """
    stop_all_npu()
    backend = FLM
    proc = subprocess.Popen(
        ["flm", "serve", companion, "-a", "1", "--pmode", pmode, "--port", "11434"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not poll_ready(backend.url, timeout=180):
            raise RuntimeError("flm serve did not become ready on :11434")
        rows = measure(backend, clips, refs, pid=proc.pid)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        sh("pkill", "-f", "flm serve")
        time.sleep(2)
    return rows


# --- measurement loop -------------------------------------------------------
def measure(backend, clips, refs, pid):
    # 2 warmup transcribes (discarded) — first-run NPU/model init cost.
    warm = clips[0]
    for _ in range(2):
        try:
            backend.transcribe(str(warm))
        except Exception as e:
            print(f"  [warn] warmup failed: {e}", file=sys.stderr)
    rows = []
    peak_rss = 0
    for clip in clips:
        name = clip.name
        ref = normalize(refs[name])
        with CpuSampler() as cpu, EnergyMeter() as em:
            text, _ = backend.transcribe(str(clip))
        latency = em.t  # wall time around the single timed call
        w, nref = wer(ref, normalize(text))
        rss = tree_rss_kb(pid) if pid else 0
        peak_rss = max(peak_rss, rss)
        rows.append({
            "clip": name,
            "lang": "ru" if name.startswith("ru") else "en",
            "latency_s": latency,
            "joules": em.joules,
            "watts": em.watts,
            "wer": w,
            "nref": nref,
            "idle_frac": cpu.idle_frac,
            "rss_kb": rss,
            "hyp": text,
        })
        j = f"{em.joules:6.1f}J" if em.joules is not None else "   n/a"
        print(f"  {backend.name} {name}: {latency:5.2f}s  WER={w:.3f}  {j}")
    return {"backend": backend.name, "model": backend.model, "rows": rows, "peak_rss_kb": peak_rss}


# --- aggregation + output ---------------------------------------------------
def aggregate(result):
    rows = result["rows"]
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    en = [r["wer"] for r in rows if r["lang"] == "en"]
    ru = [r["wer"] for r in rows if r["lang"] == "ru"]
    lat = [r["latency_s"] for r in rows]
    return {
        "backend": result["backend"],
        "model": result["model"],
        "en_wer": mean(en),
        "ru_wer": mean(ru),
        "median_latency_s": statistics.median(lat) if lat else 0.0,
        "mean_joules": mean([r["joules"] for r in rows]),
        "mean_watts": mean([r["watts"] for r in rows]),
        "peak_rss_kb": result["peak_rss_kb"],
        "mean_idle_frac": mean([r["idle_frac"] for r in rows]),
        "n_clips": len(rows),
    }


def write_outputs(results, aggs, scenario, stem="whisper-small-vs-flm-turbo", title=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # raw JSON (per-clip)
    raw = {
        "scenario": scenario,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "energy_readable": readable(),
        "aggregates": aggs,
        "backends": results,
    }
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False))

    # markdown comparison table
    def rss_gb(kb):
        return f"{kb / 1024 / 1024:.2f} GB" if kb else "n/a"
    lines = [
        f"# {title or 'Whisper-small (ours, NPU) vs FastFlowLM whisper-v3:turbo (NPU)'}",
        "",
        f"Head-to-head over **{aggs[0]['n_clips'] if aggs else 0} FLEURS clips** "
        f"(4 EN + 13 RU). Scenario: `{scenario}`. Run: {raw['timestamp']}.",
        "",
        "All backends run **sequentially** on the single-tenant NPU. Latency = wall time of the",
        "transcribe call (after 2 warmups). Energy = RAPL package J/clip. RAM = peak RSS of the",
        "serving process. CPU-idle% = mean CPU idle fraction during the call (higher = more work",
        "offloaded off the CPU, i.e. on the NPU).",
        "",
        "| backend | model | EN WER | RU WER | median latency | J/clip | peak RAM | CPU-idle% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    def num(v, fmt, dash="n/a"):
        return format(v, fmt) if v is not None else dash
    for a in aggs:
        lines.append(
            f"| {a['backend']} | {a['model']} | {num(a['en_wer'], '.3f')} | {num(a['ru_wer'], '.3f')} | "
            f"{num(a['median_latency_s'], '.2f')}s | {num(a['mean_joules'], '.1f')} | "
            f"{rss_gb(a['peak_rss_kb'])} | {num(a['mean_idle_frac'] and a['mean_idle_frac']*100, '.1f')}% |"
        )
    if not raw["energy_readable"]:
        lines += ["", "Energy is `n/a`: the RAPL counter was root-only for this run "
                  "(`sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj` to enable)."]
    lines += [""]
    (RESULTS_DIR / f"{stem}.md").write_text("\n".join(lines))
    return RESULTS_DIR / f"{stem}.md", RESULTS_DIR / f"{stem}.json"


def load_corpus():
    refs_path = ROOT / "artifacts" / "wer_clips" / "refs.json"
    refs = json.loads(refs_path.read_text())
    clip_dir = ROOT / "artifacts" / "wer_clips"
    clips = [clip_dir / n for n in sorted(refs)]
    return clips, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="ours,flm")
    ap.add_argument("--scenario", default="scenarios/asr-whisper-small.toml")
    ap.add_argument("--model-name", default="parakeet", help="model name our engine serves")
    ap.add_argument("--flm-companion", default="qwen3:0.6b",
                    help="installed LLM to serve with -a 1 so Whisper loads without an auto-pull")
    ap.add_argument("--stem", default="whisper-small-vs-flm-turbo", help="output basename")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if not readable():
        print("[warn] RAPL energy counter not readable - joules will be n/a.", file=sys.stderr)

    clips, refs = load_corpus()
    which = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"Corpus: {len(clips)} clips. Backends (sequential): {which}")

    results, aggs = [], []
    try:
        for b in which:
            print(f"\n=== backend: {b} ===")
            if b == "ours":
                res = run_ours(args.scenario, clips, refs, args.model_name)
            elif b == "flm":
                res = run_flm(clips, refs, args.flm_companion)
            else:
                print(f"  unknown backend {b!r}, skipping", file=sys.stderr)
                continue
            results.append(res)
            aggs.append(aggregate(res))
    finally:
        stop_all_npu()

    md, js = write_outputs(results, aggs, args.scenario, args.stem, args.title)
    print(f"\nWrote {md}\n      {js}\n")
    for a in aggs:
        j = f"{a['mean_joules']:.1f}" if a["mean_joules"] is not None else "n/a"
        print(f"  {a['backend']:5s} EN={a['en_wer']:.3f} RU={a['ru_wer']:.3f} "
              f"lat={a['median_latency_s']:.2f}s J/clip={j} "
              f"RSS={a['peak_rss_kb']/1024/1024:.2f}GB idle={a['mean_idle_frac']*100:.1f}%")


if __name__ == "__main__":
    main()
