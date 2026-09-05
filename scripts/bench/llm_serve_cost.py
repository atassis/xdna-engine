#!/usr/bin/env python3
"""Price the LLM serving path: tok/s, ms/token, TTFT, and the delta over the bare decode rail.

Starts its own `npu serve` on a spare port, waits for it to be READY (not merely listening), and
refuses to report a number it did not actually obtain. Both of those are the failures that killed
two earlier ad-hoc attempts: one slept 25 s and measured a dead port (completion_tokens=0, wall
0.01 s), the other polled correctly and then died inside json.loads('') instead of saying the body
was empty. A timing harness that reports a number without asserting the request returned tokens is
worse than no harness.

  python3 scripts/bench/llm_serve_cost.py [--model qwen3-0.6b] [--port 18435] [--json OUT]

Env: NPU_LOCK (argv prefix, e.g. "/path/npu_lock.sh queue --"), BIN, XDNA_ENGINE_ROOT.
The device is single-tenant: stop the system service first.
"""
import argparse, json, os, shlex, subprocess, sys, time, urllib.error, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Measured on the same fused ELF by the bare Python rail (llm-decode-first-timing). The serving
# stack's own cost -- tokenizer, chat template, sampling, HTTP, SSE framing -- is the delta to this.
BARE_RAIL_MS_PER_TOKEN = (154.0, 178.0)


def post(url, body, timeout=600):
    """POST json, returning (status, parsed). Fails LOUD on an empty or non-json body."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    if not raw.strip():
        raise SystemExit(f"FATAL: {url} returned status {status} with an EMPTY body -- "
                         "the server is up but did not answer. Not a parse error.")
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"FATAL: {url} status {status}, body is not JSON: {raw[:400]!r}")


def wait_ready(port, proc, model, budget_s=600):
    """Listening is not ready: the first request must LOAD the model (weights + ELF registration),
    which is the slow part and the one a fixed sleep gets wrong. Poll /healthz, then prove the
    generate path answers by asking for a single token."""
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < budget_s:
        if proc.poll() is not None:
            raise SystemExit(f"FATAL: server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/healthz", timeout=5):
                break
        except Exception:
            time.sleep(0.5)
    else:
        raise SystemExit(f"FATAL: /healthz never came up within {budget_s}s")
    listen_s = time.time() - t0
    t1 = time.time()
    st, r = post(base + f"/v1/completions",
                 {"model": model, "prompt": "warm", "max_tokens": 1, "temperature": 0})
    if st != 200:
        raise SystemExit(f"FATAL: warm-up request failed {st}: {r}")
    if r["usage"]["completion_tokens"] < 1:
        raise SystemExit(f"FATAL: warm-up returned {r['usage']['completion_tokens']} tokens -- "
                         "the server answered but generated nothing")
    return listen_s, time.time() - t1


def buffered(base, model, prompt, max_tokens, think=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.time()
    st, r = post(base + "/v1/chat/completions", body)
    wall = time.time() - t0
    if st != 200:
        raise SystemExit(f"FATAL: chat/completions {st}: {r}")
    u = r["usage"]
    if u["completion_tokens"] == 0:
        raise SystemExit(f"FATAL: 200 with zero completion tokens -- {r}")
    # `wall / completion_tokens` is NOT ms-per-decoded-token and must not be compared to one.
    # This rail has no prefill, so the wall also contains one full decode step per PROMPT token;
    # at a 19-token prompt and an 8-token answer that inflates the per-token figure ~3x. Report
    # the mixed number under a name that says what it is, and let the streaming arm -- which can
    # separate TTFT from the steady state -- own the per-token cost.
    return {"wall_s": wall, "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "ms_per_step_incl_prompt": 1e3 * wall / (u["prompt_tokens"] + u["completion_tokens"]),
            "ms_of_wall_per_completion_token": 1e3 * wall / u["completion_tokens"],
            "request_tok_s": u["completion_tokens"] / wall,
            "finish": r["choices"][0]["finish_reason"]}


def streamed(base, model, prompt, max_tokens, think=False):
    """TTFT is the wall time to the first chunk carrying CONTENT -- not the first SSE frame, which
    is the role-only opener and arrives before any decode step has run."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    ttft, n, done = None, 0, False
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            d = json.loads(payload)
            if d["choices"][0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    wall = time.time() - t0
    if not done:
        raise SystemExit("FATAL: stream ended without [DONE]")
    if n == 0:
        raise SystemExit("FATAL: stream carried zero content chunks")
    return {"wall_s": wall, "ttft_s": ttft, "content_chunks": n,
            "ms_per_token_after_first": 1e3 * (wall - ttft) / max(n - 1, 1),
            "tok_s": n / wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--port", type=int, default=18435)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    # Overridable so the harness can A/B two artifacts: point ENGINE_TOML at a config whose
    # scenario names the other decode dir. Same harness, same request sequence, one variable.
    cfg = os.environ.get("ENGINE_TOML") or os.path.join(
        REPO, "scripts", "smoke", "llm_serve_smoke.engine.toml")
    bin_ = os.environ.get("BIN") or os.path.join(REPO, "rust/target/release/npu")
    if not os.access(bin_, os.X_OK):
        raise SystemExit(f"FATAL: no npu binary at {bin_}")
    env = dict(os.environ)
    env.setdefault("XDNA_ENGINE_ROOT", REPO)
    env.setdefault("LD_LIBRARY_PATH", os.path.expanduser("~/.local/lib/xdna-engine"))
    env["NPU_DISPATCH_LOG"] = "1"
    prefix = shlex.split(os.environ.get("NPU_LOCK", ""))
    if not prefix:
        print("WARN: NPU_LOCK unset -- running unserialized on a single-tenant device", file=sys.stderr)

    argv = prefix + [bin_, "--config", cfg, "serve", "--port", str(a.port)]
    log = open(os.path.join(REPO, "rust/target/llm_serve_cost.serve.log"), "w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
    base = f"http://127.0.0.1:{a.port}"
    out = {"model": a.model, "port": a.port,
           "amdxdna_srcversion": open("/sys/module/amdxdna/srcversion").read().strip()
           if os.path.exists("/sys/module/amdxdna/srcversion") else None,
           "power_mode": "UNPINNED (not set by this harness); all figures are wall clock"}
    try:
        listen_s, load_s = wait_ready(a.port, proc, a.model)
        out["server_listen_s"] = round(listen_s, 2)
        out["first_request_load_s"] = round(load_s, 2)

        out["buffered"] = {}
        for mt in (16, 64, 128):
            out["buffered"][str(mt)] = buffered(base, a.model, "What is 2+2?", mt)
        out["streaming"] = streamed(base, a.model, "What is 2+2?", 64)

        # There is no prefill: every prompt token costs a full decode step, so TTFT should GROW
        # with prompt length. Stated as a prediction and tested, because the flat-in-n_past result
        # on this rail already falsified one intuition of exactly this shape.
        out["ttft_vs_prompt_len"] = []
        for words in (4, 32, 128, 400):
            p = " ".join(["token"] * words)
            s = streamed(base, a.model, p, 4)
            b = buffered(base, a.model, p, 4)
            out["ttft_vs_prompt_len"].append(
                {"words": words, "prompt_tokens": b["prompt_tokens"], "ttft_s": s["ttft_s"]})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    serve_log = open(os.path.join(REPO, "rust/target/llm_serve_cost.serve.log")).read()
    out["dispatch_lines"] = [l for l in serve_log.splitlines()
                             if "[dispatch]" in l or "dispatches " in l or "hw_contexts" in l]

    # Compare like with like: the bare-rail figure is the cost of ONE decode step, so the served
    # counterpart is the streaming steady state after the first token, never a whole-request wall
    # divided by completion tokens.
    lo, hi = BARE_RAIL_MS_PER_TOKEN
    ms = out["streaming"]["ms_per_token_after_first"]
    out["vs_bare_rail"] = {
        "bare_ms_per_decode_step": [lo, hi],
        "served_ms_per_decode_step_steady_state": round(ms, 1),
        "serving_overhead_ms_per_token": [round(ms - hi, 1), round(ms - lo, 1)],
        "note": "streaming, after the first content chunk; tokenizer/template/sampling/HTTP/SSE "
                "are inside this number and the bare rail has none of them",
    }

    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
