#!/usr/bin/env python3
"""Synthetic gate for scripts/buildtrace_merge.py -- no real build, no NPU.

Builds a tiny fake build dir: one aiecc trace.json (one edge span containing 24 identical
"aie-place-tiles" pass spans on different anchors + 1 differently-named pass, so the
same-pass-on-N-devices aggregation has something to prove), a kcc.jsonl with two
overlapping compiles (checks lane assignment), and a generator.json span. Then checks
trace.json is valid JSON covering every input event, and tree.txt aggregates the 24 spans
into one line with count=24 rather than 24 lines.

    python3 scripts/tests/buildtrace_merge_test.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MERGE = os.path.join(REPO, "scripts", "buildtrace_merge.py")

failures = []


def check(cond, msg):
    print(f"  {'ok' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


with tempfile.TemporaryDirectory() as d:
    edge_start = 1_000_000_000_000  # us
    events = [{"name": "input_with_symbols.mlir[0]", "ph": "X", "pid": 0, "tid": 0,
               "ts": edge_start, "dur": 10_000, "args": {}}]
    for i in range(24):
        events.append({"name": "aie-lower-broadcast-packet", "ph": "X", "pid": 0,
                       "tid": 1 + i % 4, "ts": edge_start + 100 * i, "dur": 50 + i,
                       "args": {"pass": "aie-lower-broadcast-packet", "anchor_op": "aie.device",
                                "anchor": f"({i},0)"}})
    events.append({"name": "aie-place-tiles", "ph": "X", "pid": 0, "tid": 5,
                   "ts": edge_start + 6000, "dur": 200, "args": {"pass": "aie-place-tiles"}})
    with open(os.path.join(d, "a.trace.json"), "w") as f:
        json.dump({"traceEvents": events,
                   "otherData": {"processStartEpochUs": edge_start}}, f)

    kcc_start = edge_start / 1e6
    with open(os.path.join(d, "kcc.jsonl"), "w") as f:
        f.write(json.dumps({"compiler": "clang", "source": "k0.cc", "rc": 0,
                            "start": kcc_start, "end": kcc_start + 0.01}) + "\n")
        f.write(json.dumps({"compiler": "clang", "source": "k1.cc", "rc": 0,
                            "start": kcc_start + 0.002, "end": kcc_start + 0.012}) + "\n")

    with open(os.path.join(d, "generator.json"), "w") as f:
        json.dump({"start": kcc_start - 0.02, "end": kcc_start - 0.005, "name": "generator"}, f)

    out = os.path.join(d, "trace.json")
    tree = os.path.join(d, "tree.txt")
    r = subprocess.run([sys.executable, MERGE, d, "-o", out, "--tree", tree],
                       capture_output=True, text=True)
    check(r.returncode == 0, f"merge exits 0 (stderr: {r.stderr[:400]})")

    with open(out) as f:
        merged = json.load(f)
    x_events = [e for e in merged["traceEvents"] if e["ph"] == "X"]
    # 1 edge + 24 + 1 pass spans, + 2 kcc, + 1 generator = 29.
    check(len(x_events) == 29, f"trace.json carries every input span, got {len(x_events)}")
    check(json.loads(json.dumps(merged)) == merged, "trace.json round-trips json.dumps/loads")

    with open(tree) as f:
        tree_text = f.read()
    check("x24" in tree_text, "the 24 identical-name pass spans collapse to one 'x24' line")
    check(tree_text.count("aie-lower-broadcast-packet") == 1,
          "the pass name appears exactly once in tree.txt (aggregated, not repeated 24x)")
    check("aie-place-tiles" in tree_text, "the lone differently-named pass still gets its own line")
    check("kernel-compiles" in tree_text, "kcc.jsonl contributes a kernel-compiles section")
    check("generator" in tree_text, "generator.json contributes a generator section")

    lane_events = [e for e in merged["traceEvents"]
                  if e.get("pid") == 900]  # PID_KCC
    tids = {e["tid"] for e in lane_events}
    check(len(tids) == 2, f"two overlapping compiles land on two separate lanes, got {tids}")

print()
if failures:
    print(f"GATE RED ({len(failures)} failed)")
    sys.exit(1)
print("GATE GREEN")
