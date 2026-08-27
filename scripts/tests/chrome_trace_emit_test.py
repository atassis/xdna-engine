#!/usr/bin/env python3
"""CPU gate for scripts/chrome_trace_emit.py -- no NPU, no real capture needed.

Two tiny synthetic mlir-aie trace_to_json-shaped fixtures (fixtures/chrome_trace/) stand in for
two sequential hw-context dispatches. Checks the three things a wrong merge would get silently
wrong: cycle->us scaling, the unclosed-B-at-buffer-end drop, and that a hw-context switch shows
up as a real time gap (not two captures overlapping because both started at cycle 0).

    python3 scripts/tests/chrome_trace_emit_test.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))
FIX = os.path.join(HERE, "fixtures", "chrome_trace")

import chrome_trace_emit as C  # noqa: E402

failures = []


def check(cond, msg):
    print(f"  {'ok' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


# clock_hz = 1e6 makes 1 cycle == 1 us, so expected values are the raw fixture numbers.
doc = C.build(
    [os.path.join(FIX, "capture_ln.json"), os.path.join(FIX, "capture_gelu.json")],
    clock_hz=1e6, labels=["ln", "gelu"], switch_us=5.0, clip_seconds=["clip0=0.05"],
)
events = doc["traceEvents"]

check(json.loads(json.dumps(doc)) == doc, "output round-trips through json.dumps/loads")

xs = [e for e in events if e["ph"] == "X"]
# capture_ln.json: 2 closed intervals on tid0 + 1 on tid1 = 3. capture_gelu.json: 1 closed
# interval + 1 unclosed trailing B that must be dropped, not extrapolated. + 1 clip-seconds bar.
check(len(xs) == 5, f"5 complete events total (3 ln + 1 gelu + 1 clip-bar), got {len(xs)}")
check(sum(1 for e in xs if e["pid"] == 0) == 3, "3 complete events on hw-context 0 (ln)")
check(sum(1 for e in xs if e["pid"] == 1) == 1, "1 complete event on hw-context 1 (gelu, 1 dropped)")

ln_events = sorted((e for e in xs if e["pid"] == 0), key=lambda e: e["ts"])
check(ln_events[0]["ts"] == 0.0 and ln_events[0]["dur"] == 10.0, "ln first interval [0,10)us")
check(ln_events[-1]["ts"] == 15.0 and ln_events[-1]["dur"] == 10.0, "ln last interval [15,25)us")

gelu_event = next(e for e in xs if e["pid"] == 1)
# ln spans [0,25) us; +5us switch gap -> gelu's [0,8) local interval starts at ts=30.
check(gelu_event["ts"] == 30.0 and gelu_event["dur"] == 8.0,
      f"gelu event offset past the switch gap (ts={gelu_event['ts']}, want 30.0)")

switches = [e for e in events if e.get("cat") == "context_switch"]
check(len(switches) == 1, f"exactly one context-switch marker, got {len(switches)}")
if switches:
    m = switches[0]
    check(m["ph"] == "i" and m["s"] == "g", "context-switch marker is a global instant event")
    check(m["ts"] == 30.0, f"marker sits at the gap's end (ts={m['ts']}, want 30.0)")
    check(m["args"]["gap_us"] == 5.0, f"marker records the 5us gap (got {m['args']['gap_us']})")
    check(m["args"]["from"] == "ln" and m["args"]["to"] == "gelu", "marker names both contexts")

check(doc["otherData"]["switch_us_source"] == "user-provided",
      "explicit --switch-us is tagged user-provided, not guessed")

proc_names = {(e["pid"]): e["args"]["name"] for e in events if e["name"] == "process_name"}
check(proc_names == {0: "ln", 1: "gelu", -1: "wall-clock: clip0"},
      f"process_name rows are the hw-context labels plus the clip-seconds bar, got {proc_names}")

clip_rows = [e for e in events if e.get("ph") == "X" and e["pid"] < 0]
check(len(clip_rows) == 1 and clip_rows[0]["dur"] == 50000.0,
      "clip-seconds row is a separate pid, ts_us = seconds * 1e6")

# No --switch-us given at all -> gap defaults to 0 and is tagged unmeasured, never guessed.
doc2 = C.build(
    [os.path.join(FIX, "capture_ln.json"), os.path.join(FIX, "capture_gelu.json")],
    clock_hz=1e6, labels=["ln", "gelu"], switch_us=None, clip_seconds=[],
)
check(doc2["otherData"]["switch_us_source"] == "unmeasured-placeholder (0)",
      "omitted --switch-us is tagged unmeasured-placeholder, not silently real")
gelu2 = next(e for e in doc2["traceEvents"] if e["ph"] == "X" and e["pid"] == 1)
check(gelu2["ts"] == 25.0, f"with no gap, gelu starts right where ln ends (ts={gelu2['ts']})")

print()
if failures:
    print(f"GATE RED ({len(failures)} failed)")
    sys.exit(1)
print("GATE GREEN")
