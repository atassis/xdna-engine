#!/usr/bin/env python3
"""Chrome Trace Event Format view of one decode dispatch, from the design's own shim BDs.

This is the MODEL half of the two-sided instrument: what the design says it will move, in
execution order, with no device and no observer effect. Every span here is derived, and the
output labels it so -- a measured trace is a separate overlay, not this file.

  python3 scripts/decode_trace_model.py <decode_fused.mlir> -o decode_model.json
  then: chrome://tracing (Load) or https://ui.perfetto.dev

Tracks:
  model: transport   one span per operator execution, width = its byte-implied floor time
  model: DDR bytes   cumulative counter across the dispatch
  measured: array    a single span at the measured array time, for scale

The gap between the last two IS the open question: the transport model accounts for 6.72 ms of
a 39.59 ms dispatch, and a static byte count cannot say whether the rest is bytes moving slowly
or a per-execution floor.
"""
import argparse, collections, json, re, sys

ELEM = {"bf16": 2, "f32": 4, "i8": 1, "i32": 4}
READ_GBPS = 52.69e9   # npu-lpddr-read-scaling-and-peak, TURBO-pinned pure-read ceiling
MEASURED_MS = 39.59   # whisper-fused-decode-step-cost dispatch term (PREBUILT 2026-08-17 ELF)

BD = re.compile(r"aie\.dma_bd\(%(\w+)\s*:\s*memref<(\d+)x(bf16|f32|i8|i32)>[^)]*?"
                r"len\s*=\s*(\d+)\s+sizes\s*=\s*\[([^\]]*)\]")
DEV = re.compile(r"aie\.device\(\w+\)\s*@(\w+)\s*\{")
CFG = re.compile(r"aiex\.configure\s+@(\w+)\s*\{")


def device_bodies(src):
    out = {}
    for m in DEV.finditer(src):
        depth = 0
        for j in range(m.end() - 1, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
        out[m.group(1)] = src[m.end():j]
    return out


def device_bytes(body):
    total = 0
    for b in BD.finditer(body):
        _arg, _n, ty, ln, sizes = b.groups()
        sz = [int(x) for x in sizes.split(",")]
        outer = sz[0] if len(sz) == 4 else 1
        total += outer * int(ln) * ELEM[ty]
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("-o", "--out", default="decode_model.json")
    a = ap.parse_args()

    src = open(a.mlir).read()
    devs = device_bodies(src)
    top = src[src.rindex("aie.device(npu2) {"):]

    # One execution is one aiex.run; a single aiex.configure block may hold many.
    order, cur = [], None
    for line in top.splitlines():
        m = CFG.search(line)
        if m:
            cur = m.group(1)
        elif "aiex.run" in line and cur:
            order.append(cur)

    nbytes = {op: device_bytes(devs[op]) for op in set(order) if op in devs}

    ev, t_us, cum = [], 0.0, 0
    ev.append({"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "model: transport (derived)"}})
    ev.append({"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "measured: array time"}})
    for op in sorted(nbytes, key=lambda o: -nbytes[o]):
        ev.append({"name": "thread_name", "ph": "M", "pid": 0, "tid": abs(hash(op)) % 10**6,
                   "args": {"name": op}})

    for i, op in enumerate(order):
        b = nbytes.get(op, 0)
        dur_us = b / READ_GBPS * 1e6
        ev.append({"name": op, "cat": "execution", "ph": "X", "pid": 0,
                   "tid": abs(hash(op)) % 10**6, "ts": t_us, "dur": max(dur_us, 0.05),
                   "args": {"exec_index": i, "ddr_bytes": b, "MB": round(b / 1e6, 4),
                            "note": "duration = bytes / 52.69 GB/s (MODEL, not measured)"}})
        t_us += dur_us
        cum += b
        ev.append({"name": "DDR bytes", "ph": "C", "pid": 0, "ts": t_us,
                   "args": {"cumulative_MB": round(cum / 1e6, 3)}})

    ev.append({"name": "array time (measured, whisper-fused-decode-step-cost)", "cat": "measured",
               "ph": "X", "pid": 1, "tid": 0, "ts": 0.0, "dur": MEASURED_MS * 1000,
               "args": {"ms": MEASURED_MS,
                        "modelled_transport_ms": round(t_us / 1000, 3),
                        "unattributed_ms": round(MEASURED_MS - t_us / 1000, 3),
                        "executions": len(order)}})

    json.dump({"traceEvents": ev, "displayTimeUnit": "ms",
               "otherData": {
                   "source": a.mlir,
                   "executions_per_dispatch": len(order),
                   "ddr_MB_per_dispatch": round(cum / 1e6, 2),
                   "modelled_transport_ms": round(t_us / 1000, 3),
                   "measured_array_ms": MEASURED_MS,
                   "read_ceiling_GBps": READ_GBPS / 1e9,
                   "WARNING": "transport spans are DERIVED from static BD sizes, not measured",
               }}, open(a.out, "w"))

    print(f"{len(order)} executions, {cum/1e6:.2f} MB, modelled transport {t_us/1000:.2f} ms "
          f"vs measured {MEASURED_MS} ms -> {MEASURED_MS - t_us/1000:.2f} ms unattributed")
    print(f"wrote {a.out}  (chrome://tracing or ui.perfetto.dev)")


if __name__ == "__main__":
    main()
