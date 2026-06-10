#!/usr/bin/env python3
"""Bench RU+EN ASR models on the voxd fixture. Measures warm inference
latency (proxy for latency-after-release) + prints transcripts for quality eyeballing."""
import sys, time, statistics, traceback
import onnx_asr

WAV = "~/voxd/tests/fixtures/sample-ru-en.wav"
DUR = 11.92  # seconds, from soundfile

# (display name, onnx-asr id, quantization or None)
MODELS = [
    ("parakeet-tdt-0.6b-v3 (RU+EN, TDT)", "nemo-parakeet-tdt-0.6b-v3", "int8"),
    ("gigaam-v3-rnnt (RU-native, RNNT)",  "gigaam-v3-rnnt",            None),
    ("t-one (RU streaming CTC)",          "t-tech/t-one",              None),
    ("nemo-fastconformer-ru-rnnt (RU)",   "nemo-fastconformer-ru-rnnt", None),
    ("canary-1b-v2 (RU+EN, AED)",         "nemo-canary-1b-v2",         "int8"),
]

def bench(name, model_id, quant):
    print(f"\n{'='*70}\n{name}  [{model_id}{', '+quant if quant else ''}]")
    try:
        t0 = time.perf_counter()
        kw = {"providers": ["CPUExecutionProvider"]}
        if quant:
            kw["quantization"] = quant
        m = onnx_asr.load_model(model_id, **kw)
        load_s = time.perf_counter() - t0
        # warm-up (lazy init / first-run cost)
        txt = m.recognize(WAV)
        # timed runs
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            txt = m.recognize(WAV)
            ts.append(time.perf_counter() - t)
        med = statistics.median(ts)
        print(f"  load: {load_s:.1f}s | warm infer: {med:.2f}s | RTF: {med/DUR:.3f} | xRT: {DUR/med:.1f}x")
        print(f"  TEXT: {txt!r}")
        return (name, med, DUR/med, txt)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return (name, None, None, f"FAILED: {e}")

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = []
    for name, mid, q in MODELS:
        if only and not any(o in mid for o in only):
            continue
        results.append(bench(name, mid, q))
    print(f"\n\n{'='*70}\nSUMMARY (vs FLM whisper-v3:turbo ~3.4s on this 11.9s clip)")
    for name, med, xrt, _ in results:
        if med:
            print(f"  {name:42s} {med:5.2f}s  {xrt:5.1f}xRT")
        else:
            print(f"  {name:42s}  FAILED")
