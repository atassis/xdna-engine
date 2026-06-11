#!/usr/bin/env python3
"""Verify the npu_asr Encoder end-to-end vs the static-ONNX reference tensors.

  --backend host   pure numpy (correctness of the package; no NPU)
  --backend npu    heavy ops on XDNA2 (matmul/dwconv/layernorm/silu)
  --blocks N       run only the first N Conformer blocks (default 16)

Checks: subsampling output, per-block output (free-running = chained, as the real
encoder runs), and the final `encoded` tensor. Run NPU mode on a freed NPU.
"""
import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_asr import Ops, Encoder, WeightStore
from npu_asr.block import conformer_block
from npu_asr.dtypes import bf16, f32
from npu_asr.verify import rel
from npu_asr import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["host", "npu"], default="host")
    ap.add_argument("--blocks", type=int, default=C.N_BLOCKS)
    ap.add_argument("--tol", type=float, default=0.08)
    ap.add_argument("--accurate", action="store_true",
                    help="keep LayerNorm+SiLU on host (approx kernels off NPU) for accuracy")
    ap.add_argument("--single-core", action="store_true", help="single_core matmul (vs whole_array)")
    a = ap.parse_args()

    ws = WeightStore()
    if a.backend == "host":
        ops = Ops.host()
    else:
        ops = Ops.on_npu(layernorm=not a.accurate, silu=not a.accurate,
                         whole_matmul=not a.single_core)
    enc = Encoder(ws, ops)
    print(f"backend={a.backend}  placement={ops.placement()}  blocks={a.blocks}\n")

    audio = ws.ref("audio_signal")[0]                 # [64,1600]
    t0 = time.perf_counter()

    # 1) subsampling
    sub = enc.subsample(audio)                         # [400,768]
    print(f"  {'subsample':14s} rel={rel(sub, ws.ref('block_in')[0]):.2e}")

    # 2) block stack (free-running / chained = the real encoder)
    x = bf16(sub)
    worst = 0.0
    for i in range(a.blocks):
        x = conformer_block(x, ws.block(i), ops, ws.cos, ws.sin)
        r = rel(f32(x), ws.ref(f"out_L{i}")[0])
        worst = max(worst, r)
        if i < 3 or i >= a.blocks - 2 or r > a.tol:
            print(f"  block {i:<2d}      rel={r:.2e}{'  **OFF**' if r > a.tol else ''}")

    # 3) final encoded (only meaningful at full depth)
    encoded = f32(x).T                                 # [768,400]
    dt = time.perf_counter() - t0
    if a.blocks == C.N_BLOCKS:
        renc = rel(encoded, ws.ref("encoded")[0])
        print(f"\n  {'ENCODED':14s} rel={renc:.2e} vs static ONNX  ({'PASS' if renc < a.tol else 'FAIL'})")
    print(f"  worst per-block rel = {worst:.2e}")
    print(f"  wall time ({a.backend}, {a.blocks} blocks) = {dt:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
