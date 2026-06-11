#!/usr/bin/env python3
"""Verify the FUSED GigaAM-v3 encoder vs the static-ONNX reference.

Uses npu_asr.fused.FusedBlock — the Conformer block with matmul-heavy ops fused on
the NPU (FFN ×2, q/k/v/out, pointwise1/2 as whole-array+epilogue dispatches; dwconv
on NPU) and cheap glue on host. Subsampling (im2col) on host.

  --blocks N   run only the first N blocks (default 16). N=1 checks the fused block vs out_L0.
Run on a freed NPU.  .venv-iron/bin/python scripts/verify_fused_encoder.py --blocks 1
"""
import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_asr import WeightStore, config as C
from npu_asr.device import NpuDevice
from npu_asr.fused import FusedBlock
from npu_asr.encoder import subsample
from npu_asr.dtypes import bf16, f32
from npu_asr.verify import rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=C.N_BLOCKS)
    ap.add_argument("--tol", type=float, default=0.08)
    ap.add_argument("--iters", type=int, default=5, help="steady-state timing iters (warm)")
    a = ap.parse_args()

    ws = WeightStore(C.ARTIFACTS)             # artifacts/encoder
    dev = NpuDevice.get()
    tb = time.perf_counter()
    blocks = [FusedBlock(dev, ws.block(i), ws.cos, ws.sin) for i in range(a.blocks)]
    build_s = time.perf_counter() - tb         # cold: weights folded/K-augmented + synced ONCE
    print(f"fused encoder: {a.blocks} blocks; matmul-heavy ops on NPU, glue on host")
    print(f"  build (weights pre-folded/synced once) = {build_s:.2f} s\n")

    audio = ws.ref("audio_signal")[0]

    def infer():
        x = bf16(subsample(audio, ws.pre_encode))
        outs = []
        for blk in blocks:
            x = blk.forward(x); outs.append(x)
        return outs

    # correctness pass (also warms the device)
    outs = infer()
    worst = 0.0
    for i in range(a.blocks):
        r = rel(f32(outs[i]), ws.ref(f"out_L{i}")[0]); worst = max(worst, r)
        if i < 2 or i >= a.blocks - 2 or r > a.tol:
            print(f"  block {i:<2d}    rel={r:.2e}{'  **OFF**' if r > a.tol else ''}")
    if a.blocks == C.N_BLOCKS:
        renc = rel(f32(outs[-1]).T, ws.ref("encoded")[0])
        print(f"\n  {'ENCODED':12s} rel={renc:.2e} vs static ONNX  ({'PASS' if renc < a.tol else 'FAIL'})")
    print(f"  worst per-block rel = {worst:.2e}")

    # steady-state (warm) latency + NPU-vs-host split
    from npu_asr.fused import NPU_DISPATCH_S, NPU_DISPATCH_N, reset_npu_prof
    infer()  # extra warmup
    reset_npu_prof()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        infer()
    warm = (time.perf_counter() - t0) / a.iters
    npu = NPU_DISPATCH_S[0] / a.iters
    ndisp = NPU_DISPATCH_N[0] // a.iters
    print(f"  STEADY-STATE inference ({a.blocks} blocks, warm) = {warm*1e3:.0f} ms/run "
          f"(vs Whisper {C.TARGET_WHISPER_S*1e3:.0f} ms, CPU {C.TARGET_CPU_S*1e3:.0f} ms)")
    print(f"  split: NPU matmul dispatch {npu*1e3:.0f} ms ({ndisp} dispatches) | "
          f"host (glue+numpy+dwconv) {(warm-npu)*1e3:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
