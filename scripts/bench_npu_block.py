#!/usr/bin/env python3
"""Latency reality-check for the host-orchestrated GigaAM-v3 encoder on XDNA2.

Times each NPU op category in isolation (warmup + N iters = the reliable per-dispatch
number), then projects to one block and to the 16-block encoder by op counts. Reports
FLOPs + effective throughput, and the gap to the targets:
  - Whisper-NPU floor to beat: ~3.3 s (11.9 s clip)
  - GigaAM-v3 on CPU:          ~0.89 s

This measures the UNFUSED, dispatch-bound path (one xclbin per op, DMA round-trip each)
— deliberately the worst case. The ObjectFifo-fused single-xclbin path removes per-op
dispatch+DMA and is the real shippable number; we bound it here from FLOPs.

Run on a freed NPU:  .venv-iron/bin/python scripts/bench_npu_block.py
"""
import sys, time
import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, "scripts")
from run_npu_block0 import NpuMatmul, NpuDwconv, NpuLayerNorm, NpuSilu, bf16, T, C, NH, HD

ITERS = 50


def timeit(fn, iters=ITERS):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    mm = NpuMatmul(); dw = NpuDwconv(); ln = NpuLayerNorm(); si = NpuSilu()
    rng = np.random.RandomState(0)
    A768 = bf16(rng.standard_normal((T, 768)))
    A3072 = bf16(rng.standard_normal((T, 3072)))
    W768x768 = bf16(rng.standard_normal((768, 768)))
    W768x1536 = bf16(rng.standard_normal((768, 1536)))
    W3072x768 = bf16(rng.standard_normal((3072, 768)))
    W768x3072 = bf16(rng.standard_normal((768, 3072)))
    glu = bf16(rng.standard_normal((C, T))); taps = bf16(rng.standard_normal((C, 5)))
    x768 = bf16(rng.standard_normal((T, C)))
    s307 = bf16(rng.standard_normal((C, T))); s1228 = bf16(rng.standard_normal((T, 4 * C)))

    # (label, per-op ms, count per block, flops per op)
    def fl(M, K, N): return 2.0 * M * K * N
    ops = [
        ("matmul 768x768  (q/k/v/out,pw2)", timeit(lambda: mm.mm(A768, W768x768)),   5, fl(400,768,768)),
        ("matmul 768x1536 (pointwise1)",    timeit(lambda: mm.mm(A768, W768x1536)),  1, fl(400,768,1536)),
        ("matmul 3072x768 (ffn linear2)",   timeit(lambda: mm.mm(A3072, W3072x768)), 2, fl(400,3072,768)),
        ("matmul 768x3072 (ffn linear1,2x)",timeit(lambda: mm.mm(A768, W768x3072)),  2, fl(400,768,3072)),
        ("dwconv k5",                       timeit(lambda: dw(glu, taps)),           1, 2.0*C*T*5),
        ("layernorm 400x768",               timeit(lambda: ln.normalize(x768)),      6, 0.0),
        ("silu 307200 (conv swish)",        timeit(lambda: si.run(s307)),            1, 0.0),
        ("silu 1228800 (ffn x2)",           timeit(lambda: si.run(s1228)),           2, 0.0),
    ]

    print(f"{'op':34s} {'ms/op':>8s} {'x/blk':>6s} {'ms/blk':>8s}")
    blk_ms = 0.0; blk_flop = 0.0
    for label, ms, cnt, flop in ops:
        print(f"{label:34s} {ms:8.3f} {cnt:6d} {ms*cnt:8.2f}")
        blk_ms += ms * cnt; blk_flop += flop * cnt

    # attention score/ctx are on host today — note separately (small FLOPs)
    attn_flop = 2 * (NH * fl(400, HD, 400))  # q@k^T + probs@v across heads
    enc_ms = blk_ms * 16
    enc_flop = (blk_flop + attn_flop) * 16

    print(f"\n[per block] NPU dispatch time = {blk_ms:.1f} ms  (matmul/dwconv/LN/SiLU on NPU; "
          f"attn score/ctx + RoPE/softmax/residual on host, not counted)")
    print(f"[encoder x16] NPU dispatch time ≈ {enc_ms/1000:.2f} s  (UNFUSED, dispatch-bound)")
    print(f"[flops] matmul ≈ {blk_flop/1e9:.1f} GFLOP/blk, encoder ≈ {enc_flop/1e9:.0f} GFLOP")
    print(f"[throughput] unfused effective ≈ {enc_flop/(enc_ms/1000)/1e9:.0f} GFLOP/s")
    print(f"[targets] Whisper-NPU floor ~3.3 s | GigaAM-CPU 0.89 s")
    # fused floor: compute-bound estimate at a conservative sustained bf16 rate
    for tops in (2e12, 5e12, 10e12):
        print(f"[fused floor @ {tops/1e12:.0f} TFLOP/s sustained] ≈ {enc_flop/tops*1000:.1f} ms compute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
