#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lever #3 vector-(b) Milestone-0 probe: a single GEMM projection as a fused full ELF, parameterised
by the batch width N, to measure whether batching B token-columns through ONE shared weight read
amortises the per-token dispatch cost (the load-bearing question of batched decode — design spec
internal notes).

The decode projections are out[M] = W[M,K] @ x[K]  (matrix x vector, M=1/token). Batching B streams
that share one weight is out[M,N] = W[M,K] @ X[K,N] — the IRON GEMM with a skinny N=B. This builds
exactly that single GEMM (no LN/bias/GELU — we want the cleanest weight-read-amortisation signal),
using the fattest decode weight (Whisper fc1: M=FF=3072, K=D=768, the 4.72 MB/layer buffer) so the
DDR-bandwidth term dominates and the amortisation (or its absence) is unambiguous.

Decisive measurement (run on device via fused_elf_probe FUSED_TIME): dispatch_ms at each N.
  per-token cost = dispatch_ms / N.
  GO   if per-token cost FALLS as N grows (dispatch_ms sub-linear in N => weight read amortised).
  KILL if per-token cost is ~flat (dispatch_ms ~ linear in N => no amortisation on this HW).

IRON GEMM arg order (get_arg_spec): A[M,K] (the matrix, here the resident weight W), B[K,N] (the
activation X), C[M,N] (out). So the runlist tuple is (gemm, "W", "X", "out"); W is a resident weight,
X is the per-token input. Output golden is the same bf16 dataflow the device runs (gate <= 0.08 in
fused_elf_probe). GEMM tiling constraint: N % (tile_n * num_aie_columns) == 0, AND the bf16 vectorized mm.cc kernel
requires tile_n % 16 == 0 (static_assert n % (2*t), t=8). With the defaults here (tile_n=16,
num_aie_columns=1) N must be a multiple of 16, so the skinny-N sweep {16,32,64,128} is valid on one
fixed array config. (N<16 per column would need a scalar/custom kernel — itself a finding.)

Run inside the IRON env (newstack_compat first; aiebu-asm on PATH). See scripts/build_gemm_probe.sh.
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 — MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf
from iron.operators.gemm.op import GEMM

BF16 = ml_dtypes.bfloat16
D = 768    # Whisper-small model dim (= GEMM K)
FF = 3072  # Whisper-small FFN inner dim (= GEMM M); fc1 weight is the fattest decode buffer


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(wdir, layer, name):
    return np.load(os.path.join(wdir, f"L{layer}", f"{name}.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="whisper_decoder weights dir (uses L<layer>/fc1.weight)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, required=True, help="batch width (token-columns); GEMM N")
    ap.add_argument("--M", type=int, default=FF)
    ap.add_argument("--K", type=int, default=D)
    ap.add_argument("--tile-m", type=int, default=64)
    ap.add_argument("--tile-k", type=int, default=64)
    ap.add_argument("--tile-n", type=int, default=16)  # bf16 vectorized mm.cc needs tile_n % 16 == 0
    ap.add_argument("--num-cols", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fuse-residual", action="store_true",
                    help="O7 (KILLED — 2-input-DMA-channel wall): preload C with a residual input")
    ap.add_argument("--m-stationary", action="store_true",
                    help="O6: M-stationary dataflow (columns split M, B broadcast) -> all 32 cores at skinny N")
    args = ap.parse_args()
    M, K, N = args.M, args.K, args.N
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L = args.layer

    # Real fc1 weight for representative DDR traffic. npy is [D, FF]; GEMM wants A=[M,K]=[FF,D]=Wfc1.T.
    Wfc1 = npy(args.weights, L, "fc1.weight").astype(np.float32)  # [768, 3072] = [D, FF]
    assert Wfc1.shape == (D, FF), f"fc1.weight is {Wfc1.shape}, expected {(D, FF)}"
    assert (M, K) == (FF, D), "this probe uses the fc1 shape (M=FF=3072, K=D=768)"
    W = bf16(Wfc1.T.copy())  # [M, K] = [3072, 768]

    rng = np.random.default_rng(args.seed)
    X = bf16(rng.standard_normal((K, N)).astype(np.float32))  # [K, N] activation (N token-columns)
    R = bf16(rng.standard_normal((M, N)).astype(np.float32)) if args.fuse_residual else None  # [M,N] residual

    ctx = AIEContext()
    gemm_kw = dict(
        M=M, K=K, N=N,
        tile_m=args.tile_m, tile_k=args.tile_k, tile_n=args.tile_n,
        num_aie_columns=args.num_cols,
        context=ctx,
    )
    if args.m_stationary:
        gemm_kw["m_stationary"] = True
    if args.fuse_residual:
        gemm_kw["fuse_residual"] = True
    gemm = GEMM(**gemm_kw)
    if args.fuse_residual:
        runlist = [(gemm, "W", "X", "R", "out")]  # A=W, B=X, RESID=R, C=out
        in_args = ["X", "R"]
    else:
        runlist = [(gemm, "W", "X", "out")]  # A=W[M,K], B=X[K,N], C=out[M,N]
        in_args = ["X"]
    fused = FusedMLIROperator("gemmprobe", runlist, input_args=in_args, output_args=["out"], context=ctx)
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    names = ("X", "out", "W") + (("R",) if args.fuse_residual else ())
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    print(f"GEMM M={M} K={K} N={N} (tile {args.tile_m}x{args.tile_k}x{args.tile_n}, cols={args.num_cols})")
    print("buffer_sizes (in,out,scratch) =", fused.buffer_sizes)
    for n, v in lay.items():
        print(f"  {n}: type={v[0]} off={int(v[1])} len={int(v[2])}")

    # device-faithful golden: out = W @ X (+ residual) in bf16 (same dataflow precision the device runs)
    out_g = W.astype(np.float32) @ X.astype(np.float32)  # [M, N]
    if args.fuse_residual:
        out_g = out_g + R.astype(np.float32)  # O7: C = residual + W@X
    out_g = bf16(out_g)

    def wbuf(name, vals):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    wbuf("W", W.reshape(-1))
    wbuf("X", X.reshape(-1))
    if args.fuse_residual:
        wbuf("R", R.reshape(-1))
    wbuf("out", out_g.reshape(-1))
    with open(os.path.join(args.out, "gemmprobe.elf"), "wb") as f:
        f.write(elf_bytes)

    weight_bytes = int(W.size * 2)
    meta = {
        "elf": "gemmprobe.elf",
        "kernel_name": "main:sequence",
        "input_size": int(in_sz),
        "output_size": int(out_sz),
        "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": in_args,
        "weights": ["W"],
        "output": "out",
        "dims": {"M": M, "K": K, "N": N, "layer": L, "weight_bytes": weight_bytes},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote ELF ({len(elf_bytes)}B) + buffers + meta.json to {args.out}  (weight={weight_bytes/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
