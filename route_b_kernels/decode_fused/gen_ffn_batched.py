#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batched fused decode — Task 1: the Whisper FFN block for B token-streams at once, as a fused full ELF.

Forked from gen_ffn.py (M=1). The whole point is to prove the LAYOUT strategy for batched decode
([[lever3-batched-gemm-amortizes]] proved GEMM-N=B amortises; this proves the inter-op plumbing):

  X[B,D] -> LN(num_channels=B) -> fc1 GEMM-N=B -> +bias -> GELU(num_channels=B) -> fc2 GEMM-N=B -> +bias

Everything stays STREAM-MAJOR `[B, feature]` so the channeled ops (LN/GELU `num_channels=B`) and the
elementwise bias add see contiguous per-stream rows. The projection GEMMs use `b_col_maj=True` (read the
activation as `[N=B, K]`) and `c_col_maj=True` (write the output as `[N=B, M]`) — so NO transpose is
needed between LN -> GEMM -> bias -> GELU. Bias is broadcast per stream by tiling it to `[B,feature]`
host-side (constant weight) and adding elementwise (ElementwiseAdd is flat, no num_channels).

N=B must be a multiple of tile_n*num_aie_columns = 16*8 = 128 (the full-array GEMM config). Gate
(generic fused_elf_probe): rel-L2(device out, buffers/out.bin) <= 0.08; golden = per-stream the exact
bf16 dataflow gen_ffn.py runs, stacked to [B,D].

Run inside the IRON env (newstack_compat first; aiebu-asm on PATH). See scripts/build_batched_decode.sh.
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes
import torch

import newstack_compat  # noqa: F401 — MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf
from iron.operators.gemm.op import GEMM
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.gelu.op import GELU
from iron.operators.elementwise_add.op import ElementwiseAdd

BF16 = ml_dtypes.bfloat16
D = 768
FF = 3072


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(wdir, layer, name):
    return np.load(os.path.join(wdir, f"L{layer}", f"{name}.npy"))


def ln(x_f32):
    t = torch.from_numpy(x_f32.astype(np.float32)).reshape(1, -1)
    return torch.nn.functional.layer_norm(t, normalized_shape=(t.shape[-1],)).numpy().reshape(-1)


def gelu_tanh(h_f32):
    t = torch.from_numpy(h_f32.astype(np.float32))
    return torch.nn.functional.gelu(t, approximate="tanh").numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, required=True, help="batch width (streams); GEMM N. Multiple of 128.")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    B = args.B
    assert B % 128 == 0, f"B={B} must be a multiple of tile_n*num_cols=128 (full-array GEMM)"
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L = args.layer

    Wfc1 = npy(args.weights, L, "fc1.weight").astype(np.float32)  # [768, 3072]
    b_fc1 = npy(args.weights, L, "fc1.bias").astype(np.float32)   # [3072]
    Wfc2 = npy(args.weights, L, "fc2.weight").astype(np.float32)  # [3072, 768]
    b_fc2 = npy(args.weights, L, "fc2.bias").astype(np.float32)   # [768]
    gf = npy(args.weights, L, "ln_final.weight").astype(np.float32)  # [768]
    bf = npy(args.weights, L, "ln_final.bias").astype(np.float32)    # [768]
    assert Wfc1.shape == (D, FF) and Wfc2.shape == (FF, D)

    # fold γf into fc1; βf into bias'. GEMM matrix A is [M, K] = Wᵀ.
    mat_fc1 = bf16((gf[:, None] * Wfc1).T.copy())     # [FF, D]
    bias_fc1 = bf16(bf @ Wfc1 + b_fc1)                # [FF]
    mat_fc2 = bf16(Wfc2.T.copy())                     # [D, FF]
    b_fc2_bf = bf16(b_fc2)                            # [D]
    # bias broadcast per stream -> [B, feature] stream-major, matching the c_col_maj GEMM output layout
    bias_fc1_b = bf16(np.tile(bias_fc1.astype(np.float32), (B, 1)))   # [B, FF]
    b_fc2_b = bf16(np.tile(b_fc2_bf.astype(np.float32), (B, 1)))      # [B, D]

    # ShimDMA limit: num_aie_columns*num_channels <= 16, so a channeled op takes <=16 channels/launch.
    # LayerNorm needs the per-stream reduction over D -> loop it over CH-stream chunks. GELU + the bias
    # adds are ELEMENTWISE (no reduction) -> run flat over the whole [B,feat] buffer (num_channels=1).
    CH = 16
    assert B % CH == 0, f"B={B} must be a multiple of {CH} (LayerNorm channel chunk)"
    ctx = AIEContext()
    # LayerNorm input is (size,) = num_channels independent groups of size/num_channels; here CH streams
    # each normalised over D -> size=CH*D, num_channels=CH, per-channel reduction length = D.
    op_ln = LayerNorm(size=CH * D, num_aie_columns=1, num_channels=CH, tile_size=D, context=ctx)
    g1 = GEMM(M=FF, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=8,
              b_col_maj=True, c_col_maj=True, context=ctx)
    add1 = ElementwiseAdd(size=B * FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)
    gelu = GELU(size=B * FF, num_aie_columns=8, num_channels=1, tile_size=FF // 8, context=ctx)
    g2 = GEMM(M=D, K=FF, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=8,
              b_col_maj=True, c_col_maj=True, context=ctx)
    add2 = ElementwiseAdd(size=B * D, tile_size=D // 8, num_aie_columns=8, context=ctx)

    chD = CH * D * 2  # BYTES per LN chunk (stream-major [CH, D], bf16); buffer slices are byte ranges
    ln_runlist = [(op_ln, f"x[{c*chD}:{(c+1)*chD}]", f"x_norm[{c*chD}:{(c+1)*chD}]") for c in range(B // CH)]
    runlist = ln_runlist + [
        (g1, "Wfc1", "x_norm", "h"),          # A=Wfc1[FF,D], B=x_norm[B,D] (col-maj), C=h[B,FF] (col-maj)
        (add1, "h", "bfc1", "h"),             # +bias (broadcast tiled), in-place; feeds GELU
        (gelu, "h", "h"),                     # [B,FF] flat elementwise
        (g2, "Wfc2", "h", "out"),             # A=Wfc2[D,FF], B=h[B,FF] (col-maj), C=out[B,D] (col-maj)
        (add2, "out", "bfc2", "out"),         # +bias
    ]
    # sliced buffers (x, x_norm — sliced by the LN channel-chunk loop) need explicit byte sizes
    bufsz = {"x": B * D * 2, "x_norm": B * D * 2}
    fused = FusedMLIROperator("ffn_b", runlist, input_args=["x"], output_args=["out"],
                              buffer_sizes=bufsz, context=ctx)
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    names = ("x", "out", "Wfc1", "bfc1", "Wfc2", "bfc2", "x_norm", "h")
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    print(f"FFN batched B={B}  buffer_sizes (in,out,scratch) =", fused.buffer_sizes)
    for n, v in lay.items():
        print(f"  {n}: type={v[0]} off={int(v[1])} len={int(v[2])}")

    # --- device-faithful golden: per stream run gen_ffn's exact bf16 dataflow, stack [B,D] ---
    rng = np.random.default_rng(args.seed)
    X = bf16(rng.standard_normal((B, D)).astype(np.float32))   # [B, D] stream-major
    out_g = np.zeros((B, D), BF16)
    for b in range(B):
        n_hw = bf16(ln(X[b].astype(np.float32)))
        h1 = bf16(mat_fc1.astype(np.float32) @ n_hw.astype(np.float32))
        h2 = bf16(h1.astype(np.float32) + bias_fc1.astype(np.float32))
        h3 = bf16(gelu_tanh(h2.astype(np.float32)))
        o1 = bf16(mat_fc2.astype(np.float32) @ h3.astype(np.float32))
        out_g[b] = bf16(o1.astype(np.float32) + b_fc2_bf.astype(np.float32))

    def wbuf(name, vals):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    wbuf("x", X.reshape(-1))
    wbuf("Wfc1", mat_fc1.reshape(-1))
    wbuf("bfc1", bias_fc1_b.reshape(-1))
    wbuf("Wfc2", mat_fc2.reshape(-1))
    wbuf("bfc2", b_fc2_b.reshape(-1))
    wbuf("out", out_g.reshape(-1))
    with open(os.path.join(args.out, "ffn_b.elf"), "wb") as f:
        f.write(elf_bytes)

    meta = {
        "elf": "ffn_b.elf",
        "kernel_name": "main:sequence",
        "input_size": int(in_sz),
        "output_size": int(out_sz),
        "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"],
        "weights": ["Wfc1", "bfc1", "Wfc2", "bfc2"],
        "output": "out",
        "dims": {"D": D, "FF": FF, "B": B, "layer": L},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote batched FFN ELF ({len(elf_bytes)}B, scratch {scratch_sz/1e6:.1f}MB) to {args.out}")


if __name__ == "__main__":
    main()
