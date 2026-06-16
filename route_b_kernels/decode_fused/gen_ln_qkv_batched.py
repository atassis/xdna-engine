#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batched fused decode — Task 2: the Whisper self-attn LN→QKV front-half for B streams as a fused ELF.

Forked from gen_ln_qkv.py (M=1) + the gen_decode.py fold. Same stream-major [B,feat] layout proven in
Task 1 ([[batched-decode-elf-ffn]]):

  X[B,D] -> LN(num_channels chunks) -> qkv GEMM-N=B (folded W'') -> +bias' -> qkv[B,QKV]

Affine LN folds into the projection (W'' = diag(γ)·Wqkv, bias' = β@Wqkv + b), and the 1/sqrt(HD) attn
scale folds onto the q rows [0:D] — EXACTLY as gen_decode.py does, so this block drops straight into the
batched layer (Task 5). Bias' is added on-device (elementwise, bias tiled to [B,QKV] host-side).

Gate (fused_elf_probe): rel-L2(device qkv, per-stream bf16 golden) <= 0.08. Run inside the IRON env.
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes
import torch

import newstack_compat  # noqa: F401 — MUST precede iron imports
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf
from iron.operators.gemm.op import GEMM
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.elementwise_add.op import ElementwiseAdd

BF16 = ml_dtypes.bfloat16
D, HD, QKV = 768, 64, 2304


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(wdir, layer, name):
    return np.load(os.path.join(wdir, f"L{layer}", f"{name}.npy")).astype(np.float32)


def ln(x_f32):
    t = torch.from_numpy(x_f32.astype(np.float32)).reshape(1, -1)
    return torch.nn.functional.layer_norm(t, normalized_shape=(t.shape[-1],)).numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, required=True, help="batch width (streams); GEMM N. Multiple of 128.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    B = args.B
    assert B % 128 == 0, f"B={B} must be a multiple of 128 (full-array GEMM)"
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L = args.layer
    scale = 1.0 / np.sqrt(HD)

    Wq, Wk, Wv = (npy(args.weights, L, n) for n in ("q.weight", "k.weight", "v.weight"))
    bq, bk, bv = (npy(args.weights, L, n) for n in ("q.bias", "k.bias", "v.bias"))
    g_s, b_s = npy(args.weights, L, "ln_self.weight"), npy(args.weights, L, "ln_self.bias")
    Wqkv = np.concatenate([Wq, Wk, Wv], axis=1)            # [768, 2304]
    assert Wqkv.shape == (D, QKV)
    # fold γ/β into the projection; fold the attn scale onto the q rows (exactly gen_decode.py:147-148)
    mat_qkv = (g_s[:, None] * Wqkv).T.copy()               # [QKV, D]
    bias_qkv = b_s @ Wqkv + np.concatenate([bq, bk, bv])   # [QKV]
    mat_qkv[0:D] *= scale
    bias_qkv[0:D] *= scale
    mat_qkv_bf = bf16(mat_qkv)
    bias_qkv_bf = bf16(bias_qkv)
    bias_qkv_b = bf16(np.tile(bias_qkv_bf.astype(np.float32), (B, 1)))   # [B, QKV]

    CH = 16
    assert B % CH == 0
    ctx = AIEContext()
    op_ln = LayerNorm(size=CH * D, num_aie_columns=1, num_channels=CH, tile_size=D, context=ctx)
    g_qkv = GEMM(M=QKV, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=8,
                 b_col_maj=True, c_col_maj=True, context=ctx)
    add = ElementwiseAdd(size=B * QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)

    chD = CH * D * 2  # BYTES per LN chunk
    ln_runlist = [(op_ln, f"x[{c*chD}:{(c+1)*chD}]", f"x_norm[{c*chD}:{(c+1)*chD}]") for c in range(B // CH)]
    runlist = ln_runlist + [
        (g_qkv, "Wqkv", "x_norm", "qkv"),     # A=W''[QKV,D], B=x_norm[B,D] col-maj, C=qkv[B,QKV] col-maj
        (add, "qkv", "bias_qkv", "qkv"),      # +bias' (broadcast tiled)
    ]
    bufsz = {"x": B * D * 2, "x_norm": B * D * 2}
    fused = FusedMLIROperator("ln_qkv_b", runlist, input_args=["x"], output_args=["qkv"],
                              buffer_sizes=bufsz, context=ctx)
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    names = ("x", "qkv", "Wqkv", "bias_qkv", "x_norm")
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    print(f"LN->QKV batched B={B}  buffer_sizes (in,out,scratch) =", fused.buffer_sizes)
    for n, v in lay.items():
        print(f"  {n}: type={v[0]} off={int(v[1])} len={int(v[2])}")

    # --- device-faithful golden: per stream, (folded W'' @ bf16(LN(x))) + bias', stacked [B,QKV] ---
    rng = np.random.default_rng(args.seed)
    X = bf16(rng.standard_normal((B, D)).astype(np.float32))
    out_g = np.zeros((B, QKV), BF16)
    for b in range(B):
        n_hw = bf16(ln(X[b].astype(np.float32)))
        qkv = bf16(bf16(mat_qkv_bf.astype(np.float32) @ n_hw.astype(np.float32)).astype(np.float32)
                   + bias_qkv_bf.astype(np.float32))
        out_g[b] = qkv

    def wbuf(name, vals):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    wbuf("x", X.reshape(-1))
    wbuf("Wqkv", mat_qkv_bf.reshape(-1))
    wbuf("bias_qkv", bias_qkv_b.reshape(-1))
    wbuf("qkv", out_g.reshape(-1))
    with open(os.path.join(args.out, "ln_qkv_b.elf"), "wb") as f:
        f.write(elf_bytes)

    meta = {
        "elf": "ln_qkv_b.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"], "weights": ["Wqkv", "bias_qkv"], "output": "qkv",
        "dims": {"D": D, "QKV": QKV, "B": B, "layer": L},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote batched LN->QKV ELF ({len(elf_bytes)}B, scratch {scratch_sz/1e6:.1f}MB) to {args.out}")


if __name__ == "__main__":
    main()
