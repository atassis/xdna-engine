#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batched fused decode — Task 4: Whisper CROSS-attention for B streams as a fused full ELF.

Forked from gen_cross_attn.py (M=1). Cross-attn vs self-attn: K/V are the per-stream ENCODER states
(resident weights, NO cache write), attention is non-causal over T_enc encoder positions, q is a single
projection (not QKV). Each of B streams has its OWN encoder output -> Kenc/Venc are [B,H,TP,HD].

  x[B,D] -> LN(chunks) -> q GEMM-N=B (+bias) -> q[B,H,HD]
  scores[B*H,TP] = GEMV(num_batches=B*H): Kenc[b,h][TP,HD] @ q[b,h][HD]
  weights = softmax(scores)        (rows=B*H)
  VencT[B*H,HD,TP] = Transpose(num_batches=B*H)
  ctx[B*H,HD] = GEMV(num_batches=B*H): VencT @ weights -> ctx[B,768]
  attn_out[B,D] = Wco GEMM-N=B (+bias)

Validation uses T_enc=T_pad (no softmax mask) at a small TP to bound the per-stream Kenc/Venc memory
(at full TP=1536, B=128 these are ~300 MB each — Task 6 caps B). Gate: rel-L2 <= 0.08. IRON env.
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes
import torch

import newstack_compat  # noqa: F401
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf
from iron.operators.gemm.op import GEMM
from iron.operators.gemv.op import GEMV
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.elementwise_add.op import ElementwiseAdd
from iron.operators.softmax.op import Softmax
from iron.operators.transpose.op import Transpose

BF16 = ml_dtypes.bfloat16
D, H, HD = 768, 12, 64


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(wdir, layer, name):
    return np.load(os.path.join(wdir, f"L{layer}", f"{name}.npy")).astype(np.float32)


def ln(x):
    t = torch.from_numpy(x.astype(np.float32)).reshape(1, -1)
    return torch.nn.functional.layer_norm(t, normalized_shape=(t.shape[-1],)).numpy().reshape(-1)


def pick_transpose_tiling(M, N):
    for s in (8, 4):
        for m in sorted((d for d in range(s, M + 1) if M % d == 0 and d % s == 0), reverse=True):
            for n in sorted((d for d in range(s, N + 1) if N % d == 0 and d % s == 0), reverse=True):
                if m * n > 8192:
                    continue
                if s == 8 and (m <= 16 or n <= 16):
                    continue
                if s == 4 and (m <= 4 or n <= 4):
                    continue
                return m, n, s
    raise ValueError(f"no valid transpose tiling for {M}x{N}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--T", type=int, default=128, help="encoder length (= T_pad, no mask in validation)")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    B, T = args.B, args.T
    TP = T  # no mask: pad == real length
    assert TP % 64 == 0 and TP % 16 == 0, "TP must be %64 and %16"
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L = args.layer
    scale = 1.0 / np.sqrt(HD)
    g_cols = 8 if B >= 128 else 1
    assert B % (16 * g_cols) == 0
    BH = B * H

    Wcq, bcq = npy(args.weights, L, "cross_q.weight"), npy(args.weights, L, "cross_q.bias")
    Wck = npy(args.weights, L, "cross_k.weight")
    Wcv, bcv = npy(args.weights, L, "cross_v.weight"), npy(args.weights, L, "cross_v.bias")
    Wco, bco = npy(args.weights, L, "cross_out.weight"), npy(args.weights, L, "cross_out.bias")
    gc, bc = npy(args.weights, L, "ln_cross.weight"), npy(args.weights, L, "ln_cross.bias")
    mat_q = (gc[:, None] * Wcq).T.copy() * scale       # [D,D]
    bias_q = (bc @ Wcq + bcq) * scale                  # [D]
    mat_q_bf, bias_q_bf = bf16(mat_q), bf16(bias_q)
    bias_q_b = bf16(np.tile(bias_q_bf.astype(np.float32), (B, 1)))
    Wco_bf, bco_bf = bf16(Wco.T.copy()), bf16(bco)
    bco_b = bf16(np.tile(bco_bf.astype(np.float32), (B, 1)))

    # per-stream encoder states -> Kenc/Venc [B,H,TP,HD]
    rng = np.random.default_rng(args.seed)

    def to_heads(M):  # [TP,D] -> [H,TP,HD]
        return M.reshape(TP, H, HD).transpose(1, 0, 2)
    Kenc_b = np.zeros((B, H, TP, HD), np.float32)
    Venc_b = np.zeros((B, H, TP, HD), np.float32)
    encs = []
    for b in range(B):
        enc = rng.standard_normal((TP, D)).astype(np.float32) * 0.5
        encs.append(enc)
        Kenc_b[b] = to_heads(enc @ Wck)
        Venc_b[b] = to_heads(enc @ Wcv + bcv)
    Kenc_bf, Venc_bf = bf16(Kenc_b), bf16(Venc_b)

    tm, tn, ts = pick_transpose_tiling(TP, HD)
    CH = 16
    assert B % CH == 0
    ctx = AIEContext()
    op_ln = LayerNorm(size=CH * D, num_aie_columns=1, num_channels=CH, tile_size=D, context=ctx)
    g_q = GEMM(M=D, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=g_cols,
               b_col_maj=True, c_col_maj=True, context=ctx)
    add_q = ElementwiseAdd(size=B * D, tile_size=D // 8, num_aie_columns=8, context=ctx)
    g_scores = GEMV(M=TP, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=TP // 8, num_batches=BH, context=ctx)
    softmax = Softmax(rows=BH, cols=TP, num_aie_columns=1, num_channels=1, rtp_vector_size=TP, mask_patch_value=0, context=ctx)
    transpose = Transpose(M=TP, N=HD, num_batches=BH, num_aie_columns=1, num_channels=1, m=tm, n=tn, s=ts, context=ctx)
    g_ctx = GEMV(M=HD, K=TP, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=BH, context=ctx)
    g_o = GEMM(M=D, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=g_cols,
               b_col_maj=True, c_col_maj=True, context=ctx)
    add_o = ElementwiseAdd(size=B * D, tile_size=D // 8, num_aie_columns=8, context=ctx)

    chD = CH * D * 2
    HS2 = BH * TP * 2
    ln_rl = [(op_ln, f"x[{c*chD}:{(c+1)*chD}]", f"x_norm[{c*chD}:{(c+1)*chD}]") for c in range(B // CH)]
    runlist = ln_rl + [
        (g_q, "Wcq", "x_norm", "q"), (add_q, "q", "bias_q", "q"),
        (g_scores, "Kenc", "q", f"scores[0:{HS2}]"),
        (softmax, "scores", "weights"),
        (transpose, "Venc", "VencT"),
        (g_ctx, "VencT", f"weights[0:{HS2}]", "ctxb"),
        (g_o, "Wco", "ctxb", "attn_out"), (add_o, "attn_out", "bias_o", "attn_out"),
    ]
    bufsz = {
        "x": B * D * 2, "x_norm": B * D * 2, "q": B * D * 2,
        "Kenc": B * H * TP * HD * 2, "Venc": B * H * TP * HD * 2, "VencT": B * H * TP * HD * 2,
        "scores": BH * TP * 2, "weights": BH * TP * 2, "ctxb": B * H * HD * 2,
    }
    fused = FusedMLIROperator("cross_attn_b", runlist, input_args=["x"], output_args=["attn_out"],
                              buffer_sizes=bufsz, context=ctx)
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    names = ("x", "attn_out", "Wcq", "bias_q", "Wco", "bias_o", "Kenc", "Venc")
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    print(f"cross_attn batched B={B} T={T} (BH={BH})  buffer_sizes (in,out,scratch) =", fused.buffer_sizes)

    # --- per-stream bf16 golden ---
    X = bf16(rng.standard_normal((B, D)).astype(np.float32))
    attn_out_g = np.zeros((B, D), BF16)
    for b in range(B):
        n_hw = bf16(ln(X[b].astype(np.float32)))
        q = bf16(bf16(mat_q_bf.astype(np.float32) @ n_hw.astype(np.float32)).astype(np.float32) + bias_q_bf.astype(np.float32)).reshape(H, HD)
        ctx_out = np.zeros((H, HD), np.float32)
        for h in range(H):
            Kr = Kenc_bf[b, h].astype(np.float32); Vr = Venc_bf[b, h].astype(np.float32)
            s = Kr @ q[h].astype(np.float32)
            wts = torch.softmax(torch.from_numpy(bf16(s).astype(np.float32)), 0).numpy()
            ctx_out[h] = bf16(wts).astype(np.float32) @ Vr
        cf = bf16(ctx_out.reshape(-1))
        attn = bf16(bf16(Wco_bf.astype(np.float32) @ cf.astype(np.float32)).astype(np.float32) + bco_bf.astype(np.float32))
        attn_out_g[b] = attn

    def wbuf(name, vals):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    wbuf("x", X.reshape(-1)); wbuf("Wcq", mat_q_bf.reshape(-1)); wbuf("bias_q", bias_q_b.reshape(-1))
    wbuf("Wco", Wco_bf.reshape(-1)); wbuf("bias_o", bco_b.reshape(-1))
    wbuf("Kenc", Kenc_bf.reshape(-1)); wbuf("Venc", Venc_bf.reshape(-1))
    wbuf("attn_out", attn_out_g.reshape(-1))
    with open(os.path.join(args.out, "cross_attn_b.elf"), "wb") as f:
        f.write(elf_bytes)

    meta = {
        "elf": "cross_attn_b.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"], "weights": ["Wcq", "bias_q", "Wco", "bias_o", "Kenc", "Venc"], "output": "attn_out",
        "dims": {"D": D, "H": H, "HD": HD, "T": T, "TP": TP, "B": B, "layer": L},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote batched cross-attn ELF ({len(elf_bytes)}B, scratch {scratch_sz/1e6:.1f}MB) to {args.out}")


if __name__ == "__main__":
    main()
