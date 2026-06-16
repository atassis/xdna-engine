#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batched fused decode — Task 3: Whisper self-attention for B streams as a fused full ELF.

Forked from gen_self_attn.py (M=1). Validates the BATCHED ATTENTION MATH: each of B streams has its own
q/k/v + its own KV cache, and attention runs per (stream,head) via num_batches=B*H. Layout:

  qkv[B,QKV] (Task-2 LN→QKV) -> extract q[B,H,HD]; scatter k,v into kcache/vcache [B,H,S,HD]
  scores[B*H,S] = GEMV(num_batches=B*H): kcache[b,h][S,HD] @ q[b,h][HD]
  weights = softmax(scores)                 (rows = B*H)
  vcacheT[B*H,HD,S] = Transpose(num_batches=B*H) of vcache
  ctx[B*H,HD] = GEMV(num_batches=B*H): vcacheT @ weights  -> ctx[B,768] stream-major
  attn_out[B,D] = Wo GEMM-N=B (b_col_maj/c_col_maj) + bias

v1 SCOPE (spec YAGNI = offline-bulk lockstep): all streams share the SAME decode position, so the KV
write offset + softmax context are CONSTANTS baked in (no per-stream scratchpad vector — that is the
later dynamic-batching scope, overlaps [[generalize-resident-scratchpad-decode]]). We validate with a
FULL cache (S = context length, P = S-1 prefilled + current), so softmax needs no mask. Gate: rel-L2
(device attn_out, per-stream bf16 golden) <= 0.08. Run inside the IRON env.
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
from iron.operators.gemv.op import GEMV
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.elementwise_add.op import ElementwiseAdd
from iron.operators.softmax.op import Softmax
from iron.operators.strided_copy.op import StridedCopy
from iron.operators.transpose.op import Transpose

BF16 = ml_dtypes.bfloat16
D, H, HD, QKV = 768, 12, 64, 2304


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
    ap.add_argument("--B", type=int, required=True, help="batch width (streams). 128 (full array) or 16 (1-col).")
    ap.add_argument("--S", type=int, default=64, help="cache/context length (= attended positions, no mask)")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    B, S = args.B, args.S
    P = S - 1  # P prefilled + current token => full cache of S, no mask needed
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L = args.layer
    scale = 1.0 / np.sqrt(HD)
    # GEMM array config: full array at B>=128 (num_cols=8, min N=128); else 1-col (min N=16).
    g_cols = 8 if B >= 128 else 1
    assert B % (16 * g_cols) == 0, f"B={B} invalid for GEMM (need multiple of {16*g_cols})"
    assert S % 8 == 0, "S must be a multiple of 8 (GEMV tiling)"
    BH = B * H

    # --- weights (QKV folded as Task 2; Wo plain) ---
    Wq, Wk, Wv = (npy(args.weights, L, n) for n in ("q.weight", "k.weight", "v.weight"))
    bq, bk, bv = (npy(args.weights, L, n) for n in ("q.bias", "k.bias", "v.bias"))
    g_s, b_s = npy(args.weights, L, "ln_self.weight"), npy(args.weights, L, "ln_self.bias")
    Wo, bo = npy(args.weights, L, "out.weight"), npy(args.weights, L, "out.bias")
    Wqkv = np.concatenate([Wq, Wk, Wv], axis=1)
    mat_qkv = (g_s[:, None] * Wqkv).T.copy()
    bias_qkv = b_s @ Wqkv + np.concatenate([bq, bk, bv])
    mat_qkv[0:D] *= scale
    bias_qkv[0:D] *= scale
    mat_qkv_bf, bias_qkv_bf = bf16(mat_qkv), bf16(bias_qkv)
    bias_qkv_b = bf16(np.tile(bias_qkv_bf.astype(np.float32), (B, 1)))
    Wo_bf, bo_bf = bf16(Wo.T.copy()), bf16(bo)
    bo_b = bf16(np.tile(bo_bf.astype(np.float32), (B, 1)))

    tm, tn, ts = pick_transpose_tiling(S, HD)

    CH = 16
    assert B % CH == 0
    ctx = AIEContext()
    op_ln = LayerNorm(size=CH * D, num_aie_columns=1, num_channels=CH, tile_size=D, context=ctx)
    g_qkv = GEMM(M=QKV, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=g_cols,
                 b_col_maj=True, c_col_maj=True, context=ctx)
    add_qkv = ElementwiseAdd(size=B * QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)
    # extract q [B,H,HD] from qkv[B,QKV] cols [0:768]; scatter k/v into [B,H,S,HD] at position P (const).
    q_ex = StridedCopy(input_sizes=(B, H, HD), input_strides=(QKV, HD, 1), input_offset=0,
                       output_sizes=(B, H, HD), output_strides=(H * HD, HD, 1), output_offset=0,
                       input_buffer_size=B * QKV, output_buffer_size=B * H * HD, num_aie_channels=1, context=ctx)
    sc_k = StridedCopy(input_sizes=(B, H, HD), input_strides=(QKV, HD, 1), input_offset=D,
                       output_sizes=(B, H, HD), output_strides=(H * S * HD, S * HD, 1), output_offset=P * HD,
                       input_buffer_size=B * QKV, output_buffer_size=B * H * S * HD, num_aie_channels=1, context=ctx)
    sc_v = StridedCopy(input_sizes=(B, H, HD), input_strides=(QKV, HD, 1), input_offset=2 * D,
                       output_sizes=(B, H, HD), output_strides=(H * S * HD, S * HD, 1), output_offset=P * HD,
                       input_buffer_size=B * QKV, output_buffer_size=B * H * S * HD, num_aie_channels=1, context=ctx)
    g_scores = GEMV(M=S, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=S // 8, num_batches=BH, context=ctx)
    softmax = Softmax(rows=BH, cols=S, num_aie_columns=1, num_channels=1, rtp_vector_size=S, mask_patch_value=0, context=ctx)
    transpose = Transpose(M=S, N=HD, num_batches=BH, num_aie_columns=1, num_channels=1, m=tm, n=tn, s=ts, context=ctx)
    g_ctx = GEMV(M=HD, K=S, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=BH, context=ctx)
    g_o = GEMM(M=D, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=g_cols,
               b_col_maj=True, c_col_maj=True, context=ctx)
    add_o = ElementwiseAdd(size=B * D, tile_size=D // 8, num_aie_columns=8, context=ctx)

    chD = CH * D * 2
    HS2 = BH * S * 2  # bytes of the full [B*H,S] scores/weights block
    ln_rl = [(op_ln, f"x[{c*chD}:{(c+1)*chD}]", f"x_norm[{c*chD}:{(c+1)*chD}]") for c in range(B // CH)]
    runlist = ln_rl + [
        (g_qkv, "Wqkv", "x_norm", "qkv"), (add_qkv, "qkv", "bias_qkv", "qkv"),
        (q_ex, "qkv", "qbuf"),
        (sc_k, "qkv", "kcache"), (sc_v, "qkv", "vcache"),
        (g_scores, "kcache", "qbuf", f"scores[0:{HS2}]"),
        (softmax, "scores", "weights"),
        (transpose, "vcache", "vcacheT"),
        (g_ctx, "vcacheT", f"weights[0:{HS2}]", "ctxb"),
        (g_o, "Wo", "ctxb", "attn_out"), (add_o, "attn_out", "bias_o", "attn_out"),
    ]
    bufsz = {
        "x": B * D * 2, "x_norm": B * D * 2, "qkv": B * QKV * 2, "qbuf": B * H * HD * 2,
        "kcache": B * H * S * HD * 2, "vcache": B * H * S * HD * 2, "vcacheT": B * H * S * HD * 2,
        "scores": BH * S * 2, "weights": BH * S * 2, "ctxb": B * H * HD * 2,
    }
    fused = FusedMLIROperator("self_attn_b", runlist, input_args=["x"], output_args=["attn_out"],
                              buffer_sizes=bufsz, context=ctx)
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    names = ("x", "attn_out", "Wqkv", "bias_qkv", "Wo", "bias_o", "kcache", "vcache")
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    print(f"self_attn batched B={B} S={S} (BH={BH})  buffer_sizes (in,out,scratch) =", fused.buffer_sizes)

    # --- inputs + prefilled cache + per-stream bf16 golden ---
    rng = np.random.default_rng(args.seed)
    X = bf16(rng.standard_normal((B, D)).astype(np.float32))
    k_past = bf16(rng.standard_normal((B, H, P, HD)).astype(np.float32) * 0.5)
    v_past = bf16(rng.standard_normal((B, H, P, HD)).astype(np.float32) * 0.5)
    kcache0 = np.zeros((B, H, S, HD), BF16); vcache0 = np.zeros((B, H, S, HD), BF16)
    kcache0[:, :, 0:P], vcache0[:, :, 0:P] = k_past, v_past

    attn_out_g = np.zeros((B, D), BF16)
    for b in range(B):
        n_hw = bf16(ln(X[b].astype(np.float32)))
        qkv = bf16(bf16(mat_qkv_bf.astype(np.float32) @ n_hw.astype(np.float32)).astype(np.float32) + bias_qkv_bf.astype(np.float32))
        q = qkv[0:D].reshape(H, HD); kc = qkv[D:2*D].reshape(H, HD); vc = qkv[2*D:3*D].reshape(H, HD)
        ctx_out = np.zeros((H, HD), np.float32)
        for h in range(H):
            K = np.concatenate([k_past[b, h].astype(np.float32), kc[h].astype(np.float32)[None]], 0)
            V = np.concatenate([v_past[b, h].astype(np.float32), vc[h].astype(np.float32)[None]], 0)
            s = K @ q[h].astype(np.float32)
            wts = torch.softmax(torch.from_numpy(bf16(s).astype(np.float32)), 0).numpy()
            ctx_out[h] = bf16(wts).astype(np.float32) @ V
        cf = bf16(ctx_out.reshape(-1))
        attn = bf16(bf16(Wo_bf.astype(np.float32) @ cf.astype(np.float32)).astype(np.float32) + bo_bf.astype(np.float32))
        attn_out_g[b] = attn

    def wbuf(name, vals):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    wbuf("x", X.reshape(-1)); wbuf("Wqkv", mat_qkv_bf.reshape(-1)); wbuf("bias_qkv", bias_qkv_b.reshape(-1))
    wbuf("Wo", Wo_bf.reshape(-1)); wbuf("bias_o", bo_b.reshape(-1))
    wbuf("kcache", kcache0.reshape(-1)); wbuf("vcache", vcache0.reshape(-1))
    wbuf("attn_out", attn_out_g.reshape(-1))
    with open(os.path.join(args.out, "self_attn_b.elf"), "wb") as f:
        f.write(elf_bytes)

    meta = {
        "elf": "self_attn_b.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"], "weights": ["Wqkv", "bias_qkv", "Wo", "bias_o", "kcache", "vcache"], "output": "attn_out",
        "dims": {"D": D, "H": H, "HD": HD, "S": S, "B": B, "P": P, "layer": L},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote batched self-attn ELF ({len(elf_bytes)}B, scratch {scratch_sz/1e6:.1f}MB) to {args.out}")


if __name__ == "__main__":
    main()
