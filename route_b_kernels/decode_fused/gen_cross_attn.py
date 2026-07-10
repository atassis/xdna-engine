#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FE1 step-4/5: Whisper CROSS-attention block as a fused full ELF (real whisper-small weights).

Cross-attn vs self-attn: K/V come from the RESIDENT encoder states (computed once per utterance →
static in scratch, NO strided-copy, NO per-token patch), attention is non-causal over T_enc encoder
positions, and the softmax mask is STATIC (compile-time rtp_vector_size, not patched). Q is projected
from the decoder state.

  x_norm = LayerNorm(x)                              # IRON layer_norm (non-affine); γ_cross folded into cross_q
  q      = (γc⊙x_norm+βc) @ cross_q + b_cq           # ×(1/√hd) scale folded into q; bias on-device
  scores = Kenc · q            (per head)            # Kenc resident [H, T, hd]
  w      = softmax(scores, mask=T_enc<padded)        # static mask (rtp_vector_size = real T_enc)
  ctx    = Venc^T · w          (per head)            # per-head Transpose then batched GEMV
  out    = ctx @ cross_out + b_co

T_enc padded 1500→1536 (must be %64 for the context-GEMV K and %16 for softmax cols); the pad rows are
masked by the static softmax width. Run inside IRON env (aiebu-asm on PATH).
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes
import torch

from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf
from iron.operators.gemv.op import GEMV
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.elementwise_add.op import ElementwiseAdd
from iron.operators.softmax.op import Softmax
from iron.operators.transpose.op import Transpose

BF16 = ml_dtypes.bfloat16
D = 768
H = 12
HD = 64


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(wdir, layer, name):
    return np.load(os.path.join(wdir, f"L{layer}", f"{name}.npy"))


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
    ap.add_argument("--t-enc", type=int, default=1500)     # real Whisper encoder length
    ap.add_argument("--t-pad", type=int, default=1536)     # %64 (ctx GEMV K) and %16 (softmax cols)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L, T, TP = args.layer, args.t_enc, args.t_pad
    assert TP % 64 == 0 and TP % 16 == 0 and TP >= T
    scale = 1.0 / np.sqrt(HD)
    SM_ROWS = 16

    # --- weights ---
    Wcq = npy(args.weights, L, "cross_q.weight").astype(np.float32)  # [768,768]
    bcq = npy(args.weights, L, "cross_q.bias").astype(np.float32)
    Wck = npy(args.weights, L, "cross_k.weight").astype(np.float32)  # [768,768] (bias 0)
    Wcv = npy(args.weights, L, "cross_v.weight").astype(np.float32)
    bcv = npy(args.weights, L, "cross_v.bias").astype(np.float32)
    Wco = npy(args.weights, L, "cross_out.weight").astype(np.float32)
    bco = npy(args.weights, L, "cross_out.bias").astype(np.float32)
    gc = npy(args.weights, L, "ln_cross.weight").astype(np.float32)
    bc = npy(args.weights, L, "ln_cross.bias").astype(np.float32)

    # fold γc + scale into cross_q; βc into bias'
    mat_q = (gc[:, None] * Wcq).T.copy() * scale     # [768,768]
    bias_q = (bc @ Wcq + bcq) * scale                # [768]
    mat_q, bias_q = bf16(mat_q), bf16(bias_q)
    mat_o, bias_o = bf16(Wco.T.copy()), bf16(bco)

    # encoder hidden states (random, real dims) -> Kenc/Venc via real cross_k/cross_v
    rng = np.random.default_rng(args.seed)
    enc = (rng.standard_normal((T, D)).astype(np.float32) * 0.5)
    Kenc = enc @ Wck                                  # [T,768] (no bias)
    Venc = enc @ Wcv + bcv                            # [T,768]
    # -> per-head [H, TP, hd], head-major, pad T->TP with zeros
    def to_heads_padded(M):
        out = np.zeros((H, TP, HD), dtype=np.float32)
        out[:, 0:T, :] = M.reshape(T, H, HD).transpose(1, 0, 2)
        return bf16(out)
    Kenc_b = to_heads_padded(Kenc)                    # [H,TP,hd]
    Venc_b = to_heads_padded(Venc)

    tm, tn, ts = pick_transpose_tiling(TP, HD)
    print(f"transpose tiling for {TP}x{HD}: m={tm} n={tn} s={ts}")

    # --- ops ---
    ctx = AIEContext()
    ln_op = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, context=ctx)
    g_q = GEMV(M=D, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    add_q = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)
    g_scores = GEMV(M=TP, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=TP // 8,
                    num_batches=H, context=ctx)
    # STATIC mask: no runtime mask patch => active width fixed to rtp_vector_size (= real T_enc).
    softmax = Softmax(rows=SM_ROWS, cols=TP, num_aie_columns=1, num_channels=1, rtp_vector_size=T, context=ctx)
    transpose = Transpose(M=TP, N=HD, num_aie_columns=2, num_channels=1, m=tm, n=tn, s=ts, context=ctx)
    g_ctx = GEMV(M=HD, K=TP, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8,
                 num_batches=H, context=ctx)
    g_o = GEMV(M=D, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    add_o = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)

    HS2 = H * TP * 2
    per_head = TP * HD * 2
    runlist = [
        (ln_op, "x", "x_norm"),
        (g_q, "Wcq", "x_norm", "q"),
        (add_q, "q", "bias_q", "q"),
        (g_scores, "Kenc", "q", f"scores[0:{HS2}]"),
        (softmax, "scores", "weights"),
    ] + [
        (transpose, f"Venc[{h * per_head}:{(h + 1) * per_head}]",
         f"VencT[{h * per_head}:{(h + 1) * per_head}]")
        for h in range(H)
    ] + [
        (g_ctx, "VencT", f"weights[0:{HS2}]", "ctx"),
        (g_o, "Wco", "ctx", "attn_out"),
        (add_o, "attn_out", "bias_o", "attn_out"),
    ]
    fused = FusedMLIROperator(
        "cross_attn", runlist, input_args=["x"], output_args=["attn_out"],
        buffer_sizes={
            "q": D * 2, "Kenc": H * TP * HD * 2, "Venc": H * TP * HD * 2, "VencT": H * TP * HD * 2,
            "scores": SM_ROWS * TP * 2, "weights": SM_ROWS * TP * 2,
        },
        context=ctx,
    )
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    names = ("x", "attn_out", "Wcq", "bias_q", "Wco", "bias_o", "Kenc", "Venc", "VencT",
             "x_norm", "q", "scores", "weights", "ctx")
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    print("buffer_sizes (in,out,scratch) =", fused.buffer_sizes)

    # --- inputs + device-faithful golden ---
    x = bf16(rng.standard_normal(D).astype(np.float32))
    n_hw = bf16(ln(x.astype(np.float32)))
    q_dev = bf16(mat_q.astype(np.float32) @ n_hw.astype(np.float32))
    q = bf16(q_dev.astype(np.float32) + bias_q.astype(np.float32)).reshape(H, HD)  # scaled
    ctx_out = np.zeros((H, HD), dtype=np.float32)
    for h in range(H):
        Kr = Kenc_b[h, 0:T, :].astype(np.float32)     # [T,hd]
        Vr = Venc_b[h, 0:T, :].astype(np.float32)
        s = Kr @ q[h].astype(np.float32)              # [T]; scale in q
        w = torch.softmax(torch.from_numpy(bf16(s).astype(np.float32)), dim=0).numpy()
        ctx_out[h] = bf16(w).astype(np.float32) @ Vr
    ctx_flat = bf16(ctx_out.reshape(-1))
    attn = bf16(mat_o.astype(np.float32) @ ctx_flat.astype(np.float32))
    attn_out = bf16(attn.astype(np.float32) + bias_o.astype(np.float32))

    def wbuf(name, vals):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    wbuf("x", x)
    wbuf("Wcq", mat_q.reshape(-1)); wbuf("bias_q", bias_q)
    wbuf("Wco", mat_o.reshape(-1)); wbuf("bias_o", bias_o)
    wbuf("Kenc", Kenc_b.reshape(-1)); wbuf("Venc", Venc_b.reshape(-1))
    wbuf("attn_out", attn_out)
    with open(os.path.join(args.out, "cross_attn.elf"), "wb") as f:
        f.write(elf_bytes)

    meta = {
        "elf": "cross_attn.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v_[0], "offset": int(v_[1]), "len": int(v_[2])} for n, v_ in lay.items()},
        "inputs": ["x"],
        "weights": ["Wcq", "bias_q", "Wco", "bias_o", "Kenc", "Venc"],
        "output": "attn_out",
        "dims": {"D": D, "H": H, "HD": HD, "T_enc": T, "T_pad": TP, "layer": L},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote ELF ({len(elf_bytes)}B) + buffers + meta.json to {args.out}  (T_enc={T}, pad={TP})")


if __name__ == "__main__":
    main()
