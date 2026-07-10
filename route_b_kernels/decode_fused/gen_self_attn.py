#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FE1 step-4/5: Whisper self-attention block as a fused full ELF (real whisper-small weights).

The dispatch-collapse prize: LN→QKV→KV-write→scores→softmax→V-transpose→context→O all in ONE dispatch,
KV cache resident in scratch, the per-token KV-write-offset + softmax mask-length patched in the ELF
(FusedElfPatcher). No RoPE, no GQA (full MHA, H=12, hd=64), Whisper biases added ON-DEVICE.

Design (the better, production-shaped one — validated at the REAL Whisper max context, multi-position):
  S = max_ctx (default 448). Both caches K-cache and V-cache are [H, S, hd] (position stride = hd), so
  the uniform FusedElfPatcher (base + num_preceding·hd·2) is correct for BOTH at any pos. The context
  GEMV needs V as [hd, S] per head, so a per-head Transpose op turns V-cache[h]=[S,hd] → [hd,S]
  (12 launches; the dedicated Transpose, not a pre-transpose hack). Validated with a PREFILLED cache at
  context_len = num_preceding+1 > 1 — real multi-position softmax + the pos>0 patch.

Scale (1/√hd) folded into the q-rows of the QKV matrix + q-portion of bias' (q only feeds scores).
Run inside IRON env (aiebu-asm on PATH).
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
from iron.operators.strided_copy.op import StridedCopy
from iron.operators.transpose.op import Transpose

BF16 = ml_dtypes.bfloat16
D = 768
H = 12
HD = 64
QKV = 2304
KV_MAGIC = 0xDEADBEE0
SM_MAGIC = 0xBA5EBA11


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(wdir, layer, name):
    return np.load(os.path.join(wdir, f"L{layer}", f"{name}.npy"))


def ln(x):
    t = torch.from_numpy(x.astype(np.float32)).reshape(1, -1)
    return torch.nn.functional.layer_norm(t, normalized_shape=(t.shape[-1],)).numpy().reshape(-1)


def pick_transpose_tiling(M, N):
    """Largest valid (m,n,s) for an MxN transpose: M%m==0, N%n==0, m%s==0, n%s==0, m*n<=8192, s in {4,8}."""
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
    ap.add_argument("--prompt-len", type=int, default=448)   # real Whisper max decode context
    ap.add_argument("--num-preceding", type=int, default=5)  # context_len = P+1 (multi-position)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L, S, P = args.layer, args.prompt_len, args.num_preceding
    assert 0 <= P < S
    scale = 1.0 / np.sqrt(HD)
    SM_ROWS = 16  # softmax rows % 16 (pad 12 heads -> 16)

    # --- weights ---
    Wq, Wk, Wv = (npy(args.weights, L, n).astype(np.float32) for n in ("q.weight", "k.weight", "v.weight"))
    bq, bk, bv = (npy(args.weights, L, n).astype(np.float32) for n in ("q.bias", "k.bias", "v.bias"))
    Wo = npy(args.weights, L, "out.weight").astype(np.float32)
    bo = npy(args.weights, L, "out.bias").astype(np.float32)
    gamma = npy(args.weights, L, "ln_self.weight").astype(np.float32)
    beta = npy(args.weights, L, "ln_self.bias").astype(np.float32)
    W_qkv = np.concatenate([Wq, Wk, Wv], axis=1)
    b_qkv = np.concatenate([bq, bk, bv])

    matrix = (gamma[:, None] * W_qkv).T.copy()       # [2304,768]
    bias_p = beta @ W_qkv + b_qkv                    # [2304]
    matrix[0:D, :] *= scale
    bias_p[0:D] *= scale
    matrix_qkv, bias_qkv = bf16(matrix), bf16(bias_p)
    matrix_o, bias_o = bf16(Wo.T.copy()), bf16(bo)

    tm, tn, ts = pick_transpose_tiling(S, HD)
    print(f"transpose tiling for {S}x{HD}: m={tm} n={tn} s={ts}")

    # --- ops ---
    ctx = AIEContext()
    ln_op = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, context=ctx)
    g_qkv = GEMV(M=QKV, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=QKV // 8, context=ctx)
    add_qkv = ElementwiseAdd(size=QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)
    sc = dict(input_sizes=(H, HD), input_strides=(HD, 1), input_offset=0,
              output_sizes=(1, H, HD), output_strides=(0, S * HD, 1), output_offset=0,
              input_buffer_size=H * HD, output_buffer_size=H * S * HD, num_aie_channels=1)
    sc_k = StridedCopy(**sc, kwargs={"output_offset_patch_marker": KV_MAGIC}, context=ctx)
    sc_v = StridedCopy(**sc, kwargs={"output_offset_patch_marker": KV_MAGIC}, context=ctx)
    g_scores = GEMV(M=S, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=S // 8,
                    num_batches=H, context=ctx)
    softmax = Softmax(rows=SM_ROWS, cols=S, num_aie_columns=1, num_channels=1, rtp_vector_size=SM_MAGIC, context=ctx)
    transpose = Transpose(M=S, N=HD, num_aie_columns=2, num_channels=1, m=tm, n=tn, s=ts, context=ctx)
    g_ctx = GEMV(M=HD, K=S, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8,
                 num_batches=H, context=ctx)
    g_o = GEMV(M=D, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    add_o = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)

    HS2 = H * S * 2            # bytes of one full [H,S] (scores/weights) row-block per ... (12 heads)
    per_head = S * HD * 2     # bytes of one head's [S,hd] cache block
    runlist = [
        (ln_op, "x", "x_norm"),
        (g_qkv, "Wqkv", "x_norm", "qkv"),
        (add_qkv, "qkv", "bias_qkv", "qkv"),
        (sc_k, "qkv[1536:3072]", "kcache"),   # k = qkv[768:1536] elems
        (sc_v, "qkv[3072:4608]", "vcache"),   # v = qkv[1536:2304] elems
        (g_scores, "kcache", "qkv[0:1536]", f"scores[0:{HS2}]"),  # q = qkv[0:768]
        (softmax, "scores", "weights"),
    ] + [
        # per-head V transpose: vcache[h]=[S,hd] -> vcacheT[h]=[hd,S]
        (transpose, f"vcache[{h * per_head}:{(h + 1) * per_head}]",
         f"vcacheT[{h * per_head}:{(h + 1) * per_head}]")
        for h in range(H)
    ] + [
        (g_ctx, "vcacheT", f"weights[0:{HS2}]", "ctx"),
        (g_o, "Wo", "ctx", "attn_out"),
        (add_o, "attn_out", "bias_o", "attn_out"),
    ]
    fused = FusedMLIROperator(
        "self_attn", runlist, input_args=["x"], output_args=["attn_out"],
        buffer_sizes={
            "qkv": QKV * 2,
            "kcache": H * S * HD * 2, "vcache": H * S * HD * 2, "vcacheT": H * S * HD * 2,
            "scores": SM_ROWS * S * 2, "weights": SM_ROWS * S * 2,
        },
        context=ctx,
    )
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    names = ("x", "attn_out", "Wqkv", "bias_qkv", "Wo", "bias_o", "kcache", "vcache", "vcacheT",
             "x_norm", "qkv", "scores", "weights", "ctx")
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    print("buffer_sizes (in,out,scratch) =", fused.buffer_sizes)
    for n in ("kcache", "vcache", "vcacheT"):
        print(f"  {n}: off={int(lay[n][1])} len={int(lay[n][2])}")

    # --- inputs + prefilled cache (P past tokens) + device-faithful golden ---
    rng = np.random.default_rng(args.seed)
    x = bf16(rng.standard_normal(D).astype(np.float32))
    n_hw = bf16(ln(x.astype(np.float32)))
    qkv_dev = bf16(matrix_qkv.astype(np.float32) @ n_hw.astype(np.float32))
    qkv_b = bf16(qkv_dev.astype(np.float32) + bias_qkv.astype(np.float32))
    q = qkv_b[0:D].reshape(H, HD)            # already scaled (fold)
    k_cur = qkv_b[D:2 * D].reshape(H, HD)
    v_cur = qkv_b[2 * D:3 * D].reshape(H, HD)

    # prefilled past K/V (random bf16), positions 0..P-1; current token at position P.
    k_past = bf16(rng.standard_normal((H, P, HD)).astype(np.float32) * 0.5)
    v_past = bf16(rng.standard_normal((H, P, HD)).astype(np.float32) * 0.5)
    kcache_init = np.zeros((H, S, HD), dtype=BF16)
    vcache_init = np.zeros((H, S, HD), dtype=BF16)
    if P > 0:
        kcache_init[:, 0:P, :] = k_past
        vcache_init[:, 0:P, :] = v_past
    # (the device strided_copy writes k_cur/v_cur at position P)

    # host golden: attention over positions 0..P (P past + current), per head, bf16 dataflow
    ctx_out = np.zeros((H, HD), dtype=np.float32)
    for h in range(H):
        K = np.concatenate([k_past[h].astype(np.float32) if P > 0 else np.zeros((0, HD)),
                            k_cur[h].astype(np.float32)[None, :]], axis=0)  # [P+1, hd]
        V = np.concatenate([v_past[h].astype(np.float32) if P > 0 else np.zeros((0, HD)),
                            v_cur[h].astype(np.float32)[None, :]], axis=0)
        s = K @ q[h].astype(np.float32)            # [P+1]; scale already in q
        w = torch.softmax(torch.from_numpy(bf16(s).astype(np.float32)), dim=0).numpy()
        ctx_out[h] = bf16(w) .astype(np.float32) @ V
    ctx_flat = bf16(ctx_out.reshape(-1))           # [768]
    attn = bf16(matrix_o.astype(np.float32) @ ctx_flat.astype(np.float32))
    attn_out = bf16(attn.astype(np.float32) + bias_o.astype(np.float32))

    def wbuf(name, vals):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    wbuf("x", x)
    wbuf("Wqkv", matrix_qkv.reshape(-1))
    wbuf("bias_qkv", bias_qkv)
    wbuf("Wo", matrix_o.reshape(-1))
    wbuf("bias_o", bias_o)
    wbuf("kcache", kcache_init.reshape(-1))
    wbuf("vcache", vcache_init.reshape(-1))
    wbuf("attn_out", attn_out)
    with open(os.path.join(args.out, "self_attn.elf"), "wb") as f:
        f.write(elf_bytes)

    meta = {
        "elf": "self_attn.elf",
        "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v_[0], "offset": int(v_[1]), "len": int(v_[2])} for n, v_ in lay.items()},
        "inputs": ["x"],
        "weights": ["Wqkv", "bias_qkv", "Wo", "bias_o", "kcache", "vcache"],
        "output": "attn_out",
        "patch": {
            "kv_cache_offsets": [int(lay["kcache"][1]), int(lay["vcache"][1])],
            "head_dim": HD, "num_preceding": P,
        },
        "dims": {"D": D, "H": H, "HD": HD, "S": S, "num_preceding": P, "layer": L},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote ELF ({len(elf_bytes)}B) + buffers + meta.json to {args.out}  (S={S}, P={P})")


if __name__ == "__main__":
    main()
