#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FE1 step-5: a FULL Whisper decoder LAYER as one fused full ELF (real whisper-small weights).

Composes the three validated blocks with residual threading, all in ONE dispatch:
    x1 = x  + self_attn(LN_self(x))                 # causal, KV cache (patched per token)
    x2 = x1 + cross_attn(LN_cross(x1))              # resident encoder K/V, static mask
    x3 = x2 + ffn(LN_final(x2))                     # GELU MLP
Each affine LN folds into its following projection (γ→weight, β→bias'); biases on-device; 1/√d folded
into the q projections. Op instances are reused where dims match (one LayerNorm(768), one GEMV(768,768)
for the projections, one Add(768) for residuals/biases) to keep the kernel count down.

Validated at the real shapes: self-attn S=448 multi-position (P prefilled past), cross-attn T_enc=1500.
This proves block composition + residual wiring; the whole-decode ELF (gen_decode.py) just stacks 12.
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
D, H, HD, QKV, FF = 768, 12, 64, 2304, 3072
KV_MAGIC, SM_MAGIC = 0xDEADBEE0, 0xBA5EBA11


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(w, L, n):
    return np.load(os.path.join(w, f"L{L}", f"{n}.npy")).astype(np.float32)


def ln(x):
    t = torch.from_numpy(x.astype(np.float32)).reshape(1, -1)
    return torch.nn.functional.layer_norm(t, normalized_shape=(t.shape[-1],)).numpy().reshape(-1)


def pick_tiling(M, N):
    for s in (8, 4):
        for m in sorted((d for d in range(s, M + 1) if M % d == 0 and d % s == 0), reverse=True):
            for n in sorted((d for d in range(s, N + 1) if N % d == 0 and d % s == 0), reverse=True):
                if m * n <= 8192 and not (s == 8 and (m <= 16 or n <= 16)):
                    return m, n, s
    raise ValueError("no tiling")


def attn_host(q, K, V):  # q[H,HD], K/V[H,T,HD] (f32) -> ctx[H,HD] with bf16 softmax
    out = np.zeros((H, HD), np.float32)
    for h in range(H):
        s = K[h] @ q[h]
        w = torch.softmax(torch.from_numpy(bf16(s).astype(np.float32)), 0).numpy()
        out[h] = bf16(w).astype(np.float32) @ V[h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-len", type=int, default=448)
    ap.add_argument("--num-preceding", type=int, default=5)
    ap.add_argument("--t-enc", type=int, default=1500)
    ap.add_argument("--t-pad", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    w, L, S, P, T, TP = a.weights, a.layer, a.prompt_len, a.num_preceding, a.t_enc, a.t_pad
    scale = 1.0 / np.sqrt(HD)
    rng = np.random.default_rng(a.seed)

    # ---------- weights + folds ----------
    g_s, b_s = npy(w, L, "ln_self.weight"), npy(w, L, "ln_self.bias")
    Wq, Wk, Wv = npy(w, L, "q.weight"), npy(w, L, "k.weight"), npy(w, L, "v.weight")
    bq, bk, bv = npy(w, L, "q.bias"), npy(w, L, "k.bias"), npy(w, L, "v.bias")
    Wso, bso = npy(w, L, "out.weight"), npy(w, L, "out.bias")
    Wqkv = np.concatenate([Wq, Wk, Wv], 1)
    mat_qkv = (g_s[:, None] * Wqkv).T.copy()
    bias_qkv = b_s @ Wqkv + np.concatenate([bq, bk, bv])
    mat_qkv[0:D] *= scale
    bias_qkv[0:D] *= scale

    g_c, b_c = npy(w, L, "ln_cross.weight"), npy(w, L, "ln_cross.bias")
    Wcq, bcq = npy(w, L, "cross_q.weight"), npy(w, L, "cross_q.bias")
    Wck = npy(w, L, "cross_k.weight")
    Wcv, bcv = npy(w, L, "cross_v.weight"), npy(w, L, "cross_v.bias")
    Wco, bco = npy(w, L, "cross_out.weight"), npy(w, L, "cross_out.bias")
    mat_cq = (g_c[:, None] * Wcq).T.copy() * scale
    bias_cq = (b_c @ Wcq + bcq) * scale

    g_f, b_f = npy(w, L, "ln_final.weight"), npy(w, L, "ln_final.bias")
    Wf1, bf1 = npy(w, L, "fc1.weight"), npy(w, L, "fc1.bias")
    Wf2, bf2 = npy(w, L, "fc2.weight"), npy(w, L, "fc2.bias")
    mat_f1 = (g_f[:, None] * Wf1).T.copy()
    bias_f1 = b_f @ Wf1 + bf1

    # encoder K/V (resident), per-head padded
    enc = rng.standard_normal((T, D)).astype(np.float32) * 0.5
    Kenc, Venc = enc @ Wck, enc @ Wcv + bcv
    def heads_pad(M):
        o = np.zeros((H, TP, HD), np.float32)
        o[:, 0:T, :] = M.reshape(T, H, HD).transpose(1, 0, 2)
        return bf16(o)
    Kenc_b, Venc_b = heads_pad(Kenc), heads_pad(Venc)

    # self KV prefill (P past)
    k_past = bf16(rng.standard_normal((H, P, HD)).astype(np.float32) * 0.5)
    v_past = bf16(rng.standard_normal((H, P, HD)).astype(np.float32) * 0.5)
    kc_init = np.zeros((H, S, HD), BF16); vc_init = np.zeros((H, S, HD), BF16)
    if P:
        kc_init[:, 0:P], vc_init[:, 0:P] = k_past, v_past

    tms, tns, tss = pick_tiling(S, HD)
    tmc, tnc, tsc = pick_tiling(TP, HD)

    # ---------- ops (reused where dims match) ----------
    ctx = AIEContext()
    op_ln = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, context=ctx)
    op_proj = GEMV(M=D, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    op_add768 = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)
    op_qkv = GEMV(M=QKV, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=QKV // 8, context=ctx)
    op_add_qkv = ElementwiseAdd(size=QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)
    sc = dict(input_sizes=(H, HD), input_strides=(HD, 1), input_offset=0, output_sizes=(1, H, HD),
              output_strides=(0, S * HD, 1), output_offset=0, input_buffer_size=H * HD,
              output_buffer_size=H * S * HD, num_aie_channels=1)
    op_sck = StridedCopy(**sc, kwargs={"output_offset_patch_marker": KV_MAGIC}, context=ctx)
    op_scv = StridedCopy(**sc, kwargs={"output_offset_patch_marker": KV_MAGIC}, context=ctx)
    op_sc_s = GEMV(M=S, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=S // 8, num_batches=H, context=ctx)
    op_sm_s = Softmax(rows=16, cols=S, num_aie_columns=1, num_channels=1, rtp_vector_size=SM_MAGIC, context=ctx)
    op_tr_s = Transpose(M=S, N=HD, num_aie_columns=2, num_channels=1, m=tms, n=tns, s=tss, context=ctx)
    op_ct_s = GEMV(M=HD, K=S, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=H, context=ctx)
    op_sc_c = GEMV(M=TP, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=TP // 8, num_batches=H, context=ctx)
    op_sm_c = Softmax(rows=16, cols=TP, num_aie_columns=1, num_channels=1, rtp_vector_size=T, context=ctx)
    op_tr_c = Transpose(M=TP, N=HD, num_aie_columns=2, num_channels=1, m=tmc, n=tnc, s=tsc, context=ctx)
    op_ct_c = GEMV(M=HD, K=TP, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=H, context=ctx)
    op_f1 = GEMV(M=FF, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=FF // 8, context=ctx)
    op_add_ff = ElementwiseAdd(size=FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)
    from iron.operators.gelu.op import GELU
    op_gelu = GELU(size=FF, num_aie_columns=8, num_channels=1, tile_size=FF // 8, context=ctx)
    op_f2 = GEMV(M=D, K=FF, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)

    HSs, HSc = H * S * 2, H * TP * 2
    phs, phc = S * HD * 2, TP * HD * 2
    rl = [
        # ---- self-attn: x1 = x + self(LN(x)) ----
        (op_ln, "x", "xn_s"),
        (op_qkv, "Wqkv", "xn_s", "qkv"), (op_add_qkv, "qkv", "bias_qkv", "qkv"),
        (op_sck, "qkv[1536:3072]", "kcache"), (op_scv, "qkv[3072:4608]", "vcache"),
        (op_sc_s, "kcache", "qkv[0:1536]", f"scs[0:{HSs}]"), (op_sm_s, "scs", "sws"),
    ] + [(op_tr_s, f"vcache[{h*phs}:{(h+1)*phs}]", f"vcT[{h*phs}:{(h+1)*phs}]") for h in range(H)] + [
        (op_ct_s, "vcT", f"sws[0:{HSs}]", "cts"),
        (op_proj, "Wso", "cts", "asf"), (op_add768, "asf", "bso", "asf"),
        (op_add768, "x", "asf", "x1"),
        # ---- cross-attn: x2 = x1 + cross(LN(x1)) ----
        (op_ln, "x1", "xn_c"),
        (op_proj, "Wcq", "xn_c", "qc"), (op_add768, "qc", "bias_cq", "qc"),
        (op_sc_c, "Kenc", "qc", f"scc[0:{HSc}]"), (op_sm_c, "scc", "swc"),
    ] + [(op_tr_c, f"Venc[{h*phc}:{(h+1)*phc}]", f"vcTc[{h*phc}:{(h+1)*phc}]") for h in range(H)] + [
        (op_ct_c, "vcTc", f"swc[0:{HSc}]", "ctc"),
        (op_proj, "Wco", "ctc", "acf"), (op_add768, "acf", "bco", "acf"),
        (op_add768, "x1", "acf", "x2"),
        # ---- ffn: x3 = x2 + ffn(LN(x2)) ----
        (op_ln, "x2", "xn_f"),
        (op_f1, "Wf1", "xn_f", "h"), (op_add_ff, "h", "bias_f1", "h"), (op_gelu, "h", "h"),
        (op_f2, "Wf2", "h", "ff"), (op_add768, "ff", "bf2", "ff"),
        (op_add768, "x2", "ff", "x3"),
    ]
    fused = FusedMLIROperator(
        "layer", rl, input_args=["x"], output_args=["x3"],
        buffer_sizes={
            "qkv": QKV * 2, "kcache": H * S * HD * 2, "vcache": H * S * HD * 2, "vcT": H * S * HD * 2,
            "scs": 16 * S * 2, "sws": 16 * S * 2,
            "Kenc": H * TP * HD * 2, "Venc": H * TP * HD * 2, "vcTc": H * TP * HD * 2,
            "scc": 16 * TP * 2, "swc": 16 * TP * 2,
        },
        context=ctx,
    )
    fused.compile()
    elf = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    wmap = ["Wqkv", "bias_qkv", "Wso", "bso", "Wcq", "bias_cq", "Wco", "bco",
            "Wf1", "bias_f1", "Wf2", "bf2", "Kenc", "Venc", "kcache", "vcache"]
    lay_names = ["x", "x3"] + wmap
    lay = {n: fused.get_layout_for_buffer(n) for n in lay_names}
    print("buffer_sizes:", fused.buffer_sizes)

    # ---------- device-faithful golden ----------
    x = bf16(rng.standard_normal(D).astype(np.float32))
    # self
    n1 = bf16(ln(x.astype(np.float32)))
    qkv = bf16(bf16(bf16(mat_qkv).astype(np.float32) @ n1.astype(np.float32)).astype(np.float32) + bf16(bias_qkv).astype(np.float32))
    q = qkv[0:D].reshape(H, HD); kcur = qkv[D:2*D].reshape(H, HD); vcur = qkv[2*D:3*D].reshape(H, HD)
    Ks = np.concatenate([k_past, kcur[:, None, :]], 1).astype(np.float32) if P else kcur[:, None, :].astype(np.float32)
    Vs = np.concatenate([v_past, vcur[:, None, :]], 1).astype(np.float32) if P else vcur[:, None, :].astype(np.float32)
    asf = bf16(bf16(attn_host(q.astype(np.float32), Ks, Vs).reshape(-1)))
    asf = bf16(bf16(bf16(Wso.T.copy()).astype(np.float32) @ asf.astype(np.float32)).astype(np.float32) + bf16(bso).astype(np.float32))
    x1 = bf16(x.astype(np.float32) + asf.astype(np.float32))
    # cross
    n2 = bf16(ln(x1.astype(np.float32)))
    qc = bf16(bf16(bf16(mat_cq).astype(np.float32) @ n2.astype(np.float32)).astype(np.float32) + bf16(bias_cq).astype(np.float32)).reshape(H, HD)
    ctc = attn_host(qc.astype(np.float32), Kenc_b[:, 0:T].astype(np.float32), Venc_b[:, 0:T].astype(np.float32)).reshape(-1)
    acf = bf16(bf16(bf16(Wco.T.copy()).astype(np.float32) @ bf16(ctc).astype(np.float32)).astype(np.float32) + bf16(bco).astype(np.float32))
    x2 = bf16(x1.astype(np.float32) + acf.astype(np.float32))
    # ffn
    n3 = bf16(ln(x2.astype(np.float32)))
    h1 = bf16(bf16(bf16(mat_f1).astype(np.float32) @ n3.astype(np.float32)).astype(np.float32) + bf16(bias_f1).astype(np.float32))
    h2 = bf16(torch.nn.functional.gelu(torch.from_numpy(h1.astype(np.float32)), approximate="tanh").numpy())
    ff = bf16(bf16(bf16(Wf2.T.copy()).astype(np.float32) @ h2.astype(np.float32)).astype(np.float32) + bf16(bf2).astype(np.float32))
    x3 = bf16(x2.astype(np.float32) + ff.astype(np.float32))

    def wb(n, v): open(os.path.join(a.out, "buffers", f"{n}.bin"), "wb").write(np.asarray(v, BF16).tobytes())
    wb("x", x); wb("x3", x3)
    wb("Wqkv", bf16(mat_qkv).reshape(-1)); wb("bias_qkv", bf16(bias_qkv))
    wb("Wso", bf16(Wso.T.copy()).reshape(-1)); wb("bso", bf16(bso))
    wb("Wcq", bf16(mat_cq).reshape(-1)); wb("bias_cq", bf16(bias_cq))
    wb("Wco", bf16(Wco.T.copy()).reshape(-1)); wb("bco", bf16(bco))
    wb("Wf1", bf16(mat_f1).reshape(-1)); wb("bias_f1", bf16(bias_f1))
    wb("Wf2", bf16(Wf2.T.copy()).reshape(-1)); wb("bf2", bf16(bf2))
    wb("Kenc", Kenc_b.reshape(-1)); wb("Venc", Venc_b.reshape(-1))
    wb("kcache", kc_init.reshape(-1)); wb("vcache", vc_init.reshape(-1))
    open(os.path.join(a.out, "layer.elf"), "wb").write(elf)

    meta = {
        "elf": "layer.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"], "weights": wmap, "output": "x3",
        "patch": {"kv_cache_offsets": [int(lay["kcache"][1]), int(lay["vcache"][1])], "head_dim": HD, "num_preceding": P},
        "dims": {"S": S, "P": P, "T_enc": T, "T_pad": TP, "layer": L},
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\nwrote layer ELF ({len(elf)}B) to {a.out}  (S={S} P={P} T={T})")


if __name__ == "__main__":
    main()
