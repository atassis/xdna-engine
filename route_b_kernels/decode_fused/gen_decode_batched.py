#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batched fused decode — Tasks 5+6: the WHOLE Whisper decoder stack (N layers) for B streams as ONE
fused full ELF. Forked from gen_decode.py (M=1), composing the device-validated batched blocks:
  self-attn ([[batched-selfattn-block]]) + residual + cross-attn (Task 4) + residual + FFN
  ([[batched-decode-elf-ffn]]) + residual, all stream-major [B,feat] with GEMM-N=B projections.

v1 SCOPE (offline-bulk lockstep): all B streams share the decode position, so the self KV-write offset
is a CONSTANT (position P) and softmax needs no mask (S = self context, TP = cross context). Per-stream
position vectors (dynamic batching) are the deferred scope ([[generalize-resident-scratchpad-decode]]).

Ops are created ONCE and reused across all layers (same shapes); per-layer weight + cache buffers.
MEMORY: self KV [B,H,S,HD] + cross Kenc/Venc [B,H,TP,HD] per layer scale with B·layers — use a small
B/S/T for 12-layer validation (the full-array B=128 12-layer needs the arena cap the plan flags).

Gate: rel-L2(device out, per-stream N-layer bf16 golden) <= 0.08. Output = x after N layers (host does
ln_post + lm-head, as M=1). Run inside the IRON env. Validate --layers 2 first, then --layers 12.
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
from iron.operators.strided_copy.op import StridedCopy
from iron.operators.transpose.op import Transpose
from iron.operators.gelu.op import GELU

BF16 = ml_dtypes.bfloat16
D, H, HD, QKV, FF = 768, 12, 64, 2304, 3072


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(w, L, n):
    return np.load(os.path.join(w, f"L{L}", f"{n}.npy")).astype(np.float32)


def ln(x):
    t = torch.from_numpy(x.astype(np.float32)).reshape(1, -1)
    return torch.nn.functional.layer_norm(t, normalized_shape=(t.shape[-1],)).numpy().reshape(-1)


def gelu_t(h):
    return torch.nn.functional.gelu(torch.from_numpy(h.astype(np.float32)), approximate="tanh").numpy()


def pick_tt(M, N, ncols=1, nch=1):
    # Largest valid (m,n,s) transpose tile. The IRON Transpose design splits N across COLUMNS and M
    # across CHANNELS (design.py taps_in_L3L2: sizes [M//nch//m, N//ncols//n, m, n]), so the tile must
    # satisfy n | (N//ncols), m | (M//nch), plus the s-floor and m*n<=8192. Prefer s=8 (efficient kernel).
    Nc, Mc = N // ncols, M // nch
    for s in (8, 4):
        for m in sorted((d for d in range(s, Mc + 1) if Mc % d == 0 and d % s == 0), reverse=True):
            for n in sorted((d for d in range(s, Nc + 1) if Nc % d == 0 and d % s == 0), reverse=True):
                if m * n > 8192:
                    continue
                if s == 8 and (m <= 16 or n <= 16):
                    continue
                if s == 4 and (m <= 4 or n <= 4):
                    continue
                return m, n, s
    raise ValueError(f"no transpose tiling {M}x{N} ncols={ncols} nch={nch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--S", type=int, default=64, help="self-attn context (full cache, no mask)")
    ap.add_argument("--T", type=int, default=128, help="cross-attn encoder length T_enc (real, masked width)")
    ap.add_argument("--t-pad", type=int, default=-1, help="cross-attn padded length (default ceil(T/64)*64)")
    ap.add_argument("--scratchpad", action="store_true", help="deep-C runtime kv_off/sm_mask params (engine mode)")
    ap.add_argument("--engine-only", action="store_true", help="skip golden + per-utterance buffers (engine fills Kenc/Venc/kcache); fast/lean build for the host driver")
    ap.add_argument("--P", type=int, default=-1, help="prefilled self positions (default S-1); current token at pos P")
    ap.add_argument("--occ", action="store_true", help="O18: spread the 1-column softmaxes/transposes across more cores (sm 1->8, tr_s 1->2, tr_c 1->2; transposes cap at 2 clean s=8 cols since N=HD=64) — fills the array residual that stays at 4/32 cores even at B>=128")
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    B, NL, S, T = a.B, a.layers, a.S, a.T
    TP = a.t_pad if a.t_pad > 0 else ((T + 63) // 64) * 64  # padded cross length (%64)
    P = a.P if a.P >= 0 else S - 1
    sp = a.scratchpad
    eng = a.engine_only
    assert P < S
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    w = a.weights
    scale = 1.0 / np.sqrt(HD)
    rng = np.random.default_rng(a.seed)
    g_cols = 8 if B >= 128 else 1
    assert B % (16 * g_cols) == 0 and S % 8 == 0 and TP % 64 == 0
    BH = B * H
    CH = 16
    assert B % CH == 0
    # O18 occupancy: lift the four 1-column ops off 4/32 cores. The transposes split N=HD=64 across
    # columns, so s=8 (n>16) caps them at 2 clean columns (n=32); the softmaxes split BH rows and reach 8.
    sm_cols = 8 if a.occ else 1
    tr_s_cols = 2 if a.occ else 1
    tr_c_cols = 2 if a.occ else 1
    tms, tns, tss = pick_tt(S, HD, tr_s_cols)
    tmc, tnc, tsc = pick_tt(TP, HD, tr_c_cols)

    # ---- ops (created once, reused for all layers) ----
    ctx = AIEContext()
    op_ln = LayerNorm(size=CH * D, num_aie_columns=1, num_channels=CH, tile_size=D, context=ctx)
    g_qkv = GEMM(M=QKV, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=g_cols, b_col_maj=True, c_col_maj=True, context=ctx)
    g_proj = GEMM(M=D, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=g_cols, b_col_maj=True, c_col_maj=True, context=ctx)
    g_f1 = GEMM(M=FF, K=D, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=g_cols, b_col_maj=True, c_col_maj=True, context=ctx)
    g_f2 = GEMM(M=D, K=FF, N=B, tile_m=64, tile_k=64, tile_n=16, num_aie_columns=g_cols, b_col_maj=True, c_col_maj=True, context=ctx)
    add_qkv = ElementwiseAdd(size=B * QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)
    add_d = ElementwiseAdd(size=B * D, tile_size=D // 8, num_aie_columns=8, context=ctx)
    add_ff = ElementwiseAdd(size=B * FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)
    gelu = GELU(size=B * FF, num_aie_columns=8, num_channels=1, tile_size=FF // 8, context=ctx)
    q_ex = StridedCopy(input_sizes=(B, H, HD), input_strides=(QKV, HD, 1), input_offset=0,
                       output_sizes=(B, H, HD), output_strides=(H * HD, HD, 1), output_offset=0,
                       input_buffer_size=B * QKV, output_buffer_size=B * H * HD, num_aie_channels=1, context=ctx)
    # scratchpad mode (engine): kv write offset + self-softmax mask are RUNTIME params (deep-C); else baked.
    sc_kw = {"output_offset_scratchpad": "kv_off"} if sp else {}
    sc_off = 0 if sp else P * HD
    sc_k = StridedCopy(input_sizes=(B, H, HD), input_strides=(QKV, HD, 1), input_offset=D,
                       output_sizes=(B, H, HD), output_strides=(H * S * HD, S * HD, 1), output_offset=sc_off,
                       input_buffer_size=B * QKV, output_buffer_size=B * H * S * HD, num_aie_channels=1, kwargs=sc_kw, context=ctx)
    sc_v = StridedCopy(input_sizes=(B, H, HD), input_strides=(QKV, HD, 1), input_offset=2 * D,
                       output_sizes=(B, H, HD), output_strides=(H * S * HD, S * HD, 1), output_offset=sc_off,
                       input_buffer_size=B * QKV, output_buffer_size=B * H * S * HD, num_aie_channels=1, kwargs=sc_kw, context=ctx)
    g_scs = GEMV(M=S, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=S // 8, num_batches=BH, context=ctx)
    sm_s = (Softmax(rows=BH, cols=S, num_aie_columns=sm_cols, num_channels=1, rtp_vector_size=S, mask_scratchpad="sm_mask", context=ctx)
            if sp else
            Softmax(rows=BH, cols=S, num_aie_columns=sm_cols, num_channels=1, rtp_vector_size=S, mask_patch_value=0, context=ctx))
    tr_s = Transpose(M=S, N=HD, num_batches=BH, num_aie_columns=tr_s_cols, num_channels=1, m=tms, n=tns, s=tss, context=ctx)
    g_cts = GEMV(M=HD, K=S, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=BH, context=ctx)
    g_scc = GEMV(M=TP, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=TP // 8, num_batches=BH, context=ctx)
    sm_c = Softmax(rows=BH, cols=TP, num_aie_columns=sm_cols, num_channels=1, rtp_vector_size=T, mask_patch_value=0, context=ctx)
    tr_c = Transpose(M=TP, N=HD, num_batches=BH, num_aie_columns=tr_c_cols, num_channels=1, m=tmc, n=tnc, s=tsc, context=ctx)
    g_ctc = GEMV(M=HD, K=TP, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=BH, context=ctx)

    chD = CH * D * 2
    HSs, HSc = BH * S * 2, BH * TP * 2
    rl = []
    bufsz = {}
    weights_to_write = {}
    util_names = []  # per-utterance buffer names (Kenc/Venc/kcache/vcache) for layout+meta
    cur = "x"
    layer_data = []

    def ln_chunks(src, dst):
        return [(op_ln, f"{src}[{c*chD}:{(c+1)*chD}]", f"{dst}[{c*chD}:{(c+1)*chD}]") for c in range(B // CH)]

    for l in range(NL):
        p = f"L{l}_"
        # ---- load + fold weights ----
        g_s, b_s = npy(w, l, "ln_self.weight"), npy(w, l, "ln_self.bias")
        Wq, Wk, Wv = npy(w, l, "q.weight"), npy(w, l, "k.weight"), npy(w, l, "v.weight")
        bq, bk, bv = npy(w, l, "q.bias"), npy(w, l, "k.bias"), npy(w, l, "v.bias")
        Wso, bso = npy(w, l, "out.weight"), npy(w, l, "out.bias")
        Wqkv = np.concatenate([Wq, Wk, Wv], 1)
        mat_qkv = (g_s[:, None] * Wqkv).T.copy(); bias_qkv = b_s @ Wqkv + np.concatenate([bq, bk, bv])
        mat_qkv[0:D] *= scale; bias_qkv[0:D] *= scale
        g_c, b_c = npy(w, l, "ln_cross.weight"), npy(w, l, "ln_cross.bias")
        Wcq, bcq = npy(w, l, "cross_q.weight"), npy(w, l, "cross_q.bias")
        Wck = npy(w, l, "cross_k.weight"); Wcv, bcv = npy(w, l, "cross_v.weight"), npy(w, l, "cross_v.bias")
        Wco, bco = npy(w, l, "cross_out.weight"), npy(w, l, "cross_out.bias")
        mat_cq = (g_c[:, None] * Wcq).T.copy() * scale; bias_cq = (b_c @ Wcq + bcq) * scale
        g_f, b_f = npy(w, l, "ln_final.weight"), npy(w, l, "ln_final.bias")
        Wf1, bf1 = npy(w, l, "fc1.weight"), npy(w, l, "fc1.bias")
        Wf2, bf2 = npy(w, l, "fc2.weight"), npy(w, l, "fc2.bias")
        mat_f1 = (g_f[:, None] * Wf1).T.copy(); bias_f1 = b_f @ Wf1 + bf1

        # per-stream encoder K/V + prefilled self KV (only materialised when NOT engine-only — in
        # engine-only mode the host driver fills Kenc/Venc/kcache/vcache per utterance).
        def heads_pad(M):  # [T,D] -> [H,TP,HD], rows >= T zero (pad masked by cross softmax rtp=T)
            o = np.zeros((H, TP, HD), np.float32)
            o[:, 0:T, :] = M.reshape(T, H, HD).transpose(1, 0, 2)
            return o
        def tile(v1, n):
            return bf16(np.tile(bf16(v1).astype(np.float32), (B, 1)))
        # shared (static) weights — always written:
        for nm, arr in [("Wqkv", bf16(mat_qkv).reshape(-1)), ("bias_qkv", tile(bias_qkv, QKV).reshape(-1)),
                        ("Wso", bf16(Wso.T.copy()).reshape(-1)), ("bso", tile(bso, D).reshape(-1)),
                        ("Wcq", bf16(mat_cq).reshape(-1)), ("bias_cq", tile(bias_cq, D).reshape(-1)),
                        ("Wco", bf16(Wco.T.copy()).reshape(-1)), ("bco", tile(bco, D).reshape(-1)),
                        ("Wf1", bf16(mat_f1).reshape(-1)), ("bias_f1", tile(bias_f1, FF).reshape(-1)),
                        ("Wf2", bf16(Wf2.T.copy()).reshape(-1)), ("bf2", tile(bf2, D).reshape(-1))]:
            weights_to_write[p + nm] = arr
        # per-utterance buffers — names registered for layout+meta; arrays written only when !engine-only.
        util_names += [p + "Kenc", p + "Venc", p + "kcache", p + "vcache"]
        if eng:
            k_past = v_past = None
        else:
            Kenc = np.zeros((B, H, TP, HD), np.float32); Venc = np.zeros((B, H, TP, HD), np.float32)
            for b in range(B):
                enc = rng.standard_normal((T, D)).astype(np.float32) * 0.5
                Kenc[b] = heads_pad(enc @ Wck); Venc[b] = heads_pad(enc @ Wcv + bcv)
            k_past = bf16(rng.standard_normal((B, H, P, HD)).astype(np.float32) * 0.5)
            v_past = bf16(rng.standard_normal((B, H, P, HD)).astype(np.float32) * 0.5)
            kc = np.zeros((B, H, S, HD), BF16); vc = np.zeros((B, H, S, HD), BF16)
            kc[:, :, 0:P], vc[:, :, 0:P] = k_past, v_past
            weights_to_write[p + "Kenc"] = bf16(Kenc).reshape(-1)
            weights_to_write[p + "Venc"] = bf16(Venc).reshape(-1)
            weights_to_write[p + "kcache"] = kc.reshape(-1)
            weights_to_write[p + "vcache"] = vc.reshape(-1)
        bufsz.update({
            p + "x_norm": B * D * 2, p + "qkv": B * QKV * 2, p + "qbuf": B * H * HD * 2,
            p + "kcache": B * H * S * HD * 2, p + "vcache": B * H * S * HD * 2, p + "vcT": B * H * S * HD * 2,
            p + "scs": HSs, p + "sws": HSs, p + "Kenc": B * H * TP * HD * 2, p + "Venc": B * H * TP * HD * 2,
            p + "vcTc": B * H * TP * HD * 2, p + "scc": HSc, p + "swc": HSc,
        })
        nxt = f"x{l+1}"
        rl += ln_chunks(cur, p + "xn_s") + [
            (g_qkv, p + "Wqkv", p + "xn_s", p + "qkv"), (add_qkv, p + "qkv", p + "bias_qkv", p + "qkv"),
            (q_ex, p + "qkv", p + "qbuf"),
            (sc_k, p + "qkv", p + "kcache"), (sc_v, p + "qkv", p + "vcache"),
            (g_scs, p + "kcache", p + "qbuf", f"{p}scs[0:{HSs}]"), (sm_s, p + "scs", p + "sws"),
            (tr_s, p + "vcache", p + "vcT"),
            (g_cts, p + "vcT", f"{p}sws[0:{HSs}]", p + "cts"),
            (g_proj, p + "Wso", p + "cts", p + "asf"), (add_d, p + "asf", p + "bso", p + "asf"),
            (add_d, cur, p + "asf", p + "x1"),
        ] + ln_chunks(p + "x1", p + "xn_c") + [
            (g_proj, p + "Wcq", p + "xn_c", p + "qc"), (add_d, p + "qc", p + "bias_cq", p + "qc"),
            (g_scc, p + "Kenc", p + "qc", f"{p}scc[0:{HSc}]"), (sm_c, p + "scc", p + "swc"),
            (tr_c, p + "Venc", p + "vcTc"),
            (g_ctc, p + "vcTc", f"{p}swc[0:{HSc}]", p + "ctc"),
            (g_proj, p + "Wco", p + "ctc", p + "acf"), (add_d, p + "acf", p + "bco", p + "acf"),
            (add_d, p + "x1", p + "acf", p + "x2"),
        ] + ln_chunks(p + "x2", p + "xn_f") + [
            (g_f1, p + "Wf1", p + "xn_f", p + "h"), (add_ff, p + "h", p + "bias_f1", p + "h"), (gelu, p + "h", p + "h"),
            (g_f2, p + "Wf2", p + "h", p + "ff"), (add_d, p + "ff", p + "bf2", p + "ff"),
            (add_d, p + "x2", p + "ff", nxt),
        ]
        bufsz.update({p + "qc": B * D * 2, p + "asf": B * D * 2, p + "acf": B * D * 2,
                      p + "cts": B * H * HD * 2, p + "ctc": B * H * HD * 2, p + "x1": B * D * 2,
                      p + "x2": B * D * 2, p + "xn_c": B * D * 2, p + "xn_f": B * D * 2,
                      p + "h": B * FF * 2, p + "ff": B * D * 2, p + "xn_s": B * D * 2})
        if not eng:
            layer_data.append(dict(mat_qkv=bf16(mat_qkv), bias_qkv=bf16(bias_qkv), Wso=bf16(Wso.T.copy()), bso=bf16(bso),
                                   mat_cq=bf16(mat_cq), bias_cq=bf16(bias_cq), Wco=bf16(Wco.T.copy()), bco=bf16(bco),
                                   mat_f1=bf16(mat_f1), bias_f1=bf16(bias_f1), Wf2=bf16(Wf2.T.copy()), bf2=bf16(bf2),
                                   Kenc=bf16(Kenc), Venc=bf16(Venc), k_past=k_past, v_past=v_past))
        cur = nxt

    out_name = cur
    bufsz["x"] = B * D * 2
    fused = FusedMLIROperator("decode_b", rl, input_args=["x"], output_args=[out_name], buffer_sizes=bufsz, context=ctx)
    fused.compile()
    elf = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    # all weight names = shared (written) + per-utterance util (engine-filled), deduped, for layout+meta.
    wnames = list(dict.fromkeys(list(weights_to_write.keys()) + util_names))
    lay = {n: fused.get_layout_for_buffer(n) for n in ["x", out_name] + wnames}

    # scratchpad mode: parse the StateTable layout (params.txt) aiecc emits, so the host knows each
    # param's byte offset + kind (deep-C; same scheme as gen_decode.py).
    scratchpad_params = {}
    if sp:
        import glob as _glob, shutil as _shutil
        _pp = sorted(_glob.glob("**/decode_b*.mlir.prj/params.txt", recursive=True), key=os.path.getmtime)
        assert _pp, "scratchpad mode but no params.txt found (StateTable not emitted)"
        _shutil.copy(_pp[-1], os.path.join(a.out, "params.txt"))
        for line in open(_pp[-1]).read().splitlines()[1:]:
            if line.strip():
                nm, idx, ty, kind = line.split()
                scratchpad_params[nm] = {"byte_offset": int(idx) * 4, "kind": kind, "dtype": ty}
        assert "kv_off" in scratchpad_params and "sm_mask" in scratchpad_params, f"params: {scratchpad_params}"

    bdir = os.path.join(a.out, "buffers")
    def wb(n, v): open(os.path.join(bdir, f"{n}.bin"), "wb").write(np.asarray(v, BF16).tobytes())

    # ---- engine-only: write shared weights + ELF + meta, skip golden/per-utterance buffers ----
    if eng:
        for nm, arr in weights_to_write.items():
            wb(nm, arr)
        open(os.path.join(a.out, "decode_b.elf"), "wb").write(elf)
        meta = {
            "elf": "decode_b.elf", "kernel_name": "main:sequence",
            "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
            "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
            "inputs": ["x"], "weights": wnames, "output": out_name,
            "dims": {"layers": NL, "B": B, "S": S, "T": T, "P": P},
            "scratchpad": {"params": scratchpad_params, "kv_param": "kv_off", "mask_param": "sm_mask",
                           "head_dim": HD, "num_preceding": P} if sp else None,
        }
        if not sp:
            del meta["scratchpad"]
        json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
        print(f"\nwrote {NL}-layer batched decode ELF [engine-only] ({len(elf)}B, scratch {scr/1e6:.1f}MB) B={B} to {a.out}")
        return

    # ---- per-stream N-layer bf16 golden ----
    def attn(q, K, V):
        o = np.zeros((H, HD), np.float32)
        for h in range(H):
            s = K[h] @ q[h]
            wts = torch.softmax(torch.from_numpy(bf16(s).astype(np.float32)), 0).numpy()
            o[h] = bf16(wts).astype(np.float32) @ V[h]
        return o
    X = bf16(rng.standard_normal((B, D)).astype(np.float32))
    outX = np.zeros((B, D), BF16)
    for b in range(B):
        x = X[b]
        for l in range(NL):
            d = layer_data[l]
            n1 = bf16(ln(x.astype(np.float32)))
            qkv = bf16(bf16(d["mat_qkv"].astype(np.float32) @ n1.astype(np.float32)).astype(np.float32) + d["bias_qkv"].astype(np.float32))
            q = qkv[0:D].reshape(H, HD); kcur = qkv[D:2*D].reshape(H, HD); vcur = qkv[2*D:3*D].reshape(H, HD)
            Ks = np.concatenate([d["k_past"][b].astype(np.float32), kcur.astype(np.float32)[:, None]], 1)
            Vs = np.concatenate([d["v_past"][b].astype(np.float32), vcur.astype(np.float32)[:, None]], 1)
            asf = bf16(attn(q.astype(np.float32), Ks, Vs).reshape(-1))
            asf = bf16(bf16(d["Wso"].astype(np.float32) @ asf.astype(np.float32)).astype(np.float32) + d["bso"].astype(np.float32))
            x1 = bf16(x.astype(np.float32) + asf.astype(np.float32))
            n2 = bf16(ln(x1.astype(np.float32)))
            qc = bf16(bf16(d["mat_cq"].astype(np.float32) @ n2.astype(np.float32)).astype(np.float32) + d["bias_cq"].astype(np.float32)).reshape(H, HD)
            ctc = attn(qc.astype(np.float32), d["Kenc"][b][:, 0:T].astype(np.float32), d["Venc"][b][:, 0:T].astype(np.float32)).reshape(-1)
            acf = bf16(bf16(d["Wco"].astype(np.float32) @ bf16(ctc).astype(np.float32)).astype(np.float32) + d["bco"].astype(np.float32))
            x2 = bf16(x1.astype(np.float32) + acf.astype(np.float32))
            n3 = bf16(ln(x2.astype(np.float32)))
            h1 = bf16(bf16(d["mat_f1"].astype(np.float32) @ n3.astype(np.float32)).astype(np.float32) + d["bias_f1"].astype(np.float32))
            h2 = bf16(gelu_t(h1.astype(np.float32)))
            ff = bf16(bf16(d["Wf2"].astype(np.float32) @ h2.astype(np.float32)).astype(np.float32) + d["bf2"].astype(np.float32))
            x = bf16(x2.astype(np.float32) + ff.astype(np.float32))
        outX[b] = x

    bdir = os.path.join(a.out, "buffers")
    def wb(n, v): open(os.path.join(bdir, f"{n}.bin"), "wb").write(np.asarray(v, BF16).tobytes())
    wb("x", X.reshape(-1)); wb(out_name, outX.reshape(-1))
    for nm, arr in weights_to_write.items():
        wb(nm, arr)
    open(os.path.join(a.out, "decode_b.elf"), "wb").write(elf)

    meta = {
        "elf": "decode_b.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"], "weights": wnames, "output": out_name,
        "dims": {"layers": NL, "B": B, "S": S, "T": T, "P": P},
    }
    if sp:
        meta["scratchpad"] = {"params": scratchpad_params, "kv_param": "kv_off", "mask_param": "sm_mask",
                              "head_dim": HD, "num_preceding": P}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\nwrote {NL}-layer batched decode ELF ({len(elf)}B, scratch {scr/1e6:.1f}MB) B={B} to {a.out}")


if __name__ == "__main__":
    main()
