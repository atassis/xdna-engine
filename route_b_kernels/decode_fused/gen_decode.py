#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FE1 step-5: WHOLE Whisper decode transformer stack (N layers) as ONE fused full ELF.

Stacks N decoder layers (each: self-attn + residual, cross-attn + residual, FFN + residual) into a
single dispatch — the dispatch-collapse end state (1 NPU dispatch for the entire per-token decoder
stack; lm-head/logits stay host per the vocab-padding constraint). Op instances are created ONCE and
reused across all layers (same dims), so 12 layers share the same kernels. Per-layer resident state:
weights, encoder K/V (static), self-attn KV cache (per-token strided-copy, patched). The
FusedElfPatcher rewrites all 2N KV-write offsets + N self-softmax masks per token.

Validate small first (--layers 2) to prove stacking + multi-layer patch, then --layers 12.
Output = x after N layers (host does ln_post affine + lm-head). Run inside IRON env.
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
from iron.operators.gemv.op import GEMV
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.elementwise_add.op import ElementwiseAdd
from iron.operators.softmax.op import Softmax
from iron.operators.strided_copy.op import StridedCopy
from iron.operators.transpose.op import Transpose
from iron.operators.gelu.op import GELU

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


def attn_host(q, K, V):
    out = np.zeros((H, HD), np.float32)
    for h in range(H):
        s = K[h] @ q[h]
        w = torch.softmax(torch.from_numpy(bf16(s).astype(np.float32)), 0).numpy()
        out[h] = bf16(w).astype(np.float32) @ V[h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--prompt-len", type=int, default=448)
    ap.add_argument("--num-preceding", type=int, default=5)
    ap.add_argument("--t-enc", type=int, default=1500)
    ap.add_argument("--t-pad", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    w, NL, S, P, T, TP = a.weights, a.layers, a.prompt_len, a.num_preceding, a.t_enc, a.t_pad
    scale = 1.0 / np.sqrt(HD)
    rng = np.random.default_rng(a.seed)
    tms, tns, tss = pick_tiling(S, HD)
    tmc, tnc, tsc = pick_tiling(TP, HD)

    # ---- ops (created once, reused for all layers) ----
    ctx = AIEContext()
    op_ln = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, context=ctx)
    op_proj = GEMV(M=D, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    op_add768 = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)
    op_qkv = GEMV(M=QKV, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=QKV // 8, context=ctx)
    op_add_qkv = ElementwiseAdd(size=QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)
    sc = dict(input_sizes=(H, HD), input_strides=(HD, 1), input_offset=0, output_sizes=(1, H, HD),
              output_strides=(0, S * HD, 1), output_offset=0, input_buffer_size=H * HD,
              output_buffer_size=H * S * HD, num_aie_channels=1)
    # Deep-C: the per-token KV-write position offset is now a runtime `addr`-kind scratchpad param
    # (shared symbol "kv_off", element units = n_self*head_dim) instead of a per-token ELF patch →
    # the decode ELF is CONSTANT across tokens (registered once; host writes the offset per dispatch).
    op_sck = StridedCopy(**sc, kwargs={"output_offset_scratchpad": "kv_off"}, context=ctx)
    op_scv = StridedCopy(**sc, kwargs={"output_offset_scratchpad": "kv_off"}, context=ctx)
    op_sc_s = GEMV(M=S, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=S // 8, num_batches=H, context=ctx)
    # Deep-C: the per-token self-softmax mask width is now a runtime `core`-kind scratchpad param
    # (symbol "sm_mask", element units = context_len = n_self) read on-tile, instead of an ELF patch.
    op_sm_s = Softmax(rows=16, cols=S, num_aie_columns=1, num_channels=1, rtp_vector_size=S, mask_scratchpad="sm_mask", context=ctx)
    op_tr_s = Transpose(M=S, N=HD, num_aie_columns=2, num_channels=1, m=tms, n=tns, s=tss, context=ctx)
    op_ct_s = GEMV(M=HD, K=S, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=H, context=ctx)
    op_sc_c = GEMV(M=TP, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=TP // 8, num_batches=H, context=ctx)
    op_sm_c = Softmax(rows=16, cols=TP, num_aie_columns=1, num_channels=1, rtp_vector_size=T, mask_patch_value=0, context=ctx)
    op_tr_c = Transpose(M=TP, N=HD, num_aie_columns=2, num_channels=1, m=tmc, n=tnc, s=tsc, context=ctx)
    op_ct_c = GEMV(M=HD, K=TP, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=H, context=ctx)
    op_f1 = GEMV(M=FF, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=FF // 8, context=ctx)
    op_add_ff = ElementwiseAdd(size=FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)
    op_gelu = GELU(size=FF, num_aie_columns=8, num_channels=1, tile_size=FF // 8, context=ctx)
    op_f2 = GEMV(M=D, K=FF, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)

    HSs, HSc, phs, phc = H * S * 2, H * TP * 2, S * HD * 2, TP * HD * 2
    rl = []
    bufsz = {}
    weights_to_write = {}   # name -> bf16 array
    patch_offsets_names = []  # cache buffer names (for patch offsets)
    cur = "x"               # running residual buffer name

    # per-layer host state for golden
    layer_data = []

    for l in range(NL):
        pre = f"L{l}_"
        # --- load + fold weights ---
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
        enc = rng.standard_normal((T, D)).astype(np.float32) * 0.5
        Kenc, Venc = enc @ Wck, enc @ Wcv + bcv
        def heads_pad(M):
            o = np.zeros((H, TP, HD), np.float32); o[:, 0:T, :] = M.reshape(T, H, HD).transpose(1, 0, 2); return bf16(o)
        Kenc_b, Venc_b = heads_pad(Kenc), heads_pad(Venc)
        k_past = bf16(rng.standard_normal((H, P, HD)).astype(np.float32) * 0.5)
        v_past = bf16(rng.standard_normal((H, P, HD)).astype(np.float32) * 0.5)
        kc = np.zeros((H, S, HD), BF16); vc = np.zeros((H, S, HD), BF16)
        if P: kc[:, 0:P], vc[:, 0:P] = k_past, v_past

        # --- register weight buffers ---
        for nm, arr in [("Wqkv", bf16(mat_qkv).reshape(-1)), ("bias_qkv", bf16(bias_qkv)),
                        ("Wso", bf16(Wso.T.copy()).reshape(-1)), ("bso", bf16(bso)),
                        ("Wcq", bf16(mat_cq).reshape(-1)), ("bias_cq", bf16(bias_cq)),
                        ("Wco", bf16(Wco.T.copy()).reshape(-1)), ("bco", bf16(bco)),
                        ("Wf1", bf16(mat_f1).reshape(-1)), ("bias_f1", bf16(bias_f1)),
                        ("Wf2", bf16(Wf2.T.copy()).reshape(-1)), ("bf2", bf16(bf2)),
                        ("Kenc", Kenc_b.reshape(-1)), ("Venc", Venc_b.reshape(-1)),
                        ("kcache", kc.reshape(-1)), ("vcache", vc.reshape(-1))]:
            weights_to_write[pre + nm] = arr
        patch_offsets_names += [pre + "kcache", pre + "vcache"]
        # explicit sizes for sliced/cache/score buffers
        bufsz.update({
            pre + "qkv": QKV * 2, pre + "kcache": H * S * HD * 2, pre + "vcache": H * S * HD * 2, pre + "vcT": H * S * HD * 2,
            pre + "scs": 16 * S * 2, pre + "sws": 16 * S * 2,
            pre + "Kenc": H * TP * HD * 2, pre + "Venc": H * TP * HD * 2, pre + "vcTc": H * TP * HD * 2,
            pre + "scc": 16 * TP * 2, pre + "swc": 16 * TP * 2,
        })

        nxt = f"x{l+1}"  # layer output residual buffer
        rl += [
            (op_ln, cur, pre + "xn_s"),
            (op_qkv, pre + "Wqkv", pre + "xn_s", pre + "qkv"), (op_add_qkv, pre + "qkv", pre + "bias_qkv", pre + "qkv"),
            (op_sck, pre + "qkv[1536:3072]", pre + "kcache"), (op_scv, pre + "qkv[3072:4608]", pre + "vcache"),
            (op_sc_s, pre + "kcache", pre + "qkv[0:1536]", f"{pre}scs[0:{HSs}]"), (op_sm_s, pre + "scs", pre + "sws"),
        ] + [(op_tr_s, f"{pre}vcache[{h*phs}:{(h+1)*phs}]", f"{pre}vcT[{h*phs}:{(h+1)*phs}]") for h in range(H)] + [
            (op_ct_s, pre + "vcT", f"{pre}sws[0:{HSs}]", pre + "cts"),
            (op_proj, pre + "Wso", pre + "cts", pre + "asf"), (op_add768, pre + "asf", pre + "bso", pre + "asf"),
            (op_add768, cur, pre + "asf", pre + "x1"),
            (op_ln, pre + "x1", pre + "xn_c"),
            (op_proj, pre + "Wcq", pre + "xn_c", pre + "qc"), (op_add768, pre + "qc", pre + "bias_cq", pre + "qc"),
            (op_sc_c, pre + "Kenc", pre + "qc", f"{pre}scc[0:{HSc}]"), (op_sm_c, pre + "scc", pre + "swc"),
        ] + [(op_tr_c, f"{pre}Venc[{h*phc}:{(h+1)*phc}]", f"{pre}vcTc[{h*phc}:{(h+1)*phc}]") for h in range(H)] + [
            (op_ct_c, pre + "vcTc", f"{pre}swc[0:{HSc}]", pre + "ctc"),
            (op_proj, pre + "Wco", pre + "ctc", pre + "acf"), (op_add768, pre + "acf", pre + "bco", pre + "acf"),
            (op_add768, pre + "x1", pre + "acf", pre + "x2"),
            (op_ln, pre + "x2", pre + "xn_f"),
            (op_f1, pre + "Wf1", pre + "xn_f", pre + "h"), (op_add_ff, pre + "h", pre + "bias_f1", pre + "h"), (op_gelu, pre + "h", pre + "h"),
            (op_f2, pre + "Wf2", pre + "h", pre + "ff"), (op_add768, pre + "ff", pre + "bf2", pre + "ff"),
            (op_add768, pre + "x2", pre + "ff", nxt),
        ]
        layer_data.append(dict(mat_qkv=mat_qkv, bias_qkv=bias_qkv, Wso=Wso, bso=bso, mat_cq=mat_cq,
                               bias_cq=bias_cq, Wco=Wco, bco=bco, mat_f1=mat_f1, bias_f1=bias_f1,
                               Wf2=Wf2, bf2=bf2, Kenc_b=Kenc_b, Venc_b=Venc_b, k_past=k_past, v_past=v_past))
        cur = nxt

    out_name = cur
    fused = FusedMLIROperator("decode", rl, input_args=["x"], output_args=[out_name],
                              buffer_sizes=bufsz, context=ctx)
    fused.compile()
    elf = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    wnames = list(weights_to_write.keys())
    lay = {n: fused.get_layout_for_buffer(n) for n in ["x", out_name] + wnames}

    # Deep-C: aiecc emits the scratchpad StateTable layout (params.txt) next to the fused MLIR; parse
    # it so the host (Rust) knows each param's byte offset (= state_table_idx*4) + kind. The ctrl
    # scratchpad is a u32 array; addr-kind values are written raw, core-kind values are written <<2
    # (firmware UPDATE_REG requirement — the core right-shifts by 2 after reading).
    import glob, shutil
    _pp = sorted(glob.glob("**/decode*.mlir.prj/params.txt", recursive=True), key=os.path.getmtime)
    scratchpad_params = {}
    if _pp:
        shutil.copy(_pp[-1], os.path.join(a.out, "params.txt"))
        for line in open(_pp[-1]).read().splitlines()[1:]:
            if line.strip():
                nm, idx, ty, kind = line.split()
                scratchpad_params[nm] = {"byte_offset": int(idx) * 4, "kind": kind, "dtype": ty}

    # ---- device-faithful golden (N-layer forward) ----
    x = bf16(rng.standard_normal(D).astype(np.float32))
    cur_x = x
    for l in range(NL):
        d = layer_data[l]
        n1 = bf16(ln(cur_x.astype(np.float32)))
        qkv = bf16(bf16(bf16(d["mat_qkv"]).astype(np.float32) @ n1.astype(np.float32)).astype(np.float32) + bf16(d["bias_qkv"]).astype(np.float32))
        q, kcur, vcur = qkv[0:D].reshape(H, HD), qkv[D:2*D].reshape(H, HD), qkv[2*D:3*D].reshape(H, HD)
        Ks = np.concatenate([d["k_past"], kcur[:, None]], 1).astype(np.float32) if P else kcur[:, None].astype(np.float32)
        Vs = np.concatenate([d["v_past"], vcur[:, None]], 1).astype(np.float32) if P else vcur[:, None].astype(np.float32)
        asf = bf16(attn_host(q.astype(np.float32), Ks, Vs).reshape(-1))
        asf = bf16(bf16(bf16(d["Wso"].T.copy()).astype(np.float32) @ asf.astype(np.float32)).astype(np.float32) + bf16(d["bso"]).astype(np.float32))
        x1 = bf16(cur_x.astype(np.float32) + asf.astype(np.float32))
        n2 = bf16(ln(x1.astype(np.float32)))
        qc = bf16(bf16(bf16(d["mat_cq"]).astype(np.float32) @ n2.astype(np.float32)).astype(np.float32) + bf16(d["bias_cq"]).astype(np.float32)).reshape(H, HD)
        ctc = attn_host(qc.astype(np.float32), d["Kenc_b"][:, 0:T].astype(np.float32), d["Venc_b"][:, 0:T].astype(np.float32)).reshape(-1)
        acf = bf16(bf16(bf16(d["Wco"].T.copy()).astype(np.float32) @ bf16(ctc).astype(np.float32)).astype(np.float32) + bf16(d["bco"]).astype(np.float32))
        x2 = bf16(x1.astype(np.float32) + acf.astype(np.float32))
        n3 = bf16(ln(x2.astype(np.float32)))
        h1 = bf16(bf16(bf16(d["mat_f1"]).astype(np.float32) @ n3.astype(np.float32)).astype(np.float32) + bf16(d["bias_f1"]).astype(np.float32))
        h2 = bf16(torch.nn.functional.gelu(torch.from_numpy(h1.astype(np.float32)), approximate="tanh").numpy())
        ff = bf16(bf16(bf16(d["Wf2"].T.copy()).astype(np.float32) @ h2.astype(np.float32)).astype(np.float32) + bf16(d["bf2"]).astype(np.float32))
        cur_x = bf16(x2.astype(np.float32) + ff.astype(np.float32))
    x_out = cur_x

    bdir = os.path.join(a.out, "buffers")
    def wb(n, v): open(os.path.join(bdir, f"{n}.bin"), "wb").write(np.asarray(v, BF16).tobytes())
    wb("x", x); wb(out_name, x_out)
    for nm, arr in weights_to_write.items():
        wb(nm, arr)
    open(os.path.join(a.out, "decode.elf"), "wb").write(elf)

    meta = {
        "elf": "decode.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"], "weights": wnames, "output": out_name,
        # Deep-C: ELF is CONSTANT (no per-token patch). Per token the host writes scratchpad params:
        #   kv_off (addr) = n_self*head_dim  (element units, raw);  sm_mask (core) = n_self+1 (<<2 by host)
        "scratchpad": {"params": scratchpad_params, "kv_param": "kv_off", "mask_param": "sm_mask",
                       "head_dim": HD, "num_preceding": P},
        "dims": {"layers": NL, "S": S, "P": P, "T_enc": T, "T_pad": TP},
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\nwrote {NL}-layer decode ELF ({len(elf)}B, scratch {scr/1e6:.1f}MB) to {a.out}")


if __name__ == "__main__":
    main()
