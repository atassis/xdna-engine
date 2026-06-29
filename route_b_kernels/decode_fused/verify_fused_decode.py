#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FE1 step-6 (pre-e2e): argmax-parity of the WHOLE 12-layer fused-ELF decode vs an f32 reference.

The open gate from the whole-decode build: the fused output is rel-L2 0.093 vs f32 (NPU softmax/gelu/ln
kernel-approx compounding). WER is argmax-driven, so the real question is: does the fused decode produce
the SAME next-token argmax as the f32 reference on a real greedy chain? This harness answers it WITHOUT
the full Rust integration — it drives the fused ELF from Python (IRON FusedFullELFCallable + per-token
patch/reload), on the REAL encoder output (artifacts/whisper-small/refs/encoded.npy), and compares the
greedy argmax sequence to a pure-f32 12-layer forward (≈ ONNX, which the host decoder matches at 1e-3).

PASS here ⇒ the fused decode is argmax-stable ⇒ WER-safe ⇒ the Rust WER plumbing is worth building.
Run inside IRON env (aiebu-asm on PATH); single-tenant (stop npu-asr/voxd first).
"""
import argparse
import os
import numpy as np
import ml_dtypes
import torch

from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf, patch_elf
from iron.operators.gemv.op import GEMV
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.elementwise_add.op import ElementwiseAdd
from iron.operators.softmax.op import Softmax
from iron.operators.strided_copy.op import StridedCopy
from iron.operators.transpose.op import Transpose
from iron.operators.gelu.op import GELU

BF16 = ml_dtypes.bfloat16
D, H, HD, QKV, FF, VOCAB = 768, 12, 64, 2304, 3072, 51865
KV_MAGIC, SM_MAGIC = 0xDEADBEE0, 0xBA5EBA11
SOT, EOT = 50258, 50257


def bf16(a): return np.asarray(a).astype(BF16)
def f32(a): return np.asarray(a).astype(np.float32)
def npy(w, L, n): return np.load(os.path.join(w, f"L{L}", n + ".npy")).astype(np.float32)
def gnpy(w, n): return np.load(os.path.join(w, n + ".npy")).astype(np.float32)
def ln_np(x, g, b, eps=1e-5):
    x = f32(x); m = x.mean(); v = x.var(); return (x - m) / np.sqrt(v + eps) * g + b


def pick_tiling(M, N):
    for s in (8, 4):
        for m in sorted((d for d in range(s, M + 1) if M % d == 0 and d % s == 0), reverse=True):
            for n in sorted((d for d in range(s, N + 1) if N % d == 0 and d % s == 0), reverse=True):
                if m * n <= 8192 and not (s == 8 and (m <= 16 or n <= 16)):
                    return m, n, s
    raise ValueError("no tiling")


def get_patch_locs(elf, magic):
    return np.where(elf == (magic & 0xFFFFFFFF))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--encoded", required=True, help="encoded.npy [T_enc,768] real encoder output")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=448)
    ap.add_argument("--t-pad", type=int, default=1536)
    ap.add_argument("--npu-logits", action="store_true",
                    help="append ln_post+proj_out on-NPU so the ELF outputs logits[VOCAB_PAD]; argmax reads "
                         "them directly (validates the e2e/NPU logits path on REAL data, T6(a) strong version)")
    ap.add_argument("--compile-only", action="store_true",
                    help="compile the fused op then exit (no device) -- validate the graph/build with services up")
    a = ap.parse_args()
    w, NL, S, TP = a.weights, a.layers, a.prompt_len, a.t_pad
    scale = 1.0 / np.sqrt(HD)
    enc = np.load(a.encoded).astype(np.float32)  # [T_enc, 768]
    T = enc.shape[0]
    assert TP >= T and TP % 64 == 0 and TP % 16 == 0
    tms, tns, tss = pick_tiling(S, HD)
    tmc, tnc, tsc = pick_tiling(TP, HD)

    emb_t = gnpy(w, "embed_tokens"); emb_p = gnpy(w, "embed_positions")
    lnp_w, lnp_b = gnpy(w, "ln_post.weight"), gnpy(w, "ln_post.bias")
    proj = gnpy(w, "proj_out.weight")  # [768, 51865]

    # ---------- per-layer folded weights + encoder K/V (REAL, from encoded.npy) ----------
    LW = []
    for l in range(NL):
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
        Kenc = enc @ Wck; Venc = enc @ Wcv + bcv  # [T,768]
        def heads_pad(M):
            o = np.zeros((H, TP, HD), np.float32); o[:, 0:T, :] = M.reshape(T, H, HD).transpose(1, 0, 2); return o
        LW.append(dict(mat_qkv=mat_qkv, bias_qkv=bias_qkv, Wso=Wso, bso=bso, mat_cq=mat_cq, bias_cq=bias_cq,
                       Wco=Wco, bco=bco, mat_f1=mat_f1, bias_f1=bias_f1, Wf2=Wf2, bf2=bf2,
                       Kenc=heads_pad(Kenc), Venc=heads_pad(Venc),
                       # raw (f32 reference, per-head, real T):
                       Kr=(enc @ Wck).reshape(T, H, HD).transpose(1, 0, 2),
                       Vr=(enc @ Wcv + bcv).reshape(T, H, HD).transpose(1, 0, 2)))

    # ---------- build the fused decode op (same structure as gen_decode.py) ----------
    ctx = AIEContext()
    o_ln = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, context=ctx)
    o_pj = GEMV(M=D, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    o_a8 = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)
    o_qk = GEMV(M=QKV, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=QKV // 8, context=ctx)
    o_aq = ElementwiseAdd(size=QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)
    scd = dict(input_sizes=(H, HD), input_strides=(HD, 1), input_offset=0, output_sizes=(1, H, HD),
               output_strides=(0, S * HD, 1), output_offset=0, input_buffer_size=H * HD,
               output_buffer_size=H * S * HD, num_aie_channels=1)
    o_sk = StridedCopy(**scd, kwargs={"output_offset_patch_marker": KV_MAGIC}, context=ctx)
    o_sv = StridedCopy(**scd, kwargs={"output_offset_patch_marker": KV_MAGIC}, context=ctx)
    o_ss = GEMV(M=S, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=S // 8, num_batches=H, context=ctx)
    o_ms = Softmax(rows=16, cols=S, num_aie_columns=1, num_channels=1, rtp_vector_size=S, mask_patch_value=SM_MAGIC, context=ctx)
    o_ts = Transpose(M=S, N=HD, num_aie_columns=2, num_channels=1, m=tms, n=tns, s=tss, context=ctx)
    o_cs = GEMV(M=HD, K=S, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=H, context=ctx)
    o_sc = GEMV(M=TP, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=TP // 8, num_batches=H, context=ctx)
    o_mc = Softmax(rows=16, cols=TP, num_aie_columns=1, num_channels=1, rtp_vector_size=T, mask_patch_value=0, context=ctx)
    o_tc = Transpose(M=TP, N=HD, num_aie_columns=2, num_channels=1, m=tmc, n=tnc, s=tsc, context=ctx)
    o_cc = GEMV(M=HD, K=TP, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=H, context=ctx)
    o_f1 = GEMV(M=FF, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=FF // 8, context=ctx)
    o_af = ElementwiseAdd(size=FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)
    o_gl = GELU(size=FF, num_aie_columns=8, num_channels=1, tile_size=FF // 8, context=ctx)
    o_f2 = GEMV(M=D, K=FF, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)

    HSs, HSc, phs, phc = H * S * 2, H * TP * 2, S * HD * 2, TP * HD * 2
    rl, bufsz, cur = [], {}, "x"
    for l in range(NL):
        p = f"L{l}_"; nxt = f"x{l+1}"
        bufsz.update({p+"qkv": QKV*2, p+"kcache": H*S*HD*2, p+"vcache": H*S*HD*2, p+"vcT": H*S*HD*2,
                      p+"scs": 16*S*2, p+"sws": 16*S*2, p+"Kenc": H*TP*HD*2, p+"Venc": H*TP*HD*2,
                      p+"vcTc": H*TP*HD*2, p+"scc": 16*TP*2, p+"swc": 16*TP*2})
        rl += [(o_ln, cur, p+"xn_s"), (o_qk, p+"Wqkv", p+"xn_s", p+"qkv"), (o_aq, p+"qkv", p+"bias_qkv", p+"qkv"),
               (o_sk, p+"qkv[1536:3072]", p+"kcache"), (o_sv, p+"qkv[3072:4608]", p+"vcache"),
               (o_ss, p+"kcache", p+"qkv[0:1536]", f"{p}scs[0:{HSs}]"), (o_ms, p+"scs", p+"sws")] + \
              [(o_ts, f"{p}vcache[{h*phs}:{(h+1)*phs}]", f"{p}vcT[{h*phs}:{(h+1)*phs}]") for h in range(H)] + \
              [(o_cs, p+"vcT", f"{p}sws[0:{HSs}]", p+"cts"), (o_pj, p+"Wso", p+"cts", p+"asf"), (o_a8, p+"asf", p+"bso", p+"asf"),
               (o_a8, cur, p+"asf", p+"x1"), (o_ln, p+"x1", p+"xn_c"), (o_pj, p+"Wcq", p+"xn_c", p+"qc"), (o_a8, p+"qc", p+"bias_cq", p+"qc"),
               (o_sc, p+"Kenc", p+"qc", f"{p}scc[0:{HSc}]"), (o_mc, p+"scc", p+"swc")] + \
              [(o_tc, f"{p}Venc[{h*phc}:{(h+1)*phc}]", f"{p}vcTc[{h*phc}:{(h+1)*phc}]") for h in range(H)] + \
              [(o_cc, p+"vcTc", f"{p}swc[0:{HSc}]", p+"ctc"), (o_pj, p+"Wco", p+"ctc", p+"acf"), (o_a8, p+"acf", p+"bco", p+"acf"),
               (o_a8, p+"x1", p+"acf", p+"x2"), (o_ln, p+"x2", p+"xn_f"), (o_f1, p+"Wf1", p+"xn_f", p+"h"),
               (o_af, p+"h", p+"bias_f1", p+"h"), (o_gl, p+"h", p+"h"), (o_f2, p+"Wf2", p+"h", p+"ff"),
               (o_a8, p+"ff", p+"bf2", p+"ff"), (o_a8, p+"x2", p+"ff", nxt)]
        cur = nxt
    out_name = cur
    weights_extra = {}
    if a.npu_logits:
        # e2e/NPU: run ln_post + proj_out on the NPU so the ELF outputs logits[VOCAB_PAD] (ln_post affine
        # folds into proj_out; o_ln is pure-normalize). Mirrors gen_decode --npu-logits. argmax stays host.
        VOCAB_PAD = 65536
        mat_proj = (lnp_w[:, None] * proj).T.copy()             # [VOCAB, D]
        bias_proj = (lnp_b @ proj).astype(np.float32)           # [VOCAB]
        mat_pad = np.zeros((VOCAB_PAD, D), np.float32); mat_pad[0:VOCAB] = mat_proj
        bias_pad = np.full(VOCAB_PAD, -1e30, np.float32); bias_pad[0:VOCAB] = bias_proj
        o_proj_out = GEMV(M=VOCAB_PAD, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=VOCAB_PAD // 8, context=ctx)
        o_add_logits = ElementwiseAdd(size=VOCAB_PAD, tile_size=VOCAB_PAD // 8, num_aie_columns=8, context=ctx)
        rl += [(o_ln, out_name, "hn"), (o_proj_out, "Wproj", "hn", "logits"), (o_add_logits, "logits", "bias_proj", "logits")]
        bufsz.update({"hn": D * 2, "logits": VOCAB_PAD * 2, "Wproj": VOCAB_PAD * D * 2, "bias_proj": VOCAB_PAD * 2})
        weights_extra = {"Wproj": bf16(mat_pad).reshape(-1), "bias_proj": bf16(bias_pad)}
        out_name = "logits"
    fused = FusedMLIROperator("decode", rl, input_args=["x"], output_args=[out_name], buffer_sizes=bufsz, context=ctx)
    print("compiling fused decode op...")
    fused.compile()
    if a.compile_only:
        import sys; print(f"compile-only: fused op compiled OK (out={out_name}, npu_logits={a.npu_logits})"); sys.exit(0)
    callable_ = fused.get_callable()  # FusedFullELFCallable
    elf_data = load_elf(fused)

    # ---------- lay weights + REAL encoder K/V into the arena; zero the self-KV caches ----------
    # XRTSubBuffer.data is the mapped numpy view into the parent arena BO.
    def put2(name, arr):
        b = callable_.get_buffer(name)
        flat = np.asarray(arr, dtype=BF16).reshape(-1)
        np.copyto(b.data, flat)
    for l in range(NL):
        p = f"L{l}_"; d = LW[l]
        for nm, arr in [("Wqkv", bf16(d["mat_qkv"]).reshape(-1)), ("bias_qkv", bf16(d["bias_qkv"])),
                        ("Wso", bf16(d["Wso"].T.copy()).reshape(-1)), ("bso", bf16(d["bso"])),
                        ("Wcq", bf16(d["mat_cq"]).reshape(-1)), ("bias_cq", bf16(d["bias_cq"])),
                        ("Wco", bf16(d["Wco"].T.copy()).reshape(-1)), ("bco", bf16(d["bco"])),
                        ("Wf1", bf16(d["mat_f1"]).reshape(-1)), ("bias_f1", bf16(d["bias_f1"])),
                        ("Wf2", bf16(d["Wf2"].T.copy()).reshape(-1)), ("bf2", bf16(d["bf2"])),
                        ("Kenc", bf16(d["Kenc"]).reshape(-1)), ("Venc", bf16(d["Venc"]).reshape(-1)),
                        ("kcache", np.zeros(H*S*HD, BF16)), ("vcache", np.zeros(H*S*HD, BF16))]:
            put2(p + nm, arr)
    for nm, arr in weights_extra.items():  # npu-logits: Wproj/bias_proj (non-per-layer)
        put2(nm, arr)
    # __call__ syncs only input+output; scratch (weights/encoder-KV/caches) must be synced once now.
    callable_.scratch_buffer.to("npu")

    # patch sites: all kcache/vcache byte offsets + softmax magic locs
    kv_offs = []
    for l in range(NL):
        for nm in ("kcache", "vcache"):
            _, off, _ = fused.get_layout_for_buffer(f"L{l}_{nm}")
            kv_offs.append(int(off))
    kv_loc_base = {}
    for off in kv_offs:
        for loc in get_patch_locs(elf_data, off + KV_MAGIC * 2):
            kv_loc_base[int(loc)] = off
    for loc in get_patch_locs(elf_data, KV_MAGIC * 2):
        kv_loc_base[int(loc)] = 0
    sm_locs = get_patch_locs(elf_data, SM_MAGIC)
    print(f"patch sites: {len(kv_loc_base)} KV, {len(sm_locs)} softmax")

    # ---------- f32 reference forward (one layer) — true ideal (≈ ONNX) ----------
    # f32 weights; the γ-fold (diag(γ)·W) is exact in f32, so this equals affine-LN + separate matmuls.
    def ref_layer(x, d, kcache_f, vcache_f, npos):
        n1 = ln_np(x, np.ones(D), np.zeros(D))  # non-affine (γ folded into mat_qkv)
        qkv = d["mat_qkv"] @ n1 + d["bias_qkv"]
        q = qkv[0:D].reshape(H, HD); k = qkv[D:2*D].reshape(H, HD); v = qkv[2*D:3*D].reshape(H, HD)
        kcache_f[:, npos, :] = k; vcache_f[:, npos, :] = v
        Kc = kcache_f[:, 0:npos+1, :]; Vc = vcache_f[:, 0:npos+1, :]
        ao = np.zeros((H, HD))
        for h in range(H):
            s = Kc[h] @ q[h]; wt = torch.softmax(torch.from_numpy(s.astype(np.float32)), 0).numpy(); ao[h] = wt @ Vc[h]
        asf = d["Wso"].T @ ao.reshape(-1) + d["bso"]
        x1 = x + asf
        n2 = ln_np(x1, np.ones(D), np.zeros(D))
        qc = (d["mat_cq"] @ n2 + d["bias_cq"]).reshape(H, HD)
        co = np.zeros((H, HD))
        for h in range(H):
            s = d["Kr"][h] @ qc[h]; wt = torch.softmax(torch.from_numpy(s.astype(np.float32)), 0).numpy(); co[h] = wt @ d["Vr"][h]
        acf = d["Wco"].T @ co.reshape(-1) + d["bco"]
        x2 = x1 + acf
        n3 = ln_np(x2, np.ones(D), np.zeros(D))
        h1 = d["mat_f1"] @ n3 + d["bias_f1"]
        h2 = torch.nn.functional.gelu(torch.from_numpy(h1.astype(np.float32)), approximate="tanh").numpy()
        ff = d["Wf2"].T @ h2 + d["bf2"]
        return x2 + ff

    def logits_of(hidden):
        nrm = ln_np(hidden, lnp_w, lnp_b)
        return nrm @ proj  # [VOCAB]

    # f32 reference greedy
    ref_kc = [np.zeros((H, S, HD)) for _ in range(NL)]; ref_vc = [np.zeros((H, S, HD)) for _ in range(NL)]
    ref_toks, tok = [], SOT
    for step in range(a.steps):
        x = emb_t[tok] + emb_p[step]
        for l in range(NL):
            x = ref_layer(x, LW[l], ref_kc[l], ref_vc[l], step)
        nt = int(np.argmax(logits_of(x)))
        ref_toks.append(nt); tok = nt
        if nt == EOT: break

    # fused-ELF greedy (device), with per-token patch/reload
    fus_toks, tok = [], SOT
    xin = callable_.get_buffer("x")
    xout = callable_.get_buffer(out_name)
    for step in range(a.steps):
        x = bf16(emb_t[tok] + emb_p[step])
        np.copyto(xin.data, x.reshape(-1))
        # patch
        off_val = step * HD * 2
        patches = {i: (base + off_val, 0xFFFFFFFF) for i, base in kv_loc_base.items()}
        patches.update({int(i): (step + 1, 0xFFFFFFFF) for i in sm_locs})
        ed = elf_data.copy(); patch_elf(ed, patches); callable_.reload_elf(ed)
        callable_()
        if a.npu_logits:
            # ELF already produced logits[VOCAB_PAD] on-NPU (ln_post+proj_out); argmax over the real vocab.
            nt = int(np.argmax(f32(xout.data[0:VOCAB])))
        else:
            hidden = f32(xout.data[0:D])
            nt = int(np.argmax(logits_of(hidden)))
        fus_toks.append(nt); tok = nt
        if nt == EOT: break

    n = min(len(ref_toks), len(fus_toks))
    match = sum(1 for i in range(n) if ref_toks[i] == fus_toks[i])
    print(f"\nref  tokens: {ref_toks}")
    print(f"fused tokens: {fus_toks}")
    print(f"\nargmax parity: {match}/{n} steps match  (fused 12-layer decode vs f32 reference)")
    if match == n:
        print("*** PARITY PASS — fused decode is argmax-stable ⇒ WER-safe ***")
    else:
        print(f"*** {n-match} mismatches — precision affects argmax; WER likely regresses ***")


if __name__ == "__main__":
    main()
