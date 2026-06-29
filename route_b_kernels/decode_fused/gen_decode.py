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
from iron.operators.elementwise_mul.op import ElementwiseMul
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
    # lever #3 coalescing toggles — DEFAULT OFF = deep-C baseline (correct). On-device numerics of the
    # combined coalescing are NOT yet validated (lever3_ab WER 0.996 → broken); these flags isolate which.
    #   --coalesce-cross : (i) store Venc pre-transposed [H,HD,TP], drop the 12 per-head op_tr_c launches.
    #   --coalesce-self  : (1) head-batch op_tr_s (num_batches=H), one launch for all H heads.
    ap.add_argument("--coalesce-cross", action="store_true", help="lever3 (i) cross-V pre-transpose")
    ap.add_argument("--coalesce-self", action="store_true", help="lever3 (1) batched self-V transpose")
    # O18 occupancy: lift the softmax off 1 column to 8 (byte-neutral). DEFAULT OFF = deep-C baseline.
    # (The transpose is already num_aie_columns=2 in this M=1 generator, so --occ = softmax 1->8 here.)
    ap.add_argument("--occ", action="store_true", help="O18: lift softmax 1->8 columns (occupancy A/B)")
    # M0.5: store the self vcache PRE-TRANSPOSED [H,HD,S] and write each new V row transposed (op_scv via a
    # 2nd addr scratchpad `vcache_off`=n_self), so op_ct_s reads it directly and op_tr_s is ELIMINATED
    # (kills the per-token self-V transpose round-trip, the #2 inter-op sink). Default OFF.
    ap.add_argument("--coalesce-self-tr", action="store_true", help="M0.5: transposed self vcache, drop op_tr_s")
    # int8 cross-K (#1 M=1 byte lever, step 1): store the resident cross-K (Kenc) int8 (halves its LPDDR
    # re-read) + run op_sc_c as int8(matrix)xbf16(vector). A FIXED per-layer scale s_k is calibrated from
    # Kenc and FOLDED into mat_cq/bias_cq so it cancels (scores = (s_k*qc).Kenc_int8 = qc.Kenc_real). The
    # host quantizes per-utterance Kenc with the same meta-provided s_k. Venc/op_ct_c stay bf16 for now.
    ap.add_argument("--int8-cross-k", action="store_true", help="int8 resident cross-K (Kenc) + int8 op_sc_c")
    # int8 cross-V (#1 M=1 byte lever, step 2): store the resident cross-V (Venc) int8 (halves ITS LPDDR
    # re-read; together with int8-cross-k the FULL cross K/V re-read is halved, -28.3 MB/token). op_ct_c
    # runs int8(matrix=Venc)xbf16(vector=attn); the per-(head,HD) scale s_v multiplies the CONTEXT OUTPUT
    # (post-GEMV, via op_mul_cv) -- can't fold into attn (that's per-t, the contracted axis). REQUIRES
    # --coalesce-cross: op_ct_c must read the resident pre-transposed Venc [H,HD,TP] directly; without it
    # op_ct_c reads the on-chip op_tr_c scratch (transposing int8 would need a separate kernel).
    ap.add_argument("--int8-cross-v", action="store_true", help="int8 resident cross-V (Venc) + int8 op_ct_c (needs --coalesce-cross)")
    # int8 WEIGHTS (the dominant LPDDR term: ~198 MB/token of bf16 GEMV matrices re-streamed each token).
    # STATIC per-output-row quant (s[m]=max_k|W[m,k]|/127, baked at build — no host/engine work): the GEMV
    # reads int8 A (halved buffer) and an op_mul applies s[m] to the output BEFORE the bias add. Split FFN
    # (Wf1/Wf2 — biggest + no attention-score sensitivity) from attention (Wqkv/Wso/Wcq/Wco — perturb cached
    # Q/K/V) so each is WER-gated independently.
    ap.add_argument("--int8-ffn", action="store_true", help="int8 static FFN weights (Wf1, Wf2)")
    ap.add_argument("--int8-attn-w", action="store_true", help="int8 static attention weights (Wqkv, Wso, Wcq, Wco)")
    # BIAS FUSION (inter-op overhead lever, the M=1 LATENCY attack — fewer ops, not fewer bytes): fold the
    # post-GEMV bias-add INTO the GEMV via K-augmentation (append the bias as one extra weight column + a
    # constant 1 to the input vector), eliminating the separate ElementwiseAdd op. EXACT (WER must == base).
    # --fuse-bias-ffn = the 2 FFN GEMVs (Wf1, Wf2); attention GEMVs are a follow-on.
    ap.add_argument("--fuse-bias-ffn", action="store_true", help="fold Wf1/Wf2 bias into the GEMV (K-aug), drop the 2 add ops")
    # --fuse-bias-attn = the 4 attention GEMVs: op_qkv (Wqkv) + op_proj (Wso/Wcq/Wco, reused 3×). Drops 4
    # bias-adds. Inputs augmented: xn_s (qkv), cts (Wso), xn_c (Wcq), ctc (Wco). Same K-aug mechanism as FFN.
    ap.add_argument("--fuse-bias-attn", action="store_true", help="fold Wqkv/Wso/Wcq/Wco bias into the GEMV (K-aug), drop 4 add ops")
    # --fuse-gelu: fold the FFN GELU into op_f1's epilogue (gelu over the m_output C-tile, in the GEMV
    # core_body). REQUIRES --fuse-bias-ffn so op_f1 outputs W·x+bias and the epilogue gives gelu(W·x+bias).
    ap.add_argument("--fuse-gelu", action="store_true", help="fold the FFN GELU into op_f1's epilogue (needs --fuse-bias-ffn)")
    # --npu-logits (e2e/NPU migration step 1): run ln_post + proj_out ON THE NPU — the ELF outputs
    # logits[VOCAB_PAD] instead of the 768-hidden, so the host drops the ~40M-MAC proj_out matmul (argmax
    # stays host for now). ln_post affine folds into proj_out (the LN op is pure-normalize).
    ap.add_argument("--npu-logits", action="store_true", help="run ln_post+proj_out on the NPU (ELF outputs logits)")
    a = ap.parse_args()
    co_cross, co_self = a.coalesce_cross, a.coalesce_self
    co_self_tr = a.coalesce_self_tr
    int8_ck = a.int8_cross_k
    int8_cv = a.int8_cross_v
    int8_ffn = a.int8_ffn
    int8_attn_w = a.int8_attn_w
    fuse_ffn = a.fuse_bias_ffn
    fuse_attn = a.fuse_bias_attn
    fuse_gelu = a.fuse_gelu
    npu_logits = a.npu_logits
    if fuse_gelu and not fuse_ffn:
        ap.error("--fuse-gelu requires --fuse-bias-ffn (the epilogue needs op_f1 to output W·x+bias)")
    VS = 64  # kernel_vector_size — K must stay a multiple of it; K-aug pads by exactly one VS block
    if fuse_ffn and int8_ffn:
        ap.error("--fuse-bias-ffn + --int8-ffn not supported together yet (int8 quant of the aug weight)")
    if fuse_attn and (int8_attn_w or int8_ck or int8_cv):
        ap.error("--fuse-bias-attn not supported with int8 attention/cross flags yet (aug-weight quant + qc/ctc scale interplay)")

    def aug_bias(mat, bias):
        """K-augment a weight [M,K] with its [M] bias -> [M, K+VS]: column K = bias, columns K+1..K+VS-1 = 0.
        With input x_aug = [x, 1, 0..0], GEMV(W_aug, x_aug) = W·x + bias in ONE op (the bias-add is gone)."""
        M, K = mat.shape
        out = np.zeros((M, K + VS), np.float32)
        out[:, 0:K] = mat
        out[:, K] = bias
        return out
    if int8_cv and not co_cross:
        ap.error("--int8-cross-v requires --coalesce-cross (op_ct_c must read the resident pre-transposed Venc)")

    def qrow(mat):
        """Static per-output-row int8 quant of a weight matrix [M,K]. Returns (int8 array [M,K], scale [M]).
        out=GEMV(int8 A)·x is then ×s[m] (op_mul) to recover W·x: s[m]·Σ_k round(W[m,k]/s[m])·x[k] ≈ W[m]·x."""
        m = mat.astype(np.float32)
        s = np.abs(m).max(axis=1) / 127.0
        s = np.where(s > 0, s, 1.0)
        q = np.clip(np.round(m / s[:, None]), -127, 127).astype(np.int8)
        return q, s.astype(np.float32)
    sm_cols = 8 if a.occ else 1
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    w, NL, S, P, T, TP = a.weights, a.layers, a.prompt_len, a.num_preceding, a.t_enc, a.t_pad
    scale = 1.0 / np.sqrt(HD)
    rng = np.random.default_rng(a.seed)
    tms, tns, tss = pick_tiling(S, HD)
    tmc, tnc, tsc = pick_tiling(TP, HD)

    # ---- ops (created once, reused for all layers) ----
    ctx = AIEContext()
    op_ln = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, context=ctx)
    op_proj = GEMV(M=D, K=D + (VS if fuse_attn else 0), num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8,
                   dtype_a=("int8" if int8_attn_w else "bf16"), context=ctx)
    op_add768 = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)
    op_mul_cq = ElementwiseMul(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)  # int8 cross-K: qc *= s_cq
    op_mul_cv = ElementwiseMul(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)  # int8 cross-V: ctc *= s_cv
    # int8-weights: per-output-row scale muls applied to the GEMV output before its bias add (size = M).
    op_mulw_d = ElementwiseMul(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)   # Wso/Wcq/Wco, Wf2 (M=D)
    op_mulw_qkv = ElementwiseMul(size=QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)  # Wqkv (M=QKV)
    op_mulw_ff = ElementwiseMul(size=FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)  # Wf1 (M=FF)
    op_qkv = GEMV(M=QKV, K=D + (VS if fuse_attn else 0), num_aie_columns=8, tile_size_input=4, tile_size_output=QKV // 8,
                  dtype_a=("int8" if int8_attn_w else "bf16"), context=ctx)
    op_add_qkv = ElementwiseAdd(size=QKV, tile_size=QKV // 8, num_aie_columns=8, context=ctx)
    sc = dict(input_sizes=(H, HD), input_strides=(HD, 1), input_offset=0, output_sizes=(1, H, HD),
              output_strides=(0, S * HD, 1), output_offset=0, input_buffer_size=H * HD,
              output_buffer_size=H * S * HD, num_aie_channels=1)
    # Deep-C: the per-token KV-write position offset is now a runtime `addr`-kind scratchpad param
    # (shared symbol "kv_off", element units = n_self*head_dim) instead of a per-token ELF patch →
    # the decode ELF is CONSTANT across tokens (registered once; host writes the offset per dispatch).
    op_sck = StridedCopy(**sc, kwargs={"output_offset_scratchpad": "kv_off"}, context=ctx)
    if co_self_tr:
        # transposed vcache [H,HD,S]: new V[h,d] -> vcache[h*HD*S + d*S + n_self]; head stride HD*S, dim
        # stride S, runtime column offset = vcache_off (= n_self, NOT n_self*HD like the kcache kv_off).
        sc_v = dict(input_sizes=(H, HD), input_strides=(HD, 1), input_offset=0, output_sizes=(1, H, HD),
                    output_strides=(0, HD * S, S), output_offset=0, input_buffer_size=H * HD,
                    output_buffer_size=H * S * HD, num_aie_channels=1)
        op_scv = StridedCopy(**sc_v, kwargs={"output_offset_scratchpad": "vcache_off"}, context=ctx)
    else:
        op_scv = StridedCopy(**sc, kwargs={"output_offset_scratchpad": "kv_off"}, context=ctx)
    op_sc_s = GEMV(M=S, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=S // 8, num_batches=H, context=ctx)
    # Deep-C: the per-token self-softmax mask width is now a runtime `core`-kind scratchpad param
    # (symbol "sm_mask", element units = context_len = n_self) read on-tile, instead of an ELF patch.
    op_sm_s = Softmax(rows=16, cols=S, num_aie_columns=sm_cols, num_channels=1, rtp_vector_size=S, mask_scratchpad="sm_mask", context=ctx)
    # lever #3 (1): head-batched self-attn V transpose (num_batches=H, ONE launch for all H heads) when
    # --coalesce-self; else per-head (num_batches=1) = deep-C baseline. vcache [H,S,HD] contiguous.
    op_tr_s = Transpose(M=S, N=HD, num_batches=(H if co_self else 1), num_aie_columns=2, num_channels=1, m=tms, n=tns, s=tss, context=ctx)
    op_ct_s = GEMV(M=HD, K=S, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=H, context=ctx)
    op_sc_c = GEMV(M=TP, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=TP // 8, num_batches=H,
                   dtype_a=("int8" if int8_ck else "bf16"), context=ctx)
    op_sm_c = Softmax(rows=16, cols=TP, num_aie_columns=sm_cols, num_channels=1, rtp_vector_size=T, mask_patch_value=0, context=ctx)
    # lever #3 (i): when --coalesce-cross, Venc is stored pre-transposed [H,HD,TP] host-side and op_ct_c
    # reads it directly (no per-token op_tr_c). Else the deep-C per-head transpose (op_tr_c) is used.
    op_tr_c = None if co_cross else Transpose(M=TP, N=HD, num_aie_columns=2, num_channels=1, m=tmc, n=tnc, s=tsc, context=ctx)
    op_ct_c = GEMV(M=HD, K=TP, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8, num_batches=H,
                   dtype_a=("int8" if int8_cv else "bf16"), context=ctx)
    # fuse_ffn: op_f1/op_f2 contract over the AUGMENTED K (real K + VS); the bias rides in the extra block.
    op_f1 = GEMV(M=FF, K=D + (VS if fuse_ffn else 0), num_aie_columns=8, tile_size_input=4, tile_size_output=FF // 8,
                 dtype_a=("int8" if int8_ffn else "bf16"), epilogue=("gelu" if fuse_gelu else "none"), context=ctx)
    op_add_ff = ElementwiseAdd(size=FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)
    op_gelu = GELU(size=FF, num_aie_columns=8, num_channels=1, tile_size=FF // 8, context=ctx)
    op_f2 = GEMV(M=D, K=FF + (VS if fuse_ffn else 0), num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8,
                 dtype_a=("int8" if int8_ffn else "bf16"), context=ctx)

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
        if int8_ck:
            # PER-UTTERANCE PER-CHANNEL int8: the scale is NOT folded at build (golden Kenc != real Kenc).
            # Instead op_mul_cq scales qc by a per-channel s_cq buffer the HOST writes from the REAL Kenc
            # each utterance: scores = Σ_d (qc[h,d]*s[h,d])*Kenc_int8 = qc·Kenc_real. mat_cq stays UNFOLDED.
            # gen_decode emits only golden placeholders (Kenc_i8, s_cq); the host overwrites both per utterance.
            kf = Kenc_b.astype(np.float32)  # [H, TP, HD]
            s_hd = np.abs(kf).max(axis=1) * 1.25 / 127.0  # [H, HD]
            s_hd = np.where(s_hd > 0, s_hd, 1.0)
            Kenc_i8 = np.clip(np.round(kf / s_hd[:, None, :]), -127, 127).astype(np.int8)
            s_cq = s_hd.reshape(-1).astype(np.float32)  # [D] golden per-channel scale (o = h*HD + d)
        if int8_cv:
            # PER-UTTERANCE PER-CHANNEL int8 for Venc (pre-transposed [H,HD,TP]; int8_cv requires co_cross).
            # op_ct_c computes ctc[h,d] = Σ_t attn[h,t]·Venc_int8[h,d,t]; op_mul_cv then scales ctc[h,d] by
            # s_cv[h,d] (POST-GEMV — the scale is per output-channel d, can't fold into attn which is per-t).
            # gen emits golden placeholders (Venc_i8, s_cv); the host overwrites both from the real Venc.
            vt = Venc_b.transpose(0, 2, 1).astype(np.float32)  # [H, HD, TP]
            s_vd = np.abs(vt).max(axis=2) / 127.0  # [H, HD] (headroom 1.0 = full int8 range)
            s_vd = np.where(s_vd > 0, s_vd, 1.0)
            Venc_i8 = np.clip(np.round(vt / s_vd[:, :, None]), -127, 127).astype(np.int8)  # [H, HD, TP]
            s_cv = s_vd.reshape(-1).astype(np.float32)  # [D] golden per-channel scale (o = h*HD + d)
        k_past = bf16(rng.standard_normal((H, P, HD)).astype(np.float32) * 0.5)
        v_past = bf16(rng.standard_normal((H, P, HD)).astype(np.float32) * 0.5)
        kc = np.zeros((H, S, HD), BF16); vc = np.zeros((H, S, HD), BF16)
        if P: kc[:, 0:P], vc[:, 0:P] = k_past, v_past

        # --- register weight buffers ---
        # int8 weights: store the per-row int8 quant + register the static `sw_*` per-row scale buffer the
        # op_mulw applies to the GEMV output. `mat` is the EXACT [M,K] matrix stored below (post-fold/T).
        def w_or_i8(mat, is_i8, sname):
            if is_i8:
                q, s = qrow(mat)
                weights_to_write[pre + sname] = bf16(s)  # static per-row [M] scale (loaded as a normal weight)
                # bf16-buffer/int8-bytes: view the M*K int8 bytes as M*K/2 bf16 slots so the element count
                # matches the GEMV's int8 A layout (a_k=K//2). wb writes the bytes through unchanged.
                return np.ascontiguousarray(q).reshape(-1).view(BF16)
            return bf16(mat).reshape(-1)
        for nm, arr in [
                        # fuse_attn: Wqkv/Wso/Wcq/Wco K-augmented with their bias; the separate bias buffers
                        # are dropped. Else unchanged (+ optional int8 attn-w).
                        ("Wqkv", bf16(aug_bias(mat_qkv, bias_qkv)).reshape(-1) if fuse_attn else w_or_i8(mat_qkv, int8_attn_w, "sw_qkv")),
                        *([] if fuse_attn else [("bias_qkv", bf16(bias_qkv))]),
                        ("Wso", bf16(aug_bias(Wso.T.copy(), bso)).reshape(-1) if fuse_attn else w_or_i8(Wso.T.copy(), int8_attn_w, "sw_so")),
                        *([] if fuse_attn else [("bso", bf16(bso))]),
                        ("Wcq", bf16(aug_bias(mat_cq, bias_cq)).reshape(-1) if fuse_attn else w_or_i8(mat_cq, int8_attn_w, "sw_cq")),
                        *([] if fuse_attn else [("bias_cq", bf16(bias_cq))]),
                        ("Wco", bf16(aug_bias(Wco.T.copy(), bco)).reshape(-1) if fuse_attn else w_or_i8(Wco.T.copy(), int8_attn_w, "sw_co")),
                        *([] if fuse_attn else [("bco", bf16(bco))]),
                        # fuse_ffn: Wf1/Wf2 are K-augmented with their bias (one extra column); the separate
                        # bias_f1/bf2 buffers are dropped (folded in). Else unchanged (+ optional int8).
                        ("Wf1", bf16(aug_bias(mat_f1, bias_f1)).reshape(-1) if fuse_ffn else w_or_i8(mat_f1, int8_ffn, "sw_f1")),
                        *([] if fuse_ffn else [("bias_f1", bf16(bias_f1))]),
                        ("Wf2", bf16(aug_bias(Wf2.T.copy(), bf2)).reshape(-1) if fuse_ffn else w_or_i8(Wf2.T.copy(), int8_ffn, "sw_f2")),
                        *([] if fuse_ffn else [("bf2", bf16(bf2))]),
                        ("Kenc", (Kenc_i8 if int8_ck else Kenc_b).reshape(-1)),
                        # (i): pre-transpose Venc to [H,HD,TP] when coalescing cross; else [H,TP,HD] (deep-C).
                        # int8_cv: Venc_i8 is already pre-transposed int8 [H,HD,TP] (co_cross enforced).
                        ("Venc", (Venc_i8 if int8_cv else (Venc_b.transpose(0, 2, 1).copy() if co_cross else Venc_b)).reshape(-1)),
                        ("kcache", kc.reshape(-1)),
                        # M0.5: vcache stored transposed [H,HD,S] when --coalesce-self-tr (op_ct_s reads it).
                        ("vcache", (vc.transpose(0, 2, 1).copy() if co_self_tr else vc).reshape(-1))]:
            weights_to_write[pre + nm] = arr
        if int8_ck:
            weights_to_write[pre + "s_cq"] = bf16(s_cq)  # per-channel qc scale; host overwrites per utterance
        if int8_cv:
            weights_to_write[pre + "s_cv"] = bf16(s_cv)  # per-channel ctc scale; host overwrites per utterance
        patch_offsets_names += [pre + "kcache", pre + "vcache"]
        # explicit sizes for sliced/cache/score buffers
        bufsz.update({
            pre + "qkv": QKV * 2, pre + "kcache": H * S * HD * 2, pre + "vcache": H * S * HD * 2,
            pre + "scs": 16 * S * 2, pre + "sws": 16 * S * 2,
            **({pre + "s_cq": D * 2} if int8_ck else {}),
            **({pre + "s_cv": D * 2} if int8_cv else {}),
            pre + "Kenc": H * TP * HD * (1 if int8_ck else 2),
            pre + "Venc": H * TP * HD * (1 if int8_cv else 2),
            pre + "scc": 16 * TP * 2, pre + "swc": 16 * TP * 2,
        })
        # int8 weights: explicit halved byte size (auto-inference defaults to bf16; cf. Kenc). bf16 weights
        # stay auto-sized. Value = M*K int8 bytes (the GEMV reads it as M*(K//2) bf16 slots).
        if int8_ffn:
            bufsz[pre + "Wf1"] = FF * D; bufsz[pre + "Wf2"] = D * FF
        if int8_attn_w:
            bufsz[pre + "Wqkv"] = QKV * D; bufsz[pre + "Wso"] = D * D
            bufsz[pre + "Wcq"] = D * D; bufsz[pre + "Wco"] = D * D
        if fuse_ffn:
            # augmented GEMV inputs: producing op writes [0:K_real], GEMV reads [0:K_real+VS]; tail [1,0..]
            # is set ONCE by the engine. xn_f feeds op_f1 (K=D), h feeds op_f2 (K=FF).
            bufsz[pre + "xn_f"] = (D + VS) * 2
            bufsz[pre + "h"] = (FF + VS) * 2
        if fuse_attn:
            # xn_s->op_qkv, cts->op_proj/Wso, xn_c->op_proj/Wcq, ctc->op_proj/Wco (all K=D).
            for b in ("xn_s", "cts", "xn_c", "ctc"):
                bufsz[pre + b] = (D + VS) * 2
        if not co_cross:  # cross-V transpose output buffer only when NOT pre-transposing Venc
            bufsz[pre + "vcTc"] = H * TP * HD * 2
        if not co_self_tr:  # self-V transpose output buffer only when NOT storing vcache transposed (M0.5)
            bufsz[pre + "vcT"] = H * S * HD * 2

        nxt = f"x{l+1}"  # layer output residual buffer
        # (1) self-V transpose: ELIMINATED (--coalesce-self-tr, vcache stored [H,HD,S], op_ct_s reads it
        # directly) | one batched launch (--coalesce-self) | H per-head launches (deep-C).
        if co_self_tr:
            self_tr = []
            v_ct = "vcache"  # already transposed [H,HD,S]
        elif co_self:
            self_tr = [(op_tr_s, pre + "vcache", pre + "vcT")]
            v_ct = "vcT"
        else:
            self_tr = [(op_tr_s, f"{pre}vcache[{h*phs}:{(h+1)*phs}]", f"{pre}vcT[{h*phs}:{(h+1)*phs}]") for h in range(H)]
            v_ct = "vcT"
        # fuse_attn: K-augmented attention inputs — the PRODUCING op writes [0:D] (byte-sliced), the GEMV
        # reads the full [D+VS]; the engine sets the [1,0..] tail. (D2 = D in bytes.)
        D2 = D * 2
        a_xns = f"xn_s[0:{D2}]" if fuse_attn else "xn_s"
        a_xnc = f"xn_c[0:{D2}]" if fuse_attn else "xn_c"
        a_cts = f"cts[0:{D2}]" if fuse_attn else "cts"
        a_ctc = f"ctc[0:{D2}]" if fuse_attn else "ctc"
        # (i) cross-V: op_ct_c reads pre-transposed Venc directly (--coalesce-cross), or H per-head op_tr_c.
        # int8_cv (co_cross enforced): op_ct_c reads int8 Venc, then op_mul_cv scales ctc by per-channel s_cv.
        cross_tr = ([(op_ct_c, pre + "Venc", f"{pre}swc[0:{HSc}]", pre + a_ctc)]
                    + ([(op_mul_cv, pre + "ctc", pre + "s_cv", pre + "ctc")] if int8_cv else []) if co_cross else
                    [(op_tr_c, f"{pre}Venc[{h*phc}:{(h+1)*phc}]", f"{pre}vcTc[{h*phc}:{(h+1)*phc}]") for h in range(H)]
                    + [(op_ct_c, pre + "vcTc", f"{pre}swc[0:{HSc}]", pre + a_ctc)])
        # int8-weights: a per-row scale mul on the GEMV output, inserted BEFORE the bias add (no-op if bf16).
        def wm(op, buf, sname, is_i8):
            return [(op, buf, pre + sname, buf)] if is_i8 else []
        # fuse_attn: drop the bias-add (folded into the aug weight); else keep it (with optional int8 mulw).
        def ba(addop, *args, mulw=None, sname=None, is_i8=False):
            return [] if fuse_attn else ((wm(mulw, args[0], sname, is_i8) if mulw else []) + [(addop, *args)])
        rl += [
            (op_ln, cur, pre + a_xns),
            (op_qkv, pre + "Wqkv", pre + "xn_s", pre + "qkv"), *ba(op_add_qkv, pre + "qkv", pre + "bias_qkv", pre + "qkv", mulw=op_mulw_qkv, sname="sw_qkv", is_i8=int8_attn_w),
            (op_sck, pre + "qkv[1536:3072]", pre + "kcache"), (op_scv, pre + "qkv[3072:4608]", pre + "vcache"),
            (op_sc_s, pre + "kcache", pre + "qkv[0:1536]", f"{pre}scs[0:{HSs}]"), (op_sm_s, pre + "scs", pre + "sws"),
        ] + self_tr + [
            (op_ct_s, pre + v_ct, f"{pre}sws[0:{HSs}]", pre + a_cts),
            (op_proj, pre + "Wso", pre + "cts", pre + "asf"), *ba(op_add768, pre + "asf", pre + "bso", pre + "asf", mulw=op_mulw_d, sname="sw_so", is_i8=int8_attn_w),
            (op_add768, cur, pre + "asf", pre + "x1"),
            (op_ln, pre + "x1", pre + a_xnc),
            (op_proj, pre + "Wcq", pre + "xn_c", pre + "qc"), *ba(op_add768, pre + "qc", pre + "bias_cq", pre + "qc", mulw=op_mulw_d, sname="sw_cq", is_i8=int8_attn_w),
        ] + ([(op_mul_cq, pre + "qc", pre + "s_cq", pre + "qc")] if int8_ck else []) + [
            (op_sc_c, pre + "Kenc", pre + "qc", f"{pre}scc[0:{HSc}]"), (op_sm_c, pre + "scc", pre + "swc"),
        ] + cross_tr + [
            (op_proj, pre + "Wco", pre + "ctc", pre + "acf"), *ba(op_add768, pre + "acf", pre + "bco", pre + "acf", mulw=op_mulw_d, sname="sw_co", is_i8=int8_attn_w),
            (op_add768, pre + "x1", pre + "acf", pre + "x2"),
            (op_ln, pre + "x2", pre + (f"xn_f[0:{D*2}]" if fuse_ffn else "xn_f")),
        ] + ([
            # fuse_ffn: bias folded into Wf1/Wf2 (K-aug). op_f1 reads aug xn_f (D+VS), writes h[0:FF];
            # op_gelu on [0:FF] (DROPPED if fuse_gelu — folded into op_f1's C-tile epilogue); op_f2 reads aug
            # h (FF+VS). The 2 bias-adds are GONE. (slices are in BYTES)
            (op_f1, pre + "Wf1", pre + "xn_f", pre + f"h[0:{FF*2}]"),
            *([] if fuse_gelu else [(op_gelu, pre + f"h[0:{FF*2}]", pre + f"h[0:{FF*2}]")]),
            (op_f2, pre + "Wf2", pre + "h", pre + "ff"),
        ] if fuse_ffn else [
            (op_f1, pre + "Wf1", pre + "xn_f", pre + "h"), *wm(op_mulw_ff, pre + "h", "sw_f1", int8_ffn), (op_add_ff, pre + "h", pre + "bias_f1", pre + "h"), (op_gelu, pre + "h", pre + "h"),
            (op_f2, pre + "Wf2", pre + "h", pre + "ff"), *wm(op_mulw_d, pre + "ff", "sw_f2", int8_ffn), (op_add768, pre + "ff", pre + "bf2", pre + "ff"),
        ]) + [
            (op_add768, pre + "x2", pre + "ff", nxt),
        ]
        layer_data.append(dict(mat_qkv=mat_qkv, bias_qkv=bias_qkv, Wso=Wso, bso=bso, mat_cq=mat_cq,
                               bias_cq=bias_cq, Wco=Wco, bco=bco, mat_f1=mat_f1, bias_f1=bias_f1,
                               Wf2=Wf2, bf2=bf2, Kenc_b=Kenc_b, Venc_b=Venc_b, k_past=k_past, v_past=v_past))
        cur = nxt

    out_name = cur
    if npu_logits:
        # e2e/NPU step 1: ln_post + proj_out on the NPU -> ELF outputs logits[VOCAB_PAD]. VOCAB_PAD pads to a
        # multiple of num_aie_columns=8 (GEMV M constraint). ln_post affine folds into proj_out (LN op is
        # pure-normalize): logits = (norm·γ + β)·W -> A=(γ·W).T [VOCAB,D], bias = β·W [VOCAB].
        VOCAB, VOCAB_PAD = 51865, 65536  # 65536 = 2*32768 (ElementwiseAdd tiling) and %8 (GEMV M)
        g_post = np.load(os.path.join(w, "ln_post.weight.npy")).astype(np.float32)
        b_post = np.load(os.path.join(w, "ln_post.bias.npy")).astype(np.float32)
        Wproj = np.load(os.path.join(w, "proj_out.weight.npy")).astype(np.float32)  # [D, VOCAB]
        mat_proj = (g_post[:, None] * Wproj).T.copy()            # [VOCAB, D]
        bias_proj = (b_post @ Wproj).astype(np.float32)          # [VOCAB]
        mat_pad = np.zeros((VOCAB_PAD, D), np.float32); mat_pad[0:VOCAB] = mat_proj
        bias_pad = np.full(VOCAB_PAD, -1e30, np.float32); bias_pad[0:VOCAB] = bias_proj  # pad rows never win argmax
        op_proj_out = GEMV(M=VOCAB_PAD, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=VOCAB_PAD // 8, context=ctx)
        op_add_logits = ElementwiseAdd(size=VOCAB_PAD, tile_size=VOCAB_PAD // 8, num_aie_columns=8, context=ctx)
        rl += [
            (op_ln, cur, "hn"),                                  # reuse op_ln (pure normalize, size=D)
            (op_proj_out, "Wproj", "hn", "logits"),
            (op_add_logits, "logits", "bias_proj", "logits"),
        ]
        weights_to_write["Wproj"] = bf16(mat_pad).reshape(-1)
        weights_to_write["bias_proj"] = bf16(bias_pad)
        bufsz["hn"] = D * 2
        bufsz["logits"] = VOCAB_PAD * 2
        out_name = "logits"
    if os.environ.get("DUMP_OPS"):
        from collections import Counter
        c = Counter(type(e[0]).__name__ for e in rl)
        print(f"# runlist: {len(rl)} entries over NL={NL} ({len(rl)//NL}/layer)")
        for nm, n in c.most_common():
            print(f"  {nm:18} {n:4}  ({n//NL}/layer)")
        print("# --- full per-layer op sequence (buffers: out <- ins) ---")
        for e in rl:
            op = e[0]; bufs = list(e[1:])
            ins = ", ".join(str(b) for b in bufs[:-1]); out = str(bufs[-1]) if bufs else "?"
            print(f"  {type(op).__name__:16} {out:14} <- {ins}")
        import sys; sys.exit(0)
    fused = FusedMLIROperator("decode", rl, input_args=["x"], output_args=[out_name],
                              buffer_sizes=bufsz, context=ctx)
    fused.compile()
    elf = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    wnames = list(weights_to_write.keys())
    # fuse_bias: the K-augmented SCRATCH inputs (xn_f, h) must be in the layout so the engine can find their
    # arena offset to write the augmentation tail. (Normal scratch buffers aren't exported.)
    _aug_sufs = (["xn_f", "h"] if fuse_ffn else []) + (["xn_s", "cts", "xn_c", "ctc"] if fuse_attn else [])
    aug_names = [f"L{li}_{suf}" for li in range(NL) for suf in _aug_sufs]
    lay = {n: fused.get_layout_for_buffer(n) for n in ["x", out_name] + wnames + aug_names}

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
    if npu_logits:
        # e2e/NPU: the ELF runs ln_post + proj_out on-NPU and outputs logits[VOCAB_PAD], so the golden
        # output must be the logits too (not the 768-hidden). Match the on-NPU op: hn = pure-normalize LN
        # (gamma folded into mat_proj), logits = hn @ mat_proj.T + bias_proj; pad rows = -1e30 (never win
        # argmax). Without this the golden was the hidden state mislabeled "logits" -> the rel-L2 gate was
        # meaningless and argmax-parity uncheckable.
        hn = ln(x_out.astype(np.float32))
        lg = (hn @ mat_proj.T).astype(np.float32) + bias_proj
        gold = np.full(VOCAB_PAD, -1e30, np.float32); gold[0:VOCAB] = lg
        x_out = bf16(gold)

    bdir = os.path.join(a.out, "buffers")
    def wb(n, v):
        v = np.asarray(v)
        # int8 buffers (quantized weights / golden K/V) write RAW bytes; everything else is bf16.
        data = v.tobytes() if v.dtype == np.int8 else np.asarray(v, BF16).tobytes()
        open(os.path.join(bdir, f"{n}.bin"), "wb").write(data)
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
        # lever #3 layout contract for the host: when coalesce_cross, the host must write each L*_Venc
        # buffer pre-transposed [H,HD,TP] (op_ct_c reads it directly; no per-token op_tr_c). coalesce_self
        # batches op_tr_s (host vcache layout unchanged). Default false = deep-C [H,TP,HD].
        "coalesce_cross": bool(co_cross), "coalesce_self": bool(co_self),
        # M0.5: when coalesce_self_tr, the host must (a) write each L*_vcache transposed [H,HD,S], and
        # (b) drive a 2nd addr scratchpad `vcache_off` = n_self (column) per token (kv_off stays n_self*HD).
        "coalesce_self_tr": bool(co_self_tr),
        **({"vcache_param": "vcache_off"} if co_self_tr else {}),
        # int8 cross-K (per-utterance per-channel): L*_Kenc are int8 [H,TP,HD]; the host quantizes Kenc per
        # utterance + writes the L*_s_cq [D] per-channel scale buffer that op_mul_cq applies to qc.
        "int8_cross_k": bool(int8_ck),
        # int8 cross-V (per-utterance per-channel; implies coalesce_cross): L*_Venc are int8 [H,HD,TP]; the
        # host quantizes Venc per utterance + writes L*_s_cv [D] that op_mul_cv applies to the ctc output.
        "int8_cross_v": bool(int8_cv),
        # int8 STATIC weights: L*_W{f1,f2} (int8_ffn) / L*_W{qkv,so,cq,co} (int8_attn_w) are int8 [M,K//2];
        # the op_mulw applies the static L*_sw_* per-row scale to the GEMV output before its bias add.
        "int8_ffn": bool(int8_ffn), "int8_attn_w": bool(int8_attn_w),
        # BIAS FUSION (K-aug): each listed per-layer input buffer L*_<suffix> is sized real_k+VS; the engine
        # writes the augmentation tail [1, 0..0] (VS elems) at element offset real_k ONCE (the producing op
        # only writes [0:real_k], so the tail persists). Lets GEMV(W_aug,x_aug)=W·x+bias with no add op.
        # e2e/NPU: ELF outputs logits[VOCAB_PAD] (ln_post+proj_out on-NPU); the engine reads them directly +
        # drops the host proj_out matmul (argmax stays host: logits[0:51865]).
        "npu_logits": bool(npu_logits),
        "fuse_bias_aug": ({**({"xn_f": D, "h": FF} if fuse_ffn else {}),
                           **({"xn_s": D, "cts": D, "xn_c": D, "ctc": D} if fuse_attn else {})}), "fuse_bias_vs": VS,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\nwrote {NL}-layer decode ELF ({len(elf)}B, scratch {scr/1e6:.1f}MB) to {a.out}")


if __name__ == "__main__":
    main()
