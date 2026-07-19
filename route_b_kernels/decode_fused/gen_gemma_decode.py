#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gemma-3 small-LLM WHOLE decode stack as ONE fused full ELF (staged, device-UNTESTED).

Mirrors the two proven references:
  * route_b_kernels/decode_fused/gen_decode.py  -- Whisper fused decode: the deep-C scratchpad KV
    addressing (constant ELF, per-token `kv_off` StridedCopy offset + `sm_mask` softmax width) that we
    ALREADY run at 1 dispatch/token on device. We reuse that mechanism verbatim.
  * IRON iron/applications/llama_3.2_1b/llama_npu.py  -- a working GQA LLM decode assembled as one fused
    ELF (RMSNorm + q/k/v/o GEMV + RoPE + StridedCopy KV cache + Repeat GQA-broadcast + batched score GEMV
    + softmax + V-transpose + context GEMV + SwiGLU FFN + tied lm-head). Gemma is the SAME graph.

This file is PREP: it is source only (writing it needs no device). The execution agent BUILDS it on the
NPU (fork instance aiecc) and gates it. Every Gemma-specific delta vs the two references is marked
`# GEMMA:`; every point that must be checked on device is marked `# VERIFY:`.

Gemma-3 decode block (verified against transformers modeling_gemma3.py, 2026-07-17), all proj BIAS-FREE:
  x0 = x
  h  = RMSNorm_input(x)                              # foldable into q/k/v GEMV weights (diag(1+w))
  q  = Wq @ h  [n_q*hd] ; k = Wk @ h [n_kv*hd] ; v = Wv @ h [n_kv*hd]
  q  = RMSNorm_q(q per head, hd) ; k = RMSNorm_k(k per head, hd)   # GEMMA: extra per-head norms
  q,k = RoPE(theta_local if sliding else theta_global)            # GEMMA: dual theta
  append k,v to KV cache at position n_past                        # StridedCopy kv_off (deep-C)
  scores = (q . K^T) * (query_pre_attn_scalar**-0.5)              # per q head, single shared KV head
  scores = softmax(scores, causal width n_past+1 [+ window lo])   # sm_mask; GEMMA: window band, see NOTE
  ctx    = scores @ V ; a = Wo @ ctx
  x1 = x0 + RMSNorm_post_attn(a)                     # GEMMA: un-foldable sandwich norm (residual after)
  hf = RMSNorm_pre_ffn(x1)                           # foldable into gate/up GEMV weights
  g  = gelu_tanh(Wgate @ hf) * (Wup @ hf) ; d = Wdown @ g          # GEMMA: GeGLU (GELU not SiLU)
  x2 = x1 + RMSNorm_post_ffn(d)                      # GEMMA: un-foldable sandwich norm
  ... x_final = RMSNorm_final(x_last) ; logits = embed^T @ x_final ; argmax

Weight folding done host-side in this generator:
  * every RMSNorm weight is stored as (1.0 + w) so IRON's weighted RMSNorm (norm(x)*w') is EXACT Gemma.
  * input_layernorm / pre_feedforward_layernorm are NOT pre-folded into the GEMV here (kept as explicit
    weighted-RMSNorm ops for a clean per-node rel-L2 gate); folding diag(1+w) into Wq/Wk/Wv/Wgate/Wup is a
    byte/op follow-on once correctness is proven (see turnkey doc step 7).

Run INSIDE the fork IRON env (scripts/toolchain_up.sh). Example:
  python route_b_kernels/decode_fused/gen_gemma_decode.py --weights <dumped_gemma_270m> \
      --out artifacts/gemma/decode_270m --layers 18
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 -- MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext
from iron.common.sequence import OperatorSequence  # migrated off the carried iron.common.fusion (FusedMLIROperator)
from iron.operators.gemv.op import GEMV
from iron.operators.rms_norm.op import RMSNorm
from iron.operators.rope.op import RoPE
from iron.operators.elementwise_add.op import ElementwiseAdd
from iron.operators.elementwise_mul.op import ElementwiseMul
from iron.operators.softmax.op import Softmax
from iron.operators.strided_copy.op import StridedCopy
from iron.operators.transpose.op import Transpose
from iron.operators.gelu.op import GELU
from iron.operators.repeat.op import Repeat

BF16 = ml_dtypes.bfloat16


def bf16(a):
    return np.asarray(a).astype(BF16)


# ---- Gemma-3 270M released-checkpoint dims (see rust/npu-gemma/src/config.rs GEMMA3_270M) ----
# GEMMA: n_q*head_dim (1024) != d_model (640); head_dim=256 explicit.
D = 640            # d_model / emb_dim
NL_DEFAULT = 18
Hq = 4             # q heads
Hkv = 1            # kv heads (GQA group = 4)
HD = 256           # head_dim
QD = Hq * HD       # 1024 (q projection width)
KVD = Hkv * HD     # 256  (k/v projection width)
FF = 2048          # intermediate_size
VOCAB = 262144     # tied embedding; 262144 % 8 == 0 -> NO pad (unlike Whisper 51865)
SWIN = 512         # sliding_window
SW_PATTERN = 6     # every 6th layer (1-based) is GLOBAL attention
QPAS = 256.0       # query_pre_attn_scalar (score scale denom; sqrt -> 1/16)
EPS = 1e-6


def is_global(layer_idx):
    # GEMMA: layer_types -- global on every SW_PATTERN-th layer (matches GemmaConfig::is_global_layer).
    return (layer_idx + 1) % SW_PATTERN == 0


def npy(w, name):
    return np.load(os.path.join(w, f"{name}.npy")).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="dir of dumped Gemma-3-270M weights (.npy); see turnkey doc step 2 for the dumper")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=NL_DEFAULT)
    ap.add_argument("--max-seq", type=int, default=2048, help="KV-cache padded capacity S (>= prompt+gen)")
    a = ap.parse_args()
    NL, S = a.layers, a.max_seq
    scale = QPAS ** -0.5
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)

    ctx = AIEContext()

    # ---- op vocabulary (created ONCE, reused across all layers; same dims per layer) ----
    # GEMMA: weighted RMSNorm. Body norm (size D) and the per-head q/k norm (size HD) are separate ops.
    op_norm = RMSNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, weighted=True, context=ctx)
    op_qk_norm = RMSNorm(size=HD, num_aie_columns=1, num_channels=1, tile_size=HD, weighted=True, context=ctx)
    # projections as GEMV (M=out, K=in). bias-free -> no ElementwiseAdd after (simpler than Whisper).
    op_q = GEMV(M=QD, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=QD // 8, context=ctx)
    op_kv = GEMV(M=KVD, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=KVD // 8, context=ctx)  # GEMMA: tile_size_output must be <= M/cols (=KVD/8); the old HD/2 violated it
    op_o = GEMV(M=D, K=QD, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    # GEMMA: dual-theta RoPE -- one op shape, TWO angle tables (host precomputes local & global).
    # decode = 1 query token -> rows = heads, angle_rows = 1.
    op_rope_q = RoPE(rows=Hq, cols=HD, angle_rows=1, context=ctx)
    op_rope_k = RoPE(rows=Hkv, cols=HD, angle_rows=1, context=ctx)
    # KV cache append: deep-C scratchpad offset "kv_off" (element units = n_past*HD), constant ELF.
    sc = dict(input_sizes=(Hkv, HD), input_strides=(HD, 1), input_offset=0,
              output_sizes=(1, Hkv, HD), output_strides=(0, S * HD, 1), output_offset=0,
              input_buffer_size=Hkv * HD, output_buffer_size=Hkv * S * HD, num_aie_channels=1)
    op_sck = StridedCopy(**sc, output_offset_parameter="kv_off", context=ctx)
    op_scv = StridedCopy(**sc, output_offset_parameter="kv_off", context=ctx)
    # GQA broadcast: Repeat single KV head -> Hq heads for the batched score/context GEMVs.
    # VERIFY: the free-by-layout alternative is a batch-stride-0 GEMV read of the single KV head (0 ops,
    # 0 bytes) -- the preferred optimization once correct; Repeat is the correctness-first mirror of llama.
    op_rep_k = Repeat(rows=Hkv, cols=S * HD, repeat=Hq // Hkv, transfer_size=HD, context=ctx)
    op_rep_v = Repeat(rows=Hkv, cols=S * HD, repeat=Hq // Hkv, transfer_size=HD, context=ctx)
    # scores = K @ q per head: GEMV M=S (context capacity), K=HD, batched over Hq heads.
    op_scores = GEMV(M=S, K=HD, num_aie_columns=8, tile_size_input=4, tile_size_output=S // 8,
                     num_batches=Hq, context=ctx)
    # GEMMA: score scale = query_pre_attn_scalar**-0.5 (== 1/16 here). Elementwise mul by a constant buffer.
    op_scale = ElementwiseMul(size=Hq * S, tile_size=S // 8, num_aie_columns=8, context=ctx)
    # softmax with deep-C runtime width param "sm_mask" (= n_past+1 causal). See NOTE for the window band.
    # GEMMA: the Softmax op requires rows % 16 == 0, so pad the Hq=4 heads to 16 (mirrors the whisper
    # gen_decode rows=16). Softmax is per-row, so the 12 dummy rows never touch the real heads; the
    # scores/scale ops fill only the first Hq rows and op_ctx (num_batches=Hq) reads only the first Hq.
    SM_ROWS = 16
    op_softmax = Softmax(rows=SM_ROWS, cols=S, num_aie_columns=1, num_channels=1, rtp_vector_size=S,
                         vector_size_parameter="sm_mask", context=ctx)
    # V transpose per head (for the context GEMV), then context = V^T @ weights.
    op_trv = Transpose(M=S, N=HD, num_aie_columns=2, num_channels=1, m=256, n=32, s=8, context=ctx)
    op_ctx = GEMV(M=HD, K=S, num_aie_columns=8, tile_size_input=4, tile_size_output=HD // 8,
                  num_batches=Hq, context=ctx)
    # FFN (GeGLU). GEMMA: GELU-tanh, not SiLU.
    op_gate = GEMV(M=FF, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=FF // 8, context=ctx)
    op_up = GEMV(M=FF, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=FF // 8, context=ctx)
    op_gelu = GELU(size=FF, num_aie_columns=8, num_channels=1, tile_size=FF // 8, context=ctx)
    op_mul_ffn = ElementwiseMul(size=FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)
    op_down = GEMV(M=D, K=FF, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    op_add = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)
    # tied lm-head: logits = embed @ x_final. VOCAB % 8 == 0 -> no pad.
    # GEMMA: VOCAB=262144 -> tile_size_output=VOCAB//8=32768 makes a 64KB (double-buffered 128KB) C tile
    # that overflows L1. Split the output finer (VOCAB//64=4096 -> 8KB/tile, 8 iterations/column) so it
    # fits; correctness-identical, just more passes on the one-shot lm-head.
    op_head = GEMV(M=VOCAB, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=VOCAB // 64, context=ctx)

    weights = {}   # name -> bf16 array
    bufsz = {}
    cache_names = []
    rl = []
    cur = "x"   # residual buffer

    def load_norm(name):
        # GEMMA: store (1 + w) so weighted RMSNorm (norm(x)*w') == Gemma RMSNorm.
        return bf16(1.0 + npy(a.weights, name))

    for l in range(NL):
        p = f"L{l}_"
        gl = is_global(l)
        # ---- weights (bias-free) ----
        weights[p + "n_in"] = load_norm(f"model.layers.{l}.input_layernorm.weight")
        weights[p + "n_qn"] = load_norm(f"model.layers.{l}.self_attn.q_norm.weight")
        weights[p + "n_kn"] = load_norm(f"model.layers.{l}.self_attn.k_norm.weight")
        weights[p + "n_pa"] = load_norm(f"model.layers.{l}.post_attention_layernorm.weight")
        weights[p + "n_pf"] = load_norm(f"model.layers.{l}.pre_feedforward_layernorm.weight")
        weights[p + "n_pff"] = load_norm(f"model.layers.{l}.post_feedforward_layernorm.weight")
        weights[p + "Wq"] = bf16(npy(a.weights, f"model.layers.{l}.self_attn.q_proj.weight")).reshape(-1)
        weights[p + "Wk"] = bf16(npy(a.weights, f"model.layers.{l}.self_attn.k_proj.weight")).reshape(-1)
        weights[p + "Wv"] = bf16(npy(a.weights, f"model.layers.{l}.self_attn.v_proj.weight")).reshape(-1)
        weights[p + "Wo"] = bf16(npy(a.weights, f"model.layers.{l}.self_attn.o_proj.weight")).reshape(-1)
        weights[p + "Wg"] = bf16(npy(a.weights, f"model.layers.{l}.mlp.gate_proj.weight")).reshape(-1)
        weights[p + "Wu"] = bf16(npy(a.weights, f"model.layers.{l}.mlp.up_proj.weight")).reshape(-1)
        weights[p + "Wd"] = bf16(npy(a.weights, f"model.layers.{l}.mlp.down_proj.weight")).reshape(-1)
        # KV cache buffers (single kv head, [1,S,HD]); host zero-inits, StridedCopy appends per token.
        weights[p + "kc"] = np.zeros(Hkv * S * HD, BF16)
        weights[p + "vc"] = np.zeros(Hkv * S * HD, BF16)
        cache_names += [p + "kc", p + "vc"]
        # angle table pointer: local vs global (host writes both tables; per-token angle row selected host-side)
        ang = "rope_local" if not gl else "rope_global"

        bufsz.update({
            p + "q": QD * 2, p + "k": KVD * 2, p + "v": KVD * 2,
            p + "kc": Hkv * S * HD * 2, p + "vc": Hkv * S * HD * 2,
            p + "kr": Hq * S * HD * 2, p + "vr": Hq * S * HD * 2, p + "vt": Hq * S * HD * 2,
            p + "sc": SM_ROWS * S * 2, p + "sw": SM_ROWS * S * 2,   # padded to 16 rows for Softmax (rows%16)
            p + "cx": QD * 2, p + "a": D * 2,
            p + "g": FF * 2, p + "u": FF * 2, p + "gh": FF * 2, p + "d": D * 2,
        })
        nxt = f"x{l+1}"
        rl += [
            (op_norm, cur, p + "n_in", p + "hn"),
            (op_q, p + "Wq", p + "hn", p + "q"),
            (op_kv, p + "Wk", p + "hn", p + "k"),
            (op_kv, p + "Wv", p + "hn", p + "v"),
            # GEMMA: per-head q/k RMSNorm (norm over head_dim), applied per head.
            *[(op_qk_norm, f"{p}q[{h*HD*2}:{(h+1)*HD*2}]", p + "n_qn", f"{p}q[{h*HD*2}:{(h+1)*HD*2}]") for h in range(Hq)],
            *[(op_qk_norm, f"{p}k[{h*HD*2}:{(h+1)*HD*2}]", p + "n_kn", f"{p}k[{h*HD*2}:{(h+1)*HD*2}]") for h in range(Hkv)],
            # GEMMA: dual-theta RoPE (angle table chosen by layer type).
            (op_rope_q, p + "q", ang, p + "q"),
            (op_rope_k, p + "k", ang, p + "k"),
            # KV append (deep-C kv_off), then GQA broadcast 1 -> Hq for the batched GEMVs.
            (op_sck, p + "k", p + "kc"),
            (op_scv, p + "v", p + "vc"),
            (op_rep_k, p + "kc", p + "kr"),
            (op_rep_v, p + "vc", p + "vr"),
            # GEMMA: sc/sw are padded to SM_ROWS(16) rows for the Softmax rows%16 rule, but scores/scale
            # /context only touch the first Hq rows -- slice the sub-region so the fused buffer layout
            # agrees on one size (mirrors whisper gen_decode's scs[0:H*S]). Softmax alone spans all 16.
            (op_scores, p + "kr", p + "q", f"{p}sc[0:{Hq*S*2}]"),
            (op_scale, f"{p}sc[0:{Hq*S*2}]", "attn_scale", f"{p}sc[0:{Hq*S*2}]"),
            (op_softmax, p + "sc", p + "sw"),
            *[(op_trv, f"{p}vr[{h*S*HD*2}:{(h+1)*S*HD*2}]", f"{p}vt[{h*S*HD*2}:{(h+1)*S*HD*2}]") for h in range(Hq)],
            (op_ctx, p + "vt", f"{p}sw[0:{Hq*S*2}]", p + "cx"),
            (op_o, p + "Wo", p + "cx", p + "a"),
            # GEMMA: post-attn sandwich norm BEFORE the residual add (un-foldable).
            (op_norm, p + "a", p + "n_pa", p + "a"),
            (op_add, cur, p + "a", p + "x1"),
            # FFN (GeGLU): pre-ffn norm -> gate/up -> gelu*up -> down.
            (op_norm, p + "x1", p + "n_pf", p + "hf"),
            (op_gate, p + "Wg", p + "hf", p + "g"),
            (op_up, p + "Wu", p + "hf", p + "u"),
            (op_gelu, p + "g", p + "g"),
            (op_mul_ffn, p + "g", p + "u", p + "gh"),
            (op_down, p + "Wd", p + "gh", p + "d"),
            # GEMMA: post-ffn sandwich norm BEFORE the residual add (un-foldable).
            (op_norm, p + "d", p + "n_pff", p + "d"),
            (op_add, p + "x1", p + "d", nxt),
        ]
        bufsz[p + "hn"] = D * 2
        bufsz[p + "hf"] = D * 2
        cur = nxt

    # final norm + tied lm-head + logits.
    weights["n_final"] = load_norm("model.norm.weight")
    weights["W_head"] = bf16(npy(a.weights, "model.embed_tokens.weight")).reshape(-1)  # tied
    weights["attn_scale"] = np.full(Hq * S, scale, BF16)
    rl += [
        (op_norm, cur, "n_final", "xf"),
        (op_head, "W_head", "xf", "logits"),
    ]
    bufsz["xf"] = D * 2
    bufsz["logits"] = VOCAB * 2

    if os.environ.get("DUMP_OPS"):
        from collections import Counter
        c = Counter(type(e[0]).__name__ for e in rl)
        print(f"# runlist: {len(rl)} entries over NL={NL} ({(len(rl)-2)//NL}/layer + 2 tail)")
        for nm, n in c.most_common():
            print(f"  {nm:16} {n:4}")
        import sys
        sys.exit(0)

    # Only declare the rope tables actually referenced: with a small --layers (e.g. 2) every layer is
    # sliding (global is every SW_PATTERN-th, first at layer 5), so rope_global would be an unused input
    # and the fused layout rejects it. The full 18-layer build declares both.
    rope_inputs = ["rope_local"] + (["rope_global"] if any(is_global(l) for l in range(NL)) else [])
    fused = OperatorSequence("gemma_decode", rl, input_args=["x", *rope_inputs],
                             output_args=["logits"], buffer_sizes=bufsz, dispatch="fused", context=ctx)
    fused.compile()

    if os.environ.get("GEMMA_SMOKE"):
        # Device smoke (needs exclusive NPU): load the freshly-compiled ELF onto the NPU and run ONE
        # dispatch with zero-filled buffers, to catch device-runtime obstacles (XRT load, hw_context
        # build, kernel dispatch/completion) BEFORE building the full weight-loaded parity harness.
        # Output is meaningless (no weights); the signal is ERT_CMD_STATE_COMPLETED, no hang/error.
        print(f"[smoke] compiled {NL}-layer ELF; loading onto NPU + running one dispatch ...", flush=True)
        c = fused.get_callable()
        c()
        print("[smoke] PASS: fused gemma decode ELF ran on NPU (ERT_CMD_STATE_COMPLETED)", flush=True)
        return

    if os.environ.get("GEMMA_PARITY"):
        # Weight-loaded greedy token-parity on device vs the host oracle. GEMMA_PARITY=<oracle_dir>
        # (dir with oracle.json: {prompt_ids, step_argmax}). Needs exclusive NPU. Uses the OperatorSequence
        # ParameterScratchpad runtime (per-token kv_off/sm_mask) -- unblocked by the rebuilt py3.14 pyxrt.
        import glob as _glob
        import struct as _struct
        oracle = json.load(open(os.path.join(os.environ["GEMMA_PARITY"], "oracle.json")))
        prompt_ids, step_argmax = oracle["prompt_ids"], oracle["step_argmax"]
        c = fused.get_callable()
        # 1) lay all weights + zero KV caches into the scratch arena, sync once.
        for nm, arr in weights.items():
            b = c.get_buffer(nm)
            np.copyto(b.data, np.asarray(arr, BF16).reshape(-1))
        c.scratch_buffer.to("npu")
        # 2) pure-Python scratchpad writer (mirrors test_utils::ParameterScratchpad: addr=raw, core=<<2).
        pp = sorted(_glob.glob(os.path.join(os.getcwd(), "**", "params.txt"), recursive=True), key=os.path.getmtime)
        params = {}
        for line in open(pp[-1]).read().splitlines()[1:]:
            nm, idx, _ty, kind = line.split()
            params[nm] = (int(idx), kind)
        import pyxrt as _pyxrt
        sbo = c.run_handle.get_ctrl_scratchpad_bo()
        smv = memoryview(sbo.map()).cast("B")           # writable byte view of the ctrl scratchpad
        def _wp(nm, val):
            idx, kind = params[nm]
            enc = ((int(val) << 2) if kind == "core" else int(val)) & 0xFFFFFFFF
            smv[idx * 4:idx * 4 + 4] = _struct.pack("<I", enc)
        # 3) rope angle tables (interleaved cos/sin, [S,HD]); Gemma dual base.
        def rope_tab(base):
            inv = 1.0 / (base ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
            fr = np.outer(np.arange(S), inv)                    # [S, HD/2]
            a = np.empty((S, HD), np.float64); a[:, 0::2] = np.cos(fr); a[:, 1::2] = np.sin(fr)
            return a.astype(BF16)
        rl_tab, rg_tab = rope_tab(10000.0), rope_tab(1000000.0)  # local 1e4, global 1e6
        emb = weights["W_head"].reshape(VOCAB, D)               # tied embedding, bf16
        norm = float(np.sqrt(D))
        xin, rloc, rglob, lg = c.get_buffer("x"), c.get_buffer("rope_local"), c.get_buffer("rope_global"), c.get_buffer("logits")
        # 4) prefill prompt then greedy-decode; compare argmax chain to the oracle.
        got, tok, npos = [], None, 0
        gen_steps = len(step_argmax)
        seq = list(prompt_ids)                                  # positions we FEED
        for i in range(len(prompt_ids) + gen_steps - 1):
            t = seq[npos]
            np.copyto(xin.data, (emb[t].astype(np.float32) * norm).astype(BF16))
            np.copyto(rloc.data, rl_tab[npos, :rloc.data.size]); np.copyto(rglob.data, rg_tab[npos, :rglob.data.size])
            _wp("kv_off", npos * HD); _wp("sm_mask", npos + 1)
            sbo.sync(_pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            c()
            nt = int(np.argmax(lg.data[:VOCAB].astype(np.float32)))
            if npos >= len(prompt_ids) - 1:                     # logits after the last prompt tok = gen[0]
                got.append(nt); seq.append(nt)
            npos += 1
        match = sum(1 for a, b in zip(got, step_argmax) if a == b)
        print(f"[parity] oracle: {step_argmax}", flush=True)
        print(f"[parity] device: {got}", flush=True)
        print(f"[parity] {match}/{len(step_argmax)} tokens match" + (" -- PASS" if match == len(step_argmax) else ""), flush=True)
        return

    # migrated off fusion.load_elf: read the built full-ELF straight from the sequence's artifact.
    elf = np.fromfile(fused.artifacts[0].filename, dtype=np.uint32).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    wnames = list(weights.keys())
    lay = {n: fused.get_layout_for_buffer(n) for n in ["x", "logits"] + wnames}

    # deep-C params.txt (kv_off addr + sm_mask core), same parse as gen_decode.py.
    import glob
    import shutil
    _pp = sorted(glob.glob("**/gemma_decode*.mlir.prj/params.txt", recursive=True), key=os.path.getmtime)
    scratchpad_params = {}
    if _pp:
        shutil.copy(_pp[-1], os.path.join(a.out, "params.txt"))
        for line in open(_pp[-1]).read().splitlines()[1:]:
            if line.strip():
                nm, idx, ty, kind = line.split()
                scratchpad_params[nm] = {"byte_offset": int(idx) * 4, "kind": kind, "dtype": ty}

    bdir = os.path.join(a.out, "buffers")

    def wb(n, v):
        open(os.path.join(bdir, f"{n}.bin"), "wb").write(np.asarray(v, BF16).tobytes())
    for nm, arr in weights.items():
        wb(nm, arr)
    open(os.path.join(a.out, "decode.elf"), "wb").write(elf)

    meta = {
        "elf": "decode.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x", "rope_local", "rope_global"], "weights": wnames, "output": "logits",
        "scratchpad": {"params": scratchpad_params, "kv_param": "kv_off", "mask_param": "sm_mask",
                       "head_dim": HD, "kv_heads": Hkv},
        "dims": {"layers": NL, "d_model": D, "q_heads": Hq, "kv_heads": Hkv, "head_dim": HD,
                 "ffn": FF, "vocab": VOCAB, "S": S, "sliding_window": SWIN, "sw_pattern": SW_PATTERN},
        # GEMMA per-token host protocol (see turnkey doc step 5):
        #   x            = embed[token] * sqrt(d_model)   (bf16 cast, absolute position n_past)
        #   rope_local/global : precomputed [S,HD] sin/cos angle tables; the row for n_past is used
        #   kv_off (addr)  = n_past * head_dim  (element units, raw)
        #   sm_mask (core) = n_past + 1   (causal width; <<2 by host per firmware UPDATE_REG)
        #   NOTE window band: for context <= sliding_window (512) local == global, so the plain causal
        #   width is EXACT for the phase-0 prompt (<512 tokens). The true low-cutoff band (mask kv <
        #   n_past-511 on local layers) is a follow-on -- add a 2nd `sm_lo` param or an additive mask row.
        "gemma_layer_types": ["global" if is_global(l) else "sliding" for l in range(NL)],
        "cache_buffers": cache_names,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\nwrote {NL}-layer Gemma decode ELF ({len(elf)}B, scratch {scr/1e6:.1f}MB) to {a.out}")


if __name__ == "__main__":
    main()
