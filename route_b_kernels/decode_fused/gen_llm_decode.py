#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decoder-LLM whole decode stack as ONE fused ELF, built from an `LlmSpec`.

Generalises gen_gemma_decode.py from one checkpoint to the spec vocabulary in llm_decode_spec.py:
a MODEL is a spec plus its weights, not a generator. Same deep-C mechanism as the shipped Whisper
fused decode (gen_decode.py): the ELF is CONSTANT across tokens; per token the host writes `x`, the
RoPE angle row, and two scratchpad params (`kv_off`, `sm_mask`), then dispatches ONCE.

Per block, all projections bias-free:
  h  = RMSNorm_in(x)
  q,k,v = Wq/Wk/Wv @ h
  q,k   = RMSNorm_qk(per head, head_dim)          # if spec.qk_norm
  q,k   = RoPE(theta)                             # dual-theta if spec.rope_theta_local
  KV cache append at n_past; GQA broadcast kv -> q heads
  scores = (K @ q) * spec.attn_scale ; softmax(width n_past+1)
  ctx = V^T @ scores ; a = Wo @ ctx
  [sandwich] a = RMSNorm_post_attn(a)             # Gemma-3 only
  x1 = x + a
  hf = RMSNorm_pre_ffn(x1)
  d  = Wdown @ (act(Wgate @ hf) * (Wup @ hf))     # act = gelu_tanh | silu
  [sandwich] d = RMSNorm_post_ffn(d)              # Gemma-3 only
  x2 = x1 + d
then RMSNorm_final + tied lm-head -> logits.

Run INSIDE the fork IRON env (scripts/toolchain_up.sh), never the wheel python. Example:
  python route_b_kernels/decode_fused/gen_llm_decode.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --out artifacts/qwen3-0.6b/decode --layers 28
"""
import argparse
import json
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import SPECS  # noqa: E402

import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext  # noqa: E402
from iron.common.fusion import FusedMLIROperator, load_elf  # noqa: E402
from iron.operators.gemv.op import GEMV  # noqa: E402
from iron.operators.rms_norm.op import RMSNorm  # noqa: E402
from iron.operators.rope.op import RoPE  # noqa: E402
from iron.operators.elementwise_add.op import ElementwiseAdd  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402
from iron.operators.softmax.op import Softmax  # noqa: E402
from iron.operators.strided_copy.op import StridedCopy  # noqa: E402
from iron.operators.transpose.op import Transpose  # noqa: E402
from iron.operators.gelu.op import GELU  # noqa: E402
from iron.operators.silu.op import SiLU  # noqa: E402
from iron.operators.repeat.op import Repeat  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = 8      # num_aie_columns for every GEMV; the spec's check() is written against this
TSI = 4       # tile_size_input


def bf16(a):
    return np.asarray(a).astype(BF16)


L1_BYTES = 65536      # AIE2P core local memory (getLocalMemorySize(), AIETargetModel.h)
L1_RESERVE = 8192     # stack + the allocator's own slack; measured headroom, not a guess (see below)


def gemv_tile_output(M, K, cols=None, tsi=None):
    """Largest legal `tile_size_output` for a GEMV that also FITS L1.

    Two independent constraints, and only the first is checked by the toolchain:

    1. design.py asserts `m_output <= M//cols` and `(M//cols) % m_output == 0`, plus
       `m_output % m_input == 0`.
    2. NOTHING checks L1 capacity. The generated core holds, per the linker map of a failing build:
       the C output tile DOUBLE-buffered (2 x m_output x 2B), the A input tile double-buffered
       (2 x m_input x K x 2B) and the B vector double-buffered (2 x K x 2B). Exceed 64 KB and aiecc
       dies with "'aie.tile' op Basic sequential allocation also failed" -- which names a tile, not a
       tile SIZE, so it reads as a placement bug rather than "your output tile is too big".

    Taking m_output = M//cols (the largest the asserts allow) blows constraint 2 on the lm-head:
    vocab 151936 -> 18992 elements -> 37984 B, double-buffered 76 KB against 64 KB of L1. Both the
    tracked gen_gemma_decode.py (vocab//8 = 32768) and a naive port hit this.

    Returns (tile_size_input, tile_size_output).
    """
    cols = COLS if cols is None else cols
    per_col = M // cols
    # m_input must shrink too: the A tile is m_input x K, so at K=3072 (the FFN down projection)
    # A+B double-buffered already exceed L1 at m_input=4 and leave the C tile nothing. Search
    # m_input downward and take the first that admits any legal C tile.
    for cand_tsi in ([tsi] if tsi is not None else (4, 2, 1)):
        if per_col % cand_tsi:
            continue
        budget = L1_BYTES - L1_RESERVE - 2 * (cand_tsi * K * 2) - 2 * (K * 2)
        if budget <= 0:
            continue
        cap = budget // 4                  # C is double-buffered, 2 bytes per element
        best = max((d for d in range(cand_tsi, per_col + 1, cand_tsi)
                    if per_col % d == 0 and d <= cap), default=0)
        if best:
            return cand_tsi, best
    raise ValueError(f"GEMV M={M} K={K}: no (tile_size_input, tile_size_output) fits L1 "
                     f"({L1_BYTES} B) with M//cols={per_col}")


def gemv(M, K, ctx, **kw):
    """GEMV tiled as large as both the design asserts AND L1 allow."""
    tsi, tso = gemv_tile_output(M, K)
    return GEMV(M=M, K=K, num_aie_columns=COLS, tile_size_input=tsi,
                tile_size_output=tso, context=ctx, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS), help="model spec name")
    ap.add_argument("--weights", required=True, help="dir of dumped .npy weights (see dump_llm_weights.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (bring-up)")
    ap.add_argument("--max-seq", type=int, default=2048, help="KV-cache padded capacity S")
    a = ap.parse_args()

    sp = SPECS[a.spec]
    sp.check(cols=COLS, tsi=TSI)
    sp.check_seq(a.max_seq)
    NL = a.layers if a.layers is not None else sp.n_layers
    S = a.max_seq
    D, FF, HD = sp.d_model, sp.ffn, sp.head_dim
    Hq, Hkv, QD, KVD, VOCAB = sp.n_q_heads, sp.n_kv_heads, sp.q_dim, sp.kv_dim, sp.vocab
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)

    def npy(name):
        return np.load(os.path.join(a.weights, f"{name}.npy")).astype(np.float32)

    def load_norm(name):
        # Gemma-3 stores RMSNorm gain as w with the kernel computing x_hat*(1+w); Qwen3 stores it
        # already absolute. IRON's weighted RMSNorm always does x_hat*w', so Gemma folds the +1 here.
        w = npy(name)
        return bf16(1.0 + w) if sp.norm_gain == "one_plus_w" else bf16(w)

    ctx = AIEContext()

    # ---- op vocabulary: created ONCE, reused across every layer (same dims per layer) ----
    op_norm = RMSNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D,
                      weighted=True, epsilon=sp.eps, context=ctx)
    op_qk_norm = RMSNorm(size=HD, num_aie_columns=1, num_channels=1, tile_size=HD,
                         weighted=True, epsilon=sp.eps, context=ctx) if sp.qk_norm else None
    op_q = gemv(QD, D, ctx)
    op_kv = gemv(KVD, D, ctx)
    op_o = gemv(D, QD, ctx)
    op_rope_q = RoPE(rows=Hq, cols=HD, angle_rows=1, context=ctx)
    op_rope_k = RoPE(rows=Hkv, cols=HD, angle_rows=1, context=ctx)
    # KV append: deep-C scratchpad offset "kv_off" (element units = n_past*HD), constant ELF.
    sc = dict(input_sizes=(Hkv, HD), input_strides=(HD, 1), input_offset=0,
              output_sizes=(1, Hkv, HD), output_strides=(0, S * HD, 1), output_offset=0,
              input_buffer_size=Hkv * HD, output_buffer_size=Hkv * S * HD, num_aie_channels=1)
    op_sck = StridedCopy(**sc, output_offset_parameter="kv_off", context=ctx)
    op_scv = StridedCopy(**sc, output_offset_parameter="kv_off", context=ctx)
    # GQA broadcast. Correctness-first; the byte-free form is a batch-stride-0 GEMV read of the kv
    # head (0 ops, 0 bytes) -- at Hq=16 x 28 layers this Repeat plus the V transpose are 41% of the
    # per-token DDR budget, so it is the first optimisation after parity, not an afterthought.
    op_rep_k = Repeat(rows=Hkv, cols=S * HD, repeat=sp.gqa_group, transfer_size=HD, context=ctx)
    op_rep_v = Repeat(rows=Hkv, cols=S * HD, repeat=sp.gqa_group, transfer_size=HD, context=ctx)
    op_scores = gemv(S, HD, ctx, num_batches=Hq)
    op_scale = ElementwiseMul(size=Hq * S, tile_size=S // COLS, num_aie_columns=COLS, context=ctx)
    op_softmax = Softmax(rows=Hq, cols=S, num_aie_columns=1, num_channels=1, rtp_vector_size=S,
                         vector_size_parameter="sm_mask", context=ctx)
    op_trv = Transpose(M=S, N=HD, num_aie_columns=2, num_channels=1, m=256, n=32, s=8, context=ctx)
    op_ctx = gemv(HD, S, ctx, num_batches=Hq)
    op_gate = gemv(FF, D, ctx)
    op_up = gemv(FF, D, ctx)
    if sp.act == "silu":
        op_act = SiLU(size=FF, num_aie_columns=COLS, tile_size=FF // COLS, context=ctx)
    else:
        op_act = GELU(size=FF, num_aie_columns=COLS, num_channels=1, tile_size=FF // COLS, context=ctx)
    op_mul_ffn = ElementwiseMul(size=FF, tile_size=FF // COLS, num_aie_columns=COLS, context=ctx)
    op_down = gemv(D, FF, ctx)
    op_add = ElementwiseAdd(size=D, tile_size=D // COLS, num_aie_columns=COLS, context=ctx)
    op_head = gemv(VOCAB, D, ctx)

    weights, bufsz, cache_names, rl = {}, {}, [], []
    cur = "x"

    for l in range(NL):
        p = f"L{l}_"
        nm = sp.norm_weight_names(l)
        for key, tensor in nm.items():
            weights[p + key] = load_norm(tensor)
        for key, tensor in (("Wq", "self_attn.q_proj"), ("Wk", "self_attn.k_proj"),
                            ("Wv", "self_attn.v_proj"), ("Wo", "self_attn.o_proj"),
                            ("Wg", "mlp.gate_proj"), ("Wu", "mlp.up_proj"), ("Wd", "mlp.down_proj")):
            weights[p + key] = bf16(npy(f"model.layers.{l}.{tensor}.weight")).reshape(-1)
        weights[p + "kc"] = np.zeros(Hkv * S * HD, BF16)
        weights[p + "vc"] = np.zeros(Hkv * S * HD, BF16)
        cache_names += [p + "kc", p + "vc"]
        ang = "rope_global" if sp.is_global(l) else "rope_local"

        bufsz.update({
            p + "q": QD * 2, p + "k": KVD * 2, p + "v": KVD * 2,
            p + "kc": Hkv * S * HD * 2, p + "vc": Hkv * S * HD * 2,
            p + "kr": Hq * S * HD * 2, p + "vr": Hq * S * HD * 2, p + "vt": Hq * S * HD * 2,
            p + "sc": Hq * S * 2, p + "sw": Hq * S * 2,
            p + "cx": QD * 2, p + "a": D * 2,
            p + "g": FF * 2, p + "u": FF * 2, p + "gh": FF * 2, p + "d": D * 2,
            p + "hn": D * 2, p + "hf": D * 2,
        })
        nxt = f"x{l+1}"
        qk = []
        if sp.qk_norm:
            qk = [*[(op_qk_norm, f"{p}q[{h*HD*2}:{(h+1)*HD*2}]", p + "n_qn",
                     f"{p}q[{h*HD*2}:{(h+1)*HD*2}]") for h in range(Hq)],
                  *[(op_qk_norm, f"{p}k[{h*HD*2}:{(h+1)*HD*2}]", p + "n_kn",
                     f"{p}k[{h*HD*2}:{(h+1)*HD*2}]") for h in range(Hkv)]]
        rl += [
            (op_norm, cur, p + "n_in", p + "hn"),
            (op_q, p + "Wq", p + "hn", p + "q"),
            (op_kv, p + "Wk", p + "hn", p + "k"),
            (op_kv, p + "Wv", p + "hn", p + "v"),
            *qk,
            (op_rope_q, p + "q", ang, p + "q"),
            (op_rope_k, p + "k", ang, p + "k"),
            (op_sck, p + "k", p + "kc"),
            (op_scv, p + "v", p + "vc"),
            (op_rep_k, p + "kc", p + "kr"),
            (op_rep_v, p + "vc", p + "vr"),
            (op_scores, p + "kr", p + "q", p + "sc"),
            (op_scale, p + "sc", "attn_scale", p + "sc"),
            (op_softmax, p + "sc", p + "sw"),
            *[(op_trv, f"{p}vr[{h*S*HD*2}:{(h+1)*S*HD*2}]", f"{p}vt[{h*S*HD*2}:{(h+1)*S*HD*2}]")
              for h in range(Hq)],
            (op_ctx, p + "vt", p + "sw", p + "cx"),
            (op_o, p + "Wo", p + "cx", p + "a"),
        ]
        if sp.sandwich_norms:
            rl.append((op_norm, p + "a", p + "n_pa", p + "a"))
        rl += [
            (op_add, cur, p + "a", p + "x1"),
            (op_norm, p + "x1", p + "n_pf", p + "hf"),
            (op_gate, p + "Wg", p + "hf", p + "g"),
            (op_up, p + "Wu", p + "hf", p + "u"),
            (op_act, p + "g", p + "g"),
            (op_mul_ffn, p + "g", p + "u", p + "gh"),
            (op_down, p + "Wd", p + "gh", p + "d"),
        ]
        if sp.sandwich_norms:
            rl.append((op_norm, p + "d", p + "n_pff", p + "d"))
        rl.append((op_add, p + "x1", p + "d", nxt))
        bufsz[p + "x1"] = D * 2
        cur = nxt

    weights["n_final"] = load_norm("model.norm.weight")
    weights["W_head"] = bf16(npy("model.embed_tokens.weight")).reshape(-1)   # tied
    weights["attn_scale"] = np.full(Hq * S, sp.attn_scale, BF16)
    rl += [(op_norm, cur, "n_final", "xf"), (op_head, "W_head", "xf", "logits")]
    bufsz["xf"] = D * 2
    bufsz["logits"] = VOCAB * 2

    if os.environ.get("DUMP_OPS"):
        from collections import Counter
        c = Counter(type(e[0]).__name__ for e in rl)
        print(f"# {sp.name}: runlist {len(rl)} entries over NL={NL} "
              f"({(len(rl)-2)//NL}/layer + 2 tail)")
        for k, v in c.most_common():
            print(f"  {k:16} {v:5}")
        return

    inputs = ["x", "rope_global"] + (["rope_local"] if sp.rope_theta_local is not None else [])
    fused = FusedMLIROperator(f"{sp.name.replace('-','_').replace('.','_')}_decode", rl,
                              input_args=inputs, output_args=["logits"],
                              buffer_sizes=bufsz, context=ctx)
    fused.compile()
    elf = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    wnames = list(weights.keys())
    lay = {n: fused.get_layout_for_buffer(n) for n in ["x", "logits"] + wnames}

    import glob
    import shutil
    # The project dir suffix moved from `.mlir.prj` to `.mlir.d`; match params.txt wherever aiecc
    # put it under the build tree rather than pinning a suffix that has already changed once.
    _pp = sorted(glob.glob("**/params.txt", recursive=True), key=os.path.getmtime)
    scratchpad_params = {}
    if _pp:
        shutil.copy(_pp[-1], os.path.join(a.out, "params.txt"))
        for line in open(_pp[-1]).read().splitlines()[1:]:
            if line.strip():
                n_, idx, ty, kind = line.split()
                scratchpad_params[n_] = {"byte_offset": int(idx) * 4, "kind": kind, "dtype": ty}

    bdir = os.path.join(a.out, "buffers")
    for n_, arr in weights.items():
        open(os.path.join(bdir, f"{n_}.bin"), "wb").write(np.asarray(arr, BF16).tobytes())
    open(os.path.join(a.out, "decode.elf"), "wb").write(elf)

    meta = {
        "spec": sp.name, "elf": "decode.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": inputs, "weights": wnames, "output": "logits",
        "scratchpad": {"params": scratchpad_params, "kv_param": "kv_off", "mask_param": "sm_mask",
                       "head_dim": HD, "kv_heads": Hkv},
        "dims": {"layers": NL, "d_model": D, "q_heads": Hq, "kv_heads": Hkv, "head_dim": HD,
                 "ffn": FF, "vocab": VOCAB, "S": S,
                 "sliding_window": sp.sliding_window, "sw_pattern": sp.sw_pattern},
        # Per-token host protocol (the ELF is constant; only these change):
        #   x        = embed[token], scaled by sqrt(d_model) iff embed_scale == "sqrt_d_model"
        #   rope_*   = precomputed [S,HD] angle tables; the row for n_past is used
        #   kv_off   = n_past * head_dim   (addr kind, element units, raw)
        #   sm_mask  = n_past + 1          (core kind, causal width; host writes it <<2)
        "host_protocol": {"embed_scale": sp.embed_scale, "attn_scale": float(sp.attn_scale),
                          "act": sp.act, "norm_gain": sp.norm_gain, "eps": sp.eps,
                          "rope_theta_global": sp.rope_theta_global,
                          "rope_theta_local": sp.rope_theta_local},
        "layer_types": ["global" if sp.is_global(l) else "sliding" for l in range(NL)],
        "cache_buffers": cache_names,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\nwrote {NL}-layer {sp.name} decode ELF ({len(elf)}B, scratch {scr/1e6:.1f}MB) to {a.out}")


if __name__ == "__main__":
    main()
