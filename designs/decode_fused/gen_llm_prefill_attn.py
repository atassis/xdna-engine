#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The attention half of a batched prefill layer: scores + softmax + context, at batch M.

The MLP block (gen_llm_prefill_mlp.py) exercises the projections. This exercises the term it does
not: the one whose traffic grows with the CONTEXT rather than with the batch, and which the
architecture doc names as the next thing to measure rather than model.

Per q head h (kv head h // gqa_group), over a full window of S positions:

    sc[h] = q[h]  [M, HD] @ kc[kv]^T  [HD, S]     GEMM, b_col_maj (kc is stored [S, HD])
    sw    = softmax(sc)                            ONE invocation over all Hq*M rows at once
    cx[h] = sw[h] [M, S]  @ vc[kv]    [S, HD]      GEMM, plain (vc is stored [S, HD])

Two things this pins down that were open in the design:

* **The softmax is ONE run, not Hq runs.** Its `rows` axis is just "independent rows to normalise",
  so stacking every head's [M, S] score block into one [Hq*M, S] buffer lets a single invocation
  cover the whole layer. That is only expressible because each head's GEMM writes a contiguous
  slice of the same buffer.
* **q must be HEAD-MAJOR, `[Hq, M, HD]`, and the projections produce it TOKEN-major, `[M, QD]`.**
  A head's columns are not contiguous in the token-major tile contract, and the runlist's slice
  syntax addresses byte ranges. At M=1 the question does not arise -- q is [QD] and every head IS
  contiguous. This is a real seam between the two halves of a prefill layer, and it is taken here
  as an INPUT in head-major order rather than papered over: closing it needs either a per-head
  projection or an explicit rearrange, and that choice should be made with this measurement in hand.

NON-CAUSAL, deliberately, and scoped: a full window with no mask isolates the batched attention
dataflow from the causal-mask brick the architecture still owes. `gen_self_attn_batched.py` scoped
its own v1 the same way and for the same reason.

kc/vc are random bf16 here, not projected from real weights: this measures the dataflow and its
cost, and the arithmetic gate is against a CPU golden of the same random tensors.
"""
import argparse
import json
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import SPECS  # noqa: E402

import newstack_compat  # noqa: F401,E402
from iron.common import AIEContext  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemm.op import GEMM  # noqa: E402
from iron.operators.softmax.op import Softmax  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = int(os.environ.get("PREFILL_COLS", "8"))
TILE_M = TILE_K = 64


def bf16(a):
    return np.asarray(a).astype(BF16)


def pick_tile_n(Nout, label):
    """Largest tile_n with `Nout % (tile_n*COLS) == 0` and `tile_n % 16 == 0` (mm.cc's real rule).

    K007: the shape is picked HERE, so the modulus is checked HERE, naming the offending number --
    rather than surfacing as a ValueError from the operator or, worse, a C++ static_assert.
    """
    for tn in (64, 48, 32, 16):
        if tn % 16 == 0 and Nout % (tn * COLS) == 0:
            return tn
    raise ValueError(f"{label}: Nout={Nout} admits no tile_n in (64,48,32,16) with "
                     f"Nout % (tile_n*{COLS}) == 0 and tile_n % 16 == 0. "
                     f"Divisors of Nout that would work at {COLS} columns: "
                     f"{[t for t in range(16, 65, 16) if Nout % (t*COLS) == 0] or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b", choices=sorted(SPECS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=2048, help="KV window S (the COMPILED width)")
    a = ap.parse_args()

    sp = SPECS[a.spec]
    M, S = a.batch, a.seq
    Hq, Hkv, HD = sp.n_q_heads, sp.n_kv_heads, sp.head_dim
    grp = Hq // Hkv
    if M % (TILE_M * 4):
        raise ValueError(f"batch={M} is not a multiple of tile_m*n_aie_rows={TILE_M*4}")
    if S % TILE_K:
        raise ValueError(f"seq={S} is not a multiple of tile_k={TILE_K} (the ctx GEMM's K)")
    tn_sc, tn_cx = pick_tile_n(S, "scores"), pick_tile_n(HD, "ctx")
    print(f"[shape] {sp.name} M={M} S={S} Hq={Hq} Hkv={Hkv} HD={HD} gqa={grp}  "
          f"tile_n scores={tn_sc} ctx={tn_cx} cols={COLS}")

    if os.environ.get("AIE_DEVICE"):
        import aie.utils as _aie_utils
        from aie.iron.device import from_name as _from_name
        _aie_utils.set_current_device(_from_name(os.environ["AIE_DEVICE"], n_cols=None))

    ctx = AIEContext()
    op_sc = GEMM(M=M, K=HD, N=S, tile_m=TILE_M, tile_k=TILE_K, tile_n=tn_sc,
                 num_aie_columns=COLS, b_col_maj=True, context=ctx)
    op_cx = GEMM(M=M, K=S, N=HD, tile_m=TILE_M, tile_k=TILE_K, tile_n=tn_cx,
                 num_aie_columns=COLS, b_col_maj=False, context=ctx)
    # ONE softmax over every head's rows at once -- see the module docstring.
    op_sm = Softmax(rows=Hq * M, cols=S, num_aie_columns=COLS, num_channels=1, context=ctx)

    rl, bufsz = [], {}
    bufsz["q"] = Hq * M * HD * 2
    bufsz["kc"] = Hkv * S * HD * 2
    bufsz["vc"] = Hkv * S * HD * 2
    bufsz["sc"] = Hq * M * S * 2
    bufsz["sw"] = Hq * M * S * 2
    bufsz["cx"] = Hq * M * HD * 2
    for h in range(Hq):
        kv = h // grp
        qs = f"q[{h*M*HD*2}:{(h+1)*M*HD*2}]"
        ks = f"kc[{kv*S*HD*2}:{(kv+1)*S*HD*2}]"
        ss = f"sc[{h*M*S*2}:{(h+1)*M*S*2}]"
        rl.append((op_sc, qs, ks, ss))
    rl.append((op_sm, "sc", "sw"))
    for h in range(Hq):
        kv = h // grp
        ws = f"sw[{h*M*S*2}:{(h+1)*M*S*2}]"
        vs = f"vc[{kv*S*HD*2}:{(kv+1)*S*HD*2}]"
        cs = f"cx[{h*M*HD*2}:{(h+1)*M*HD*2}]"
        rl.append((op_cx, ws, vs, cs))

    rng = np.random.default_rng(13)
    Q = bf16(rng.standard_normal((Hq, M, HD)).astype(np.float32) * (HD ** -0.5))
    KC = bf16(rng.standard_normal((Hkv, S, HD)).astype(np.float32))
    VC = bf16(rng.standard_normal((Hkv, S, HD)).astype(np.float32))

    fused = OperatorSequence(f"prefill_attn_{sp.name}_m{M}_s{S}_c{COLS}", rl,
                             input_args=["q"], output_args=["cx"],
                             buffer_sizes=bufsz, context=ctx, share_designs=True)
    fused.compile()

    # ---- CPU golden, same bf16 dataflow ----
    out = np.zeros((Hq, M, HD), np.float32)
    for h in range(Hq):
        kv = h // grp
        s = np.asarray(Q[h], np.float32) @ np.asarray(KC[kv], np.float32).T   # [M, S]
        s = np.asarray(bf16(s), np.float32)
        e = np.exp(s - s.max(-1, keepdims=True))
        p = np.asarray(bf16(e / e.sum(-1, keepdims=True)), np.float32)
        out[h] = np.asarray(bf16(p @ np.asarray(VC[kv], np.float32)), np.float32)

    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    b = os.path.join(a.out, "buffers")
    open(os.path.join(b, "q.bin"), "wb").write(Q.tobytes())
    open(os.path.join(b, "kc.bin"), "wb").write(KC.reshape(-1).tobytes())
    open(os.path.join(b, "vc.bin"), "wb").write(VC.reshape(-1).tobytes())
    open(os.path.join(b, "cx.bin"), "wb").write(bf16(out).tobytes())
    elf = load_elf(fused).view(np.uint8).tobytes()
    open(os.path.join(a.out, "prefill_attn.elf"), "wb").write(elf)
    in_sz, out_sz, scr = fused.buffer_sizes
    lay = {n: fused.get_layout_for_buffer(n) for n in ("q", "cx", "kc", "vc")}
    json.dump({
        "elf": "prefill_attn.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["q"], "weights": ["kc", "vc"], "output": "cx",
        "dims": {"spec": sp.name, "M": M, "S": S, "q_heads": Hq, "kv_heads": Hkv, "head_dim": HD,
                 "tile_n_scores": tn_sc, "tile_n_ctx": tn_cx, "cols": COLS, "causal": False},
        "macs": int(2 * Hq * M * S * HD),
    }, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[ok] wrote {len(elf)}B ELF, scratch {scr/1e6:.1f} MB, "
          f"{2*Hq*M*S*HD:,} MACs/dispatch, {len(rl)} runlist entries -> {a.out}")


if __name__ == "__main__":
    main()
