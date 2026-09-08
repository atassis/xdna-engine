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
from gemm_tile_registry import registry  # noqa: E402
from prefill_ref import attn_block, f32, gate_block  # noqa: E402

import newstack_compat  # noqa: F401,E402
from iron.common import AIEContext  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemm.op import GEMM  # noqa: E402
from iron.operators.softmax.op import Softmax  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = int(os.environ.get("PREFILL_COLS", "8"))
# No TILE_M/TILE_K constant and no local pick_tile_n: `gemm_tile_registry` holds a measured tiling
# per SHAPE, and `scores` (K=head_dim, N=S) and `ctx` (K=S, N=head_dim) are two very different
# shapes -- which is exactly why this file used to need a hand-written tile_n for each.


def bf16(a):
    return np.asarray(a).astype(BF16)


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
    # K007: `Registry.lookup` runs the same `gemm_tiling_rejection` the spec checks raise from, so
    # the batch modulus, mm.cc's static_asserts, the K/N divisibility and both capacity budgets are
    # all named here, at the point the shape is picked. An unswept shape raises with the sweep
    # command; nothing falls back to a guessed triple.
    reg = registry()
    ch_sc = reg.lookup(M, HD, S, b_col_maj=True, label="scores")
    ch_cx = reg.lookup(M, S, HD, b_col_maj=False, label="ctx")
    tn_sc, tn_cx = ch_sc.tile_n, ch_cx.tile_n
    print(f"[shape] {sp.name} M={M} S={S} Hq={Hq} Hkv={Hkv} HD={HD} gqa={grp}  "
          f"scores={ch_sc}  ctx={ch_cx}  cols={COLS}")

    if os.environ.get("AIE_DEVICE"):
        import aie.utils as _aie_utils
        from aie.iron.device import from_name as _from_name
        _aie_utils.set_current_device(_from_name(os.environ["AIE_DEVICE"], n_cols=None))

    ctx = AIEContext()
    op_sc = GEMM(M=M, K=HD, N=S, b_col_maj=True, context=ctx, **ch_sc.gemm_kwargs)
    op_cx = GEMM(M=M, K=S, N=HD, b_col_maj=False, context=ctx, **ch_cx.gemm_kwargs)
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

    tile_sig = (f"sc{ch_sc.tile_m}x{ch_sc.tile_k}x{ch_sc.tile_n}c{ch_sc.cols}"
                f"_cx{ch_cx.tile_m}x{ch_cx.tile_k}x{ch_cx.tile_n}c{ch_cx.cols}")
    fused = OperatorSequence(f"prefill_attn_{sp.name}_m{M}_s{S}_c{COLS}_{tile_sig}", rl,
                             input_args=["q"], output_args=["cx"],
                             buffer_sizes=bufsz, context=ctx, share_designs=True)
    fused.compile()

    # ---- Two references off ONE dataflow (prefill_ref.attn_block) ----
    # bf16 for the probe's rel-L2; f32 for Tier 1, same bf16 q/kc/vc with nothing narrowed between.
    out = attn_block(Q, KC, VC, bf16)
    ref32 = attn_block(Q, KC, VC, f32)

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
                 "tile_n_scores": tn_sc, "tile_n_ctx": tn_cx, "cols": COLS, "causal": False,
                 "tiles": {"scores": {"tile": [ch_sc.tile_m, ch_sc.tile_k, ch_sc.tile_n],
                                      "cols": ch_sc.cols, "source": ch_sc.source},
                           "ctx": {"tile": [ch_cx.tile_m, ch_cx.tile_k, ch_cx.tile_n],
                                   "cols": ch_cx.cols, "source": ch_cx.source}}},
        "macs": int(2 * Hq * M * S * HD),
        "gate": gate_block(a.out, {"cx": np.asarray(ref32, np.float32).reshape(Hq, M, HD)},
                           {"cx": np.asarray(out, np.float32).reshape(Hq, M, HD)}),
    }, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[ok] wrote {len(elf)}B ELF, scratch {scr/1e6:.1f} MB, "
          f"{2*Hq*M*S*HD:,} MACs/dispatch, {len(rl)} runlist entries -> {a.out}")


if __name__ == "__main__":
    main()
