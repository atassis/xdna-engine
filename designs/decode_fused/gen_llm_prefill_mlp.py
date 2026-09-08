#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One qwen3 MLP block at BATCH M, as a fused full ELF -- the first batched-prefill brick on this rail.

The decode path runs this block at M=1 through GEMV. This runs the same arithmetic at M>1 through
GEMM, in the TOKEN-MAJOR orientation the whole prefill design is built on:

    C[M, Nout] = A[M, K] @ B[K, Nout]        A = the [M, D] activation tile
                                             B = the weight, b_col_maj, read in its STORED [Nout, K]

`b_col_maj=True` is load-bearing and not a detail. Our weights are stored `[Nout, K]` because that
is what decode's GEMV wants, and prefill must read THE SAME BYTES: the whole architecture rests on
prefill and decode sharing one arena, so a transposed second copy would double a 1.110 GiB weight
arena and delete the reason for sharing it. `iron.operators.swiglu_prefill` -- upstream's ready-made
version of this block -- does NOT set it, which is why this file exists rather than calling that.

Also demonstrated here, because both are claims the architecture doc makes and neither had been run:
  * RMSNorm at M rows costs nothing new -- `num_aie_columns` already splits ROWS, so the same
    operator normalises 1 row or 256, and `tile_size` stays the normalised width D.
  * the batch is legal at M=256 under `M % (tile_m*4)`, with every N satisfying
    `N % (tile_n*num_aie_columns)`. Both are asserted here, at the point the shape is picked (K007).

Emits the meta.json/buffers layout `rust/npu-probes/src/bin/fused_elf_probe.rs` consumes, so the
device run needs no new host code.

Run inside the fork IRON env. AIE_DEVICE=npu2 keeps the build off the device lock.
"""
import argparse
import json
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import SPECS, gemm_l1_bytes  # noqa: E402
from gemm_tile_registry import registry  # noqa: E402
from prefill_ref import f32, gate_block, mlp_block  # noqa: E402

import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports
from iron.common import AIEContext  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemm.op import GEMM  # noqa: E402
from iron.operators.rms_norm.op import RMSNorm  # noqa: E402
from iron.operators.silu.op import SiLU  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = int(os.environ.get("PREFILL_COLS", "8"))
# The GEMM tiling is no longer a constant here -- `gemm_tile_registry` holds a measured triple per
# SHAPE, and a shape that has never been swept raises rather than falling back to 64/64/64. COLS
# still splits the non-GEMM ops (RMSNorm/SiLU/mul).


def bf16(a):
    return np.asarray(a).astype(BF16)


def pick_tiles(batch, shapes):
    """The registry's tiling for every GEMM this block builds, checked where the shape is picked.

    K007/K008 are still enforced here, but not by a second copy of the rules: `Registry.lookup`
    runs `llm_decode_spec.gemm_tiling_rejection` -- the same function `check_prefill_projections`
    raises from -- so the modulus, the mm.cc `static_assert`s, the 64 KB L1 and the 512 KB MemTile
    are all named before an operator is constructed. A shape absent from the registry raises with
    the sweep command that fills it; nothing falls back to a default triple.
    """
    reg = registry()
    out = {}
    for label, K, N in shapes:
        ch = reg.lookup(batch, K, N, b_col_maj=True, label=label)
        out[label] = ch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b", choices=sorted(SPECS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    sp = SPECS[a.spec]
    D, FF, M = sp.d_model, sp.ffn, a.batch
    ch = pick_tiles(M, [("gate_up", D, FF), ("down", FF, D)])
    l1 = gemm_l1_bytes(ch["gate_up"].tile_m, ch["gate_up"].tile_k, ch["gate_up"].tile_n)
    print(f"[shape] {sp.name} L{a.layer}  M={M}  D={D}  FF={FF}  "
          f"gate_up={ch['gate_up']}  down={ch['down']}  L1={l1}B ({100*l1/65536:.0f}%)")

    if os.environ.get("AIE_DEVICE"):
        import aie.utils as _aie_utils
        from aie.iron.device import from_name as _from_name
        _aie_utils.set_current_device(_from_name(os.environ["AIE_DEVICE"], n_cols=None))

    def npy(name):
        return np.load(os.path.join(a.weights, f"{name}.npy")).astype(np.float32)

    p = f"model.layers.{a.layer}."
    Wg = npy(p + "mlp.gate_proj.weight")     # [FF, D] as stored -- decode's GEMV layout
    Wu = npy(p + "mlp.up_proj.weight")       # [FF, D]
    Wd = npy(p + "mlp.down_proj.weight")     # [D, FF]
    nw = npy(p + "post_attention_layernorm.weight")   # qwen3's pre-FFN norm
    assert Wg.shape == (FF, D) and Wd.shape == (D, FF), f"unexpected {Wg.shape} {Wd.shape}"

    ctx = AIEContext()
    # RMSNorm splits ROWS across columns and normalises `tile_size` wide, so the SAME operator that
    # does one row at M=1 does M rows here -- the batch is a size, not a new op-type.
    op_norm = RMSNorm(size=M * D, num_aie_columns=COLS, num_channels=1, tile_size=D,
                      weighted=True, epsilon=sp.eps, context=ctx)
    # shared by gate and up: same shape, one design, one registry entry
    op_gu = GEMM(M=M, K=D, N=FF, b_col_maj=True, context=ctx, **ch["gate_up"].gemm_kwargs)
    op_down = GEMM(M=M, K=FF, N=D, b_col_maj=True, context=ctx, **ch["down"].gemm_kwargs)
    op_silu = SiLU(size=M * FF, num_aie_columns=COLS, tile_size=FF // COLS, context=ctx)
    op_mul = ElementwiseMul(size=M * FF, num_aie_columns=COLS, tile_size=FF // COLS, context=ctx)

    rl = [(op_norm, "x", "n_pf", "hf"),
          (op_gu, "hf", "Wg", "g"),
          (op_gu, "hf", "Wu", "u"),
          (op_silu, "g", "gs"),
          (op_mul, "gs", "u", "gh"),
          (op_down, "gh", "Wd", "out")]

    rng = np.random.default_rng(7)
    X = bf16(rng.standard_normal((M, D)).astype(np.float32) * 0.5)
    weights = {"n_pf": bf16(nw), "Wg": bf16(Wg).reshape(-1),
               "Wu": bf16(Wu).reshape(-1), "Wd": bf16(Wd).reshape(-1)}

    tile_sig = "_".join(f"{k}{c.tile_m}x{c.tile_k}x{c.tile_n}c{c.cols}"
                        for k, c in sorted(ch.items()))
    fused = OperatorSequence(f"prefill_mlp_{sp.name}_m{M}_c{COLS}_l{a.layer}_{tile_sig}", rl,
                             input_args=["x"], output_args=["out"], context=ctx,
                             share_designs=True)
    fused.compile()

    # ---- Two references off ONE dataflow (prefill_ref.mlp_block) ----
    # bf16: rounded where the device rounds, for the probe's rel-L2. f32: the SAME bf16 inputs with
    # no intermediate narrowed, which is what Tier 1 gates against -- a bf16 reference would agree
    # with the device on exactly the rounding the gate exists to measure.
    ref_args = (X, bf16(nw), bf16(Wg), bf16(Wu), bf16(Wd), sp.eps, sp.act)
    out = mlp_block(*ref_args, bf16)
    ref32 = mlp_block(*ref_args, f32)

    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    bdir = os.path.join(a.out, "buffers")
    open(os.path.join(bdir, "x.bin"), "wb").write(X.tobytes())
    for n_, arr in weights.items():
        open(os.path.join(bdir, f"{n_}.bin"), "wb").write(np.asarray(arr, BF16).tobytes())
    open(os.path.join(bdir, "out.bin"), "wb").write(out.tobytes())
    elf = load_elf(fused).view(np.uint8).tobytes()
    open(os.path.join(a.out, "prefill_mlp.elf"), "wb").write(elf)
    in_sz, out_sz, scr = fused.buffer_sizes
    names = ["x", "out", *weights]
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    json.dump({
        "elf": "prefill_mlp.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"], "weights": list(weights), "output": "out",
        "dims": {"spec": sp.name, "layer": a.layer, "M": M, "d_model": D, "ffn": FF,
                 "tiles": {k: {"tile": [c.tile_m, c.tile_k, c.tile_n], "cols": c.cols,
                               "source": c.source} for k, c in ch.items()},
                 "cols": COLS},
        # DDR bytes this block moves per dispatch, so a device timing converts to GB/s without the
        # caller re-deriving it. Weights are M-independent; activations are not. Splitting the two
        # is what separates "transport-bound" from "compute-bound" on an M sweep.
        "bytes": {"weights": int(3 * D * FF * 2),
                  "activations": int(M * (5 * D + 8 * FF) * 2)},
        # MACs the block actually issues, so a device timing converts straight to MAC/s without
        # the caller re-deriving it -- this is the number that decides how far batching pays.
        "macs": int(2 * M * D * FF + M * FF * D),
        "gate": gate_block(a.out, {"out": np.asarray(ref32, np.float32).reshape(M, D)},
                           {"out": np.asarray(out, np.float32).reshape(M, D)}),
    }, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[ok] wrote {len(elf)}B ELF, scratch {scr/1e6:.1f} MB, "
          f"{2*M*D*FF + M*FF*D:,} MACs/dispatch -> {a.out}")


if __name__ == "__main__":
    main()
