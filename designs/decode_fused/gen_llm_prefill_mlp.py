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
from llm_decode_spec import SPECS  # noqa: E402

import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports
from iron.common import AIEContext  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemm.op import GEMM  # noqa: E402
from iron.operators.rms_norm.op import RMSNorm  # noqa: E402
from iron.operators.silu.op import SiLU  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = int(os.environ.get("PREFILL_COLS", "8"))
TILE_M = TILE_K = TILE_N = 64


def bf16(a):
    return np.asarray(a).astype(BF16)


def check_batch(batch, shapes):
    """K007: assert every GEMM modulus HERE, where the shape is picked, naming the offending number.

    `iron/operators/gemm/op.py` raises for M/K/N, but only after the caller has committed; and its
    tile checks are WEAKER than the kernel's (`op.py` tests `tile_m >= 8` while
    `aie_kernels/aie2p/mm.cc` static_asserts `m % (2*r) == 0`, r=8 on the bfp16-emulation path that
    is IRON's GEMM default). So the kernel's rule is the one checked here.
    """
    if batch % (TILE_M * 4):
        raise ValueError(f"batch={batch} is not a multiple of tile_m*n_aie_rows={TILE_M * 4} "
                         f"(n_aie_rows is hardcoded 4 in iron/operators/gemm/design.py)")
    if TILE_M % 16 or TILE_N % 16:
        raise ValueError(f"tile_m={TILE_M}/tile_n={TILE_N}: mm.cc static_asserts m%(2*r)==0 and "
                         f"n%(2*t)==0 with r=t=8 on the bfp16 path; op.py's own >=8 check is weaker")
    if TILE_K % 8:
        raise ValueError(f"tile_k={TILE_K}: mm.cc static_asserts k%s==0 with s=8")
    for label, K, N in shapes:
        if K % TILE_K:
            raise ValueError(f"{label}: K={K} not a multiple of tile_k={TILE_K}")
        if N % (TILE_N * COLS):
            raise ValueError(f"{label}: N={N} not a multiple of tile_n*num_aie_columns="
                             f"{TILE_N * COLS}; the widest legal column count is "
                             f"{max((c for c in (8, 4, 2, 1) if N % (TILE_N * c) == 0), default=0)}")
    # K008: nothing in the toolchain checks GEMM's L1 occupancy. A, B and C are all double-buffered
    # bf16 tiles, plus gemm/design.py's stack_size=0xD00 per worker.
    l1 = 4 * (TILE_M * TILE_K + TILE_K * TILE_N + TILE_M * TILE_N) + 0xD00
    if l1 > 65536:
        raise ValueError(f"tile ({TILE_M},{TILE_K},{TILE_N}) needs {l1} B of L1 against 65536; "
                         f"aiecc would report this as \"'aie.tile' op Basic sequential allocation "
                         f"also failed\", naming a tile and not a size")
    return l1


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
    l1 = check_batch(M, [("gate/up", D, FF), ("down", FF, D)])
    print(f"[shape] {sp.name} L{a.layer}  M={M}  D={D}  FF={FF}  "
          f"tiles=({TILE_M},{TILE_K},{TILE_N}) cols={COLS}  L1={l1}B ({100*l1/65536:.0f}%)")

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
    gemm_kw = dict(tile_m=TILE_M, tile_k=TILE_K, tile_n=TILE_N, num_aie_columns=COLS,
                   b_col_maj=True, context=ctx)
    op_gu = GEMM(M=M, K=D, N=FF, **gemm_kw)      # shared by gate and up: same shape, one design
    op_down = GEMM(M=M, K=FF, N=D, **gemm_kw)
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

    fused = OperatorSequence(f"prefill_mlp_{sp.name}_m{M}_c{COLS}_l{a.layer}", rl,
                             input_args=["x"], output_args=["out"], context=ctx,
                             share_designs=True)
    fused.compile()

    # ---- CPU golden: the same bf16 dataflow, rounded where the device rounds ----
    def rms(v, w):
        f = np.asarray(v, np.float32)
        s = f / np.sqrt((f * f).mean(-1, keepdims=True) + sp.eps)
        return bf16(s * np.asarray(w, np.float32))
    hf = rms(X, bf16(nw))
    g = bf16(np.asarray(hf, np.float32) @ np.asarray(bf16(Wg), np.float32).T)
    u = bf16(np.asarray(hf, np.float32) @ np.asarray(bf16(Wu), np.float32).T)
    gf = np.asarray(g, np.float32)
    gs = bf16(gf / (1.0 + np.exp(-gf)))
    gh = bf16(np.asarray(gs, np.float32) * np.asarray(u, np.float32))
    out = bf16(np.asarray(gh, np.float32) @ np.asarray(bf16(Wd), np.float32).T)

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
                 "tile": [TILE_M, TILE_K, TILE_N], "cols": COLS},
        # DDR bytes this block moves per dispatch, so a device timing converts to GB/s without the
        # caller re-deriving it. Weights are M-independent; activations are not. Splitting the two
        # is what separates "transport-bound" from "compute-bound" on an M sweep.
        "bytes": {"weights": int(3 * D * FF * 2),
                  "activations": int(M * (5 * D + 8 * FF) * 2)},
        # MACs the block actually issues, so a device timing converts straight to MAC/s without
        # the caller re-deriving it -- this is the number that decides how far batching pays.
        "macs": int(2 * M * D * FF + M * FF * D),
    }, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[ok] wrote {len(elf)}B ELF, scratch {scr/1e6:.1f} MB, "
          f"{2*M*D*FF + M*FF*D:,} MACs/dispatch -> {a.out}")


if __name__ == "__main__":
    main()
