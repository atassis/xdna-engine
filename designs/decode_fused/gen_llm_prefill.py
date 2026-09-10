#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A WHOLE batched-prefill layer stack for a decoder LLM, as ONE fused ELF.

gen_llm_prefill_mlp.py and gen_llm_prefill_attn.py are the two halves, each gated on device at
M=256. This composes them into the real thing: for a chunk of M tokens at absolute base position
`base`, N layers of

    h        = RMSNorm_in(x)                       rows=M
    q,k,v    = GEMM(h, Wq/Wk/Wv)                   token-major C[M,Nout] = A[M,K] @ B[K,Nout]
    q,k      = per-head RMSNorm(qk_norm), then RoPE over M angle rows
    kc/vc   += k,v at rows [base, base+M)
    ctx      = attention over the compiled window S
    x       += GEMM(ctx, Wo)
    x       += MLP(RMSNorm(x))

Output is `x` after N layers. There is NO lm-head: prefill's product is the KV side effect, and
the shipped M=1 decode step produces the logits for the last position.

=== Causality: the per-row width vector IS the mask ===

The scores buffer is `[Hq*M, S]` and softmax's `rows` axis is just "independent rows to
normalise", so row `r = h*M + i` is token `i` of the chunk under head `h`. Row `i` may attend
absolute positions `<= base + i`, which is a width of `base + i + 1` -- the same width under every
head, since a head changes WHICH scores a row is over, never how many positions precede it. A
scalar cannot say that; a per-row width vector says it exactly.

So there is no separate mask buffer and no additive triangle. `Softmax(vector_size_source="rows")`
(IRON branch prefill/causal-softmax) streams one int32 per row alongside the scores, `mask_bf16`
writes -inf past it, and the diagonal block and the zeroed tail of the cache are killed by the
same mechanism. `sm_widths` is a THIRD declared input buffer, `[Hq*M]` int32, written per chunk by
the host; the ELF is still constant across chunks.

  --causal rows   (default) true causality, as above.
  --causal none   every row attends the whole compiled window S, zeroed cache tail included. Kept
                  as the A/B control -- it costs nothing (`sm_kw` is empty and `sm_widths` is not
                  declared), and it is the arm that isolates a dataflow bug from a mask bug. The
                  KV written by layer 0 is correct in this arm too (K/V are appended before any
                  softmax runs), but layer 1's input is not, so the KV of layers >= 1 is NOT the
                  KV the M=1 path would leave. Gate the DATAFLOW with it, never seed a decode.

=== The head-axis seam, and what it costs ===

Attention wants q HEAD-major `[Hq, M, HD]` -- the runlist addresses byte ranges, so a head's slice
must be contiguous -- and the projections produce it TOKEN-major `[M, QD]`, where a head's columns
are strided. At M=1 the question does not arise. Two closures were on the table for the q side,
and the choice is a byte count off the shapes:

  per-head projection   16 GEMMs at N=HD=128 (tile_n=16). Each re-streams the whole A tile, so
                        the q projection costs 16 * M*D*2 = 8.0 MB/layer of activation reads at
                        M=256, where one GEMM at N=QD reads A once for 0.5 MB.
  explicit rearrange    one StridedCopy (pure DMA, 0% compute) from [M,Hq,HD] to [Hq,M,HD] AFTER
                        RoPE: 1 MB read + 1 MB written = 2.0 MB/layer, on top of that same 0.5 MB.

2.5 < 8.0, so: rearrange. The CONTEXT side needs a second one either way -- the ctx GEMM writes
C[M,HD] per head, so `cx` comes back head-major and the o projection wants [M, QD] -- which makes
4.0 MB/layer of zero-compute DMA in total. That is 2.6% of the layer's 154.8 MB of operand bytes
at M=256/S=2048 (see meta.json's `bytes`), against the 43% the score matrix takes, so it is a wart
to fix after the mask and the window, not before them. The way to delete it outright is a GEMM
that can write C with a stride, which IRON does not have.

RoPE and the KV append both want TOKEN-major, so the rearrange sits AFTER both, on q only; k and v
are never rearranged.

=== Batched RoPE: the convention trap, verified against design.py ===

`rope/design.py::core_body` acquires ONE angle row and applies it to `rows/angle_rows` CONSECUTIVE
tensor rows. `rope/reference.py:143` uses `cos.repeat(rep, 1)`, which TILES -- row r gets angle row
`r % angle_rows`. The two agree only at `angle_rows == 1` or `angle_rows == rows`. Everything here
is validated against design.py; the CPU golden below implements the BLOCK convention.

The block convention is exactly what the batched projection already produces: with `rows = M*Hq`
and `angle_rows = M`, angle row `t` covers `Hq` consecutive tensor rows, and the token-major GEMM
output `[M, QD]` IS `[M][Hq][HD]`. So batched RoPE needs no rearrange and no new operator.

=== Why the QKV projection is THREE GEMMs and not one ===

Decode fuses Wq|Wk|Wv into one GEMV writing one `qkv` buffer, and RoPEs q+k in one run because
they are adjacent there. At M>1 that fusion is not expressible: token-major C[M, QD+2*KVD] puts a
token's v rows between its k rows and the NEXT token's q rows, so neither the q head rows nor the
k head rows are a contiguous byte slice -- and both the per-head qk-norm (different gain for q and
k) and RoPE (must not touch v) address rows by slice. Three GEMMs reading three slices of the SAME
`L{l}_Wqkv` buffer keep the shared arena and cost one extra pair of A reads, 1.0 MB/layer.

=== Arena sharing with the decode ELF ===

Prefill and decode must place every weight and cache buffer at IDENTICAL scratch offsets: they
share one FusedArena, and the weights are 1.110 GiB of it. IRON assigns scratch offsets by
first-appearance order in the runlist, so two graphs do not agree by accident. This build reads
the decode artifact's meta.json, reconstructs its scratch arena exactly (including gap fillers for
decode-only intermediates its layout does not name), passes that as `OperatorSequence(...,
scratch_order=...)`, and FAILS THE BUILD if any shared buffer lands at a different offset.

That also means this generator emits NO weight .bin files. The bytes are decode's, byte for byte,
and meta.json's `weights_from` names the directory. One consequence worth stating because it is
silent otherwise: decode folds `attn_scale` into `L{l}_n_qn` (SCALE_IN_QNORM, on by default), so
prefill inherits the scale by reading the same buffer and must NOT apply it again. Verified
against the artifact's own bytes, not assumed.

Run inside the fork IRON env with IRON pointed at a checkout carrying BOTH `scratch_order` and
`vector_size_source` (wt-iron-causal, branch prefill/causal-softmax -- prefill/scratch-order
merged with prefill/rowwise-softmax). AIE_DEVICE=npu2 keeps the build off the device lock -- see
the comment in gen_llm_decode.py.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import SPECS  # noqa: E402
from gemm_tile_registry import registry  # noqa: E402
from prefill_ref import (f32, gate_block, layer_stack, npy_weights,  # noqa: E402
                         rope_block as _rope_block, softmax_rows as _softmax_rows)

import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports
from iron.common import AIEContext  # noqa: E402
from iron.common.kv_layout import KVLayout  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemm.op import GEMM  # noqa: E402
from iron.operators.rms_norm.op import RMSNorm  # noqa: E402
from iron.operators.rope.op import RoPE  # noqa: E402
from iron.operators.softmax.op import Softmax  # noqa: E402
from iron.operators.silu.op import SiLU  # noqa: E402
from iron.operators.gelu.op import GELU  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402
from iron.operators.elementwise_add.op import ElementwiseAdd  # noqa: E402
from iron.operators.strided_copy.op import StridedCopy  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = int(os.environ.get("PREFILL_COLS", "8"))
# There is no TILE_M/TILE_K/TILE_N constant here any more. It used to be `64` for every GEMM in
# the model -- q (K=1024,N=2048), o (K=2048,N=1024), gate/up (K=1024,N=3072), down (K=3072,N=1024),
# scores (K=128,N=2048), ctx (K=2048,N=128) -- one triple over six shapes, picked because it
# divides and fits rather than because anything measured it. `ctx` already had to override it to
# tile_n=16, which is the standing proof that one size does not fit. Each GEMM now asks
# `gemm_tile_registry` for its own shape and RAISES if that shape has never been swept; the
# registry's lookup checks the answer against `gemm_tiling_rejection` before returning it, so K007
# is enforced at the point the shape is picked exactly as before.
# COLS still sets the column split for every NON-GEMM op (RMSNorm/RoPE/Softmax/elementwise); the
# GEMMs take theirs from the registry, which may legitimately differ per shape.
# StridedCopy sizes its ObjectFifo at `transfer_size` elements and, unset, that is the WHOLE
# tensor: 512 KB-1 MB here, against a 512 KB MemTile. Chunk it. 16384 bf16 elements = 32 KB is
# comfortably inside one MemTile with room for the forwarded pair, and every tensor this file
# copies is a multiple of it.
XFER_ELEMS = int(os.environ.get("PREFILL_XFER_ELEMS", "16384"))
# The causal mask, as a buffer name. One int32 per softmax row, host-written per chunk.
SM_WIDTHS = "sm_widths"


def bf16(a):
    return np.asarray(a).astype(BF16)


def rope_table(base, rows, head_dim, theta):
    """[rows, head_dim] bf16 angle table for absolute positions base..base+rows-1.

    Same derivation and the same INTERLEAVED [cos, sin, cos, sin, ...] packing as
    verify_llm_decode.rope_row, one row per position instead of one row per dispatch. NOT the
    half-split [cos..., sin...] packing mlir-air's examples use.
    """
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64)[:half] / head_dim))
    ang = np.arange(base, base + rows, dtype=np.float64)[:, None] * inv[None, :]
    t = np.empty((rows, head_dim), np.float32)
    t[:, 0::2] = np.cos(ang)
    t[:, 1::2] = np.sin(ang)
    return bf16(t)


def causal_widths(base, M, S, heads):
    """The per-row unmasked widths for one chunk: `[heads*M]` int32, row `r = h*M + i`.

    Row `i` of a chunk starting at absolute position `base` attends positions `<= base + i`, so
    its width is `base + i + 1` -- the same under every head. Clamped to `[1, S]` because
    `mask_bf16` loops `for (j = width; j < cols; j++)` over the raw i32 it is handed: a width past
    `S` masks nothing, and a width of 0 leaves the row all -inf, whose softmax is NaN rather than
    a small number. The clamp at S is reachable -- the last chunk of a full window has
    `base + M - 1 == S - 1`, so its final width is exactly S.
    """
    w = np.clip(np.arange(M, dtype=np.int64) + base + 1, 1, S).astype(np.int32)
    return np.tile(w, heads)


def pick_transfer(total_elems, target=None):
    """Largest divisor of `total_elems` at or below `target`, for StridedCopy's ObjectFifo size."""
    target = XFER_ELEMS if target is None else target
    best = max((d for d in range(1, target + 1) if total_elems % d == 0), default=0)
    if not best:
        raise ValueError(f"no ObjectFifo transfer size <= {target} divides {total_elems}")
    return best


def decode_arena_plan(meta_path):
    """Reconstruct the decode ELF's scratch arena as an ordered (name, size) list.

    The decode meta's `layout` names only inputs, the output and the WEIGHT/cache buffers -- its
    per-layer intermediates (`L0_q`, `L0_sc`, `L0_sw`, `L0_cx`, ...) are scratch too and are not
    listed. They show up here as GAPS between consecutive offsets, and the gaps reconcile exactly
    (the first is 4096 B = L0_q at QD*2; the second is 135168 B = sc+sw+cx). So a filler buffer per
    gap reproduces decode's offsets for every shared name without needing to know what decode put
    there.

    Returns (meta, order, sizes, reserved_bytes) where `order` is the scratch_order prefix.
    """
    meta = json.load(open(meta_path))
    scratch = sorted((v["offset"], n, v["len"]) for n, v in meta["layout"].items()
                     if v["type"] == "scratch")
    if not scratch:
        raise ValueError(f"{meta_path}: no scratch buffers in layout")
    order, sizes, cursor, pads = [], {}, 0, 0
    for off, name, length in scratch:
        if off < cursor:
            raise ValueError(f"{meta_path}: {name} at {off} overlaps the previous buffer "
                             f"ending at {cursor} -- the layout is not a packed sequence and "
                             f"scratch_order cannot reproduce it")
        if off > cursor:
            filler = f"__decode_gap{pads}"
            order.append(filler)
            sizes[filler] = off - cursor
            pads += 1
        order.append(name)
        sizes[name] = length
        cursor = off + length
    return meta, order, sizes, cursor


def build_graph(spec_name, NL, M, S, causal, dec_meta_path, cols=COLS, do_compile=True):
    """Construct the fused prefill graph. Returns (spec, fused, dims).

    `do_compile=False` stops after the buffer layout, which is what the shared-arena assert needs
    -- seconds instead of an aiecc run, so the arena contract is checkable on every edit.
    """
    sp = SPECS[spec_name]
    D, FF, HD = sp.d_model, sp.ffn, sp.head_dim
    Hq, Hkv, QD, KVD = sp.n_q_heads, sp.n_kv_heads, sp.q_dim, sp.kv_dim
    grp = sp.gqa_group

    # The KV block size comes FROM THE DECODE ARTIFACT and is never re-derived here. Prefill writes
    # the cache decode reads, so the two must agree on its addressing, and two generators deriving
    # `T` independently is exactly how they came to disagree: decode blocked at T=128 on
    # 2026-09-10 while this file still wrote flat [Hkv, S, HD], and every batched-prefill
    # generation came back as one token repeated -- a correct KV write read through the wrong
    # layout. An absent `kv_block` means the pre-blocking flat cache, for which KVLayout(T=S)
    # reduces to exactly the strides this file used to hardcode.
    dm_early = json.load(open(dec_meta_path)) if dec_meta_path else None
    kv_T = (dm_early["dims"].get("kv_block") or S) if dm_early else S
    kvl = KVLayout(Hkv=Hkv, S=S, HD=HD, T=kv_T)
    print(f"[gen] prefill KV layout: T={kv_T} "
          + ("(flat [Hkv,S,HD])" if kv_T == S else
             f"(blocked [S/T,Hkv,T,HD], head_stride={kvl.head_stride}, "
             f"block_stride={kvl.block_stride})"))

    # Wqkv's ROW ORDER, likewise taken from the decode artifact rather than assumed.
    # decode_layer_dp's attn column c reads ONE contiguous run of (gqa+2) head blocks -- its gqa
    # query heads, then its own k head, then its own v head -- so decode REORDERS the stock
    # [Wq|Wk|Wv] rows at build time (gen_llm_decode.py, wqkv_head_major=True). Prefill reads the
    # same buffer out of the same arena, so it reads the same order, as three blocked operands over
    # one matrix.
    #
    # `wqkv_head_major` in dims is the artifact SAYING so; artifacts built before that field
    # existed are read off `norms`, which decode packs in the same branch and only in that branch.
    # Either way `check_shared_weights()` verifies the answer against the .npy source, because this
    # is the third thing decode changed under prefill in one week -- the blocked KV cache, the
    # packed norms, this -- and all three were silent: a reordered weight is a plausible wrong
    # answer, never an error.
    hm = False
    if dm_early:
        stated = dm_early["dims"].get("wqkv_head_major")
        hm = bool(stated) if stated is not None else any(
            k.endswith("_norms") for k in dm_early["layout"])
    # One kv head's run of rows, and where each role's block sits inside it.
    qkv_group_rows = (grp + 2) * HD
    qkv_blocking = {  # role -> (rows in one block, first row of the first block)
        "q": (grp * HD, 0), "k": (HD, grp * HD), "v": (HD, (grp + 1) * HD),
    } if hm else {}
    print(f"[gen] prefill Wqkv row order: "
          + (f"head-major, {grp}+1+1 head blocks of {HD} rows per kv head"
             if hm else "stock [Wq|Wk|Wv]"))

    # Op-types this graph does not carry. Named here rather than left to produce a plausible
    # wrong answer: the golden below has no sandwich norms either, so the two would AGREE and the
    # gate would pass on a model this graph cannot run.
    if sp.sandwich_norms:
        raise ValueError(f"{sp.name}: sandwich norms (attn/FFN output normalised before the "
                         f"residual add) are not in this graph's vocabulary")
    if sp.rope_theta_local is not None:
        raise ValueError(f"{sp.name}: dual-theta RoPE needs a second angle table; this graph "
                         f"declares one `rope` input")
    if not sp.qk_norm:
        # Not a missing brick so much as a missing multiply: `attn_scale` rides on the shared
        # q-norm gain (decode's SCALE_IN_QNORM), so a spec without a q-norm has nowhere to put it
        # and would produce unscaled scores.
        raise ValueError(f"{sp.name}: no per-head q-norm, so attn_scale has no gain to ride on; "
                         f"this graph has no separate score scale")

    # ---- K007: every shape constraint asserted where the shape is picked ----
    # The GEMM tilings come from the registry below, at the point each GEMM is constructed --
    # `Registry.lookup` runs the same `gemm_tiling_rejection` these checks do, so a shape that
    # reaches an operator has already had its modulus, L1 and MemTile budgets named. What is left
    # here is everything that is NOT a GEMM.
    # RoPE. `rows = M*heads, angle_rows = M` makes design.py's block quotient exactly the head
    # count, which is what makes the block convention the RIGHT one for a token-major tensor --
    # so these three moduli are the whole contract, and the fourth condition (that the buffer is
    # token-major) is structural and lives in the runlist.
    if HD % 32 or HD < 32:
        raise ValueError(f"rope: cols=head_dim={HD} must be a multiple of 32 and >= 32")
    if M % cols:
        raise ValueError(f"rope: angle_rows=M={M} must be divisible by num_aie_columns={cols}")
    for label, heads in (("q", Hq), ("k", Hkv)):
        if (M * heads) % cols:
            raise ValueError(f"rope {label}: rows=M*heads={M * heads} must be divisible by "
                             f"num_aie_columns={cols}")
    for label, size in (("norm", M * D), ("qk-norm q", M * QD), ("qk-norm k", M * KVD)):
        tile = D if label == "norm" else HD
        if size % (cols * tile):
            raise ValueError(f"RMSNorm {label}: size={size} not a multiple of "
                             f"num_aie_columns*tile_size={cols * tile}")
    if (Hq * M) % cols:
        raise ValueError(f"Softmax rows=Hq*M={Hq * M} not divisible by num_aie_columns={cols}")
    if S % 16:
        raise ValueError(f"Softmax cols=S={S} must be a multiple of 16")

    if os.environ.get("AIE_DEVICE"):
        import aie.utils as _aie_utils
        from aie.iron.device import from_name as _from_name
        _aie_utils.set_current_device(_from_name(os.environ["AIE_DEVICE"], n_cols=None))

    ctx = AIEContext()
    # PREFILL_BFP16=0 turns OFF IRON GEMM's default bfp16 emulation.
    #
    # Not a tuning knob -- a NUMERICS one, and it is the axis that decides whether batched prefill
    # can ever be token-identical to the M=1 decode path. GEMM defaults
    # `emulate_bf16_mmul_with_bfp16=True`, which converts both operands to `v64bfp16ebs8`: block
    # float with ONE exponent shared across 8 elements. The decode path's GEMV has no bfp16
    # anywhere (grep `iron/operators/gemv/`, `aie_kernels/aie2p/mv.cc`) -- it does plain bf16 MACs.
    # So the two paths compute the same expression in different formats, and measured on device
    # 2026-09-08 that is worth rel-L2 1.3e-2 on layer 0's K AND V, compounding to 2.2e-1 by
    # layer 27. Setting this to 0 costs the 4x mmul throughput and is what makes the arithmetic
    # comparable.
    emulate = os.environ.get("PREFILL_BFP16", "1") == "1"
    # PREFILL_ACC=1 keeps the GEMM's K-reduction in f32.
    #
    # The second numerics axis, and measured to be the bigger one. By DEFAULT `C_l1_ty` is the
    # OUTPUT dtype -- bf16 -- so the reduction loop rounds to bf16 once per k-tile, 16 times at
    # K=1024/tile_k=64. `gemm/design.py:180` says so outright: prio_accuracy "will accumulate in
    # place with a f32 buffer, which will be converted to bf16 after the reduction loop finishes".
    # Decode's GEMV has no such split -- it carries all of K in f32 -- so the default GEMM cannot
    # reproduce it however the operands are formatted.
    prio_acc = os.environ.get("PREFILL_ACC", "0") == "1"
    # PREFILL_ROUND_EVEN=0 makes the GEMM round bf16 outputs FLOOR instead of nearest-even.
    #
    # The third numerics axis, and the one that is a genuine ASYMMETRY rather than a choice.
    # `aie_kernels/aie2p/mm.cc:222-227` takes conv_even under -DROUND_CONV_EVEN and floor
    # otherwise; `aie_kernels/aie2p/mv.cc` -- the kernel the M=1 decode path actually runs --
    # contains the string "rounding" ZERO times, so it never sets the register and inherits
    # whatever is in it (documented default floor; in a fused ELF, whatever the previous kernel on
    # that core left). One ULP of bf16 is 2^-8 = 3.9e-3 relative, which is the order of the
    # residual disagreement between the two paths. Matching the modes is a precondition for token
    # identity; it is NOT a claim that floor is the better mode -- see K001.
    round_even = os.environ.get("PREFILL_ROUND_EVEN", "1") == "1"

    # Every GEMM's tiling, from the measured registry. A shape with no entry RAISES here, naming
    # the sweep that fills it -- there is deliberately no fallback triple, because a fallback is
    # the hardcoded constant this replaced with an extra indirection. `GEMM_TILES_OVERRIDE`
    # (JSON, keyed by registry key or by the op label below) is the experiment escape hatch.
    reg = registry()
    tiles = {}

    def gemm_for(label, K, Nout, b_col_maj=True, blocking=None):
        """One GEMM at the registry's tiling for its shape, checked twice on the way through.

        `blocking` is `(rows_per_block, block_stride)` when B's rows are not one contiguous slab in
        the shared arena -- the KV cache under `kvl`, or one role's head rows inside a head-major
        `Wqkv`. `None` is the contiguous case every other call builds.
        """
        ch = reg.lookup(M, K, Nout, emulate=emulate, prio_accuracy=prio_acc,
                        b_col_maj=b_col_maj, label=label)
        # The spec-shaped check as well as the registry's own: it names the SPEC and the op, which
        # is the message a shape error should carry, and it costs nothing.
        sp.check_prefill_projections(M, ((label, K, Nout),), tile_m=ch.tile_m, tile_k=ch.tile_k,
                                     tile_n=ch.tile_n, cols=ch.cols, bfp16=emulate,
                                     prio_accuracy=prio_acc)
        tiles[label] = {"tile": [ch.tile_m, ch.tile_k, ch.tile_n], "cols": ch.cols,
                        "source": ch.source, "K": K, "N": Nout,
                        "measured": ch.measured}
        blk = dict(b_block_rows=blocking[0], b_block_stride=blocking[1]) if blocking else {}
        return GEMM(M=M, K=K, N=Nout, b_col_maj=b_col_maj, context=ctx,
                    emulate_bf16_mmul_with_bfp16=emulate, prio_accuracy=prio_acc,
                    round_conv_even=round_even, **blk, **ch.gemm_kwargs)

    op_norm = RMSNorm(size=M * D, num_aie_columns=cols, num_channels=1, tile_size=D,
                      weighted=True, epsilon=sp.eps, context=ctx)
    op_qn = RMSNorm(size=M * QD, num_aie_columns=cols, num_channels=1, tile_size=HD,
                    weighted=True, epsilon=sp.eps, context=ctx)
    op_kn = RMSNorm(size=M * KVD, num_aie_columns=cols, num_channels=1, tile_size=HD,
                    weighted=True, epsilon=sp.eps, context=ctx)
    op_rq = RoPE(rows=M * Hq, cols=HD, angle_rows=M, num_aie_columns=cols, context=ctx)
    op_rk = RoPE(rows=M * Hkv, cols=HD, angle_rows=M, num_aie_columns=cols, context=ctx)
    op_gq = gemm_for("q", D, QD, blocking=(
        (qkv_blocking["q"][0], qkv_group_rows * D) if hm else None))
    # ONE design serves k and v (same shape, and under head-major the same block geometry -- only
    # the slice base differs), and one serves gate and up, so the label is the pair.
    # `GEMM_TILES_OVERRIDE` keys on these labels or on the registry key.
    op_gkv = gemm_for("kv", D, KVD, blocking=(
        (qkv_blocking["k"][0], qkv_group_rows * D) if hm else None))
    op_o = gemm_for("o", QD, D)
    op_gu = gemm_for("gate_up", D, FF)
    op_down = gemm_for("down", FF, D)
    # scores: B is the kv cache read as [N=S, K=HD] -> b_col_maj. ctx: the SAME bytes read as
    # [K=S, N=HD] -> plain. Either way the blocked axis is the physical leading one, positions, so
    # one `b_blocked` serves both and the descriptor rewrite lives in the operator.
    kv_blocking = (kvl.T, kvl.block_stride) if kvl.T != kvl.S else None
    op_sc = gemm_for("scores", HD, S, blocking=kv_blocking)
    op_cx = gemm_for("ctx", S, HD, b_col_maj=False, blocking=kv_blocking)
    tn_sc, tn_cx = tiles["scores"]["tile"][2], tiles["ctx"]["tile"][2]
    print("[tiles] " + "  ".join(
        f"{k}={v['tile'][0]}x{v['tile'][1]}x{v['tile'][2]}@{v['cols']}c({v['source']})"
        for k, v in sorted(tiles.items())))
    # ONE softmax over every head's rows at once: its `rows` axis is just "independent rows to
    # normalise", and every head's [M, S] block is a contiguous slice of the same buffer. That is
    # also what makes the causal mask a plain vector: row Hq*M is (head, token) flattened, and the
    # width depends only on the token half.
    sm_kw = dict(vector_size_source="rows") if causal == "rows" else {}
    # PREFILL_ATTN_ORDER is a CONFIGURE-COST CONTROL, not a feature. Both arms split the softmax
    # per head -- legal because softmax is per ROW and every head's sc/sw/widths slice is
    # contiguous -- so the two arms run IDENTICAL ops over IDENTICAL bytes with IDENTICAL designs,
    # and differ only in runlist ORDER, hence only in how many contiguous same-design blocks the
    # dispatch configures. `grouped` is 2 blocks for the 32 ops, `interleaved` is 32.
    #
    # It exists because D009's 51.0-61.9 us per configure is measured on DECODE and the prefill
    # regime cell is `p`. If prefill's per-configure cost really is ~55 us, +30 configures/layer
    # costs ~46 ms; if the per-layer residual is un-overlapped objectFIFO fill/drain at ~350 us a
    # configure, it costs ~294 ms. The arms separate those by 6x, which no drift can hide.
    # `off` (the default) is the shipped single whole-buffer softmax, unchanged.
    attn_order = os.environ.get("PREFILL_ATTN_ORDER", "off")
    if attn_order not in ("off", "grouped", "interleaved"):
        raise ValueError(f"PREFILL_ATTN_ORDER={attn_order!r}; want off|grouped|interleaved")
    n_il = int(os.environ.get("PREFILL_ATTN_HEADS", Hq))
    op_sm = Softmax(rows=Hq * M, cols=S, num_aie_columns=cols, num_channels=1,
                    context=ctx, **sm_kw)
    # One design serves every head: same rows, same cols. Only how many times the runlist SWITCHES
    # to it changes between the arms.
    op_sm_head = (Softmax(rows=M, cols=S, num_aie_columns=cols, num_channels=1,
                          context=ctx, **sm_kw) if attn_order != "off" else None)
    if sp.act == "silu":
        op_act = SiLU(size=M * FF, num_aie_columns=cols, tile_size=FF // cols, context=ctx)
    else:
        op_act = GELU(size=M * FF, num_aie_columns=cols, num_channels=1,
                      tile_size=FF // cols, context=ctx)
    op_mul = ElementwiseMul(size=M * FF, num_aie_columns=cols, tile_size=FF // cols, context=ctx)
    op_add = ElementwiseAdd(size=M * D, num_aie_columns=cols, tile_size=D // cols, context=ctx)
    # KV append. The cache is [Hkv, S, HD] and `kv_off` is an element-unit BD offset, so M
    # consecutive positions are M contiguous rows per head -- the M=1 BD with an extra outer
    # dimension, not a new mechanism. The SOURCE is token-major [M, Hkv, HD], so the (M, Hkv) axes
    # swap in the descriptor: input walks h fastest within a token, output walks m fastest within
    # a head.
    if kvl.T == kvl.S:
        # Flat [Hkv, S, HD]: M consecutive positions are M contiguous rows per head.
        kv_in_sizes, kv_in_strides = (M, Hkv, HD), (Hkv * HD, HD, 1)
        kv_out_sizes, kv_out_strides = (M, Hkv, HD), (HD, kvl.head_stride, 1)
    else:
        # Blocked [S/T, Hkv, T, HD]. A chunk starts at a multiple of M and M is a whole number of
        # blocks, so the write stays block-aligned and the descriptor gains one outer dimension
        # rather than needing a scatter.
        if M % kvl.T:
            raise ValueError(
                f"prefill batch M={M} is not a whole number of KV blocks (T={kvl.T}); the append "
                f"would straddle a block boundary mid-descriptor")
        nb = M // kvl.T
        kv_in_sizes = (nb, Hkv, kvl.T, HD)
        kv_in_strides = (kvl.T * Hkv * HD, HD, Hkv * HD, 1)
        kv_out_sizes = (nb, Hkv, kvl.T, HD)
        kv_out_strides = (kvl.block_stride, kvl.head_stride, HD, 1)
    op_kvapp = StridedCopy(
        input_sizes=kv_in_sizes, input_strides=kv_in_strides, input_offset=0,
        output_sizes=kv_out_sizes, output_strides=kv_out_strides, output_offset=0,
        input_buffer_size=M * Hkv * HD, output_buffer_size=kvl.total_elems,
        transfer_size=pick_transfer(M * Hkv * HD), num_aie_channels=1,
        output_offset_parameter="kv_off", context=ctx)
    # The head-axis seam, both directions. See the module docstring for why these exist and what
    # they cost; they are pure DMA and do 0% compute.
    op_q2h = StridedCopy(
        input_sizes=(Hq, M, HD), input_strides=(HD, QD, 1), input_offset=0,
        output_sizes=(Hq, M, HD), output_strides=(M * HD, HD, 1), output_offset=0,
        input_buffer_size=M * QD, output_buffer_size=M * QD,
        transfer_size=pick_transfer(M * QD), num_aie_channels=1, context=ctx)
    op_h2t = StridedCopy(
        input_sizes=(M, Hq, HD), input_strides=(HD, M * HD, 1), input_offset=0,
        output_sizes=(M, Hq, HD), output_strides=(Hq * HD, HD, 1), output_offset=0,
        input_buffer_size=M * QD, output_buffer_size=M * QD,
        transfer_size=pick_transfer(M * QD), num_aie_channels=1, context=ctx)

    # ---- buffers ----
    # Every prefill intermediate is ONE buffer shared by all layers: the sequence runs layers one
    # at a time, so nothing outlives its layer. Decode declares them per layer; at M=256 that
    # would be 28 * 43 MB of arena for no reason.
    bufsz = {
        "h": M * D * 2, "q": M * QD * 2, "k": M * KVD * 2, "v": M * KVD * 2,
        "qh": M * QD * 2, "sc": Hq * M * S * 2, "sw": Hq * M * S * 2,
        "cx": M * QD * 2, "cxt": M * QD * 2, "a": M * D * 2, "xs": M * D * 2,
        "hf": M * D * 2, "g": M * FF * 2, "gs": M * FF * 2, "u": M * FF * 2,
        "gh": M * FF * 2, "d": M * D * 2,
    }
    if attn_order != "off":
        # Slicing an INPUT needs its size declared: calculate_buffer_layout takes a plain buffer's
        # size from the arg spec, and a sliced one only from here. Same size the whole-buffer arm
        # gets from op_sm's spec, so the input arena is byte-identical between the arms.
        bufsz[SM_WIDTHS] = Hq * M * 4
    prefill_local = sorted(bufsz)
    dec_meta, dec_order, dec_sizes, dec_reserved = (None, [], {}, 0)
    if dec_meta_path:
        dec_meta, dec_order, dec_sizes, dec_reserved = decode_arena_plan(dec_meta_path)
        for name, length in dec_sizes.items():
            if name in bufsz:
                raise ValueError(f"decode scratch name {name!r} collides with a prefill "
                                 f"intermediate of the same name")
            bufsz[name] = length

    def qkv_slab(p, role, op):
        """One projection's operand inside `L*_Wqkv`.

        Stock, the three roles are three contiguous slabs. Head-major, each role's rows are one
        block per kv head, so the operand runs from its FIRST block to the end of its last one and
        the operator's blocked descriptor picks its own rows out of the span -- `op.b_elems` is
        that span, asked of the operator rather than recomputed here.
        """
        if not hm:
            base, span = ({"q": (0, QD), "k": (QD, KVD), "v": (QD + KVD, KVD)}[role][0] * D,
                          {"q": QD, "k": KVD, "v": KVD}[role] * D)
        else:
            base, span = qkv_blocking[role][1] * D, op.b_elems
        return f"{p}Wqkv[{base * 2}:{(base + span) * 2}]"

    def kv_slab(buf, kv):
        """A kv head's slab: from its base to the end of its LAST block, not `S*HD` -- across
        blocks the head's positions are `block_stride` apart with the other heads in between.
        `kvl` owns both numbers, and both reduce to the flat `kv*S*HD` slice at T == S."""
        base = kvl.head_base(kv)
        return f"{buf}[{base * 2}:{(base + kvl.head_span) * 2}]"

    def attn_norms(p):
        """This layer's (input-norm, q-norm, k-norm) operands, as the decode arena actually holds
        them.

        Decode packs the three into ONE `norms` buffer under FUSE_DECODE_LAYER (its default since
        2026-09-10) and keeps them separate otherwise. Prefill has to read whichever it finds,
        because naming a buffer the shared arena does NOT have is not an error: it allocates a
        prefill-local scratch buffer that nothing ever fills, so the norm weight reads as ZERO,
        every position attends identically, and the model emits one token repeated. The general
        form of that trap is checked below; this is the instance that paid for the check.
        """
        if p + "norms" in dec_sizes:
            b = p + "norms"
            return (f"{b}[0:{D * 2}]", f"{b}[{D * 2}:{(D + HD) * 2}]",
                    f"{b}[{(D + HD) * 2}:{(D + 2 * HD) * 2}]")
        return (p + "n_in", p + "n_qn", p + "n_kn")

    rl, cache_names = [], []
    for l in range(NL):
        p = f"L{l}_"
        src = "x" if l == 0 else "xs"
        dst = "xout" if l == NL - 1 else "xs"
        wq, wk, wv = (qkv_slab(p, "q", op_gq), qkv_slab(p, "k", op_gkv),
                      qkv_slab(p, "v", op_gkv))
        # Decode pads Wo by two rows for swiglu_mlp_dp's fuse_o tiling; the projection is the
        # first D rows and the pad rows are zero, so prefill reads the unpadded prefix.
        wo = f"{p}Wo[0:{D * QD * 2}]"
        w_nin, w_nqn, w_nkn = attn_norms(p)
        rl += [
            (op_norm, src, w_nin, "h"),
            (op_gq, "h", wq, "q"),
            (op_gkv, "h", wk, "k"),
            (op_gkv, "h", wv, "v"),
        ]
        rl += [
            (op_qn, "q", w_nqn, "q"),
            (op_kn, "k", w_nkn, "k"),
            (op_rq, "q", "rope", "q"),
            (op_rk, "k", "rope", "k"),
            # K after qk-norm AND after RoPE; V raw, projection only. Different points in the
            # pipeline, and the M=1 path a decode step resumes from depends on both.
            (op_kvapp, "k", p + "kc"),
            (op_kvapp, "v", p + "vc"),
            (op_q2h, "q", "qh"),
        ]
        # The widths buffer is an INPUT of the softmax step, not a side channel: op.get_arg_spec()
        # puts it between in and out, so it is the middle name here.
        def score(h):
            return (op_sc, f"qh[{h * M * HD * 2}:{(h + 1) * M * HD * 2}]",
                    kv_slab(p + "kc", h // grp),
                    f"sc[{h * M * S * 2}:{(h + 1) * M * S * 2}]")

        def soft(h):
            sl = f"sc[{h * M * S * 2}:{(h + 1) * M * S * 2}]"
            out = f"sw[{h * M * S * 2}:{(h + 1) * M * S * 2}]"
            w = f"{SM_WIDTHS}[{h * M * 4}:{(h + 1) * M * 4}]"
            return (op_sm_head, sl, w, out) if causal == "rows" else (op_sm_head, sl, out)

        if attn_order == "interleaved":
            # Only the first `n_il` heads alternate; the rest stay grouped. The knob exists because
            # a configure costs ~80 KB of instruction stream, so interleaving all 16 heads built a
            # 157 MB ELF that the driver refuses to allocate a BO for (CREATE_BO EAGAIN,
            # reproducible). +2 configures per interleaved head per layer.
            for h in range(n_il):
                rl += [score(h), soft(h)]
            rl += [score(h) for h in range(n_il, Hq)]
            rl += [soft(h) for h in range(n_il, Hq)]
        elif attn_order == "grouped":
            rl += [score(h) for h in range(Hq)] + [soft(h) for h in range(Hq)]
        else:
            rl += [score(h) for h in range(Hq)]
            rl.append((op_sm, "sc", SM_WIDTHS, "sw") if causal == "rows"
                      else (op_sm, "sc", "sw"))
        for h in range(Hq):
            rl.append((op_cx, f"sw[{h * M * S * 2}:{(h + 1) * M * S * 2}]",
                       kv_slab(p + "vc", h // grp),
                       f"cx[{h * M * HD * 2}:{(h + 1) * M * HD * 2}]"))
        rl += [
            (op_h2t, "cx", "cxt"),
            (op_o, "cxt", wo, "a"),
            (op_add, src, "a", "xs"),
            (op_norm, "xs", p + "n_pf", "hf"),
            (op_gu, "hf", p + "Wg", "g"),
            (op_gu, "hf", p + "Wu", "u"),
            (op_act, "g", "gs"),
            (op_mul, "gs", "u", "gh"),
            (op_down, "gh", p + "Wd", "d"),
            (op_add, "xs", "d", dst),
        ]
        cache_names += [p + "kc", p + "vc"]

    # `sm_widths` goes LAST so x and rope keep the input-arena offsets the non-causal arm gives
    # them: add_buffers walks input_args in order, and the host's x/rope writes are the same in
    # both arms.
    inputs = ["x", "rope"] + ([SM_WIDTHS] if causal == "rows" else [])

    # Every buffer this graph READS and never writes has to come from somewhere -- a host input, or
    # the decode arena. One that comes from neither is a prefill-local scratch buffer nothing fills:
    # it reads as ZERO, which for a weight is silent all the way to the tokens. That is precisely
    # how prefill went on naming L*_n_in/n_qn/n_kn after decode packed the three into L*_norms; the
    # build stayed green, the shared-arena assert stayed green (it compares the buffers both sides
    # DO declare), and every batched generation came back as one token repeated. Checkable only on
    # the shared-arena path -- with --no-arena-share nothing is provided and every weight is an
    # orphan by construction.
    if dec_meta_path:
        read, written = set(), set()
        for op, *bufs in rl:
            for spec, nm in zip(op.get_arg_spec(), bufs):
                base = nm.split("[")[0]
                (written if spec.direction in ("out", "inout") else read).add(base)
        orphans = sorted(read - written - set(dec_sizes) - set(inputs))
        if orphans:
            raise ValueError(
                f"{len(orphans)} buffer(s) are read by the prefill graph, never written by it, and "
                f"not provided by the decode arena or the host: {orphans}. They would be allocated "
                f"as zeroed scratch and read as zero. If the decode artifact renamed or packed "
                f"them, follow it here")
    # Every knob that changes the GRAPH must be in the name: IRON keys the cached artifact by
    # it, so an arm whose name collides with an earlier one silently RUNS THE EARLIER
    # BINARY. Measured here 2026-09-08: a PREFILL_BFP16=0 rebuild produced an ELF with the
    # same md5 as the bfp16 arm, so the numerics A/B measured nothing until the flag went
    # into the name. `gen_llm_decode.py`'s sequence_name() carries the same warning.
    name = (f"prefill_{sp.name.replace('-', '_').replace('.', '_')}"
            f"_m{M}_s{S}_l{NL}_c{cols}_{causal}_bfp{int(emulate)}_acc{int(prio_acc)}_re{int(round_even)}")
    if causal != "none":
        name += f"_{causal}"
    # The two configure-cost arms differ ONLY in runlist order, so every other name component is
    # identical between them -- exactly the collision that silently links the earlier arm's ELF.
    if attn_order != "off":
        name += f"_ao{attn_order}" + (f"{n_il}" if attn_order == "interleaved" else "")
    # The tiling is now a per-shape lookup, so it is a GRAPH knob like the three above and has to
    # be in the name for the same reason: a re-sweep that moves one GEMM's tile must not link the
    # previous tiling's ELF out of the artifact cache. Hashed rather than spelled out -- seven
    # ops * four numbers does not belong in a filename, and the tiles themselves are in meta.json.
    tile_sig = ";".join(f"{k}:{v['tile']}x{v['cols']}" for k, v in sorted(tiles.items()))
    name += "_t" + hashlib.md5(tile_sig.encode()).hexdigest()[:8]
    fused = OperatorSequence(name, rl, input_args=inputs, output_args=["xout"],
                             buffer_sizes=bufsz, context=ctx, share_designs=True,
                             scratch_order=(dec_order or None))
    if do_compile:
        fused.compile()
    else:
        (fused.subbuffer_layout, fused.buffer_sizes,
         fused.slice_info) = fused.calculate_buffer_layout()

    # The whole point of scratch_order: verify it, do not trust it. A silent disagreement here is
    # decode reading prefill's `Wg` as its `kc`.
    if dec_meta is not None:
        for nm in dec_order:
            if nm.startswith("__decode_gap"):
                continue
            got = fused.get_layout_for_buffer(nm)
            want = dec_meta["layout"][nm]
            if (got[0], int(got[1]), int(got[2])) != ("scratch", want["offset"], want["len"]):
                raise ValueError(
                    f"shared-arena mismatch on {nm!r}: prefill places it at "
                    f"{got[0]} offset {got[1]} len {got[2]}, the decode artifact at "
                    f"scratch offset {want['offset']} len {want['len']} -- the two ELFs cannot "
                    f"share one FusedArena")

    dims = dict(NL=NL, M=M, S=S, inputs=inputs, cache_names=cache_names,
                tn_sc=tn_sc, tn_cx=tn_cx, tiles=tiles, cols=cols, causal=causal,
                kv_block=kvl.T, wqkv_head_major=hm,
                sm_widths=(SM_WIDTHS if causal == "rows" else None), sm_rows=Hq * M,
                shared=[n for n in dec_order if not n.startswith("__decode_gap")],
                reserved=dec_reserved, prefill_local=prefill_local,
                rl=rl, runlist_len=len(rl), per_layer=len(rl) // NL,
                # TWO different numbers, and conflating them understated the configure count by
                # 31x. `n_designs` is how many designs get BUILT -- `share_designs` collapses
                # operators reporting the same design_key onto one. `n_configures` is how many
                # setups the dispatch PAYS for, and D009 prices that one: a configure covers a
                # CONTIGUOUS same-design block, offsets free inside it, so one design reached at
                # three separate points in the runlist costs three. Runs inside a block ride free.
                n_designs=len(fused.unique_designs()[0]),
                n_configures=count_configures(rl, fused))
    return sp, fused, dims


def count_configures(runlist, fused):
    """Contiguous same-design blocks in `runlist` -- the unit D009 prices, at 51.0-61.9 us each.

    Uses the sequence's OWN design assignment (`unique_designs()[1]`) rather than a second notion
    of identity here, so this cannot drift from what the build actually configures.
    """
    _, design_of = fused.unique_designs()
    ids = [design_of[id(op)] for op, *_ in runlist]
    return 1 + sum(1 for a, b in zip(ids, ids[1:]) if a != b) if ids else 0


def operand_traffic(op, bufs, resolve):
    """Bytes each of `op`'s operands actually moves, one per name in `bufs`.

    Not the slice length. A blocked operand's slice runs from its first block to the end of its
    last, past the peer matrices interleaved in between -- a KV head's slab spans 7.5x the bytes
    the head holds, and a head-major Wqkv role spans ~2.8x its own rows. StridedCopy is counted off
    its access pattern and GEMM off its shapes; everything else still reads its slice, which for a
    contiguous operand is the same number.
    """
    if isinstance(op, StridedCopy):
        return [int(np.prod(op.input_sizes)) * 2] * len(bufs)
    if isinstance(op, GEMM):
        return [op.M * op.K * 2, op.K * op.N * 2, op.M * op.N * 2]
    out = []
    for name in bufs:
        if "[" in name:
            lo, hi = name[name.index("[") + 1:-1].split(":")
            out.append(int(hi) - int(lo))
        else:
            out.append(int(resolve(name)[2]))
    return out


def op_census(runlist, resolve, n_layers):
    """Bytes and invocation count PER OPERATOR TYPE, for one layer.

    `operand_bytes` above splits by BUFFER class, which answers "what kind of traffic is this" and
    not "which op should I attack". Device timing put 27.6% of a layer outside the two block
    artifacts (MLP 36.0%, attention 36.4%) without saying which of the remaining ops it is; this
    is the byte-side companion to that split. It is an ATTRIBUTION BY BYTES, not a measurement:
    ops do not all run at the same GB/s, and the two zero-compute rearranges in particular are
    pure DMA. Read it to rank suspects, then measure the winner.
    """
    per = {}
    for op, *bufs in runlist:
        kind = type(op).__name__
        b = sum(operand_traffic(op, bufs, resolve))
        e = per.setdefault(kind, [0, 0])
        e[0] += 1
        e[1] += b
    total = sum(v[1] for v in per.values())
    rows = sorted(per.items(), key=lambda kv: -kv[1][1])
    print(f"[census] per LAYER ({n_layers} built), by operator type:")
    print(f"[census] {'op':<22} {'calls':>6} {'MB':>10} {'share':>7}")
    for k, (n, b) in rows:
        print(f"[census] {k:<22} {n // n_layers:>6} {b / n_layers / 2**20:>10.2f} "
              f"{100 * b / total:>6.1f}%")
    print(f"[census] {'TOTAL':<22} {len(runlist) // n_layers:>6} "
          f"{total / n_layers / 2**20:>10.2f} {100.0:>6.1f}%")


def operand_bytes(runlist, resolve, shared):
    """Bytes the graph touches, per class, COMPUTED off the runlist instead of a hand formula.

    A floor, not a measurement: one pass per named operand, so it ignores the broadcast of A to
    every column and any MemTile reuse, in both directions. Its job is to let a device timing be
    read as GB/s without the caller re-deriving the byte count, and to keep the three terms
    separable -- weights are M-independent, activations are linear in M, and the score matrix
    grows with the compiled WINDOW rather than with the batch, which is the term the MLP block
    could not exercise.

    Per-operand bytes come from `operand_traffic`, which is where the slice-length trap lives:
    `get_layout_for_buffer`'s third element is a LENGTH for a plain buffer and an absolute END
    offset for a slice (iron/common/sequence.py), and reading it as a length made a 40 MB layer
    read as 93 GB.
    """
    out = {"weights": 0, "cache": 0, "scores": 0, "activations": 0}

    def klass(name):
        base = name.split("[")[0]
        if base.endswith("_kc") or base.endswith("_vc"):
            return "cache"
        if base in ("sc", "sw"):
            return "scores"
        return "weights" if base in shared else "activations"

    for op, *bufs in runlist:
        for name, moved in zip(bufs, operand_traffic(op, bufs, resolve)):
            out[klass(name)] += moved
    return out


# --------------------------------------------------------------------------------------------
# CPU goldens. The dataflow itself lives in prefill_ref.py, once, parameterised on where it
# narrows; this file only supplies the weights and picks the arm.
# --------------------------------------------------------------------------------------------
def golden_widths(sp, M, S, base, causal):
    """The softmax widths the golden must use.

    The device is handed `[Hq*M]`; a head's block is `[M, S]` and every head shares the same M
    widths, so slicing the first M off the flat vector is the same thing and keeps the two
    derivations from drifting.
    """
    return causal_widths(base, M, S, sp.n_q_heads)[:M] if causal == "rows" else None


def golden(sp, weights_dir, NL, M, S, base, causal, X, table, rnd=bf16):
    """Full-stack CPU golden. Returns (xout, [(kc_slab, vc_slab) per layer]).

    `rnd=bf16` rounds where the device rounds -- the probe's reference. `rnd=f32` keeps every
    intermediate wide on the same bf16 inputs -- Tier 1's reference (see prefill_ref's header for
    why a bf16 golden cannot serve as one).
    """
    return layer_stack(sp, npy_weights(weights_dir), NL, M, S, base, X, table,
                       golden_widths(sp, M, S, base, causal), rnd)


def check_shared_weights(dec_meta_path, weights_dir, sp, dims):
    """Every claim this graph makes about a decode weight buffer, checked against the .npy source.

    Prefill does not own these bytes -- it borrows decode's arena -- so every read is an assumption
    about a layout decode chose and can change. Three of them changed under prefill in one week and
    all three were silent, because a mis-read weight is a plausible wrong answer and never an error:
    decode blocked the KV cache, packed `n_in|n_qn|n_kn` into `norms`, and reordered `Wqkv` into
    head-major. This is the check that turns the next one into a build failure. Layer 0 only: these
    are per-layer transforms applied uniformly, so a disagreement shows there.

    Skipped when the caller has no weights (`--no-golden`); it costs one layer's tensors otherwise.
    """
    bdir = os.path.join(os.path.dirname(os.path.abspath(dec_meta_path)), "buffers")
    npy = lambda t: np.load(os.path.join(weights_dir, f"model.layers.0.{t}.weight.npy"))
    raw = lambda n: np.fromfile(os.path.join(bdir, f"L0_{n}.bin"), dtype=BF16)
    same = lambda a, b: np.array_equal(np.asarray(a).view(np.uint16),
                                       np.asarray(np.asarray(b).astype(BF16)).view(np.uint16))
    D, HD, grp = sp.d_model, sp.head_dim, sp.gqa_group
    QD, KVD = sp.q_dim, sp.kv_dim
    bad = []

    def claim(ok, what):
        if not ok:
            bad.append(what)

    n = raw("norms") if os.path.exists(os.path.join(bdir, "L0_norms.bin")) else None
    if n is not None:
        claim(same(n[:D], npy("input_layernorm")), "norms[0:D] is the input layernorm")
        # decode folds attn_scale into the q-norm gain (SCALE_IN_QNORM); prefill applies no scale
        # of its own, so the FOLDED tensor is what it must be reading.
        claim(same(n[D:D + HD], np.asarray(npy("self_attn.q_norm"), np.float32) * sp.attn_scale),
              "norms[D:D+HD] is the q-norm with attn_scale folded in")
        claim(same(n[D + HD:D + 2 * HD], npy("self_attn.k_norm")), "norms[D+HD:] is the k-norm")
    claim(same(raw("n_pf"), npy("post_attention_layernorm").reshape(-1)),
          "n_pf is the post-attention layernorm")
    claim(same(raw("Wo")[:D * QD], npy("self_attn.o_proj").reshape(-1)),
          "Wo's first D rows are o_proj (decode pads the tail for fuse_o)")
    for nm, t in (("Wg", "mlp.gate_proj"), ("Wu", "mlp.up_proj"), ("Wd", "mlp.down_proj")):
        claim(same(raw(nm), npy(t).reshape(-1)), f"{nm} is {t} unreordered")

    w = raw("Wqkv").reshape(-1, D)
    q, k, v = (npy(f"self_attn.{r}_proj") for r in ("q", "k", "v"))
    if dims["wqkv_head_major"]:
        ok = True
        for c in range(sp.n_kv_heads):
            b = c * (grp + 2) * HD
            for g in range(grp):
                ok &= same(w[b + g * HD:b + (g + 1) * HD], q[(c * grp + g) * HD:(c * grp + g + 1) * HD])
            ok &= same(w[b + grp * HD:b + (grp + 1) * HD], k[c * HD:(c + 1) * HD])
            ok &= same(w[b + (grp + 1) * HD:b + (grp + 2) * HD], v[c * HD:(c + 1) * HD])
        claim(ok, f"Wqkv is head-major ({grp} q + 1 k + 1 v head blocks per kv head)")
    else:
        claim(same(w[:QD], q) and same(w[QD:QD + KVD], k) and same(w[QD + KVD:], v),
              "Wqkv is the stock [Wq|Wk|Wv] stacking")
    if bad:
        raise SystemExit("ERROR: the decode arena does not hold what this graph assumes:\n  - "
                         + "\n  - ".join(bad)
                         + "\nFollow the decode generator's layout here, or rebuild the decode "
                           "artifact from a generator that agrees with this one.")
    print(f"[gen] shared-weight contract verified against {weights_dir} (layer 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b", choices=sorted(SPECS))
    ap.add_argument("--weights", help="dir of dumped .npy weights (needed only for --golden)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=2048, help="compiled KV window S")
    ap.add_argument("--base", type=int, default=0,
                    help="absolute position of the chunk's first token; the ELF is constant "
                         "across chunks, so this only shapes the emitted inputs and golden")
    ap.add_argument("--causal", default="rows", choices=("rows", "none"),
                    help="`rows` (default) is true causality via per-row softmax widths; `none` "
                         "is the non-causal A/B control -- see the module docstring")
    ap.add_argument("--decode-meta", default=None,
                    help="decode artifact meta.json to pin the shared scratch arena against; "
                         "omit (or --no-arena-share) to build a standalone arena")
    ap.add_argument("--no-arena-share", action="store_true")
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--layout-only", action="store_true",
                    help="stop after the buffer layout + shared-arena assert; no aiecc, no ELF")
    a = ap.parse_args()

    sp = SPECS[a.spec]
    dec_meta_path = None if a.no_arena_share else a.decode_meta
    if dec_meta_path and not os.path.isfile(dec_meta_path):
        raise SystemExit(f"ERROR: --decode-meta {dec_meta_path} does not exist")
    if dec_meta_path:
        dm = json.load(open(dec_meta_path))
        # kv_block is checked because its ABSENCE from this list is what let a blocked decode and
        # a flat prefill share an arena, agree on every offset, and disagree about what the bytes
        # in it mean -- caught only on device, as one token repeated.
        dkb = dm["dims"].get("kv_block")
        if dkb and a.batch % dkb:
            raise SystemExit(f"ERROR: decode artifact kv_block={dkb} does not divide --batch "
                             f"{a.batch}; the KV append would straddle a block boundary")
        for key, ours in (("S", a.seq), ("layers", a.layers), ("d_model", sp.d_model),
                          ("head_dim", sp.head_dim), ("kv_heads", sp.n_kv_heads)):
            theirs = dm["dims"][key]
            if key == "layers":
                if a.layers > theirs:
                    raise SystemExit(f"ERROR: --layers {a.layers} exceeds the decode artifact's "
                                     f"{theirs}; the shared arena has no L{theirs}+ buffers")
                continue
            if theirs != ours:
                raise SystemExit(f"ERROR: decode artifact {key}={theirs}, this build {ours} -- "
                                 f"the shared cache/weight buffers would not match")

    sp_, fused, dims = build_graph(a.spec, a.layers, a.batch, a.seq, a.causal,
                                   dec_meta_path, do_compile=not a.layout_only)
    if dec_meta_path and a.weights:
        check_shared_weights(dec_meta_path, a.weights, sp, dims)
    M, S, NL = dims["M"], dims["S"], dims["NL"]
    if a.layout_only:
        in_sz, out_sz, scr = fused.buffer_sizes
        print(f"[layout] {NL} layers, M={M} S={S}: {dims['runlist_len']} runlist entries "
              f"({dims['per_layer']}/layer), in {in_sz}B out {out_sz}B scratch {scr/1e6:.1f}MB; "
              f"{len(dims['shared'])} shared buffers verified against the decode arena"
              if dec_meta_path else
              f"[layout] {NL} layers, M={M} S={S}: {dims['runlist_len']} runlist entries, "
              f"{dims['n_designs']} designs, scratch {scr/1e6:.1f}MB (standalone arena)")
        print(f"[layout] {dims['n_designs']} designs built; "
              f"{dims['n_configures']} CONFIGURES paid "
              f"({dims['n_configures']/NL:.1f}/layer) -- the unit D009 prices")
        op_census(dims["rl"], fused.get_layout_for_buffer, NL)
        return
    D, FF, HD = sp.d_model, sp.ffn, sp.head_dim
    Hq, Hkv, QD = sp.n_q_heads, sp.n_kv_heads, sp.q_dim

    rng = np.random.default_rng(11)
    X = bf16(rng.standard_normal((M, D)).astype(np.float32) * 0.02)
    table = rope_table(a.base, M, HD, sp.rope_theta_global)

    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    bdir = os.path.join(a.out, "buffers")
    open(os.path.join(bdir, "x.bin"), "wb").write(X.tobytes())
    open(os.path.join(bdir, "rope.bin"), "wb").write(table.tobytes())
    if dims["sm_widths"]:
        widths = causal_widths(a.base, M, S, Hq)
        want = fused.get_layout_for_buffer(SM_WIDTHS)[2]
        if widths.nbytes != want:
            raise SystemExit(f"ERROR: {SM_WIDTHS} is {widths.nbytes}B here and {want}B in the "
                             f"layout -- AIERuntimeArgSpec.dtype defaults to bfloat16, so this is "
                             f"what an unset dtype looks like")
        open(os.path.join(bdir, f"{SM_WIDTHS}.bin"), "wb").write(widths.tobytes())

    golden_files, gate = {}, None
    if not a.no_golden:
        if not a.weights:
            raise SystemExit("ERROR: --weights is required unless --no-golden")
        xout, slabs = golden(sp, a.weights, NL, M, S, a.base, a.causal, X, table, bf16)
        gdir = os.path.join(bdir, "golden")
        os.makedirs(gdir, exist_ok=True)
        open(os.path.join(gdir, "xout.bin"), "wb").write(xout.tobytes())
        golden_files["xout"] = "buffers/golden/xout.bin"
        for l, (kslab, vslab) in enumerate(slabs):
            for tag, arr in (("kc", kslab), ("vc", vslab)):
                fn = f"L{l}_{tag}_slab.bin"
                open(os.path.join(gdir, fn), "wb").write(np.asarray(arr, BF16).tobytes())
                golden_files[f"L{l}_{tag}"] = f"buffers/golden/{fn}"
        # The Tier 1 reference: SECOND pass, same inputs, nothing narrowed in between. A second
        # full CPU forward is the honest price -- reusing the bf16 pass's intermediates would make
        # the two references share exactly the rounding the gate is meant to see.
        x32, slab32 = golden(sp, a.weights, NL, M, S, a.base, a.causal, X, table, f32)
        refs = {"xout": np.asarray(x32, np.float32).reshape(M, D)}
        floors = {"xout": np.asarray(xout, np.float32).reshape(M, D)}
        for l, ((kslab, vslab), (kf, vf)) in enumerate(zip(slab32, slabs)):
            refs[f"L{l}_kc"] = np.asarray(kslab, np.float32).reshape(Hkv, M, HD)
            refs[f"L{l}_vc"] = np.asarray(vslab, np.float32).reshape(Hkv, M, HD)
            floors[f"L{l}_kc"] = np.asarray(kf, np.float32).reshape(Hkv, M, HD)
            floors[f"L{l}_vc"] = np.asarray(vf, np.float32).reshape(Hkv, M, HD)
        gate = gate_block(a.out, refs, floors)

    elf = load_elf(fused).view(np.uint8).tobytes()
    open(os.path.join(a.out, "prefill.elf"), "wb").write(elf)
    in_sz, out_sz, scr = fused.buffer_sizes

    scratchpad_params = {}
    pp = sorted(glob.glob("**/params.txt", recursive=True), key=os.path.getmtime)
    if pp:
        shutil.copy(pp[-1], os.path.join(a.out, "params.txt"))
        for line in open(pp[-1]).read().splitlines()[1:]:
            if line.strip():
                n_, idx, ty, kind = line.split()
                scratchpad_params[n_] = {"byte_offset": int(idx) * 4, "kind": kind, "dtype": ty}

    lay_names = [*dims["inputs"], "xout", *dims["shared"], *dims["prefill_local"]]
    lay = {n: fused.get_layout_for_buffer(n) for n in lay_names}
    dec_ref = None
    if dec_meta_path:
        import hashlib
        elf_path = os.path.join(os.path.dirname(dec_meta_path),
                                json.load(open(dec_meta_path))["elf"])
        md5 = (hashlib.md5(open(elf_path, "rb").read()).hexdigest()
               if os.path.isfile(elf_path) else None)
        dec_ref = {"meta": os.path.abspath(dec_meta_path), "elf_md5": md5,
                   "reserved_bytes": dims["reserved"]}

    byte_classes = operand_bytes(dims["rl"], fused.get_layout_for_buffer, set(dims["shared"]))
    meta = {
        "spec": sp.name, "elf": "prefill.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])}
                   for n, v in lay.items()},
        "inputs": dims["inputs"], "output": "xout",
        "weights": dims["shared"],
        # No weight .bin files are emitted: the bytes ARE decode's, at decode's offsets.
        "weights_from": (os.path.join(os.path.dirname(os.path.abspath(dec_meta_path)), "buffers")
                         if dec_meta_path else None),
        "cache_buffers": dims["cache_names"],
        "arena_shared": bool(dec_meta_path),
        "decode_artifact": dec_ref,
        "causal": dims["causal"] == "rows",
        "causal_mode": dims["causal"],
        # Everything the host needs to fill the mask. dtype is stated because it is the one field
        # that is silently wrong when omitted: AIERuntimeArgSpec defaults to bfloat16 and the
        # buffer layout is sized off it, so an unset dtype under-allocates this by 2x.
        "mask_widths": None if dims["causal"] != "rows" else {
            "buffer": SM_WIDTHS,
            "dtype": "int32",
            "rows": dims["sm_rows"],
            "len": dims["sm_rows"] * 4,
            "row_index": "r = h * M + i, h in [0, q_heads), i in [0, M)",
            "rule": "widths[h*M + i] = clamp(base + i + 1, 1, S)",
            "note": "one width per softmax row; row r attends scores[r, :widths[r]] and "
                    "mask_bf16 writes -inf over the rest. This IS the causal mask -- there is no "
                    "triangle buffer and no scalar width.",
        },
        "scratchpad": {
            "params": scratchpad_params,
            "kv_param": "kv_off",
            # No scalar causal width in either arm: `rows` streams a per-row vector instead, and
            # `none` masks nothing at all.
            "mask_param": None,
            "head_dim": HD, "kv_heads": Hkv,
        },
        "dims": {"layers": NL, "M": M, "S": S, "d_model": D, "ffn": FF,
                 "q_heads": Hq, "kv_heads": Hkv, "head_dim": HD, "q_dim": QD,
                 # Spelled exactly as decode spells it, and load-bearing rather than descriptive:
                 # the Rust host reads `dims.kv_block` off THIS artifact for the per-chunk kv_off,
                 # and an artifact that omits it is read as FLAT (artifact.rs defaults it to
                 # max_seq). A prefill built blocked but silent about it primes the right values at
                 # flat addresses, which no gate below the token can see. This dict is a hand-
                 # written literal, not `dims` above -- adding a key there does not reach here.
                 "kv_block": dims["kv_block"],
                 "tiles": dims["tiles"],
                 "tile_sources": sorted({v["source"] for v in dims["tiles"].values()}),
                 "tile_n_scores": dims["tn_sc"],
                 "tile_n_ctx": dims["tn_cx"], "cols": dims["cols"],
                 "runlist": dims["runlist_len"], "runlist_per_layer": dims["per_layer"],
                 "designs": dims["n_designs"], "configures": dims["n_configures"]},
        "host_protocol": {
            "batch": M,
            "x": f"[{M}, {D}] bf16 token-major embeddings for this chunk "
                 f"(embed_scale={sp.embed_scale})",
            "rope": f"[{M}, {HD}] bf16, one row per absolute position base..base+{M}-1, "
                    f"INTERLEAVED [cos, sin, cos, sin, ...], theta="
                    f"{sp.rope_theta_global}",
            "kv_off": "base * head_dim, element units, addr kind, written raw",
            SM_WIDTHS: (f"[{dims['sm_rows']}] int32 = q_heads({Hq}) * M({M}), row r = h*M + i "
                        f"holding clamp(base + i + 1, 1, {S}); a plain input-arena write, not a "
                        f"scratchpad parameter" if dims["causal"] == "rows" else None),
            "xout": f"[{M}, {D}] bf16 hidden states after {NL} layers; prefill emits NO logits",
            "attn_scale_folded_into": "L*_n_qn (decode's SCALE_IN_QNORM); do NOT apply it again",
            "chunking": f"pad the final chunk to M={M}; pad tokens sit at the END so their KV "
                        f"lands past n_past, where the decode mask already kills it",
            "sync": "kc/vc are written by the device; sync the scratch arena to host before the "
                    "M=1 step reads them",
        },
        "golden": golden_files or None,
        "gate": gate,
        "golden_layout": {"xout": [M, D],
                          "kv_slab": [Hkv, M, HD],
                          "note": "a kv slab is cache[h, base:base+M, :] for every kv head h; "
                                  "the rest of the cache is untouched"},
        "bytes": byte_classes,
        "macs": int(NL * (M * D * (QD + 2 * sp.kv_dim) + M * QD * D + 3 * M * D * FF
                          + 2 * Hq * M * S * HD)),
        "limitations": ([] if dims["causal"] == "rows" else [
            "NOT causal: --causal none attends the whole compiled window including the zeroed "
            "tail of the cache. The KV of layer 0 is still correct (K/V are appended before any "
            "softmax runs); the KV of layers >= 1 is not, because it is computed from a "
            "non-causal layer-0 output. This arm is the A/B control, not a seed for a decode.",
        ]) + ([
            f"KV cache is blocked (T={dims['kv_block']}): the scores/ctx GEMMs address it in "
            "place, "
            "through a rewritten B descriptor (GEMM's b_block_rows/b_block_stride), so blocking "
            "costs no extra bytes here -- the discarded alternative, gathering it back flat, cost "
            f"{2 * NL * (Hkv * S * HD * 2) / 2**20:.1f} MB per prefill of zero-compute DMA.",
        ] if dims["kv_block"] != S else []) + [
            "The two head-axis rearranges cost 4.0 MB/layer at M=256 of zero-compute DMA.",
            f"The widths cost a THIRD shim DMA channel per softmax core ({dims['cols']} here; "
            "read off the generated MLIR: one BD per core, sm_rows/cores int32 each, not one "
            "transfer per row). Measured in op.py at 8 cores OK / 16 failing for DMA capacity, so "
            "this shape has no headroom left in that dimension -- a wider softmax needs the "
            "broadcast-one-buffer form that file names. On-core it is one objectFIFO "
            "acquire/release per row; unmeasured.",
        ],
    }
    from gen_llm_decode import toolchain_provenance
    prov = toolchain_provenance()
    if prov:
        meta["toolchain"] = prov
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[ok] {NL}-layer {sp.name} prefill ELF ({len(elf)}B), M={M} S={S} "
          f"causal={dims['causal']}, {dims['runlist_len']} runlist entries "
          f"({dims['per_layer']}/layer), scratch {scr/1e6:.1f} MB "
          f"({'shared with ' + os.path.basename(os.path.dirname(dec_meta_path)) if dec_meta_path else 'standalone'})"
          f" -> {a.out}")


if __name__ == "__main__":
    main()
