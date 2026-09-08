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
import json
import os
import shutil
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
from iron.operators.rope.op import RoPE  # noqa: E402
from iron.operators.softmax.op import Softmax  # noqa: E402
from iron.operators.silu.op import SiLU  # noqa: E402
from iron.operators.gelu.op import GELU  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402
from iron.operators.elementwise_add.op import ElementwiseAdd  # noqa: E402
from iron.operators.strided_copy.op import StridedCopy  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = int(os.environ.get("PREFILL_COLS", "8"))
TILE_M = TILE_K = TILE_N = 64
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


def pick_tile_n(Nout, label, cols=COLS):
    """Largest tile_n with `Nout % (tile_n*cols) == 0` and `tile_n % 16 == 0` (mm.cc's real rule).

    K007: the shape is picked HERE, so the modulus is checked HERE, naming the offending number.
    """
    for tn in (64, 48, 32, 16):
        if tn % 16 == 0 and Nout % (tn * cols) == 0:
            return tn
    raise ValueError(f"{label}: Nout={Nout} admits no tile_n in (64,48,32,16) with "
                     f"Nout % (tile_n*{cols}) == 0 and tile_n % 16 == 0")


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
    tn_sc, tn_cx = pick_tile_n(S, "scores"), pick_tile_n(HD, "ctx")
    sp.check_prefill_projections(
        M, (("q", D, QD), ("k", D, KVD), ("v", D, KVD), ("o", QD, D),
            ("gate", D, FF), ("up", D, FF), ("down", FF, D)),
        tile_m=TILE_M, tile_k=TILE_K, tile_n=TILE_N, cols=cols)
    sp.check_prefill_projections(
        M, (("scores", HD, S), ("ctx", S, HD)),
        tile_m=TILE_M, tile_k=TILE_K, tile_n=TILE_N, cols=cols,
        tile_n_overrides={"scores": tn_sc, "ctx": tn_cx})
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
    gemm_kw = dict(tile_m=TILE_M, tile_k=TILE_K, num_aie_columns=cols, context=ctx)
    op_norm = RMSNorm(size=M * D, num_aie_columns=cols, num_channels=1, tile_size=D,
                      weighted=True, epsilon=sp.eps, context=ctx)
    op_qn = RMSNorm(size=M * QD, num_aie_columns=cols, num_channels=1, tile_size=HD,
                    weighted=True, epsilon=sp.eps, context=ctx)
    op_kn = RMSNorm(size=M * KVD, num_aie_columns=cols, num_channels=1, tile_size=HD,
                    weighted=True, epsilon=sp.eps, context=ctx)
    op_rq = RoPE(rows=M * Hq, cols=HD, angle_rows=M, num_aie_columns=cols, context=ctx)
    op_rk = RoPE(rows=M * Hkv, cols=HD, angle_rows=M, num_aie_columns=cols, context=ctx)
    op_gq = GEMM(M=M, K=D, N=QD, tile_n=TILE_N, b_col_maj=True, **gemm_kw)
    op_gkv = GEMM(M=M, K=D, N=KVD, tile_n=TILE_N, b_col_maj=True, **gemm_kw)
    op_o = GEMM(M=M, K=QD, N=D, tile_n=TILE_N, b_col_maj=True, **gemm_kw)
    op_gu = GEMM(M=M, K=D, N=FF, tile_n=TILE_N, b_col_maj=True, **gemm_kw)
    op_down = GEMM(M=M, K=FF, N=D, tile_n=TILE_N, b_col_maj=True, **gemm_kw)
    # scores: B is the kv cache stored [S, HD], i.e. [N, K] -> read b_col_maj.
    # ctx:    B is the same cache read as [K=S, N=HD] -> plain.
    op_sc = GEMM(M=M, K=HD, N=S, tile_n=tn_sc, b_col_maj=True, **gemm_kw)
    op_cx = GEMM(M=M, K=S, N=HD, tile_n=tn_cx, b_col_maj=False, **gemm_kw)
    # ONE softmax over every head's rows at once: its `rows` axis is just "independent rows to
    # normalise", and every head's [M, S] block is a contiguous slice of the same buffer. That is
    # also what makes the causal mask a plain vector: row Hq*M is (head, token) flattened, and the
    # width depends only on the token half.
    sm_kw = dict(vector_size_source="rows") if causal == "rows" else {}
    op_sm = Softmax(rows=Hq * M, cols=S, num_aie_columns=cols, num_channels=1,
                    context=ctx, **sm_kw)
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
    op_kvapp = StridedCopy(
        input_sizes=(M, Hkv, HD), input_strides=(Hkv * HD, HD, 1), input_offset=0,
        output_sizes=(M, Hkv, HD), output_strides=(HD, S * HD, 1), output_offset=0,
        input_buffer_size=M * Hkv * HD, output_buffer_size=Hkv * S * HD,
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
    prefill_local = sorted(bufsz)
    dec_meta, dec_order, dec_sizes, dec_reserved = (None, [], {}, 0)
    if dec_meta_path:
        dec_meta, dec_order, dec_sizes, dec_reserved = decode_arena_plan(dec_meta_path)
        for name, length in dec_sizes.items():
            if name in bufsz:
                raise ValueError(f"decode scratch name {name!r} collides with a prefill "
                                 f"intermediate of the same name")
            bufsz[name] = length

    rl, cache_names = [], []
    for l in range(NL):
        p = f"L{l}_"
        src = "x" if l == 0 else "xs"
        dst = "xout" if l == NL - 1 else "xs"
        wq = f"{p}Wqkv[0:{QD * D * 2}]"
        wk = f"{p}Wqkv[{QD * D * 2}:{(QD + KVD) * D * 2}]"
        wv = f"{p}Wqkv[{(QD + KVD) * D * 2}:{(QD + 2 * KVD) * D * 2}]"
        # Decode pads Wo by two rows for swiglu_mlp_dp's fuse_o tiling; the projection is the
        # first D rows and the pad rows are zero, so prefill reads the unpadded prefix.
        wo = f"{p}Wo[0:{D * QD * 2}]"
        rl += [
            (op_norm, src, p + "n_in", "h"),
            (op_gq, "h", wq, "q"),
            (op_gkv, "h", wk, "k"),
            (op_gkv, "h", wv, "v"),
        ]
        rl += [
            (op_qn, "q", p + "n_qn", "q"),
            (op_kn, "k", p + "n_kn", "k"),
            (op_rq, "q", "rope", "q"),
            (op_rk, "k", "rope", "k"),
            # K after qk-norm AND after RoPE; V raw, projection only. Different points in the
            # pipeline, and the M=1 path a decode step resumes from depends on both.
            (op_kvapp, "k", p + "kc"),
            (op_kvapp, "v", p + "vc"),
            (op_q2h, "q", "qh"),
        ]
        for h in range(Hq):
            kv = h // grp
            rl.append((op_sc, f"qh[{h * M * HD * 2}:{(h + 1) * M * HD * 2}]",
                       f"{p}kc[{kv * S * HD * 2}:{(kv + 1) * S * HD * 2}]",
                       f"sc[{h * M * S * 2}:{(h + 1) * M * S * 2}]"))
        # The widths buffer is an INPUT of the softmax step, not a side channel: op.get_arg_spec()
        # puts it between in and out, so it is the middle name here.
        rl.append((op_sm, "sc", SM_WIDTHS, "sw") if causal == "rows"
                  else (op_sm, "sc", "sw"))
        for h in range(Hq):
            kv = h // grp
            rl.append((op_cx, f"sw[{h * M * S * 2}:{(h + 1) * M * S * 2}]",
                       f"{p}vc[{kv * S * HD * 2}:{(kv + 1) * S * HD * 2}]",
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
    name = f"prefill_{sp.name.replace('-', '_').replace('.', '_')}_m{M}_s{S}_l{NL}_c{cols}"
    if causal != "none":
        name += f"_{causal}"
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
                tn_sc=tn_sc, tn_cx=tn_cx, cols=cols, causal=causal,
                sm_widths=(SM_WIDTHS if causal == "rows" else None), sm_rows=Hq * M,
                shared=[n for n in dec_order if not n.startswith("__decode_gap")],
                reserved=dec_reserved, prefill_local=prefill_local,
                rl=rl, runlist_len=len(rl), per_layer=len(rl) // NL)
    return sp, fused, dims


def operand_bytes(runlist, resolve, shared):
    """Bytes the graph touches, per class, COMPUTED off the runlist instead of a hand formula.

    A floor, not a measurement: one pass per named operand, so it ignores the broadcast of A to
    every column and any MemTile reuse, in both directions. Its job is to let a device timing be
    read as GB/s without the caller re-deriving the byte count, and to keep the three terms
    separable -- weights are M-independent, activations are linear in M, and the score matrix
    grows with the compiled WINDOW rather than with the batch, which is the term the MLP block
    could not exercise.

    StridedCopy is counted off its own access pattern: its operand is the whole [Hkv, S, HD]
    cache while the descriptor moves M rows of it.

    Slice lengths are parsed from the name rather than taken from `get_layout_for_buffer`, whose
    third element is a LENGTH for a plain buffer and an absolute END offset for a slice
    (iron/common/sequence.py: `return buf_type, parent_start + start, parent_start + end`).
    Reading it as a length here made a 40 MB layer read as 93 GB.
    """
    out = {"weights": 0, "cache": 0, "scores": 0, "activations": 0}

    def klass(name):
        base = name.split("[")[0]
        if base.endswith("_kc") or base.endswith("_vc"):
            return "cache"
        if base in ("sc", "sw"):
            return "scores"
        return "weights" if base in shared else "activations"

    def length(name):
        if "[" in name:
            lo, hi = name[name.index("[") + 1:-1].split(":")
            return int(hi) - int(lo)
        return int(resolve(name)[2])

    for op, *bufs in runlist:
        moved = int(np.prod(op.input_sizes)) * 2 if isinstance(op, StridedCopy) else None
        for b in bufs:
            out[klass(b)] += moved if moved is not None else length(b)
    return out


# --------------------------------------------------------------------------------------------
# CPU golden -- the same bf16 dataflow, rounded where the device rounds.
# --------------------------------------------------------------------------------------------
def _rms(v, w, eps):
    f = np.asarray(v, np.float32)
    s = f / np.sqrt((f * f).mean(-1, keepdims=True) + eps)
    return bf16(s * np.asarray(w, np.float32))


def _mm(a, b_t):
    """bf16(A @ B^T) with B stored [Nout, K] -- the b_col_maj read the device does."""
    return bf16(np.asarray(a, np.float32) @ np.asarray(b_t, np.float32).T)


def _rope_block(x, table, heads):
    """RoPE in design.py's BLOCK convention: token t's angle row covers `heads` consecutive rows.

    x is [M, heads, HD] token-major; `table` is [M, HD] interleaved [cos, sin, ...].
    """
    f = np.asarray(x, np.float32)
    ang = np.asarray(table, np.float32)
    cos, sin = ang[:, 0::2][:, None, :], ang[:, 1::2][:, None, :]
    half = f.shape[-1] // 2
    x1, x2 = f[..., :half], f[..., half:]
    return bf16(np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1))


def _softmax_rows(s, widths):
    """Row-wise softmax of `[rows, cols]`, with one unmasked width per row.

    `widths` is None (attend everything) or a `[rows]` int vector: row `i` is softmaxed over
    `s[i, :widths[i]]` and the tail is zero. That models `mask_bf16` writing -inf past the width
    and `softmax_bf16` then exponentiating the whole row -- the MASK, not the device's bf16
    rounding, and it assumes aie::exp2 returns exactly 0 at -inf.
    """
    f = np.asarray(s, np.float32)
    if widths is not None:
        keep = np.arange(f.shape[-1])[None, :] < np.asarray(widths, np.int64)[:, None]
        f = np.where(keep, f, -np.inf)
    e = np.exp(f - f.max(-1, keepdims=True))
    e = np.nan_to_num(e, nan=0.0)
    return bf16(e / e.sum(-1, keepdims=True))


def golden(sp, weights_dir, NL, M, S, base, causal, X, table):
    """Full-stack CPU golden. Returns (xout, [(kc_slab, vc_slab) per layer]).

    kc/vc slabs are [Hkv, M, HD] -- the rows this chunk writes, which is what a host gate compares
    against `cache[h, base:base+M, :]`. The rest of the cache is untouched and still zero.
    """
    D, FF, HD = sp.d_model, sp.ffn, sp.head_dim
    Hq, Hkv, grp = sp.n_q_heads, sp.n_kv_heads, sp.gqa_group
    # The device is handed `[Hq*M]`; a head's block is `[M, S]` and every head shares the same M
    # widths, so slicing the first M off the flat vector is the same thing and keeps the two
    # derivations from drifting.
    widths = causal_widths(base, M, S, Hq)[:M] if causal == "rows" else None

    def npy(n):
        return np.load(os.path.join(weights_dir, f"{n}.npy")).astype(np.float32)

    x = np.asarray(X, np.float32)
    slabs = []
    for l in range(NL):
        pre = f"model.layers.{l}."
        n_in = bf16(npy(pre + "input_layernorm.weight"))
        n_pf = bf16(npy(pre + "post_attention_layernorm.weight"))
        # attn_scale rides on the q-norm gain, because that is how the SHARED decode buffer stores
        # it (SCALE_IN_QNORM). Applying it again here would double-scale against the device.
        n_qn = bf16(npy(pre + "self_attn.q_norm.weight") * sp.attn_scale)
        n_kn = bf16(npy(pre + "self_attn.k_norm.weight"))
        Wq = bf16(npy(pre + "self_attn.q_proj.weight"))
        Wk = bf16(npy(pre + "self_attn.k_proj.weight"))
        Wv = bf16(npy(pre + "self_attn.v_proj.weight"))
        Wo = bf16(npy(pre + "self_attn.o_proj.weight"))
        Wg = bf16(npy(pre + "mlp.gate_proj.weight"))
        Wu = bf16(npy(pre + "mlp.up_proj.weight"))
        Wd = bf16(npy(pre + "mlp.down_proj.weight"))

        h = _rms(x, n_in, sp.eps)
        q = _mm(h, Wq).reshape(M, Hq, HD)
        k = _mm(h, Wk).reshape(M, Hkv, HD)
        v = _mm(h, Wv).reshape(M, Hkv, HD)
        q = _rms(q, n_qn, sp.eps)
        k = _rms(k, n_kn, sp.eps)
        q = _rope_block(q, table, Hq)
        k = _rope_block(k, table, Hkv)
        kc = np.zeros((Hkv, S, HD), np.float32)
        vc = np.zeros((Hkv, S, HD), np.float32)
        kc[:, base:base + M] = np.asarray(k, np.float32).transpose(1, 0, 2)
        vc[:, base:base + M] = np.asarray(v, np.float32).transpose(1, 0, 2)
        slabs.append((bf16(kc[:, base:base + M]), bf16(vc[:, base:base + M])))
        cx = np.empty((Hq, M, HD), np.float32)
        for hh in range(Hq):
            kv = hh // grp
            s = _mm(np.asarray(q, np.float32)[:, hh], bf16(kc[kv]))
            p = _softmax_rows(s, widths)
            cx[hh] = np.asarray(bf16(np.asarray(p, np.float32) @ vc[kv]), np.float32)
        cxt = cx.transpose(1, 0, 2).reshape(M, Hq * HD)
        a = _mm(cxt, Wo)
        x1 = bf16(x + np.asarray(a, np.float32))
        hf = _rms(x1, n_pf, sp.eps)
        g = np.asarray(_mm(hf, Wg), np.float32)
        u = np.asarray(_mm(hf, Wu), np.float32)
        # exp(-g) overflows f32 below g = -88, which a 28-layer stack reaches; g/inf is -0.0,
        # the right limit, so the warning is the only thing to suppress. NOT rewritten to a
        # two-sided sigmoid: that is the same function but a different f32 rounding, and it moved
        # this golden away from gen_llm_prefill_mlp.py's, which is the one device-gated at M=256.
        with np.errstate(over="ignore"):
            gs = np.asarray(bf16(g / (1.0 + np.exp(-g))) if sp.act == "silu"
                            else bf16(0.5 * g * (1.0 + np.tanh(0.7978845608 *
                                                               (g + 0.044715 * g ** 3)))),
                            np.float32)
        gh = bf16(gs * u)
        d = _mm(gh, Wd)
        x = np.asarray(bf16(np.asarray(x1, np.float32) + np.asarray(d, np.float32)), np.float32)
    return bf16(x), slabs


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
    M, S, NL = dims["M"], dims["S"], dims["NL"]
    if a.layout_only:
        in_sz, out_sz, scr = fused.buffer_sizes
        print(f"[layout] {NL} layers, M={M} S={S}: {dims['runlist_len']} runlist entries "
              f"({dims['per_layer']}/layer), in {in_sz}B out {out_sz}B scratch {scr/1e6:.1f}MB; "
              f"{len(dims['shared'])} shared buffers verified against the decode arena"
              if dec_meta_path else
              f"[layout] {NL} layers, M={M} S={S}: {dims['runlist_len']} runlist entries, "
              f"scratch {scr/1e6:.1f}MB (standalone arena)")
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

    golden_files = {}
    if not a.no_golden:
        if not a.weights:
            raise SystemExit("ERROR: --weights is required unless --no-golden")
        xout, slabs = golden(sp, a.weights, NL, M, S, a.base, a.causal, X, table)
        gdir = os.path.join(bdir, "golden")
        os.makedirs(gdir, exist_ok=True)
        open(os.path.join(gdir, "xout.bin"), "wb").write(xout.tobytes())
        golden_files["xout"] = "buffers/golden/xout.bin"
        for l, (kslab, vslab) in enumerate(slabs):
            for tag, arr in (("kc", kslab), ("vc", vslab)):
                fn = f"L{l}_{tag}_slab.bin"
                open(os.path.join(gdir, fn), "wb").write(np.asarray(arr, BF16).tobytes())
                golden_files[f"L{l}_{tag}"] = f"buffers/golden/{fn}"

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
                 "tile": [TILE_M, TILE_K, TILE_N], "tile_n_scores": dims["tn_sc"],
                 "tile_n_ctx": dims["tn_cx"], "cols": dims["cols"],
                 "runlist": dims["runlist_len"], "runlist_per_layer": dims["per_layer"]},
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
        ]) + [
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
