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
from types import SimpleNamespace

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import SPECS  # noqa: E402
from gemm_tile_registry import registry  # noqa: E402
import pack_cache  # noqa: E402
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
from iron.operators.softmax.design import SM_VEC_LEN  # noqa: E402
from iron.operators.silu.op import SiLU  # noqa: E402
from iron.operators.gelu.op import GELU  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402
from iron.operators.elementwise_add.op import ElementwiseAdd  # noqa: E402
from iron.operators.strided_copy.op import StridedCopy  # noqa: E402
from iron.common.quant import row_stride_bytes  # noqa: E402
from iron.operators.dequant_rows.op import DequantRows  # noqa: E402

# Decode's ACT_POLY, mirrored: SiLU and sigmoid in f32 polynomial math instead of the SFU tanh LUT.
# The two halves must agree, or prefill and decode compute different activations over one cache.
ACT_POLY = os.environ.get("ACT_POLY", "0") == "1"
# The same functions in native bf16 MACs over two-limb operands (act_poly's *Fast), prefill only.
ACT_FAST = ACT_POLY and os.environ.get("PREFILL_ACT_FAST", "0") == "1"
if ACT_FAST:
    from iron.operators.act_poly.op import SiLUFast as SiLUAct, SigmoidFast as SigmoidAct  # noqa: E402
elif ACT_POLY:
    from iron.operators.act_poly.op import SiLUPoly as SiLUAct, SigmoidPoly as SigmoidAct  # noqa: E402
else:
    SiLUAct = SiLU
    from iron.operators.sigmoid.op import Sigmoid as SigmoidAct  # noqa: E402

BF16 = ml_dtypes.bfloat16

# Cut the layer stack into this many DISPATCH VARIANTS inside ONE ELF. 1 (the default) is the
# single control code every build had, byte-for-byte. This is the prefill twin of
# gen_llm_decode.py's DECODE_SEGMENTS and is cheaper than it: decode gives each segment its own
# OperatorSequence and its own arena, while `extra_runlists` puts every variant in one arena
# computed over their union (sequence.py::calculate_buffer_layout walks `self.runlists.values()`),
# so the residual seam stays SCRATCH and needs no arg on either side. `xs` is already one shared
# buffer across prefill layers, so a cut needs no renaming either.
#
# What it buys is the driver's 2 s TDR watchdog (aie2_tdr.c). A 48-layer M=256 dispatch moves
# 39.94 GiB and cannot finish inside it; four variants of 12 layers move 9.99 GiB each.
PREFILL_SEGMENTS = int(os.environ.get("PREFILL_SEGMENTS", "1"))

# Hold the GEMM's A operand in L2 across its N loop, instead of re-reading it from DDR
# `N//(tile_n*cols)` times (iron/operators/gemm/design.py's `pattern_repeat`). The operator falls
# back per site when A+B+C do not co-fit a 512 KiB MemTile, or when there is no N loop to hoist.
# Opt-in, default OFF: it removes 3.92 GiB of 16.00 at M=64 and costs 0.00% in time, measured over
# 10 alternated pairs (-0.38%, CI [-0.86, +0.11]). An energy lever, not a latency one -- the DDR
# fetch it deletes sits upstream of L2, and the L2->L1 stream the cores actually consume is
# byte-identical either way, so it buys nothing here and stays off until that changes.
A_RESIDENT = os.environ.get("A_RESIDENT", "0") == "1"
if PREFILL_SEGMENTS < 1:
    raise SystemExit(f"PREFILL_SEGMENTS={PREFILL_SEGMENTS} must be >= 1")
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
# Per delta-rule call, the real tokens it steps: an int32 in each call's dk-wide bf16 slot.
GDR_COUNT = "gdr_count"
# aie.dma_bd's stride range, [1:1048576] elements: a 20-bit field of the shim BD.
BD_STRIDE_MAX = 1 << 20
# The causal mask, as a buffer name. One int32 per softmax row, host-written per chunk.
SM_WIDTHS = "sm_widths"
# Batch prefill past a sliding-window's circular KV ring. Default 0: byte-identical MLIR to
# before this existed. A sliding (w < S) geometry then goes read-first over [ring W | chunk M]
# instead of append-then-read, which is the only order safe past the wrap for M > 1: today's
# append-then-read destroys ring data before later rows in the same chunk can read it.
SLIDING_RING = os.environ.get("PREFILL_SLIDING_RING", "0") == "1"
# The ring mask, one (hole_lo, hole_hi, width) int32 triple per softmax row -- SM_WIDTHS's sibling
# for the rows_hole softmax mode.
SM_RING = "sm_ring"


def bf16(a):
    return np.asarray(a).astype(BF16)


def rope_table(base, rows, head_dim, theta, partial=None):
    """[rows, head_dim] bf16 angle table for absolute positions base..base+rows-1.

    Same derivation and the same INTERLEAVED [cos, sin, cos, sin, ...] packing as
    verify_llm_decode.rope_row, one row per position instead of one row per dispatch. NOT the
    half-split [cos..., sin...] packing mlir-air's examples use.

    `partial` is `LlmSpec.rope_partial_rotary` for rope_type "proportional": zero the inverse
    frequency past `int(partial * head_dim // 2)` pairs, keeping the full head_dim width -- a zero
    frequency is the identity rotation. The exponent's denominator stays head_dim regardless (that
    is what makes it "proportional"). Same rule as verify_llm_decode.rope_row and
    rust/npu-engine/src/llm/npu_decode.rs::rope_row, batched over rows instead of one position.
    """
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64)[:half] / head_dim))
    if partial is not None:
        inv[int(partial * head_dim // 2):] = 0.0
    ang = np.arange(base, base + rows, dtype=np.float64)[:, None] * inv[None, :]
    t = np.empty((rows, head_dim), np.float32)
    t[:, 0::2] = np.cos(ang)
    t[:, 1::2] = np.sin(ang)
    return bf16(t)


def ring_mask_rows(base, m, w):
    """`[m, 3]` int32 (hole_lo, hole_hi, width) triples for a ring chunk at `base`, capacity `w`.

    Row i of a chunk at `base` attends the ring's tail `[hole_lo, hole_hi)` EXCLUDED plus its own
    batch column `[0, width)`. `b0 = base % w`: `hole_lo = b0`; `hole_hi = w` while `base < w`,
    else `b0 + i + 1`; `width = w + i + 1`. Independently derived (never copied) against Rust's
    `ring_mask_block` (`rust/npu-engine/src/llm/npu_prefill.rs`) -- pin the two against each other.
    """
    b0 = base % w
    i = np.arange(m, dtype=np.int64)
    hole_lo = np.full(m, b0)
    hole_hi = np.full(m, w) if base < w else b0 + i + 1
    return np.stack([hole_lo, hole_hi, w + i + 1], 1).astype(np.int32)


def causal_widths(base, M, S, heads):
    """The per-row unmasked widths for one chunk: `[heads*M]` int32, row `r = h*M + i`.

    Row `i` of a chunk starting at absolute position `base` attends positions `<= base + i`, so
    its width is `base + i + 1` -- the same under every head. Clamped to `[1, S]` because
    `mask_bf16` loops `for (j = width; j < cols; j++)` over the raw i32 it is handed: a width past
    `S` masks nothing, and a width of 0 leaves the row all -inf, whose softmax is NaN rather than
    a small number. The clamp at S is reachable -- the last chunk of a full window has
    `base + M - 1 == S - 1`, so its final width is exactly S.

    ONE clamp serves every geometry, including a narrowed (sliding) one whose own softmax runs at
    `cols = w < S`: while `base + M <= w` the width is `base + i + 1` either way, so the global
    clamp and a per-window `min(base + i + 1, w)` are the same vector. That inequality is not an
    assumption here -- `npu_prefill.rs::batchable_window` is what holds it, and it holds it because
    past `w` the valid slots stop being a prefix at all and no width, per-window or not, names
    them.
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


# ---- quantized weight sourcing (Task 5) -----------------------------------------------------
# gemm_for()'s registry tiling (tile_m/k/n, cols) comes from (M, K, N, emulate, prio_accuracy)
# alone, same lookup a plain bf16 GEMM at that shape gets -- weight_dtype/group_size are an
# ADDITIONAL axis GEMM validates against whichever tile that lookup picked (tile_k must be a
# whole number of groups; see iron/operators/gemm/op.py:_validate_weight_dtype), not a second
# tiling decision.
QUANT_TENSOR = {
    "Wq": "self_attn.q_proj.weight", "Wk": "self_attn.k_proj.weight",
    "Wv": "self_attn.v_proj.weight", "Wo": "self_attn.o_proj.weight",
    "Wg": "mlp.gate_proj.weight", "Wu": "mlp.up_proj.weight", "Wd": "mlp.down_proj.weight",
}


def read_quant_manifest(src_dir):
    """(dtype, group_size, scale_dtype) exactly as `src_dir`'s own quant.json records them -- read,
    never hardcoded, so a dump that changes its own group_size cannot silently drift from this file.

    `layout` and `scale_dtype` are read for the same reason, and `layout` is REFUSED rather than
    returned: `repack_gemm_weight` parses the row-packed form only (it takes no `layout` argument),
    so a `row_group_planar` dump -- where the scales sit outside the row -- would be read with
    payload bytes where the header belongs, silently, at every row. Every Gemma-4-12B quantized dump
    on this box is planar, which is the case that would hit it.
    """
    with open(os.path.join(src_dir, "quant.json")) as fh:
        m = json.load(fh)
    layout = m.get("layout", "header_first")
    if layout != "header_first":
        raise ValueError(
            f"{src_dir}/quant.json declares layout={layout!r}, which prefill's GEMM path cannot "
            f"read: iron.common.quant.repack_gemm_weight permutes the row-packed form and has no "
            f"{layout!r} parser. Re-dump this checkpoint at layout='header_first', or teach the "
            f"repack the layout -- do not point this build at it as-is")
    return m["dtype"], int(m["group_size"]), m.get("scale_dtype", "f32")


def build_quant_plan(quant_weights, quant_attn_o_weights):
    """{'qkv': (dtype, group, dir), 'mlp': (...), 'o': (...)}, or {} for the all-bf16 build every
    spec had before this axis existed.

    Two directories, not one: Gemma-4-12B's own mix is most sites at one group_size, `attn_o` at
    another, and neither dump states that mix on its own -- weights_int8g32/quant.json and
    weights_int8g64/quant.json each pack EVERY site at their own uniform group (verified
    2026-09-12), so the per-site MIX is which directory a site reads from, not a manifest field.
    """
    if not quant_weights:
        return {}
    dtype, group, sdt = read_quant_manifest(quant_weights)
    o_dir = quant_attn_o_weights or quant_weights
    o_dtype, o_group, o_sdt = ((dtype, group, sdt) if o_dir == quant_weights
                               else read_quant_manifest(o_dir))
    return {"qkv": (dtype, group, sdt, quant_weights), "mlp": (dtype, group, sdt, quant_weights),
            "o": (o_dtype, o_group, o_sdt, o_dir)}


def quant_source_files(src_dir, prefix, tensor):
    """The dumped file(s) for `{prefix}{tensor}`, in K order: `[]` if neither form exists, one
    plain `{tensor}.npy`, or decode's own `k_chunks_for` split (`.kchunk0.npy ..`) -- baked into
    the dump itself (down always, o only on a global/K=8192 layer; verified 2026-09-12: layer 5's
    `o_proj` is two kchunks, layer 0's is one plain file). Prefill's GEMM has no reason to
    replicate this split -- it tiles K internally -- but the dump only offers it this way, so a
    quantized o/down is packed per chunk, same as its full-K sibling is packed whole.
    """
    if os.path.exists(os.path.join(src_dir, f"{prefix}{tensor}.npy")):
        return [f"{prefix}{tensor}.npy"]
    files, n = [], 0
    while os.path.exists(os.path.join(src_dir, f"{prefix}{tensor}.kchunk{n}.npy")):
        files.append(f"{prefix}{tensor}.kchunk{n}.npy")
        n += 1
    return files


def pack_quant_weights(quant_pack, bdir, quant_plan):
    """Write `bdir/<buf>.bin` for every `quant_pack` entry (Task 5): unlike x/rope/sm_widths,
    these are STATIC model weights, packed ONCE, here, at build time. Each is a permutation of the
    same per-tensor dump decode's own GEVM reads (`iron.common.quant.repack_gemm_weight`; nothing
    is requantized), cut to the exact tile config the op that owns it was built with, so a
    mismatch between the two would be a coding error in this file, not a device numerics gap.

    Content-addressed cache (pack_cache.py): repack_gemm_weight is a pure function of the dump
    bytes + these params + the two files that decide them, so the same checkpoint + packing layout
    hits instead of re-permuting ~6.5 GB/build. PREFILL_PACK_CACHE=0 falls back to always-pack,
    byte-identical either way.
    """
    from iron.common import quant as quant_mod

    use_cache = pack_cache.cache_enabled()
    hits = misses = 0
    if use_cache:
        store = pack_cache.ContentStore(pack_cache.default_cache_dir())
        packer_digest = pack_cache.sha256_file(quant_mod.__file__)
        generator_digest = pack_cache.sha256_file(os.path.abspath(__file__))
        src_digests = {}
    for e in quant_pack:
        dest = os.path.join(bdir, f"{e['buf']}.bin")
        if use_cache:
            src_path = os.path.join(e["src_dir"], e["src_file"])
            if src_path not in src_digests:
                src_digests[src_path] = pack_cache.sha256_file(src_path)
            key = pack_cache.content_key(e, src_digests[src_path], packer_digest,
                                         generator_digest)
            if store.materialize(key, dest):
                hits += 1
                continue
            misses += 1
        row_packed = np.load(os.path.join(e["src_dir"], e["src_file"]))
        _, mmul_s, mmul_t = e["mmul"]
        packed = quant_mod.repack_gemm_weight(row_packed, e["N"], e["K"], e["tile_k"],
                                              e["tile_n"], e["group_size"], e["weight_dtype"],
                                              mmul_s, mmul_t, e["cols"],
                                              scale_dtype=e["scale_dtype"])
        packed.tofile(dest)
        if use_cache:
            store.put(key, dest)
    cache_note = f" (cache: {hits} hit, {misses} miss)" if use_cache else ""
    print(f"[gen] packed {len(quant_pack)} quantized weight buffer(s) into {bdir} "
          f"(sites: {sorted({s for s in quant_plan})}){cache_note}")


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


def build_graph(spec_name, NL, M, S, causal, dec_meta_path, cols=COLS, do_compile=True,
                quant_plan=None, kv_alloc=0):
    """Construct the fused prefill graph. Returns (spec, fused, dims).

    `do_compile=False` stops after the buffer layout, which is what the shared-arena assert needs
    -- seconds instead of an aiecc run, so the arena contract is checkable on every edit.

    `quant_plan` is `build_quant_plan()`'s `{site: (dtype, group_size, src_dir)}` map, or `None`
    for the all-bf16 graph every spec had before this axis existed.

    `kv_alloc` is decode's `KV_ALLOC` under its own name: the KV cache is ALLOCATED for that many
    positions while attention still computes over `S`. It exists so the two halves can share one
    cache when decode was built for a wide capacity -- `check_shared_layout_agrees` compares buffer
    LENGTHS, so a prefill sizing `kc`/`vc` by its own narrower window cannot bind to that arena at
    all. 0 means capacity is the window, byte for byte the pre-existing build.
    """
    quant_plan = quant_plan or {}
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
    KVA = kv_alloc or S
    if KVA < S:
        raise ValueError(f"--kv-alloc {KVA} < --seq {S}: it is the CAPACITY the cache is allocated "
                         f"for, never a window. Attention computes over --seq; a capacity under it "
                         f"would put the window's own positions past the end of the cache")
    if KVA % kv_T:
        raise ValueError(f"--kv-alloc {KVA} is not a whole number of kv_block={kv_T} blocks; the "
                         f"blocked layout addresses a partial trailing block at strides no "
                         f"consumer computes")
    kvl = KVLayout(Hkv=Hkv, S=KVA, HD=HD, T=kv_T)
    print(f"[gen] prefill KV layout: T={kv_T} capacity={KVA} window={S} "
          + ("(flat [Hkv,S,HD])" if kv_T == KVA else
             f"(blocked [S/T,Hkv,T,HD], head_stride={kvl.head_stride}, "
             f"block_stride={kvl.block_stride})"))

    # Per-geometry KV capacity, read off decode's OWN build rather than re-derived: a sliding
    # geometry (SLIDING_KV_CIRCULAR) is allocated at `sliding_window`, not S, and prefill's
    # KV-append must target the SAME narrower buffer or it walks past the end of it -- the exact
    # defect this pairs against (`check_prefill_pairing`'s `kv_windows` check).
    sliding_kv_circular = bool(dm_early["dims"].get("sliding_kv_circular")) if dm_early else False
    sliding_window = dm_early["dims"].get("sliding_window") if dm_early else None

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
    print(f"[gen] prefill Wqkv row order: " + ("head-major" if hm else "stock [Wq|Wk|Wv]"))

    # Every ATTENTION geometry this build touches: (head_dim, n_kv_heads, has_v_proj), one entry
    # per distinct triple over all NL layers. Uniform for every spec except Gemma-4-12B, whose
    # global layers are (512, 1, False) against the sliding (256, 8, True) -- has_v_proj is its own
    # axis (attention_k_eq_v), not derived from the other two, for the reason
    # gen_llm_decode.py:1262-1265 gives: a model where the two splits do not coincide would silently
    # mis-key on a two-part key.
    geoms = sorted({(sp.head_dim_for(l), sp.n_kv_heads_for(l), sp.has_v_proj(l))
                    for l in range(NL)})
    for hd, hkv, has_v in geoms:
        g_grp = Hq // hkv
        print(f"[gen] geometry head_dim={hd} n_kv_heads={hkv} has_v_proj={has_v}: "
              + (f"head-major, {g_grp}+{2 if has_v else 1} head blocks of {hd} rows per kv head"
                 if hm else "stock [Wq|Wk|Wv]"))

    if not sp.qk_norm:
        # Not a missing brick so much as a missing multiply: `attn_scale` rides on the shared
        # q-norm gain (decode's SCALE_IN_QNORM), so a spec without a q-norm has nowhere to put it
        # and would produce unscaled scores.
        raise ValueError(f"{sp.name}: no per-head q-norm, so attn_scale has no gain to ride on; "
                         f"this graph has no separate score scale")

    # ---- K007: every shape constraint asserted where the shape is picked, PER GEOMETRY ----
    # The GEMM tilings come from the registry below, at the point each GEMM is constructed --
    # `Registry.lookup` runs the same `gemm_tiling_rejection` these checks do, so a shape that
    # reaches an operator has already had its modulus, L1 and MemTile budgets named. What is left
    # here is everything that is NOT a GEMM.
    # RoPE. `rows = M*heads, angle_rows = M` makes design.py's block quotient exactly the head
    # count, which is what makes the block convention the RIGHT one for a token-major tensor --
    # so these three moduli are the whole contract, and the fourth condition (that the buffer is
    # token-major) is structural and lives in the runlist.
    for hd, hkv, has_v in geoms:
        qd, kvd = Hq * hd, hkv * hd
        if hd % 32 or hd < 32:
            raise ValueError(f"rope: cols=head_dim={hd} must be a multiple of 32 and >= 32")
        if M % cols:
            raise ValueError(f"rope: angle_rows=M={M} must be divisible by num_aie_columns={cols}")
        for label, heads in (("q", Hq), ("k", hkv)):
            if (M * heads) % cols:
                raise ValueError(f"rope {label} @hd={hd}: rows=M*heads={M * heads} must be "
                                 f"divisible by num_aie_columns={cols}")
        for label, size in (("norm", M * D), ("qk-norm q", M * qd), ("qk-norm k", M * kvd)):
            tile = D if label == "norm" else hd
            if size % (cols * tile):
                raise ValueError(f"RMSNorm {label} @hd={hd}: size={size} not a multiple of "
                                 f"num_aie_columns*tile_size={cols * tile}")
        if (Hq * M) % cols:
            raise ValueError(f"Softmax rows=Hq*M={Hq * M} not divisible by num_aie_columns={cols}")
        # Softmax cols is THIS geometry's window `w`, not the global S -- same derivation attn_ops
        # uses below. A narrowed geometry's own window is the shape that reaches the op.
        is_global_geom = hd == sp.global_head_dim and hkv == sp.global_n_kv_heads
        w = (S if (is_global_geom or not sliding_kv_circular or sliding_window is None)
             else sliding_window)
        if w % 16:
            raise ValueError(f"Softmax cols=w={w} @hd={hd} must be a multiple of 16")

    if os.environ.get("AIE_DEVICE"):
        import aie.utils as _aie_utils
        from aie.iron.device import from_name as _from_name
        _aie_utils.set_current_device(_from_name(os.environ["AIE_DEVICE"], n_cols=None))

    ctx = AIEContext()
    # ALLOC_SCHEME_ALL: EXPERIMENT ONLY, device-wide allocation_scheme for the tile-sharing
    # investigation on aiecc-dma-lowering-may-be-superlinear item 1 -- not a default, not wired
    # for real. Applies to every operator; GEMM_ALLOC_SCHEME (gemm_for) overrides it for GEMM
    # specifically if both are set.
    alloc_all = os.environ.get("ALLOC_SCHEME_ALL")
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

    def quant_kwargs(site):
        """weight_dtype/group_size kwargs for `site` ("qkv"/"o"/"mlp"), or `{}` for the plain
        bf16 operand every caller had before this axis existed -- the same shape decode's own
        precision plane hands its operators."""
        if site not in quant_plan:
            return {}
        dtype, group, _, _ = quant_plan[site]
        return dict(weight_dtype=dtype, group_size=group)

    # Buffers this build packs itself (Task 5), one entry per (layer, chunk): where the packed
    # bytes come from and the exact tile config they were packed for. Consumed by main(), after
    # compile, to write `buffers/<name>.bin` -- never serialized into meta.json itself.
    quant_pack = []

    def b_bytes(op):
        """`op`'s B (weight) operand size in BYTES. `get_arg_spec` already states it in the unit
        GEMM actually declares -- packed int8 bytes when quantized, bf16 elements otherwise -- so
        this is the one place that knows which unit applies, not a second `K*N*2` guess that goes
        stale the moment a call site quantizes."""
        n = 1
        for d in op.get_arg_spec()[1].shape:
            n *= d
        return n if op.weight_dtype != "bf16" else n * 2

    def gemm_for(label, K, Nout, b_col_maj=True, blocking=None, extra=None, site=None):
        """One GEMM at the registry's tiling for its shape, checked twice on the way through.

        `blocking` is `(rows_per_block, block_stride)` when B's rows are not one contiguous slab in
        the shared arena -- the KV cache under `kvl`, or one role's head rows inside a head-major
        `Wqkv`. `None` is the contiguous case every other call builds.

        `site` selects a quantized weight_dtype/group_size from `quant_plan`, or `None` for plain
        bf16. A quantized weight is packed fresh (see weight_gemm/qkv_operand below), never
        decode's aliased arena slab, so `blocking` never applies to one.
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
        qkw = quant_kwargs(site)
        # A bf16 [N, K] B walks N in steps of tile_n*cols rows, one BD stride of that many rows;
        # gemm_tiling_rejection has no stride rule, and aiecc refuses the BD only at the end.
        step = ch.tile_n * ch.cols
        if b_col_maj and not qkw and not blocking and Nout > step and step * K > BD_STRIDE_MAX:
            raise ValueError(f"GEMM {label} M={M} K={K} N={Nout}: tile_n({ch.tile_n})*cols({ch.cols})"
                             f"*K = {step * K} exceeds the BD stride range {BD_STRIDE_MAX}; pick "
                             f"tile_n*cols <= {BD_STRIDE_MAX // K}")
        blk = dict(b_block_rows=blocking[0], b_block_stride=blocking[1]) \
            if (blocking and not qkw) else {}
        blk.update(extra or {})
        blk.update(qkw)
        # GEMM_ALLOC_SCHEME: EXPERIMENT ONLY, for the tile-sharing investigation on
        # aiecc-dma-lowering-may-be-superlinear item 1 -- not a default, not wired for real.
        alloc = os.environ.get("GEMM_ALLOC_SCHEME") or alloc_all
        if alloc:
            blk["allocation_scheme"] = alloc
        return GEMM(M=M, K=K, N=Nout, b_col_maj=b_col_maj, context=ctx,
                    emulate_bf16_mmul_with_bfp16=emulate, prio_accuracy=prio_acc,
                    round_conv_even=round_even, a_resident=A_RESIDENT,
                    **blk, **ch.gemm_kwargs)

    op_norm = RMSNorm(size=M * D, num_aie_columns=cols, num_channels=1, tile_size=D,
                      weighted=True, epsilon=sp.eps, context=ctx, allocation_scheme=alloc_all)
    # PREFILL_MERGE_QKNORM: run the q-norm as `grp` chunks of the K-NORM'S OWN DESIGN instead of
    # one wider design of its own -- see `attn_ops` below for the per-geometry construction. Both
    # normalise independent head_dim tiles, q_dim is exactly grp*kv_dim BY CONSTRUCTION (grp is
    # q_heads//kv_heads for the SAME kv_heads that gives kv_dim, so the two cancel regardless of
    # which geometry), and a chunk boundary at a multiple of head_dim never splits a tile -- so the
    # arithmetic is identical and the q chunks and the k run become grp+1 ADJACENT runs of one
    # design, which D009 charges as ONE configure instead of two. Predicted -28 configures on a
    # 28-layer graph; at the measured 193.2 us that is -5.4 ms, and it was a forward test of that
    # rate in the REDUCTION direction (it was measured by adding). MEASURED -4.87 ms, 3 alternated
    # rounds, non-overlapping -- 90% of prediction -- and 14/14 on gate_llm.sh --tier2-prefill.
    # Default ON since 2026-09-10; =0 restores the two-design form.
    merge_qknorm = os.environ.get("PREFILL_MERGE_QKNORM", "1") == "1"
    # PREFILL_HEAD_SEAM=0 keeps the two head-axis rearranges. With them (the default) scores reads
    # its A operand as a per-head SLICE of the token-major `q` and ctx writes its C the same way
    # into `cxt`, at row pitch q_dim -- so `op_q2h` and `op_h2t` disappear entirely, with their two
    # configures per layer and their 4.0 MB/layer of zero-compute DMA. Head selection stays a
    # buffer SLICE, so the descriptor is head-independent and one design still serves all Hq.
    seam = os.environ.get("PREFILL_HEAD_SEAM", "1") == "1"

    # ---- attention op vocabulary, keyed on the layer's (head_dim, n_kv_heads, has_v_proj) ----
    # Uniform for every shipped spec except Gemma-4-12B, which has two geometries. A memoized
    # factory keeps the uniform case UNCHANGED (one cache entry, same objects every layer) and
    # costs the non-uniform case one more entry rather than a rewrite -- same shape as
    # gen_llm_decode.py's `attn_ops` (gen_llm_decode.py:1259).
    multi_geom = len(geoms) > 1
    _attn_cache = {}
    # One (kv_param, head_dim, window, mask_param) entry per DISTINCT geometry, appended as each is
    # built -- emitted verbatim as `scratchpad.kv_windows`, mirroring gen_llm_decode.py's
    # `geom_slots`. `mask_param` has no real counterpart here (prefill masks with the per-row
    # `mask_widths` vector, not a scratchpad scalar); it names this entry's OWN `kv_param` so the
    # field still resolves to a declared scratchpad parameter, and nothing reads it as a mask --
    # `check_prefill_pairing` matches `kv_windows` entries on `(head_dim, window)` alone.
    geom_slots = []
    # attn_ops (below) builds a per-WINDOW softmax and needs `sm_kw`/`attn_order` to do it -- see
    # `_win_cache`.
    sm_kw = dict(vector_size_source="rows") if causal == "rows" else {}
    # One owner for the chunking policy, decode's: `SOFTMAX_SEGMENT` unset keeps the unchunked op,
    # which at `cols` over ~7k no longer fits L1 -- in+out objectFIFOs are `cols` wide and double
    # buffered, so 8192 columns alone ask for the whole 64 KB.
    from gen_llm_decode import softmax_segment
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
    # Softmax designs, keyed by WINDOW rather than by full geometry key -- two geometries sharing a
    # window must share the SAME design object or gain a configure they didn't have before (mirrors
    # gen_llm_decode.py's `_win_cache`). Every shipped spec has one window (w == S everywhere), so
    # this produces exactly the one shared design it always did; only SLIDING_KV_CIRCULAR narrowing
    # one geometry's `w` away from `S` (Gemma-4-12B) makes it produce a second.
    _win_cache = {}

    def attn_ops(hd, hkv, has_v):
        key = (hd, hkv, has_v)
        if key in _attn_cache:
            return _attn_cache[key]
        qd, kvd, g_grp = Hq * hd, hkv * hd, Hq // hkv
        if not has_v and not sp.v_norm:
            raise ValueError(f"{sp.name}: head_dim={hd} has no v_proj (attention_k_eq_v) and no "
                             f"v_norm to derive v from k -- this graph has no other mechanism to "
                             f"populate V")
        if merge_qknorm and qd != g_grp * kvd:
            raise ValueError(f"PREFILL_MERGE_QKNORM needs q_dim ({qd}) == gqa_group ({g_grp}) * "
                             f"kv_dim ({kvd}) at head_dim={hd}; this geometry does not split evenly")
        sfx = f"_hd{hd}" if multi_geom else ""
        op_qn = RMSNorm(size=M * qd, num_aie_columns=cols, num_channels=1, tile_size=hd,
                        weighted=True, epsilon=sp.eps, context=ctx, allocation_scheme=alloc_all)
        op_kn = RMSNorm(size=M * kvd, num_aie_columns=cols, num_channels=1, tile_size=hd,
                        weighted=True, epsilon=sp.eps, context=ctx, allocation_scheme=alloc_all)
        # Gemma-4's gainless value-norm rides the SAME design as op_kn (decode's identical move,
        # gen_llm_decode.py:1305-1317): a weighted RMSNorm fed decode's shared `ones_h{hd}` gain is
        # bit-identical to an unweighted one, and reusing the object -- not building a `weighted=
        # False` twin -- makes the v-norm and k-norm runs one contiguous same-design block.
        op_vn = op_kn if sp.v_norm else None
        # An ordinary partial rotary (Qwen3.5) turns the first `rot` of each head row, in place.
        rot = sp.rope_rotary_dim or hd
        rkw = dict(row_stride=hd) if rot != hd else {}
        op_rq = RoPE(rows=M * Hq, cols=rot, angle_rows=M, num_aie_columns=cols, context=ctx,
                    allocation_scheme=alloc_all, **rkw)
        op_rk = RoPE(rows=M * hkv, cols=rot, angle_rows=M, num_aie_columns=cols, context=ctx,
                    allocation_scheme=alloc_all, **rkw)
        # One kv head's run of rows, and where each role's block sits inside it. A layer with no
        # v_proj (attention_k_eq_v) concatenates grp+1 blocks, not grp+2 -- verified against the
        # decode dump: L*_Wqkv is 62,914,560 B at sliding (grp+2=4 blocks) and 66,846,720 B at
        # global (grp+1=17 blocks), not 70,778,880 (what grp+2 would give there).
        qkv_rows = (g_grp + (2 if has_v else 1)) * hd
        blocking = {"q": (g_grp * hd, 0), "k": (hd, g_grp * hd)}
        if has_v:
            blocking["v"] = (hd, (g_grp + 1) * hd)
        op_gq = gemm_for(f"q{sfx}", D, qd, blocking=(
            (blocking["q"][0], qkv_rows * D) if hm else None), site="qkv")
        op_gkv = gemm_for(f"kv{sfx}", D, kvd, blocking=(
            (blocking["k"][0], qkv_rows * D) if hm else None), site="qkv")
        op_o = gemm_for(f"o{sfx}", qd, D, site="o")
        # This geometry's own KV capacity: S for the global geometry (identified by matching
        # spec.global_head_dim/global_n_kv_heads, same test gen_llm_decode.py's attn_ops uses), or
        # decode's `sliding_window` for every other one once SLIDING_KV_CIRCULAR narrowed it.
        # `kv_T` stays the block size derived from decode's flat `dims.kv_block`; capped to this
        # geometry's own capacity the same way decode's `T_g = min(T, w)` is.
        is_global_geom = hd == sp.global_head_dim and hkv == sp.global_n_kv_heads
        w = (S if (is_global_geom or not sliding_kv_circular or sliding_window is None)
             else sliding_window)
        # `--kv-alloc` widens the CAPACITY of a geometry whose window can grow. A circular sliding
        # geometry wraps at `w` by construction, so its capacity IS `w` -- `w != S` is exactly that
        # condition, see `w`'s own derivation above.
        KVA_g = (kv_alloc or w) if w == S else w
        T_g = min(kv_T, w)
        kv_slot = "kv_off" if not geom_slots else f"kv_off{len(geom_slots)}"
        geom_slots.append((kv_slot, hd, w, kv_slot))
        # scores: B is the kv cache read as [N=w, K=hd] -> b_col_maj. ctx: the SAME bytes read as
        # [K=w, N=hd] -> plain. N/K is THIS geometry's own window `w`, matching `kv_slab()`'s own
        # w-sized span -- an S-wide operand here would read past a narrowed geometry's buffer.
        kvl_g = KVLayout(Hkv=hkv, S=KVA_g, HD=hd, T=T_g)
        # The same layout over the WINDOW. `head_span` is the extent an operand covering the
        # window may address, so under a wider capacity it has to be asked of the window's layout:
        # the scores/ctx operands read `w` positions, while the append writes the whole cache.
        kvl_w = KVLayout(Hkv=hkv, S=w, HD=hd, T=T_g)
        kv_blk = (kvl_g.T, kvl_g.block_stride) if kvl_g.T != kvl_g.S else None
        op_sc = gemm_for(f"scores{sfx}", hd, w, blocking=kv_blk,
                         extra=dict(a_row_stride=qd) if seam else {})
        op_cx = gemm_for(f"ctx{sfx}", w, hd, b_col_maj=False, blocking=kv_blk,
                         extra=dict(c_row_stride=qd) if seam else {})
        # Softmax, shared across every geometry at this WINDOW (see `_win_cache`'s own comment).
        # `rows` comes from Hq/M alone, not from hd/hkv, so the shared design is correct even where
        # two distinct geometries land on the same window.
        if w not in _win_cache:
            _win_cache[w] = (
                Softmax(rows=Hq * M, cols=w, num_aie_columns=cols, num_channels=1,
                        context=ctx, allocation_scheme=alloc_all,
                        segment=softmax_segment(w), **sm_kw),
                Softmax(rows=M, cols=w, num_aie_columns=cols, num_channels=1,
                        context=ctx, allocation_scheme=alloc_all,
                        segment=softmax_segment(w), **sm_kw)
                if attn_order != "off" else None,
            )
        op_sm_w, op_sm_head_w = _win_cache[w]
        # KV append. The cache is [hkv, S, hd] and `kv_off` is an element-unit BD offset, so M
        # consecutive positions are M contiguous rows per head -- the M=1 BD with an extra outer
        # dimension, not a new mechanism. The SOURCE is token-major [M, hkv, hd], so the (M, hkv)
        # axes swap in the descriptor: input walks h fastest within a token, output walks m
        # fastest within a head.
        # A batch lands inside ONE block whenever the block is a whole number of batches, and the
        # descriptor is then the unblocked one at this geometry's own `head_stride`: chunk starts
        # are M-aligned, so `q0%T + m < T` and the block term is constant across the batch. Only a
        # batch SPANNING blocks needs the outer block dimension.
        if kvl_g.T == kvl_g.S or (M <= kvl_g.T and kvl_g.T % M == 0):
            kv_in_sizes, kv_in_strides = (M, hkv, hd), (hkv * hd, hd, 1)
            kv_out_sizes, kv_out_strides = (M, hkv, hd), (hd, kvl_g.head_stride, 1)
        else:
            if M % kvl_g.T:
                raise ValueError(
                    f"prefill batch M={M} straddles a KV block boundary (T={kvl_g.T}) at "
                    f"head_dim={hd}: it neither divides the block nor is a whole number of them")
            nb = M // kvl_g.T
            kv_in_sizes = (nb, hkv, kvl_g.T, hd)
            kv_in_strides = (kvl_g.T * hkv * hd, hd, hkv * hd, 1)
            kv_out_sizes = (nb, hkv, kvl_g.T, hd)
            kv_out_strides = (kvl_g.block_stride, kvl_g.head_stride, hd, 1)
        op_kvapp = StridedCopy(
            input_sizes=kv_in_sizes, input_strides=kv_in_strides, input_offset=0,
            output_sizes=kv_out_sizes, output_strides=kv_out_strides, output_offset=0,
            input_buffer_size=M * hkv * hd, output_buffer_size=kvl_g.total_elems,
            transfer_size=pick_transfer(M * hkv * hd), num_aie_channels=1,
            # THIS geometry's slot, the spelling `geom_slots` puts in the meta -- the two need
            # different runtime values (different capacity, different head_dim), and a meta naming
            # a slot the ELF never declared does not load at all. First geometry keeps the bare
            # `kv_off` for the same reason decode's does: the pre-list host fallback reads it.
            output_offset_parameter=kv_slot, context=ctx)
        # The head-axis seam, both directions -- only built (and only ever used) when `not seam`.
        # See the module docstring for why these exist and what they cost; pure DMA, 0% compute.
        op_q2h = op_h2t = None
        if not seam:
            op_q2h = StridedCopy(
                input_sizes=(Hq, M, hd), input_strides=(hd, qd, 1), input_offset=0,
                output_sizes=(Hq, M, hd), output_strides=(M * hd, hd, 1), output_offset=0,
                input_buffer_size=M * qd, output_buffer_size=M * qd,
                transfer_size=pick_transfer(M * qd), num_aie_channels=1, context=ctx)
            op_h2t = StridedCopy(
                input_sizes=(M, Hq, hd), input_strides=(hd, M * hd, 1), input_offset=0,
                output_sizes=(M, Hq, hd), output_strides=(Hq * hd, hd, 1), output_offset=0,
                input_buffer_size=M * qd, output_buffer_size=M * qd,
                transfer_size=pick_transfer(M * qd), num_aie_channels=1, context=ctx)
        # ---- ring variant (sec 2.1): sliding (w < S) geometries only, opt-in, default off. ----
        # Read-first over the concat [ring w | this chunk's own M rows] instead of append-then-
        # read: `sc_ring`/`cx_ring` are today's `op_sc`/`op_cx` widened to a row of w+M columns
        # (c_row_stride/a_row_stride skip the extra M), `sc_batch`/`cx_batch` are the SAME shapes
        # against the chunk's OWN fresh k/v (b_block_rows=1, b_block_stride=hkv*hd -- the token-
        # major stride `blocking["k"]` already uses elsewhere in this function), and `op_add_cx`
        # sums the two context partials. `op_kvapp` is reused unchanged, only moved later in the
        # runlist (sec 2.1 row 7).
        ring = SLIDING_RING and w < S
        op_sc_ring = op_sc_batch = op_sm_ring = op_cx_ring = op_cx_batch = op_add_cx = None
        if ring:
            # K007, named at the point w and M are picked: the two divisibility preconditions the
            # ring algebra and the softmax kernel each need, and the arms the ring runlist is
            # written for.
            if not seam or attn_order != "off" or causal != "rows":
                raise ValueError(
                    "PREFILL_SLIDING_RING needs PREFILL_HEAD_SEAM=1, PREFILL_ATTN_ORDER=off and "
                    "--causal rows -- the ring runlist is written for the default arms only")
            if w % M:
                raise ValueError(f"PREFILL_SLIDING_RING: window w={w} @hd={hd} is not a multiple "
                                 f"of M={M} (W % M == 0); a chunk's hole/commit would wrap the "
                                 f"ring end")
            row_w = w + M
            if row_w % SM_VEC_LEN:
                raise ValueError(f"PREFILL_SLIDING_RING: w+M={row_w} @hd={hd} is not a multiple "
                                 f"of {SM_VEC_LEN} (softmax SM_VEC_LEN); the kernel would drop a "
                                 f"tail")
            # b_block_rows=1: B's physical rows are single hd-wide TOKENS of the fresh (token-
            # major, [M, hkv*hd]) k/v buffer, block_stride=hkv*hd apart -- not `blocking["k"]`
            # above, which blocks a head-major WEIGHT's output rows and does not apply here.
            op_sc_ring = gemm_for(f"sc_ring{sfx}", hd, w, blocking=kv_blk,
                                  extra=dict(a_row_stride=qd, c_row_stride=row_w))
            op_sc_batch = gemm_for(f"sc_batch{sfx}", hd, M, blocking=(1, hkv * hd),
                                   extra=dict(a_row_stride=qd, c_row_stride=row_w))
            op_cx_ring = gemm_for(f"cx_ring{sfx}", w, hd, b_col_maj=False, blocking=kv_blk,
                                  extra=dict(a_row_stride=row_w, c_row_stride=qd))
            op_cx_batch = gemm_for(f"cx_batch{sfx}", M, hd, b_col_maj=False,
                                   blocking=(1, hkv * hd),
                                   extra=dict(a_row_stride=row_w, c_row_stride=qd))
            # One design per geometry, not per window (unlike `_win_cache` above): every shipped
            # spec has exactly one sliding geometry, so this is the shared-design case already;
            # a second sliding geometry at the same window would cost an extra configure here,
            # same tradeoff `_win_cache` exists to avoid -- not built because nothing needs it yet.
            op_sm_ring = Softmax(rows=Hq * M, cols=row_w, num_aie_columns=cols, num_channels=1,
                                 context=ctx, allocation_scheme=alloc_all,
                                 vector_size_source="rows_hole")
            op_add_cx = ElementwiseAdd(size=M * qd, num_aie_columns=cols, tile_size=qd // cols,
                                       context=ctx, allocation_scheme=alloc_all)
        g = SimpleNamespace(hd=hd, hkv=hkv, has_v=has_v, qd=qd, kvd=kvd, grp=g_grp, sfx=sfx, w=w,
                            qkv_rows=qkv_rows, blocking=blocking, kvl=kvl_g, kvl_win=kvl_w,
                            op_qn=op_qn, op_kn=op_kn, op_vn=op_vn, op_rq=op_rq, op_rk=op_rk,
                            op_gq=op_gq, op_gkv=op_gkv, op_o=op_o, op_sc=op_sc, op_cx=op_cx,
                            op_sm=op_sm_w, op_sm_head=op_sm_head_w,
                            op_kvapp=op_kvapp, op_q2h=op_q2h, op_h2t=op_h2t,
                            ring=ring, op_sc_ring=op_sc_ring, op_sc_batch=op_sc_batch,
                            op_sm_ring=op_sm_ring, op_cx_ring=op_cx_ring, op_cx_batch=op_cx_batch,
                            op_add_cx=op_add_cx)
        _attn_cache[key] = g
        return g

    # PREFILL_FUSE_SILU: the gate GEMM applies SiLU to its own C tile before it leaves L1, so the
    # standalone SiLU op and `g`'s whole DDR round-trip disappear. Configure-NEUTRAL by
    # construction -- gate and up stop sharing one design (the epilogue changes it), so their
    # shared configure becomes two while SiLU's one goes away -- which makes this a clean read of
    # what the BYTES alone are worth: -3.0 MiB/layer, -84 MiB/dispatch.
    fuse_silu = os.environ.get("PREFILL_FUSE_SILU", "0") == "1" and sp.act == "silu"
    op_gu = gemm_for("gate_up", D, FF, site="mlp")
    # PREFILL_EPI_ELEMS=0 builds the NULL CONTROL: same fused design, same call, no arithmetic.
    epi_n = os.environ.get("PREFILL_EPI_ELEMS")
    op_gate = gemm_for("gate_up", D, FF, extra=dict(
        epilogue="silu", **({"epilogue_elems": int(epi_n)} if epi_n is not None else {}),
    ), site="mlp") if fuse_silu else None
    op_down = gemm_for("down", FF, D, site="mlp")
    for hd, hkv, has_v in geoms:
        attn_ops(hd, hkv, has_v)   # build now so [tiles] below reports every geometry
    print("[tiles] " + "  ".join(
        f"{k}={v['tile'][0]}x{v['tile'][1]}x{v['tile'][2]}@{v['cols']}c({v['source']})"
        for k, v in sorted(tiles.items())))
    # ONE softmax per WINDOW over every head's rows at once (see `_win_cache` in attn_ops): its
    # `rows` axis is just "independent rows to normalise", and every head's [M, w] block is a
    # contiguous slice of the same buffer. That is also what makes the causal mask a plain vector:
    # row Hq*M is (head, token) flattened, and the width depends only on the token half.
    if sp.act == "silu":
        op_act = SiLUAct(size=M * FF, num_aie_columns=cols, tile_size=FF // cols, context=ctx,
                         allocation_scheme=alloc_all)
    else:
        op_act = GELU(size=M * FF, num_aie_columns=cols, num_channels=1,
                      tile_size=FF // cols, context=ctx, allocation_scheme=alloc_all)
    op_mul = ElementwiseMul(size=M * FF, num_aie_columns=cols, tile_size=FF // cols, context=ctx,
                            allocation_scheme=alloc_all)
    op_add = ElementwiseAdd(size=M * D, num_aie_columns=cols, tile_size=D // cols, context=ctx,
                            allocation_scheme=alloc_all)
    # TIMING PROBE (see commit): PREFILL_NO_LAYER_SCALAR=1 drops the per-row dispatches below to
    # price their count. Changes numerics -- never for shipping.
    no_layer_scalar = os.environ.get("PREFILL_NO_LAYER_SCALAR", "0") == "1"
    # PREFILL_LSCALE_BCAST=1: ElementwiseMul.rows=M broadcasts the [D] gain against [M, D] in one
    # dispatch instead of M per-row calls (iron/elementwise-mul-broadcast) -- output-identical, a
    # real candidate. Needs that IRON tree on PYTHONPATH; the stock tree has no `rows` kwarg.
    lscale_bcast = os.environ.get("PREFILL_LSCALE_BCAST", "0") == "1"
    op_lscale = (ElementwiseMul(size=D, tile_size=D // cols, num_aie_columns=cols, context=ctx,
                                **(dict(rows=M) if lscale_bcast else {}))
                 if sp.layer_scalar else None)
    # ---- buffers ----
    # Every prefill intermediate is ONE buffer shared by all layers: the sequence runs layers one
    # at a time, so nothing outlives its layer. Decode declares them per layer; at M=256 that
    # would be 28 * 43 MB of arena for no reason.
    # q/k/v/cxt/sc/sw are geometry-shaped: IRON's calculate_buffer_layout rejects one buffer NAME
    # declared at two different operator shapes (a GEMM's own arg spec, not just this file's
    # `bufsz`), so a multi-geometry build needs one buffer per geometry, suffixed exactly like the
    # tile labels (`sfx` in `attn_ops`) -- unsuffixed when every spec has one geometry, which keeps
    # the uniform case's buffer set (and arena) byte-identical to before this axis existed. sc/sw
    # are sized off THIS geometry's own window `g.w`, not the global S -- a narrowed geometry's
    # scores/softmax tile is `[Hq*M, w]`, not `[Hq*M, S]` (see attn_ops).
    bufsz = {
        "h": M * D * 2,
        "a": M * D * 2, "xs": M * D * 2,
        "hf": M * D * 2, "gs": M * FF * 2, "u": M * FF * 2,
        "gh": M * FF * 2, "d": M * D * 2,
    }
    for hd, hkv, has_v in geoms:
        g = attn_ops(hd, hkv, has_v)
        gsfx = g.sfx
        qd, kvd = Hq * hd, hkv * hd
        bufsz[f"q{gsfx}"] = M * qd * 2
        bufsz[f"k{gsfx}"] = M * kvd * 2
        bufsz[f"v{gsfx}"] = M * kvd * 2
        bufsz[f"cxt{gsfx}"] = M * qd * 2
        # A ring geometry's row grows by M (the concat [ring w | this chunk's own M]); sc_batch/
        # cx_batch write the extra columns in place, so it is one wider buffer, not two.
        row_w = g.w + M if g.ring else g.w
        bufsz[f"sc{gsfx}"] = Hq * M * row_w * 2
        bufsz[f"sw{gsfx}"] = Hq * M * row_w * 2
        if g.ring:
            bufsz[f"cxt2{gsfx}"] = M * qd * 2
        if not seam:
            bufsz[f"qh{gsfx}"] = M * qd * 2
            bufsz[f"cx{gsfx}"] = M * qd * 2
    if attn_order != "off":
        # Slicing an INPUT needs its size declared: calculate_buffer_layout takes a plain buffer's
        # size from the arg spec, and a sliced one only from here. Same size the whole-buffer arm
        # gets from op_sm's spec, so the input arena is byte-identical between the arms.
        bufsz[SM_WIDTHS] = Hq * M * 4
    if not fuse_silu:
        bufsz["g"] = M * FF * 2
    dec_meta, dec_order, dec_sizes, dec_reserved = (None, [], {}, 0)
    if dec_meta_path:
        dec_meta, dec_order, dec_sizes, dec_reserved = decode_arena_plan(dec_meta_path)
        # Scratch for a K-split weight (weight_gemm below), added ONLY when the shared arena
        # actually holds one -- every shipped uniform-geometry spec never does, and this keeps
        # their arena byte-identical to before weight_gemm existed. Both down and o produce a
        # D-wide C, so one triple covers either, reused across layers and across the two roles.
        # `kpart2` is the pairwise-tree fold's third slot (weight_gemm below, n=4 case).
        # weight_gemm asserts this D-sizing against whatever Nout a call actually needs, so a
        # future K-split at a different width (Wg/Wu, FF-wide) fails loud instead of overrunning
        # this.
        if any(f"L{l}_{base}k0" in dec_sizes for l in range(NL) for base in ("Wd", "Wo")):
            bufsz["kacc"] = M * D * 2
            bufsz["kpart"] = M * D * 2
            bufsz["kpart2"] = M * D * 2
        for name, length in dec_sizes.items():
            if name in bufsz:
                raise ValueError(f"decode scratch name {name!r} collides with a prefill "
                                 f"intermediate of the same name")
            bufsz[name] = length
    # `prefill_local` is resolved AFTER every op below (including a quantized site's packed
    # weight buffers, registered into `bufsz` as they are built) rather than snapshotted here --
    # everything in `bufsz` that isn't one of decode's own shared names is prefill's, regardless
    # of how late it was added.

    def qkv_slab(p, role, geom):
        """One projection's operand inside `L*_Wqkv`, at THIS layer's geometry (decode's shared
        bf16 arena only -- see `qkv_operand` for the quantized case, which never reads this).

        Stock, the three roles are three contiguous slabs. Head-major, each role's rows are one
        block per kv head, so the operand runs from its FIRST block to the end of its last one and
        the operator's blocked descriptor picks its own rows out of the span -- `b_elems` is that
        span, asked of the operator rather than recomputed here.
        """
        op = {"q": geom.op_gq, "k": geom.op_gkv, "v": geom.op_gkv}[role]
        if not hm:
            base, span = ({"q": (0, geom.qd), "k": (geom.qd, geom.kvd),
                          "v": (geom.qd + geom.kvd, geom.kvd)}[role][0] * D,
                          {"q": geom.qd, "k": geom.kvd, "v": geom.kvd}[role] * D)
        else:
            base, span = geom.blocking[role][1] * D, op.b_elems
        return f"{p}Wqkv[{base * 2}:{(base + span) * 2}]"

    def qkv_rl(p, role, geom, layer, out):
        """Runlist entries for one of q/k/v: the GEMM over its operand, preceded by the
        DequantRows that makes the operand when decode holds `Wqkv` as int4 rows."""
        op = {"q": geom.op_gq, "k": geom.op_gkv, "v": geom.op_gkv}[role]
        if "qkv" in quant_plan or hm or not dec_meta_path:
            return [(op, "h", qkv_operand(p, role, geom, layer), out)]
        row0, n = {"q": (0, geom.qd), "k": (geom.qd, geom.kvd),
                   "v": (geom.qd + geom.kvd, geom.kvd)}[role]
        total = geom.qd + (2 if geom.has_v else 1) * geom.kvd
        pre, w = weight_rows(f"{p}Wqkv", total, row0, n, D)
        return pre + [(op, "h", w, out)]

    def qkv_operand(p, role, geom, layer):
        """Q/K/V's weight operand: `qkv_slab` (decode's head-major `Wqkv` slice) when bf16, or a
        buffer this build packs itself from the "qkv" site's own dump when quantized.

        These never share bytes even at the same (K, N, dtype, group_size): decode's arena
        interleaves q/k/v by head for its OWN GEVM, and a packed GEMM operand is a tile-planar
        permutation of the row form (`iron.common.quant.repack_gemm_weight`'s whole job) -- two
        different layouts neither side can read as the other. So a quantized qkv reads its own
        per-role `.npy` straight off the dump, no interleave needed at all.
        """
        if "qkv" not in quant_plan:
            return qkv_slab(p, role, geom)
        dtype, group, scale_dtype, src_dir = quant_plan["qkv"]
        op = {"q": geom.op_gq, "k": geom.op_gkv, "v": geom.op_gkv}[role]
        prefix = f"{sp.weight_prefix}layers.{layer}."
        tensor = QUANT_TENSOR[{"q": "Wq", "k": "Wk", "v": "Wv"}[role]]
        srcs = quant_source_files(src_dir, prefix, tensor)
        if len(srcs) != 1:
            raise ValueError(f"{prefix}{tensor}: expected exactly one dumped file (qkv never "
                             f"K-splits), found {len(srcs)} under {src_dir}")
        name = f"{p}W{role}_qp"
        nbytes = b_bytes(op)
        bufsz[name] = nbytes
        quant_pack.append(dict(buf=name, src_dir=src_dir, src_file=srcs[0],
                               N=op.N, K=op.K, tile_k=op.tile_k, tile_n=op.tile_n,
                               group_size=group, weight_dtype=dtype, cols=op.num_aie_columns,
                               scale_dtype=scale_dtype, mmul=op._mmul_rst))
        return f"{name}[0:{nbytes}]"

    def kv_slab(buf, kv, geom):
        """A kv head's slab: from its base to the end of its LAST block, not `S*hd` -- across
        blocks the head's positions are `block_stride` apart with the other heads in between.
        The span is the WINDOW's, which is what the scores/ctx operands address; the base is shared
        (both layouts carry the same `T`, so the same `head_stride`). Both reduce to the flat
        `kv*S*hd` slice at T == S."""
        base = geom.kvl.head_base(kv)
        return f"{buf}[{base * 2}:{(base + geom.kvl_win.head_span) * 2}]"

    def attn_norms(p, hd):
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
            return (f"{b}[0:{D * 2}]", f"{b}[{D * 2}:{(D + hd) * 2}]",
                    f"{b}[{(D + hd) * 2}:{(D + 2 * hd) * 2}]")
        return (p + "n_in", p + "n_qn", p + "n_kn")

    # Decode's weights as int4 rows (its GEMV format) are read through DequantRows into ONE shared
    # bf16 matrix `wdq`, right before the GEMM that consumes it: a second packed copy in GEMM's
    # layout does not fit the 4 GiB arena beside a 4B model's decode weights. Which buffers are int4
    # is read off decode's own byte lengths, never assumed.
    _dq_cache = {}
    wdq_elems = [0]

    def dq_group(buf, rows, K):
        n = dec_sizes.get(buf)
        if n is None:
            raise ValueError(f"{buf}: not in the decode arena")
        if n == rows * K * 2:
            return None
        hits = [g for g in (32, 64, 128) if n == rows * row_stride_bytes(K, g, "int4")]
        if len(hits) != 1:
            raise ValueError(f"{buf}: {n} B is neither bf16 nor one int4 group size for "
                             f"[{rows}, {K}] (matches: {hits})")
        return hits[0]

    def weight_rows(buf, total_rows, row0, nrows, K, out_stride=None, out_col=0):
        """(runlist entries, operand) for rows [row0, row0+nrows) of decode's [total_rows, K]
        weight `buf`: its bf16 slice as is, or a DequantRows of the int4 rows into `wdq`."""
        g = dq_group(buf, total_rows, K)
        if g is None:
            return [], f"{buf}[{row0 * K * 2}:{(row0 + nrows) * K * 2}]"
        key = (nrows, K, g, out_stride, out_col)
        if key not in _dq_cache:
            _dq_cache[key] = DequantRows(rows=nrows, K=K, group_size=g, num_aie_columns=cols,
                                         out_stride=out_stride, out_col=out_col, context=ctx)
        op = _dq_cache[key]
        width = out_stride or K
        wdq_elems[0] = max(wdq_elems[0], nrows * width)
        out = f"wdq[0:{nrows * width * 2}]"
        return [(op, f"{buf}[{row0 * op.row_stride}:{(row0 + nrows) * op.row_stride}]", out)], out

    # A K-split weight's chunk GEMM + its accumulate-add, memoized by (chunk_K, Nout): every layer
    # needing a chunked down-projection shares one pair of designs, exactly like `attn_ops` shares
    # one attention op set per geometry.
    _chunked_cache = {}

    def weight_gemm(p, base_name, plain_op, K, Nout, a_buf, out_buf, label, site=None, layer=None):
        """Runlist entries computing `plain_op(a_buf, p+base_name) -> out_buf`.

        Reads `p+base_name` whole when the decode arena holds it as one buffer. Gemma-4's
        down-projection (K=15360, every layer) and global layers' o-projection (K=8192) exceed
        GEVM's L1 budget at K-chunk 1, so decode's own generator ALWAYS splits those two weights
        (`k_chunks_for`, llm_decode_spec.py:317) regardless of build flags -- verified against the
        dump: L0/L5 of decode_l6 both carry `Wdk0..Wdk3`/`Wok0..Wok1`, never a plain `Wd`/`Wo`.
        Prefill's GEMM has no L1 problem at this K (it tiles K internally via tile_k) and reads the
        chunks itself: one GEMM per chunk, each taking its K-slice of the token-major `a_buf` via
        `a_row_stride` -- the same strided-read mechanism `PREFILL_HEAD_SEAM` already uses for a
        head slice of `q` -- with the partial C tiles summed by ElementwiseAdd. All `n` chunk GEMMs
        share ONE op object (same shape, same stride), so they collapse to one configure, matching
        every other per-head/per-chunk loop in this file.

        When `site` is quantized (`layer` then required), NONE of the above applies: `plain`
        (decode's arena name) is never read at all. A packed weight is a different byte layout
        from decode's row-packed GEVM operand at the same (K, N, dtype, group_size) --
        `iron.common.quant.repack_gemm_weight`'s whole job is that permutation -- so this packs
        fresh from the same per-tensor `.npy` dump decode's own GEVM reads, into a buffer THIS
        build owns (`buffers/<name>.bin`, uploaded from prefill's own artifact dir as "a
        prefill-only weight", `rust/npu-engine/src/llm/npu_decode.rs`), never decode's shared
        arena. The dump is chunked exactly where decode's own `k_chunks_for` chunks it (down
        always, o only on a global layer) -- Prefill has no L1 reason to re-chunk on its own, but
        the dump does not offer an unchunked K=15360/K=8192 tensor to begin with, so it packs and
        folds each chunk the same way the decode-arena branch above does.
        """
        plain = p + base_name
        if site in quant_plan:
            dtype, group, scale_dtype, src_dir = quant_plan[site]
            prefix = f"{sp.weight_prefix}layers.{layer}."
            tensor = QUANT_TENSOR[base_name]
            srcs = quant_source_files(src_dir, prefix, tensor)
            n = len(srcs)
            if not n:
                raise ValueError(f"{prefix}{tensor}: no plain or .kchunkN.npy file under "
                                 f"{src_dir}")
            if K % n:
                raise ValueError(f"{plain}: K={K} not divisible by its own {n} dumped chunks")
            chunk_k = K // n
            if n == 1:
                # No split at all: `plain_op` (the caller's already-built full-K op, site= already
                # baked in) is exactly the right shape -- packing a second, redundant "_k{K}of1"
                # design would be pure waste.
                chunk_op, add_op = plain_op, None
            else:
                # C stays bf16 regardless of B's weight_dtype (GEMM's own "A and C stay bf16"),
                # so this is the SAME D-wide kacc/kpart/kpart2 the decode-arena branch below
                # checks -- guarded the same way, for the same reason.
                want = M * Nout * 2
                have = bufsz.get("kacc")
                if have != want:
                    raise ValueError(f"{plain}: K-split scratch (kacc/kpart/kpart2) is {have}B, "
                                     f"sized for a different Nout than this call's {Nout} "
                                     f"({want}B needed)")
                ckey = (chunk_k, Nout, site)
                if ckey not in _chunked_cache:
                    chunk_op_ = gemm_for(f"{label}_k{chunk_k}of{n}", chunk_k, Nout,
                                        extra=dict(a_row_stride=K), site=site)
                    add_op_ = ElementwiseAdd(size=M * Nout, num_aie_columns=cols,
                                            tile_size=Nout // cols, context=ctx,
                                            allocation_scheme=alloc_all)
                    _chunked_cache[ckey] = (chunk_op_, add_op_)
                chunk_op, add_op = _chunked_cache[ckey]

            def a_slice(i):
                if n == 1:
                    return a_buf
                lo = i * chunk_k
                return f"{a_buf}[{lo * 2}:{(lo + chunk_op.a_elems) * 2}]"

            nbytes = b_bytes(chunk_op)

            def wk(i):
                name = f"{plain}_qp{f'_c{i}' if n > 1 else ''}"
                bufsz[name] = nbytes
                quant_pack.append(dict(buf=name, src_dir=src_dir, src_file=srcs[i],
                                       N=Nout, K=chunk_k, tile_k=chunk_op.tile_k,
                                       tile_n=chunk_op.tile_n, group_size=group,
                                       weight_dtype=dtype, cols=chunk_op.num_aie_columns,
                                       scale_dtype=scale_dtype, mmul=chunk_op._mmul_rst))
                return f"{name}[0:{nbytes}]"
        elif not dec_meta_path:
            return [(plain_op, a_buf, f"{plain}[0:{K * Nout * 2}]", out_buf)]
        elif plain in dec_sizes:
            pre, w = weight_rows(plain, Nout, 0, Nout, K)
            return pre + [(plain_op, a_buf, w, out_buf)]
        else:
            n = 0
            while f"{plain}k{n}" in dec_sizes:
                n += 1
            if not n:
                raise ValueError(f"{plain}: neither a plain buffer nor {plain}k0.. chunks exist in "
                                 f"the decode shared arena")
            if K % n:
                raise ValueError(f"{plain}: K={K} not divisible by its own {n} decode-arena chunks")
            chunk_k = K // n
            if dq_group(f"{plain}k0", Nout, chunk_k) is not None:
                # int4 chunks land side by side in one [Nout, K] matrix: one full-K GEMM, no fold.
                pre = []
                for i in range(n):
                    pre += weight_rows(f"{plain}k{i}", Nout, 0, Nout, chunk_k, out_stride=K,
                                       out_col=i * chunk_k)[0]
                return pre + [(plain_op, a_buf, f"wdq[0:{Nout * K * 2}]", out_buf)]
            if n > 1:
                # kacc/kpart/kpart2 are pre-sized D-wide, above, for the only two roles the shared
                # arena chunks today (Wd, Wo). A future K-split at a different Nout (Wg/Wu -> FF)
                # would silently write an FF-wide tile into this D-wide scratch; fail loud instead,
                # before spending a GEMM tile lookup on a shape we are about to reject anyway.
                want = M * Nout * 2
                have = bufsz.get("kacc")
                if have != want:
                    raise ValueError(f"{plain}: K-split scratch (kacc/kpart/kpart2) is {have}B, "
                                     f"sized for a different Nout than this call's {Nout} "
                                     f"({want}B needed)")
            ckey = (chunk_k, Nout)
            if ckey not in _chunked_cache:
                chunk_op = gemm_for(f"{label}_k{chunk_k}of{n}", chunk_k, Nout,
                                    extra=dict(a_row_stride=K))
                add_op = (ElementwiseAdd(size=M * Nout, num_aie_columns=cols, tile_size=Nout // cols,
                                         context=ctx) if n > 1 else None)
                _chunked_cache[ckey] = (chunk_op, add_op)
            chunk_op, add_op = _chunked_cache[ckey]

            def a_slice(i):
                lo = i * chunk_k
                return f"{a_buf}[{lo * 2}:{(lo + chunk_op.a_elems) * 2}]"

            def wk(i):
                return f"{plain}k{i}[0:{chunk_k * Nout * 2}]"

        # Pairwise-tree fold, mirroring split_over_k's grouping (gen_llm_decode.py): bf16 rounds
        # on every add, so combining adjacent chunks first keeps the rounding depth at
        # ceil(log2(n)) instead of this file's old n-1 linear chain. Measured on Gemma-4's real
        # n=4 down_proj chunks (int8/g32 dump, M=256 activation): tree-vs-f32-truth rel-L2
        # 3.03e-3, linear-vs-truth 3.15e-3, tree-vs-linear (the decode/prefill disagreement)
        # 2.88e-3 -- the same order as this file's own PREFILL_ROUND_EVEN ULP effect (3.9e-3),
        # so the chain was a second, avoidable source of prefill/decode divergence.
        if n == 1:
            return [(chunk_op, a_slice(0), wk(0), out_buf)]
        if n == 2:
            return [
                (chunk_op, a_slice(0), wk(0), "kacc"),
                (chunk_op, a_slice(1), wk(1), "kpart"),
                (add_op, "kacc", "kpart", out_buf),
            ]
        if n == 4:
            return [
                (chunk_op, a_slice(0), wk(0), "kacc"),
                (chunk_op, a_slice(1), wk(1), "kpart"),
                (add_op, "kacc", "kpart", "kacc"),        # kacc = chunk0 + chunk1
                (chunk_op, a_slice(2), wk(2), "kpart"),
                (chunk_op, a_slice(3), wk(3), "kpart2"),
                (add_op, "kpart", "kpart2", "kpart"),      # kpart = chunk2 + chunk3
                (add_op, "kacc", "kpart", out_buf),
            ]
        raise ValueError(f"{plain}: pairwise K-split fold not implemented for n={n} chunks "
                         f"(only 1, 2 and 4 are used by any shipped spec)")

    # Dual-theta RoPE: a spec with a local/global theta split declares TWO host-written angle
    # tables instead of one, and each layer's q/k RoPE reads whichever is_global() says -- global
    # and sliding layers rotate the same head_dim by different theta values (and, on Gemma-4,
    # `rope_global` alone is also partial-rotary; see main()'s table construction).
    dual_rope = sp.rope_theta_local is not None

    def ang_buf(l):
        return ("rope_global" if sp.is_global(l) else "rope_local") if dual_rope else "rope"

    # Gated DeltaNet (Qwen3.5's linear attention), M tokens a chunk: the in_proj rows land in a
    # [taps-1+M, LCH] conv window `cwp` behind the history rows copied in from decode's `cw`, the
    # conv runs in place, and the delta rule steps GDR_T tokens a call with its state in L1. The
    # output lands in `cxt`, where o_proj reads it, as in decode. A padded last chunk must not reach
    # the state: the host writes each call's real-token count into `gdr_count`, and the history
    # handed back is window rows [n_valid, n_valid + taps-1), picked by the `hist_off` parameter
    # before the conv overwrites them.
    lin_layers = [l for l in range(NL) if sp.mixer_for(l) == "linear_attention"]
    if lin_layers:
        from iron.operators.conv1d_step.op import Conv1dStep
        from iron.operators.gated_delta_rule.op import GatedDeltaRule
        if not dec_meta_path:
            raise ValueError(f"{sp.name}: the DeltaNet state lives in decode's arena; build with "
                             f"--decode-meta")
        LDK, LTAPS = sp.lin_head_dim, sp.lin_conv_taps
        LKD, LVD = sp.lin_k_heads * LDK, sp.lin_v_heads * LDK
        LCH = 2 * LKD + LVD
        LZAB = LVD + max(2 * sp.lin_v_heads, LDK)   # decode's z | a | b row, a|b padded to dk
        LAB = LZAB - LVD
        HIST = (LTAPS - 1) * LCH
        GDR_T = int(os.environ.get("PREFILL_GDR_TOKENS", "16"))
        # The state as bf16 limbs, native MACs: 7x the emulated-f32 step, 16 mantissa bits not 24.
        GDR_LIMBS = os.environ.get("PREFILL_GDR_LIMBS", "1") == "1"
        if LVD != QD:
            raise ValueError(f"{sp.name}: the DeltaNet output takes o_proj's slot, so its value "
                             f"width ({LVD}) must equal q_dim ({QD})")
        if M % GDR_T:
            raise ValueError(f"M={M} is not a whole number of GDR_T={GDR_T} delta-rule steps")
        # The widest conv tile whose window, output and weight fifos fit L1, double-buffered if
        # any tile allows it, else single (a long window: 259 rows at M=256).
        conv_fits = [(d, t) for d in (2, 1) for t in (256, 128, 64, 32) if (LCH // cols) % t == 0
                     and d * (2 * (LTAPS - 1 + M) + LTAPS) * t * 2 <= 56 * 1024]
        if not conv_fits:
            raise ValueError(f"no conv tile of a [{LTAPS - 1 + M}, t] window fits L1")
        conv_depth, conv_tc = conv_fits[0]
        op_lqkv = gemm_for("lin_qkv", D, LCH)
        op_lz = gemm_for("lin_z", D, LVD)
        op_lab = gemm_for("lin_ab", D, LAB)
        op_hist = StridedCopy(input_sizes=(HIST,), input_strides=(1,), input_offset=0,
                              output_sizes=(HIST,), output_strides=(1,), output_offset=0,
                              input_buffer_size=HIST, output_buffer_size=HIST,
                              transfer_size=pick_transfer(HIST), num_aie_channels=1, context=ctx)
        op_hist_out = StridedCopy(input_sizes=(HIST,), input_strides=(1,), input_offset=0,
                                  output_sizes=(HIST,), output_strides=(1,), output_offset=0,
                                  input_buffer_size=(LTAPS - 1 + M) * LCH, output_buffer_size=HIST,
                                  transfer_size=pick_transfer(HIST), num_aie_channels=1,
                                  input_offset_parameter="hist_off", context=ctx)
        op_conv = Conv1dStep(channels=LCH, taps=LTAPS, tokens=M, tile_channels=conv_tc,
                             depth=conv_depth if conv_depth != 2 else None,
                             num_aie_columns=cols, context=ctx)
        op_mix_act = SiLUAct(size=M * LCH, num_aie_columns=cols, tile_size=LCH // cols,
                             context=ctx, allocation_scheme=alloc_all)
        op_gdr = GatedDeltaRule(v_heads=sp.lin_v_heads, k_heads=sp.lin_k_heads, dk=LDK, dv=LDK,
                                ab_len=LAB, ab_off=0, mixed_len=LCH, q_off=0, k_off=LKD,
                                v_off=2 * LKD, tokens=GDR_T, l2_qk=True, counted=True,
                                limbs=GDR_LIMBS or None, num_aie_columns=cols, context=ctx)
        op_gnorm = RMSNorm(size=M * LVD, num_aie_columns=cols, num_channels=1, tile_size=LDK,
                           weighted=True, epsilon=sp.eps, context=ctx, allocation_scheme=alloc_all)
        op_z_act = SiLUAct(size=M * LVD, num_aie_columns=cols, tile_size=LVD // cols, context=ctx,
                           allocation_scheme=alloc_all)
        op_o_mul = ElementwiseMul(size=M * LVD, num_aie_columns=cols, tile_size=LVD // cols,
                                  context=ctx, allocation_scheme=alloc_all)
        bufsz.update({"cwp": (LTAPS - 1 + M) * LCH * 2, "lz": M * LVD * 2, "lab": M * LAB * 2,
                      "lo": M * LVD * 2, GDR_COUNT: (M // GDR_T) * LDK * 2})

    def deltanet_rl(p, out):
        el = lambda buf, lo, n: f"{buf}[{lo * 2}:{(lo + n) * 2}]"
        rows = el("cwp", HIST, M * LCH)
        rl_ = [(op_hist, el(p + "cw", 0, HIST), el("cwp", 0, HIST))]
        pre, w = weight_rows(p + "Wlqkv", LCH, 0, LCH, D)
        rl_ += pre + [(op_lqkv, "h", w, rows),
                      (op_hist_out, "cwp", el(p + "cw", 0, HIST)),
                      (op_conv, "cwp", p + "cvw", "cwp"),
                      (op_mix_act, rows, rows)]
        pre, w = weight_rows(p + "Wlzab", LZAB, 0, LVD, D)
        rl_ += pre + [(op_lz, "h", w, "lz")]
        pre, w = weight_rows(p + "Wlzab", LZAB, LVD, LAB, D)
        rl_ += pre + [(op_lab, "h", w, "lab")]
        for c in range(M // GDR_T):
            rl_.append((op_gdr, el("lab", c * GDR_T * LAB, GDR_T * LAB), p + "gdp",
                        el(GDR_COUNT, c * LDK, LDK),
                        el("cwp", HIST + c * GDR_T * LCH, GDR_T * LCH), p + "S", p + "S",
                        el("lo", c * GDR_T * LVD, GDR_T * LVD)))
        return rl_ + [(op_gnorm, "lo", p + "gnw", "lo"), (op_z_act, "lz", "lz"),
                      (op_o_mul, "lo", "lz", out)]

    if sp.attn_output_gate:
        if not seam:
            raise ValueError("the attention output gate multiplies the token-major context; "
                             "build with PREFILL_HEAD_SEAM=1")
        op_gate_act = SigmoidAct(size=M * QD, num_aie_columns=cols, num_channels=1,
                                 tile_size=QD // cols, context=ctx, allocation_scheme=alloc_all) \
            if not ACT_POLY else SigmoidAct(size=M * QD, num_aie_columns=cols, tile_size=QD // cols,
                                             context=ctx, allocation_scheme=alloc_all)
        op_gate_mul = ElementwiseMul(size=M * QD, num_aie_columns=cols, tile_size=QD // cols,
                                     context=ctx, allocation_scheme=alloc_all)
        bufsz["gt"] = M * QD * 2

    rl, cache_names, layer_starts, ring_layers = [], [], [], []
    for l in range(NL):
        layer_starts.append(len(rl))
        p = f"L{l}_"
        src = "x" if l == 0 else "xs"
        dst = "xout" if l == NL - 1 else "xs"
        g = attn_ops(sp.head_dim_for(l), sp.n_kv_heads_for(l), sp.has_v_proj(l))
        hd, grp = g.hd, g.grp
        # Geometry-suffixed buffer names -- see the bufsz comment above for why one shared "q"
        # cannot serve two geometries.
        qb, kb, vb, cxtb = f"q{g.sfx}", f"k{g.sfx}", f"v{g.sfx}", f"cxt{g.sfx}"
        qhb, cxb = f"qh{g.sfx}", f"cx{g.sfx}"
        w_nin, w_nqn, w_nkn = attn_norms(p, hd)
        lin = sp.mixer_for(l) == "linear_attention"
        if lin:
            rl += [(op_norm, src, w_nin, "h")] + deltanet_rl(p, cxtb)
        else:
            rl += ([(op_norm, src, w_nin, "h")] + qkv_rl(p, "q", g, l, qb)
                   + qkv_rl(p, "k", g, l, kb) + (qkv_rl(p, "v", g, l, vb) if g.has_v else []))
            if sp.attn_output_gate:
                pre, w = weight_rows(p + "Wgate", g.qd, 0, g.qd, D)
                rl += pre + [(g.op_gq, "h", w, "gt"), (op_gate_act, "gt", "gt")]
            if sp.v_norm and not g.has_v:
                # attention_k_eq_v: no v_proj at all. v_norm reads the RAW k projection -- before
                # qk-norm and RoPE, which mutate `k` in place below -- and writes `v`; that IS the
                # copy, so no separate copy operator (mirrors gen_llm_decode.py:1907-1922 exactly).
                rl.append((g.op_vn, kb, f"ones_h{hd}", vb))
            elif sp.v_norm:
                rl.append((g.op_vn, vb, f"ones_h{hd}", vb))
            qn_runs = ([(g.op_kn, f"{qb}[{i * M * g.kvd * 2}:{(i + 1) * M * g.kvd * 2}]", w_nqn,
                         f"{qb}[{i * M * g.kvd * 2}:{(i + 1) * M * g.kvd * 2}]") for i in range(grp)]
                       if merge_qknorm else [(g.op_qn, qb, w_nqn, qb)])
            rl += qn_runs + [
                (g.op_kn, kb, w_nkn, kb),
                (g.op_rq, qb, ang_buf(l), qb),
                (g.op_rk, kb, ang_buf(l), kb),
            ]
            if not g.ring:
                # K after qk-norm AND after RoPE; V raw, projection only. Different points in the
                # pipeline, and the M=1 path a decode step resumes from depends on both. Moved past
                # the context read on a ring layer -- see the `g.ring` branch below.
                rl += [(g.op_kvapp, kb, p + "kc"), (g.op_kvapp, vb, p + "vc")]
            rl += [] if seam else [(g.op_q2h, qb, qhb)]
            # The widths buffer is an INPUT of the softmax step, not a side channel: op.get_arg_spec()
            # puts it between in and out, so it is the middle name here.
            scb, swb = f"sc{g.sfx}", f"sw{g.sfx}"

            if g.ring:
                # Read-first ring runlist (sec 2.1): concat [ring g.w | this chunk's own M rows], one
                # (hole_lo, hole_hi, width) mask per row, commit to the ring AFTER the context is read
                # (K007 already refused this geometry unless seam=True). Grouped by row type (all Hq
                # sc_ring, then all Hq sc_batch, ...) so each type is one configure (D009).
                row_w = g.w + M

                def qslice_ring(h):
                    return f"{qb}[{h * hd * 2}:{(h * hd + g.op_sc_ring.a_elems) * 2}]"

                for h in range(Hq):
                    rl.append((g.op_sc_ring, qslice_ring(h), kv_slab(p + "kc", h // grp, g),
                               f"{scb}[{h * M * row_w * 2}:"
                               f"{h * M * row_w * 2 + g.op_sc_ring.c_elems * 2}]"))
                for h in range(Hq):
                    kg = h // grp
                    rl.append((g.op_sc_batch, qslice_ring(h),
                               f"{kb}[{kg * hd * 2}:{kg * hd * 2 + g.op_sc_batch.b_elems * 2}]",
                               f"{scb}[{h * M * row_w * 2 + g.w * 2}:"
                               f"{h * M * row_w * 2 + g.w * 2 + g.op_sc_batch.c_elems * 2}]"))
                rl.append((g.op_sm_ring, scb, SM_RING, swb))
                for h in range(Hq):
                    cx_out = f"{cxtb}[{h * hd * 2}:{(h * hd + g.op_cx_ring.c_elems) * 2}]"
                    rl.append((g.op_cx_ring,
                               f"{swb}[{h * M * row_w * 2}:"
                               f"{h * M * row_w * 2 + g.op_cx_ring.a_elems * 2}]",
                               kv_slab(p + "vc", h // grp, g), cx_out))
                cxt2b = f"cxt2{g.sfx}"
                for h in range(Hq):
                    kg = h // grp
                    cx2_out = f"{cxt2b}[{h * hd * 2}:{(h * hd + g.op_cx_batch.c_elems) * 2}]"
                    rl.append((g.op_cx_batch,
                               f"{swb}[{h * M * row_w * 2 + g.w * 2}:"
                               f"{h * M * row_w * 2 + g.w * 2 + g.op_cx_batch.a_elems * 2}]",
                               f"{vb}[{kg * hd * 2}:{kg * hd * 2 + g.op_cx_batch.b_elems * 2}]",
                               cx2_out))
                rl.append((g.op_add_cx, cxtb, cxt2b, cxtb))
                # KV commit LAST: past the wrap, appending before the context read above would
                # overwrite ring slots this chunk's own later rows still need (sec 1.2/1.4).
                # `check_kv_append_order` below is the build-time assert that catches this reordered.
                rl += [(g.op_kvapp, kb, p + "kc"), (g.op_kvapp, vb, p + "vc")]
            else:
                def qslice(h):
                    """Head h's queries: a strided slice of token-major `q`, or the head-major copy."""
                    if not seam:
                        return f"{qhb}[{h * M * hd * 2}:{(h + 1) * M * hd * 2}]"
                    return f"{qb}[{h * hd * 2}:{(h * hd + g.op_sc.a_elems) * 2}]"

                def score(h):
                    return (g.op_sc, qslice(h), kv_slab(p + "kc", h // grp, g),
                            f"{scb}[{h * M * g.w * 2}:{(h + 1) * M * g.w * 2}]")

                def soft(h):
                    sl = f"{scb}[{h * M * g.w * 2}:{(h + 1) * M * g.w * 2}]"
                    out = f"{swb}[{h * M * g.w * 2}:{(h + 1) * M * g.w * 2}]"
                    wid = f"{SM_WIDTHS}[{h * M * 4}:{(h + 1) * M * 4}]"
                    return (g.op_sm_head, sl, wid, out) if causal == "rows" else (g.op_sm_head, sl, out)

                if attn_order == "interleaved":
                    # Only the first `n_il` heads alternate; the rest stay grouped. The knob exists
                    # because a configure costs ~80 KB of instruction stream, so interleaving all 16
                    # heads built a 157 MB ELF that the driver refuses to allocate a BO for (CREATE_BO
                    # EAGAIN, reproducible). +2 configures per interleaved head per layer.
                    for h in range(n_il):
                        rl += [score(h), soft(h)]
                    rl += [score(h) for h in range(n_il, Hq)]
                    rl += [soft(h) for h in range(n_il, Hq)]
                elif attn_order == "grouped":
                    rl += [score(h) for h in range(Hq)] + [soft(h) for h in range(Hq)]
                else:
                    rl += [score(h) for h in range(Hq)]
                    rl.append((g.op_sm, scb, SM_WIDTHS, swb) if causal == "rows"
                              else (g.op_sm, scb, swb))
                for h in range(Hq):
                    cx_out = (f"{cxtb}[{h * hd * 2}:{(h * hd + g.op_cx.c_elems) * 2}]" if seam
                              else f"{cxb}[{h * M * hd * 2}:{(h + 1) * M * hd * 2}]")
                    rl.append((g.op_cx, f"{swb}[{h * M * g.w * 2}:{(h + 1) * M * g.w * 2}]",
                               kv_slab(p + "vc", h // grp, g), cx_out))
            if sp.attn_output_gate:
                rl.append((op_gate_mul, cxtb, "gt", cxtb))
        rl += ([] if seam or lin else [(g.op_h2t, cxb, cxtb)]) + \
            weight_gemm(p, "Wo", g.op_o, g.qd, D, cxtb, "a", f"o_hd{hd}", site="o", layer=l) + \
            ([(op_norm, "a", p + "n_pa", "a")] if sp.sandwich_norms else []) + [
            (op_add, src, "a", "xs"),
            (op_norm, "xs", p + "n_pf", "hf"),
        ] + (weight_gemm(p, "Wg", op_gate, D, FF, "hf", "gs", "gate_up", site="mlp", layer=l) +
             weight_gemm(p, "Wu", op_gu, D, FF, "hf", "u", "gate_up", site="mlp", layer=l)
             if fuse_silu else
             weight_gemm(p, "Wg", op_gu, D, FF, "hf", "g", "gate_up", site="mlp", layer=l) +
             weight_gemm(p, "Wu", op_gu, D, FF, "hf", "u", "gate_up", site="mlp", layer=l) +
             [(op_act, "g", "gs")]) + [
            (op_mul, "gs", "u", "gh"),
        ] + weight_gemm(p, "Wd", op_down, FF, D, "gh", "d", "down", site="mlp", layer=l) + \
            ([(op_norm, "d", p + "n_pff", "d")] if sp.sandwich_norms else []) + [
            (op_add, "xs", "d", dst),
        ]
        if op_lscale is not None:
            # `L{l}_ls` is named scratch in decode's own meta.json (a per-layer weight, not a
            # shared intermediate), so scratch_order needs it declared even when an arm/probe
            # below skips or collapses the per-row dispatch -- otherwise every later shared
            # buffer shifts off decode's offsets.
            bufsz[p + "ls"] = D * 2
            if lscale_bcast:
                # One dispatch: op_lscale.get_arg_spec() (rows=M) wants the whole [M, D] buffer
                # on A and C, unsliced -- not the per-row slice the loop below uses.
                rl.append((op_lscale, dst, p + "ls", dst))
            elif not no_layer_scalar:
                # Last statement of the layer, after both residual adds -- in place on `dst`,
                # against decode's shared `p+"ls"` (same arena slot, same D-wide value every row).
                rl += [(op_lscale, f"{dst}[{r * D * 2}:{(r + 1) * D * 2}]", p + "ls",
                        f"{dst}[{r * D * 2}:{(r + 1) * D * 2}]") for r in range(M)]
        cache_names += [p + "cw", p + "S"] if lin else [p + "kc", p + "vc"]
        ring_layers.append(False if lin else g.ring)

    if wdq_elems[0]:
        bufsz["wdq"] = wdq_elems[0] * 2
    uses_ring = any(ring_layers)
    # (head_dim, capacity) per ring-enabled geometry -- what `check_prefill_pairing` (artifact.rs)
    # matches a circular decode geometry against before accepting a prefill past its window.
    ring_geoms = sorted({(attn_ops(hd, hkv, has_v).hd, attn_ops(hd, hkv, has_v).w)
                         for hd, hkv, has_v in geoms if attn_ops(hd, hkv, has_v).ring})
    if uses_ring:
        check_kv_append_order(rl, layer_starts, NL, ring_layers)
        print(f"[layout] read-before-write assert OK ({sum(ring_layers)}/{NL} ring layer(s))")

    # `sm_widths` goes LAST so x and rope keep the input-arena offsets the non-causal arm gives
    # them: add_buffers walks input_args in order, and the host's x/rope writes are the same in
    # both arms. `sm_ring` after it for the same reason.
    inputs = ["x"] + (["rope_local", "rope_global"] if dual_rope else ["rope"]) + \
        ([SM_WIDTHS] if causal == "rows" else []) + ([SM_RING] if uses_ring else []) + \
        ([GDR_COUNT] if lin_layers else [])

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
        # A THIRD source, beside the decode arena and the host: a quantized site's own packed
        # weight, filled by THIS build (main(), after compile) rather than left for a request to
        # write -- see weight_gemm/qkv_operand's quantized branch and quant_pack above.
        quant_names = {e["buf"] for e in quant_pack}
        orphans = sorted(read - written - set(dec_sizes) - set(inputs) - quant_names)
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
    if merge_qknorm:
        name += "_mqn"
    if seam:
        name += "_nseam"
    if fuse_silu:
        name += "_fsilu"
    if uses_ring:
        name += "_ring"
    # The tiling is now a per-shape lookup, so it is a GRAPH knob like the three above and has to
    # be in the name for the same reason: a re-sweep that moves one GEMM's tile must not link the
    # previous tiling's ELF out of the artifact cache. Hashed rather than spelled out -- seven
    # ops * four numbers does not belong in a filename, and the tiles themselves are in meta.json.
    tile_sig = ";".join(f"{k}:{v['tile']}x{v['cols']}" for k, v in sorted(tiles.items()))
    name += "_t" + hashlib.md5(tile_sig.encode()).hexdigest()[:8]
    if PREFILL_SEGMENTS > NL:
        raise SystemExit(f"PREFILL_SEGMENTS={PREFILL_SEGMENTS} exceeds the {NL} layers there are "
                         f"to split; a segment boundary only exists at a layer boundary")
    # Contiguous and near-equal, remainder to the earliest segments -- 48 over 4 is 12/12/12/12 and
    # 48 over 5 is 10/10/10/9/9. Same rule as DECODE_SEGMENTS so the two read alike.
    _q, _r = divmod(NL, PREFILL_SEGMENTS)
    cuts, _a = [], 0
    for _i in range(PREFILL_SEGMENTS):
        _b = _a + _q + (1 if _i < _r else 0)
        cuts.append((_a, _b))
        _a = _b
    seg_rls = [rl[layer_starts[la]:(layer_starts[lb] if lb < NL else len(rl))] for la, lb in cuts]
    # A graph knob, so it goes in the NAME for the reason this block already gives: IRON keys the
    # cached artifact by name and an arm that collides runs the earlier binary.
    if len(cuts) > 1:
        name += f"_seg{len(cuts)}"
    if A_RESIDENT:
        name += "_ares"
    if lin_layers:
        name += f"_gdr{GDR_T}" + ("l" if GDR_LIMBS else "")
    if _dq_cache:
        name += "_dq"
    if ACT_POLY:
        name += "_actf" if ACT_FAST else "_actp"
    # Both lscale arms change the RUNLIST (156 -> 93 entries/layer at M=64) while leaving every
    # other name component identical, which is the collision this block's header warns about --
    # and it bit: an A/B on 2026-09-19 compiled two arms to one md5 and measured nothing.
    if lscale_bcast:
        name += "_lsb"
    if no_layer_scalar:
        name += "_nols"
    print(f"[layout] name={name}")
    fused = OperatorSequence(name, seg_rls[0], input_args=inputs, output_args=["xout"],
                             buffer_sizes=bufsz, context=ctx, share_designs=True,
                             scratch_order=(dec_order or None),
                             extra_runlists={f"seg{i}": seg_rls[i] for i in range(1, len(cuts))})
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
    check_operand_bounds(rl, fused)

    # meta.json's tile_n_scores/tile_n_ctx are ONE representative number; a multi-geometry build
    # reports the base (sliding) geometry's and the full per-geometry breakdown is in `tiles`.
    _base_sfx = f"_hd{HD}" if multi_geom else ""
    tn_sc = tiles[f"scores{_base_sfx}"]["tile"][2]
    tn_cx = tiles[f"ctx{_base_sfx}"]["tile"][2]
    # Resolved HERE, not snapshotted earlier: every op above (including a quantized site's own
    # packed-weight buffers) has had its chance to add to `bufsz` by now, and everything in it
    # that is not one of decode's own shared names is prefill's, regardless of how late it was
    # added.
    prefill_local = sorted(n for n in bufsz if n not in dec_sizes)
    dims = dict(NL=NL, M=M, S=S, inputs=inputs, cache_names=cache_names,
                recurrent_counts=(dict(buffer=GDR_COUNT, tokens_per_call=GDR_T, calls=M // GDR_T,
                                       slot_bytes=LDK * 2, hist_param="hist_off",
                                       hist_row_elems=LCH) if lin_layers else None),
                tn_sc=tn_sc, tn_cx=tn_cx, tiles=tiles, cols=cols, causal=causal,
                kv_block=kvl.T, wqkv_head_major=hm, geom_slots=geom_slots,
                sm_widths=(SM_WIDTHS if causal == "rows" else None), sm_rows=Hq * M,
                uses_ring=uses_ring, ring_geoms=ring_geoms,
                shared=[n for n in dec_order if not n.startswith("__decode_gap")],
                reserved=dec_reserved, prefill_local=prefill_local, quant_pack=quant_pack,
                rl=rl, runlist_len=len(rl), per_layer=len(rl) // NL,
                segments=cuts, seg_kernels=["sequence"] + [f"seg{i}" for i in
                                                            range(1, len(cuts))],
                # TWO different numbers, and conflating them understated the configure count by
                # 31x. `n_designs` is how many designs get BUILT -- `share_designs` collapses
                # operators reporting the same design_key onto one. `n_configures` is how many
                # setups the dispatch PAYS for, and D009 prices that one: a configure covers a
                # CONTIGUOUS same-design block, offsets free inside it, so one design reached at
                # three separate points in the runlist costs three. Runs inside a block ride free.
                n_designs=len(fused.unique_designs()[0]),
                n_configures=count_configures(rl, fused))
    return sp, fused, dims


def check_kv_append_order(rl, layer_starts, NL, ring_layers):
    """K019-shaped build-time assert: a ring layer's runlist must read `L{l}_kc`/`L{l}_vc` before
    its own `op_kvapp` write (read-first, sec 1.2/2.1); every other layer must write before it
    reads (today's append-then-read). Catches the hazard sec 1.2 names by construction: swap
    `op_kvapp` back above the score GEMMs on a ring layer and this raises, naming the read that now
    runs after the write which just overwrote it.
    """
    for l in range(NL):
        lo = layer_starts[l]
        hi = layer_starts[l + 1] if l + 1 < NL else len(rl)
        for tag in ("kc", "vc"):
            name = f"L{l}_{tag}"
            write_idx, read_idx = [], []
            for i in range(lo, hi):
                op, *bufs = rl[i]
                for spec, nm in zip(op.get_arg_spec(), bufs):
                    if nm.split("[")[0] != name:
                        continue
                    (write_idx if spec.direction in ("out", "inout") else read_idx).append(i)
            if not write_idx:
                raise ValueError(f"{name}: op_kvapp never writes it in layer {l}'s runlist slice")
            w0 = write_idx[0]
            if ring_layers[l]:
                bad = [i for i in read_idx if i > w0]
                if bad:
                    raise ValueError(
                        f"{name}: ring layer {l} reads it at runlist index {bad[0]}, AFTER its "
                        f"own op_kvapp write at {w0} -- past the wrap this overwrites positions "
                        f"the chunk's later rows still need (sec 1.2)")
            else:
                bad = [i for i in read_idx if i < w0]
                if bad:
                    raise ValueError(
                        f"{name}: layer {l} reads it at runlist index {bad[0]}, BEFORE its own "
                        f"op_kvapp write at {w0}")


def check_operand_bounds(runlist, fused):
    """Every operand's own declared extent against the bytes its buffer actually holds.

    The `scratch_order` check above compares each shared buffer's (offset, len) with decode's meta
    and prefill INHERITS both from it, so it passes by construction and cannot see a descriptor
    that addresses PAST the end of a buffer whose length the two halves agree on. IRON does not
    own that either: `calculate_buffer_layout` records a slice's (start, end) and never compares
    it to the parent's length or to the arg spec's own byte count. Two operand failures, both
    checked here: wider than the slice it was handed, and leaving the buffer from that slice's
    start.

    What it catches, verified 2026-09-16 against decode_int4g32qat_s6912_l48_rg_mc: a scores/ctx
    GEMM built at the global S over a sliding geometry declares 3538944 B of B per head against a
    524288 B slab, and kv head 7 reaches 7208960 B in an `L0_kc` decode sized to 4194304.
    """
    worst = {}
    for op, *bufs in runlist:
        for spec, nm in zip(op.get_arg_spec(), bufs):
            base = nm.split("[")[0]
            buflen = fused.subbuffer_layout[base][2]
            if "[" in nm:
                lo, hi = (int(x) for x in nm[nm.index("[") + 1:-1].split(":"))
            else:
                lo, hi = 0, buflen
            need = int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize
            # Ranked so a span that leaves the BUFFER outranks one that merely overruns its slice
            # into a sibling: same arithmetic, but only the first corrupts the shared arena.
            rank = (max(lo + need - buflen, 0), max(need - (hi - lo), 0))
            if not any(rank):
                continue
            key = (type(op).__name__, base)
            if rank > worst.get(key, ((0, 0),))[0]:
                worst[key] = (rank, nm, need, hi - lo, lo, buflen)
    if not worst:
        return
    lines = [f"  {kind} on {nm}: declares {need} B from offset {lo}, reaching {lo + need} -- "
             f"slice holds {have} B and buffer {base!r} is {buflen} B"
             for (kind, base), (_, nm, need, have, lo, buflen) in sorted(worst.items())]
    raise ValueError(
        f"{len(worst)} operand(s) address past the buffer they were given:\n" + "\n".join(lines))


def count_configures(runlist, fused):
    """Contiguous same-design blocks in `runlist` -- the unit D009 prices, at 51.0-61.9 us each.

    Uses the sequence's OWN design assignment (`unique_designs()[1]`) rather than a second notion
    of identity here, so this cannot drift from what the build actually configures.
    """
    _, design_of = fused.unique_designs()
    ids = [design_of[id(op)] for op, *_ in runlist]
    return 1 + sum(1 for a, b in zip(ids, ids[1:]) if a != b) if ids else 0


def gemm_l3_repeats(op):
    """How many times the GEMM's runtime sequence re-streams each operand FROM DDR: (A, B).

    Neither is a reuse count. `iron/operators/gemm/design.py` fills A with
    `pattern_repeat=n_c_col_tiles_per_core` and re-issues `B_prods[col].fill()` inside the
    `tile_row` loop, so each operand is re-READ once per output tile along the axis the OTHER
    operand is tiled on. The MemTile is a conduit here, not a cache.
    """
    cols = op.num_aie_columns
    return (max(1, op.N // (op.tile_n * cols)),      # n_c_col_tiles_per_core
            max(1, op.M // (op.tile_m * 4)))         # n_c_row_tiles_per_core; n_aie_rows == 4


def gemm_b_bytes(op):
    """B's size in the unit GEMM declares it -- packed bytes when quantized, bf16 otherwise.

    Same rule as the builder's own `b_bytes`; read off `get_arg_spec` so the two cannot disagree.
    """
    n = 1
    for d in op.get_arg_spec()[1].shape:
        n *= d
    return n if op.weight_dtype != "bf16" else n * 2


def operand_traffic(op, bufs, resolve):
    """Bytes each of `op`'s operands actually moves, one per name in `bufs`.

    Not the slice length. A blocked operand's slice runs from its first block to the end of its
    last, past the peer matrices interleaved in between -- a KV head's slab spans 7.5x the bytes
    the head holds, and a head-major Wqkv role spans ~2.8x its own rows. StridedCopy is counted off
    its access pattern and GEMM off its shapes; everything else still reads its slice, which for a
    contiguous operand is the same number.

    CORRECTED 2026-09-16. GEMM used to return `[M*K*2, K*N*2, M*N*2]`: bf16 assumed for all three
    operands, one pass each. Both halves were wrong and they partly cancelled, which is why the
    total looked plausible. A quantized B read as bf16 is 3.56x its packed size, so `L0_Wg_qp`
    (35.16 MiB on disk) was counted at 112.50; and ignoring `gemm_l3_repeats` hid a 30x re-read of
    A at the gate/up/down sites. Corrected, gemma4-12b prefill at M=256/S=6912 reads 38.67 GiB per
    dispatch against the 35.38 GiB this function used to report, and the composition changes
    completely: A 16.33 GiB where one pass is 1.73, B 13.73 where one pass is 8.02.
    """
    if isinstance(op, StridedCopy):
        return [int(np.prod(op.input_sizes)) * 2] * len(bufs)
    if isinstance(op, GEMM):
        a_rep, b_rep = gemm_l3_repeats(op)
        wa = np.dtype(ml_dtypes.bfloat16 if op.dtype_in == "bf16" else op.dtype_in).itemsize
        wc = np.dtype(ml_dtypes.bfloat16 if op.dtype_out == "bf16" else op.dtype_out).itemsize
        return [op.M * op.K * wa * a_rep, gemm_b_bytes(op) * b_rep, op.M * op.N * wc]
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
    gemm_l3_reread_census(runlist, n_layers, total)


def gemm_l3_reread_census(runlist, n_layers, total):
    """The GEMM operands, split by A/B/C and by how often each is re-read from DDR.

    The line above ranks by OPERATOR, which cannot show this: a site is expensive here because the
    runtime sequence re-issues its fill, not because the operator is slow. `gemm_l3_repeats` owns
    the two factors.
    """
    sites, a_tot, a_once, b_tot, b_once, c_tot = {}, 0, 0, 0, 0, 0
    for op, *bufs in runlist:
        if not isinstance(op, GEMM):
            continue
        a_rep, b_rep = gemm_l3_repeats(op)
        a, b, c = op.M * op.K * 2, gemm_b_bytes(op), op.M * op.N * 2
        a_tot += a * a_rep; a_once += a
        b_tot += b * b_rep; b_once += b
        c_tot += c
        key = (bufs[1].split("[")[0].split("_", 1)[-1], a_rep, b_rep)
        sites[key] = sites.get(key, 0) + a * a_rep + b * b_rep + c
    if not sites:
        return
    mb = lambda v: v / n_layers / 2**20
    print(f"[census] GEMM L3 traffic, per layer -- A x{'N//(tile_n*cols)':>18}, "
          f"B x{'M//(tile_m*4)':>14}")
    print(f"[census]   A {mb(a_tot):>8.2f} MB   one pass {mb(a_once):>8.2f}   "
          f"re-read {a_tot - a_once:>12} B ({100 * (a_tot - a_once) / total:.1f}% of the graph)")
    print(f"[census]   B {mb(b_tot):>8.2f} MB   one pass {mb(b_once):>8.2f}   "
          f"re-read {b_tot - b_once:>12} B ({100 * (b_tot - b_once) / total:.1f}% of the graph)")
    print(f"[census]   C {mb(c_tot):>8.2f} MB")
    print(f"[census] {'worst re-read sites':<24}{'Arep':>5}{'Brep':>5}{'MB/layer':>10}")
    for (nm, ar, br), v in sorted(sites.items(), key=lambda kv: -kv[1])[:8]:
        print(f"[census]   {nm:<22}{ar:>5}{br:>5}{mb(v):>10.2f}")


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

    def klass(name, is_gemm_b):
        base = name.split("[")[0]
        if base.endswith("_kc") or base.endswith("_vc"):
            return "cache"
        if base == "sc" or base == "sw" or base.startswith(("sc_hd", "sw_hd")):
            return "scores"
        # A GEMM's B operand IS the weight, whoever owns the buffer. Membership in `shared` used
        # to stand in for this and answered a different question -- prefill repacks its own
        # quantized weights (`*_qp`, 6.81 GB of them), so none of them are decode's and all of
        # them read as activations. That is how `bytes.weights` came to report 0.3% of a stream
        # that is 61.6% weights.
        return "weights" if (is_gemm_b or base in shared) else "activations"

    for op, *bufs in runlist:
        gemm = isinstance(op, GEMM)
        for i, (name, moved) in enumerate(zip(bufs, operand_traffic(op, bufs, resolve))):
            out[klass(name, gemm and i == 1)] += moved
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
    if sp.mixer_for(0) != "full_attention":
        print(f"[check] shared-weight contract NOT verified: layer 0 is {sp.mixer_for(0)} and this "
              "check reads only an attention layer's buffers")
        return
    bdir = os.path.join(os.path.dirname(os.path.abspath(dec_meta_path)), "buffers")
    npy = lambda t: np.load(os.path.join(weights_dir, f"{sp.weight_prefix}layers.0.{t}.weight.npy"))
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
    # WHICH norm `n_pf` holds depends on the spec. With sandwich norms the layer carries three --
    # `n_pa` post-attention, `n_pf` PRE-feedforward, `n_pff` post-feedforward -- and `n_pf` is the
    # middle one; without them there is one and it is the post-attention norm. Claiming the
    # post-attention norm unconditionally reported gemma4-12b's correct arena as wrong.
    if sp.sandwich_norms:
        for nm, leaf in (("n_pa", "post_attention_layernorm"),
                         ("n_pf", "pre_feedforward_layernorm"),
                         ("n_pff", "post_feedforward_layernorm")):
            if os.path.exists(os.path.join(bdir, f"L0_{nm}.bin")):
                claim(same(raw(nm), npy(leaf).reshape(-1)), f"{nm} is the {leaf}")
    else:
        claim(same(raw("n_pf"), npy("post_attention_layernorm").reshape(-1)),
              "n_pf is the post-attention layernorm")
    # A weight is checked against the SAME dump file decode packed from, and the discriminator is
    # BYTES, never a dtype label: for a quantized site the arena holds the dump's packed bytes
    # verbatim (verified byte-for-byte on gemma4-12b -- L0_Wg, L0_Wu and L0_Wdk0 each equal their
    # own .npy exactly), so the widths agree and a raw compare is exact; for an unpacked one the
    # arena is bf16 against an f32 dump, the widths differ, and only then is the cast meaningful.
    # Reading a packed buffer as bf16 -- what this did before -- compares int4 payload against a
    # bf16 reference and fails every claim on any quantized arena.
    #
    # Both sides can split over K, and independently: decode splits into `<nm>k0..`, each a
    # contiguous K-slice that weight_gemm's `wk(i)` reads, and the DUMP offers `.kchunk0.npy ..`
    # for the same tensor. They are paired in K order.
    rawb = lambda p: np.fromfile(p, dtype=np.uint8)

    def arena_parts(nm):
        whole = os.path.join(bdir, f"L0_{nm}.bin")
        if os.path.exists(whole):
            return [whole]
        parts = []
        while os.path.exists(os.path.join(bdir, f"L0_{nm}k{len(parts)}.bin")):
            parts.append(os.path.join(bdir, f"L0_{nm}k{len(parts)}.bin"))
        return parts

    def check_weight(nm, t, what):
        parts = arena_parts(nm)
        if not parts:
            bad.append(f"{nm} is in the arena neither whole nor as k-chunks")
            return
        files = quant_source_files(weights_dir, f"{sp.weight_prefix}layers.0.", f"{t}.weight")
        if not files:
            bad.append(f"{nm}: {t} is in {weights_dir} neither whole nor as kchunks")
            return
        if len(parts) != len(files):
            bad.append(f"{nm} is {len(parts)} arena part(s) against {len(files)} dump file(s)")
            return
        for i, (part, fn) in enumerate(zip(parts, files)):
            ref = np.load(os.path.join(weights_dir, fn))
            tag = f"{nm}{'' if len(parts) == 1 else f'k{i}'}"
            a = rawb(part)
            if a.nbytes >= ref.nbytes:
                # Widths agree (or the arena is longer): the dump is already in the arena's own
                # packed representation. `[:ref.nbytes]` generalises the old `[:D * QD]` slice --
                # decode pads Wo's tail for fuse_o, and the pad sits past the reference's bytes.
                claim(np.array_equal(a[:ref.nbytes], ref.reshape(-1).view(np.uint8)),
                      f"{tag} is {what} unreordered (packed bytes)")
            else:
                # Arena narrower than the dump: an unpacked site, bf16 against the f32 dump.
                # `[:ref.size]` is the element-side twin of the packed path's `[:ref.nbytes]`, and
                # carries the old `[:D * QD]` slice: decode pads Wo's tail for fuse_o.
                claim(same(np.fromfile(part, dtype=BF16)[:ref.size], ref.reshape(-1)),
                      f"{tag} is {what} unreordered")

    check_weight("Wo", "self_attn.o_proj", "o_proj")
    for nm, t in (("Wg", "mlp.gate_proj"), ("Wu", "mlp.up_proj"), ("Wd", "mlp.down_proj")):
        check_weight(nm, t, t)

    if sp.layer_scalar:
        # layer_scalar_name() has no ".weight" suffix, unlike every other leaf `npy()` assumes --
        # a bespoke load. decode's `L0_ls` buffer holds it broadcast D-wide (same value every row).
        ls_val = np.load(os.path.join(weights_dir, f"{sp.layer_scalar_name(0)}.npy")).reshape(-1)[0]
        claim(same(raw("ls"), np.full(D, ls_val, np.float32)),
              "ls is the layer_scalar gain, broadcast D-wide")

    # Same packed-vs-bf16 split as the weights above. On a packed arena the stock stacking IS a
    # byte concatenation of the three dumps -- verified byte-for-byte on gemma4-12b -- and the
    # bf16 read below cannot see it: it reinterprets int4 payload as bf16 and reports a correct
    # arena as wrong.
    qkv_files = [quant_source_files(weights_dir, f"{sp.weight_prefix}layers.0.",
                                    f"self_attn.{r}_proj.weight") for r in ("q", "k", "v")]
    wqkv_raw = rawb(os.path.join(bdir, "L0_Wqkv.bin"))
    qkv_packed = all(len(f) == 1 for f in qkv_files) and wqkv_raw.nbytes == sum(
        np.load(os.path.join(weights_dir, f[0])).nbytes for f in qkv_files)
    if qkv_packed:
        if dims["wqkv_head_major"]:
            # A head's ROWS are not a contiguous byte range once packed: the dump keeps payload
            # and scales in per-tensor planes. Not expressible from this data -- say so rather
            # than emit a claim that cannot be true.
            print("[check] Wqkv head-major over a packed arena is not checkable here "
                  "(per-head byte offsets are not defined by a planar dump) -- skipped")
        else:
            cat = np.concatenate([np.load(os.path.join(weights_dir, f[0])).reshape(-1).view(np.uint8)
                                  for f in qkv_files])
            claim(np.array_equal(wqkv_raw, cat),
                  "Wqkv is the stock [Wq|Wk|Wv] stacking (packed bytes)")
        q = k = v = w = None
    else:
        w = raw("Wqkv").reshape(-1, D)
        q, k, v = (npy(f"self_attn.{r}_proj") for r in ("q", "k", "v"))
    if w is not None and dims["wqkv_head_major"]:
        ok = True
        for c in range(sp.n_kv_heads):
            b = c * (grp + 2) * HD
            for g in range(grp):
                ok &= same(w[b + g * HD:b + (g + 1) * HD], q[(c * grp + g) * HD:(c * grp + g + 1) * HD])
            ok &= same(w[b + grp * HD:b + (grp + 1) * HD], k[c * HD:(c + 1) * HD])
            ok &= same(w[b + (grp + 1) * HD:b + (grp + 2) * HD], v[c * HD:(c + 1) * HD])
        claim(ok, f"Wqkv is head-major ({grp} q + 1 k + 1 v head blocks per kv head)")
    elif w is not None:
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
    ap.add_argument("--kv-alloc", type=int, default=int(os.environ.get("KV_ALLOC", "0")),
                    help="allocate kc/vc for this many positions while attention computes over "
                         "--seq, so a narrow-window prefill can share a wide-capacity decode's "
                         "arena. Defaults to $KV_ALLOC, the name decode reads, so one exported "
                         "value drives both generators. 0 = capacity is the window.")
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
    ap.add_argument("--quant-weights",
                    help="dir of a row-packed quantized (.npy) weight dump (decode's own dump "
                         "format -- quant.json + per-tensor .npy/.kchunkN.npy); enables GEMM's "
                         "weight_dtype path for qkv/gate/up/down. Omit for an all-bf16 build.")
    ap.add_argument("--quant-attn-o-weights",
                    help="separate quantized dump for attn_o (o_proj) only, when it uses a "
                         "different group_size than --quant-weights (defaults to it)")
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
        #
        # dkb == S is the KVLayout(T=S) degenerate case: one block, byte-identical to the flat
        # layout, no boundary to straddle. Exempted here to match the KV-append descriptor below
        # (`if kvl_g.T == kvl_g.S`), which imposes no divisibility requirement either.
        dkb = dm["dims"].get("kv_block")
        if dkb and dkb != dm["dims"]["S"] and a.batch % dkb:
            raise SystemExit(f"ERROR: decode artifact kv_block={dkb} does not divide --batch "
                             f"{a.batch}; the KV append would straddle a block boundary")
        # `kc`/`vc` are sized by the CAPACITY, so it is the capacity -- `--kv-alloc`, which
        # defaults to the window -- that has to be matched. A narrower attention window over a
        # wider shared cache is exactly what the flag exists for.
        #
        # Decode's own capacity is per GEOMETRY and lives in `kv_windows`; `dims.S` is its WINDOW
        # and equals the capacity only on an artifact built without KV_ALLOC. Take the widest
        # geometry, which is the one `--kv-alloc` names, and fall back to `dims.S` for every
        # artifact that predates the field.
        dec_caps = [e.get("window") for e in (dm.get("scratchpad", {}) or {}).get("kv_windows", [])
                    if e.get("window")]
        dec_capacity = max(dec_caps) if dec_caps else dm["dims"]["S"]
        for key, ours in (("S", a.kv_alloc or a.seq), ("layers", a.layers), ("d_model", sp.d_model),
                          ("head_dim", sp.head_dim), ("kv_heads", sp.n_kv_heads)):
            theirs = dec_capacity if key == "S" else dm["dims"][key]
            if key == "layers":
                if a.layers > theirs:
                    raise SystemExit(f"ERROR: --layers {a.layers} exceeds the decode artifact's "
                                     f"{theirs}; the shared arena has no L{theirs}+ buffers")
                continue
            if theirs != ours:
                hint = (f"; pass --kv-alloc {theirs} to compute over a {a.seq}-wide window of its "
                        f"cache" if key == "S" and not a.kv_alloc and a.seq < theirs else "")
                raise SystemExit(f"ERROR: decode artifact {key}={theirs}, this build {ours} -- "
                                 f"the shared cache/weight buffers would not match{hint}")

    quant_plan = build_quant_plan(a.quant_weights, a.quant_attn_o_weights)
    sp_, fused, dims = build_graph(a.spec, a.layers, a.batch, a.seq, a.causal,
                                   dec_meta_path, do_compile=not a.layout_only,
                                   quant_plan=quant_plan, kv_alloc=a.kv_alloc)
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
    dual_rope = sp.rope_theta_local is not None

    # prefill_ref.layer_stack is a plain pre-norm, single-theta, uniform-geometry golden -- it has
    # no sandwich-norm, dual-theta, v-norm or layer-scalar arm. Building one of those specs with a
    # golden would silently compare the device against a dataflow it does not run.
    golden_gaps = [n for n, on in (("sandwich_norms", sp.sandwich_norms), ("dual-theta RoPE",
                   dual_rope), ("v_norm", sp.v_norm), ("layer_scalar", sp.layer_scalar),
                   ("non-uniform geometry", not sp.geometry_is_uniform()),
                   ("linear-attention mixers", sp.mixer_types is not None),
                   ("attention output gate", sp.attn_output_gate)) if on]
    if golden_gaps and not a.no_golden:
        raise SystemExit(f"ERROR: {sp.name} sets {golden_gaps}, which prefill_ref.layer_stack "
                         f"does not model yet -- pass --no-golden (the device graph itself has no "
                         f"such restriction)")

    rng = np.random.default_rng(11)
    X = bf16(rng.standard_normal((M, D)).astype(np.float32) * 0.02)
    # `table` doubles as the single-theta case's whole input and dual-theta's GLOBAL half -- Gemma-4
    # rotates only 0.25 of the global layers' frequency pairs (rope_type "proportional"), over that
    # geometry's OWN head_dim, which may differ from the base `HD` (512 vs 256 on Gemma-4).
    global_hd = sp.global_head_dim if sp.global_head_dim is not None else HD
    table = rope_table(a.base, M, sp.rope_rotary_dim or global_hd, sp.rope_theta_global,
                       partial=sp.rope_partial_rotary)

    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    bdir = os.path.join(a.out, "buffers")
    open(os.path.join(bdir, "x.bin"), "wb").write(X.tobytes())
    if dual_rope:
        open(os.path.join(bdir, "rope_local.bin"), "wb").write(
            rope_table(a.base, M, HD, sp.rope_theta_local).tobytes())
        open(os.path.join(bdir, "rope_global.bin"), "wb").write(table.tobytes())
    else:
        open(os.path.join(bdir, "rope.bin"), "wb").write(table.tobytes())
    if dims["sm_widths"]:
        widths = causal_widths(a.base, M, S, Hq)
        want = fused.get_layout_for_buffer(SM_WIDTHS)[2]
        if widths.nbytes != want:
            raise SystemExit(f"ERROR: {SM_WIDTHS} is {widths.nbytes}B here and {want}B in the "
                             f"layout -- AIERuntimeArgSpec.dtype defaults to bfloat16, so this is "
                             f"what an unset dtype looks like")
        open(os.path.join(bdir, f"{SM_WIDTHS}.bin"), "wb").write(widths.tobytes())
    if GDR_COUNT in dims["inputs"]:
        # A full chunk: every delta-rule call steps all of its tokens.
        n_calls = fused.get_layout_for_buffer(GDR_COUNT)[2] // (sp.lin_head_dim * 2)
        counts = np.zeros((n_calls, sp.lin_head_dim // 2), np.int32)
        counts[:, 0] = M // n_calls
        open(os.path.join(bdir, f"{GDR_COUNT}.bin"), "wb").write(counts.tobytes())
    if dims["uses_ring"]:
        # One shared sm_ring, head-major like sm_widths (the triple is head-independent, sec 1.3).
        # First ring geometry's capacity -- every shipped spec has exactly one.
        ring_w = dims["ring_geoms"][0][1]
        rows = np.tile(ring_mask_rows(a.base, M, ring_w), (Hq, 1))
        want = fused.get_layout_for_buffer(SM_RING)[2]
        if rows.nbytes != want:
            raise SystemExit(f"ERROR: {SM_RING} is {rows.nbytes}B here and {want}B in the layout")
        open(os.path.join(bdir, f"{SM_RING}.bin"), "wb").write(rows.tobytes())

    if dims["quant_pack"]:
        pack_quant_weights(dims["quant_pack"], bdir, quant_plan)

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
    # `geom_slots` names the slots the META will advertise; `params.txt` is what the ELF actually
    # declares. They are produced by different halves of the build and nothing else compares them,
    # so a geometry whose `output_offset_parameter` was never threaded through ships an artifact
    # that cannot load -- which is how this one was found, at `LlmArtifact::load_prefill`.
    undeclared = [n for n, _, _, _ in dims["geom_slots"] if n not in scratchpad_params]
    if undeclared:
        raise SystemExit(f"ERROR: geom_slots names scratchpad parameter(s) {undeclared} that the "
                         f"ELF does not declare (params.txt has {sorted(scratchpad_params)})")

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
        # decode's shared names carry no .bin here (the bytes ARE decode's, at decode's offsets,
        # `weights_from` below) -- a quantized site's own buffers DO, in THIS artifact's own
        # `buffers/` dir, because they are a different byte layout from anything decode holds
        # (see weight_gemm's quantized branch). Rust's loader already has a path for exactly this
        # ("a prefill-only weight stays legal", npu_decode.rs) -- it just had no producer before.
        # Cache names are excluded too: decode ships no `.bin` for them (they're zero-filled by
        # length, not read), and `cache_buffers` below is where a consumer looks for them.
        "weights": [n for n in dims["shared"] if n not in dims["cache_names"]]
        + sorted({e["buf"] for e in dims["quant_pack"]}),
        "weights_from": (os.path.join(os.path.dirname(os.path.abspath(dec_meta_path)), "buffers")
                         if dec_meta_path else None),
        "cache_buffers": dims["cache_names"],
        # Not position-indexed: zeroed before a request's first chunk, never resumed mid-prefix.
        "recurrent_buffers": [n for n in dims["cache_names"] if n.endswith(("_cw", "_S"))],
        # Per chunk, what keeps pad rows out of that state; see host_protocol's gdr_count/hist_off.
        "recurrent_counts": dims["recurrent_counts"],
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
        # Present only on a build that ran a sliding (w < S) geometry read-first over the ring
        # (PREFILL_SLIDING_RING=1). `geoms` names which (head_dim, capacity) it covers --
        # `check_prefill_pairing` (artifact.rs) accepts a circular decode geometry past its window
        # only when this names that geometry and `capacity % M == 0`.
        "mask_ring": None if not dims["uses_ring"] else {
            "buffer": SM_RING,
            "dtype": "int32",
            "rows": dims["sm_rows"],
            "len": dims["sm_rows"] * 3 * 4,
            "row_index": "r = h * M + i, h in [0, q_heads), i in [0, M)",
            "geoms": [{"head_dim": hd, "capacity": ww} for hd, ww in dims["ring_geoms"]],
            "rule": "hole_lo = base % w; hole_hi = w if base < w else hole_lo + i + 1; "
                    "width = w + i + 1",
            "note": "one (hole_lo, hole_hi, width) triple per softmax row over the concat "
                    "[ring w | this chunk's own M rows]; mask_hole_bf16 writes -inf on "
                    "[hole_lo, hole_hi) and [width, w+M).",
        },
        "scratchpad": {
            "params": scratchpad_params,
            "kv_param": "kv_off",
            # No scalar causal width in either arm: `rows` streams a per-row vector instead, and
            # `none` masks nothing at all.
            "mask_param": None,
            "kv_params": [{"param": n, "head_dim": hd} for n, hd, _, _ in dims["geom_slots"]],
            # Per-geometry KV capacity, the pairing check reads this to catch a decode geometry
            # narrower than S with no matching prefill capacity -- see `geom_slots`'s own comment.
            "kv_windows": [{"kv_param": n, "head_dim": hd, "window": ww, "mask_param": mp}
                           for n, hd, ww, mp in dims["geom_slots"]],
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
                 # The dispatch variants, in the order the host must run them. One entry is the
                 # unsegmented artifact every build produced before; N entries mean N dispatches
                 # of `main:<kernel>` against ONE hw_context and ONE arena, residual crossing in
                 # scratch. A host that ignores this runs segment 0 and returns a tenth of a
                 # forward pass, which looks like a bad answer rather than a missing one.
                 "segments": [{"layers": [la, lb], "kernel": k}
                              for (la, lb), k in zip(dims["segments"], dims["seg_kernels"])],
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
            **({"rope_local": f"[{M}, {HD}] bf16, one row per absolute position base..base+{M}-1, "
                              f"INTERLEAVED [cos, sin, cos, sin, ...], theta={sp.rope_theta_local}",
                "rope_global": f"[{M}, {global_hd}] bf16, same layout, theta={sp.rope_theta_global}"
                              f"{f', partial={sp.rope_partial_rotary}' if sp.rope_partial_rotary else ''}"}
              if dual_rope else
              {"rope": f"[{M}, {sp.rope_rotary_dim or HD}] bf16, one row per absolute position "
                       f"base..base+{M}-1, INTERLEAVED [cos, sin, cos, sin, ...], "
                       f"theta={sp.rope_theta_global}"}),
            **({"rope_rotary_dim": sp.rope_rotary_dim} if sp.rope_rotary_dim else {}),
            "kv_off": "base * head_dim, element units, addr kind, written raw",
            **({GDR_COUNT: "one int32 per delta-rule call at byte c * 2 * lin_head_dim: "
                           "clamp(n_valid - c * GDR_T, 0, GDR_T), n_valid = the chunk's real tokens",
                "hist_off": "n_valid * (2 * lin_k_heads + lin_v_heads) * lin_head_dim, element "
                            "units: the conv history handed back is the window rows after the last "
                            "real token"} if GDR_COUNT in dims["inputs"] else {}),
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
            # Which sentence is true here is decided by `uses_ring` -- the SAME value that emits the
            # `mask_ring` block below, not a re-derivation. A capped-at-the-window limitation
            # inherited onto a ring artifact reads as a property of the graph while describing the
            # one it replaced.
            f"batched prefill covers positions [0, {min(ww for _, _, ww, _ in dims['geom_slots'])}) "
            f"only, not the full S={S}: past the narrowest geometry's capacity its circular cache "
            "holds a wrapped interval and mask_bf16's suffix mask cannot express one. The host "
            "stops there (npu_prefill.rs::batchable_window) and finishes the prompt stepwise."
            if not dims["uses_ring"] else
            f"batched prefill spans the prompt past the sliding window (ring capacities "
            f"{sorted({ww for _, ww in dims['ring_geoms']})}), because the wrapped interval is "
            "masked per row by mask_hole_bf16 rather than by a suffix width. Stage 1: a chunk "
            "needing >= 2 pad rows past a capacity is skipped and the tail finishes stepwise "
            "(npu_prefill.rs::ring_tail_end), so a prompt whose length is not a whole number of "
            "M-chunks still ends per-token.",
        ] if dims["causal"] == "rows" and any(ww < S for _, _, ww, _ in dims["geom_slots"]) else []) + ([
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
    # The artifact's own graph identity, recorded and PRINTED. meta.json already carried the
    # md5 of the decode it pairs with but never its own, so two arms that collided on one name
    # linked one binary and every device-free check -- meta's entry counts, the build log, the
    # layout census -- kept reporting the Python-side runlist they were built from. All three
    # agreed with each other and none of them described what ran. This is the line that makes a
    # collision visible in the log of the build that caused it.
    import hashlib as _hl
    meta["elf_md5"] = _hl.md5(elf).hexdigest()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[ok] elf md5 {meta['elf_md5']} -- compare arms on THIS, never on the name")
    print(f"[ok] {NL}-layer {sp.name} prefill ELF ({len(elf)}B), M={M} S={S} "
          f"causal={dims['causal']}, {dims['runlist_len']} runlist entries "
          f"({dims['per_layer']}/layer), scratch {scr/1e6:.1f} MB "
          f"({'shared with ' + os.path.basename(os.path.dirname(dec_meta_path)) if dec_meta_path else 'standalone'})"
          f" -> {a.out}")


if __name__ == "__main__":
    main()
