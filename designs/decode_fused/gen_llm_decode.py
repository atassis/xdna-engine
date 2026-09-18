#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decoder-LLM whole decode stack as ONE fused ELF, built from an `LlmSpec`.

Generalises the retired gen_gemma_decode.py (see git history) from one checkpoint to the spec
vocabulary in llm_decode_spec.py:
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
  python designs/decode_fused/gen_llm_decode.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --out artifacts/qwen3-0.6b/decode --layers 28
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer_blob import write_blob  # noqa: E402
from llm_decode_spec import (SPECS, C_TILE_GRANULE, L1_BYTES, L1_RESERVE,  # noqa: E402,F401
                             gemv_fits, gemv_tile_output, k_chunks_for, operator_rejects)

# Dataflow switches, read ONCE at module scope. They are consumed in three different functions
# (graph construction, the runlist, and the meta writer), and defining them next to their first
# use put them out of scope in the others -- four times in one session, because Python does not
# complain until the single run that matters.
# DEFAULTS FLIPPED ON 2026-09-07 (owner call) after a four-arm A/B, interleaved, 28 layers:
#
#   arm                        MB/token   ms/token   tok/s
#   base, no flags              3105.99     137.90    7.25
#   GQA_GROUPED_K+V             2166.47     100.03   10.00
#   TMV_CTX alone               2036.18      98.50   10.15
#   TMV_CTX + GQA_GROUPED_K     1566.42      79.13   12.64   <- the default then
#
# The 79.13 is the 2026-09-07 state and is STALE as a current rate: the configure cut and the two
# data-parallel fusions below took it to 49.35-49.55 ms/token (bench_llm_decode.py, 25 reps, sd
# 0.21-0.53), which is what the installed artifact measures. Kept as the row of the four-arm A/B
# it belongs to, not as the rail's rate.
#
# ~1.25x over the previous best arm, and 1.20x AHEAD of AMD's own mlir-air Qwen3 example (95.2
# ms/token on this box), which we were 1.44x behind on 2026-09-05. Correctness: 8/8 vs the bf16
# oracle at every step but one, and that one is a THREE-WAY EXACT bf16 tie (279/9625/15344 all at
# 16.7500, argmax broken by index) which the un-grouped arm happens to win by exactly one ulp.
# Determinism is bit-identical across passes. Full record: the 2026-09-07 log note in the journal.
#
# Each flag still takes "0" to turn it OFF, so every arm above is still reachable for A/B.
GROUPED_K = os.environ.get("GQA_GROUPED_K", "1") == "1"
# NOT flipped: TMV_CTX subsumes the v-side grouping (TMatVec reads vc per kv head itself, so the
# Repeat would materialise a `vr` nothing consumes). Setting this with TMV_CTX on is a no-op, and
# reading a null result from toggling it as evidence about grouping would be a mistake.
GROUPED_V = os.environ.get("GQA_GROUPED_V", "0") == "1"
# Context step as a transposed-A reduction over the V cache rows, deleting op_trv outright.
TMV_CTX = os.environ.get("TMV_CTX", "1") == "1"
TMV_RPC_DEFAULT = 64
TMV_RPC = int(os.environ.get("TMV_RPC", str(TMV_RPC_DEFAULT)))
# Block the SCORES GEMV's K cache, per geometry, where doing so wins a delivery. Default on and
# byte-identical on every geometry it does not win one for -- see scores_block_size(), which only
# returns a blocked T when the operator's own group_reuse verdict differs between the two layouts.
# "0" pins every geometry flat, which is the A/B control arm for the blocked one.
SCORES_KV_BLOCK = os.environ.get("SCORES_KV_BLOCK", "1") == "1"

import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports (new-mlir-aie port shim)
# Row-batch the scores GEMV by DEFAULT. SCORES_ROWBATCH is a ROWS COUNT read by IRON's
# decode_layer_dp/op.py, not a boolean, and IRON defaults it to 1 = OFF; aie::reduce_add_v folds at
# most 4 vectors, so 4 is the maximum. Measured on device at -1.218 / -2.426 / -4.742 ms per token
# at window 1024 / 2048 / 4096, 6/6 points faster, identical dispatch count. Gated 2026-09-12: both
# arms PASS tier2 with identical top-5 logits at the divergence step, which is the token-for-token
# parity the reordered bf16 accumulation needed.
#
# It is set HERE, not in scripts/build_llm_decode.sh, because verify_llm_decode.py imports
# build_graph from this file and never runs that script -- setting it there passed the gate
# VACUOUSLY, on IRON's default of 1. And not in IRON itself, which is a shared checkout.
# Override with SCORES_ROWBATCH=1.
os.environ.setdefault("SCORES_ROWBATCH", "4")  # noqa: E402
# Read back for sequence_name(): decode_layer_dp/op.py's own artifact name now tags a >1 rowbatch
# (see its `name` property), and this file's name must not diverge from it -- an rb=1 and an rb=4
# build were sharing this function's name entirely, the same collision TMV_CTX/GROUPED_K are
# named for above.
SCORES_ROWBATCH = int(os.environ["SCORES_ROWBATCH"])

from iron.common import AIEContext  # noqa: E402
from iron.common.kv_layout import KVLayout, derive_block_size  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemv.op import GEMV  # noqa: E402
from iron.operators.gemv.design import (MAX_GROUP_REUSE, group_reuse_n_vec,  # noqa: E402
                                        split_run as gemv_split_run)
# The packer MOVED in IRON 6a347dc ("the weight packer has two operators now, move it to
# iron/common"), and the workspace carries trees on both sides of it: integration-stack has only
# the new path, the vendored designs/iron_operators only the old. Try both and, if neither is
# there, say so naming both paths -- a bare ModuleNotFoundError names a module and not the
# mis-pointed IRON tree, which is the failure scripts/build_llm_decode.sh's gate exists to prevent.
try:                                                                            # noqa: E402
    from iron.common.quant import (                                            # noqa: E402
        quantize_weight, row_stride_bytes, derive_row_group, widest_chunk,      # noqa: E402
        dequantize_weight_chunked, _planar_to_rows)                            # noqa: E402
except ModuleNotFoundError:                                                     # noqa: E402
    try:                                                                        # noqa: E402
        from iron.operators.gemv.quant import quantize_weight, row_stride_bytes  # noqa: E402
        derive_row_group = widest_chunk = None  # pre-move tree: no row_group_planar  # noqa: E402
        dequantize_weight_chunked = None  # pre-move tree: no row_parallel_down either  # noqa: E402
        _planar_to_rows = None  # pre-move tree: no row_group_planar either  # noqa: E402
    except ModuleNotFoundError as e:                                            # noqa: E402
        raise ModuleNotFoundError(
            "no weight packer in this IRON tree: tried iron.common.quant (post-6a347dc) and "
            "iron.operators.gemv.quant (pre-6a347dc). Point IRON at a tree carrying one of them."
        ) from e
import precision  # noqa: E402
from iron.operators.rms_norm.op import RMSNorm  # noqa: E402
from iron.operators.rope.op import RoPE  # noqa: E402
from iron.operators.elementwise_add.op import ElementwiseAdd  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402
from iron.operators.softmax.op import Softmax  # noqa: E402
from iron.operators.strided_copy.op import StridedCopy  # noqa: E402
from iron.operators.transpose.op import Transpose  # noqa: E402
from iron.operators.tmatvec.op import TMatVec  # noqa: E402
from iron.operators.gelu.op import GELU  # noqa: E402
from iron.operators.silu.op import SiLU  # noqa: E402
from iron.operators.repeat.op import Repeat  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = 8      # num_aie_columns for every GEMV; the spec's check() is written against this
TSI = 4       # tile_size_input


def bf16(a):
    return np.asarray(a).astype(BF16)


# PRECISION. The per-site weight-format plan is declarative and lives in designs/decode_fused/
# precision.py: `PRECISION=<preset>`, `PRECISION='{"mlp": "int8a/g128"}'`, or a path to such a
# file. The legacy QUANT_MLP_DTYPE / QUANT_ATTN_DTYPE / QUANT_HEAD_DTYPE / QUANT_*_GROUP /
# QUANT_CLIP_SEARCH variables still resolve to a plan, and setting both forms is refused.
#
# WHICH COMBINATIONS ARE BUILDABLE is a property of this graph, not of the formats: it depends on
# which weights share an ObjectFifo, which operator declares each buffer, and how many shim
# channels are left. None of that is knowable here, so the plan is CHECKED in build_graph once
# the fused arms are decided, and `precision.check()` names the rule it refuses on.
PRECISION_PLAN, PRECISION_PROV = precision.plan_from_env()


def _spec(site):
    return PRECISION_PLAN.get(site, precision.BF16_SPEC)


# Row layout of a quantized weight -- see iron/common/quant.py. Not a site-level precision choice
# (every site shares one on-wire row shape); set from the dump's own quant.json declaration in
# build_graph, before any GEMV is constructed, and read here rather than threaded as a parameter
# because _quant_kw/_pack are module-level and the dump isn't known until a spec_name is chosen.
_BUILD_STATE = {"layout": "header_first", "scale_dtype": "f32"}


def _quant_kw(site, force_header_first=False):
    """`weight_dtype`/`group_size`/`layout` kwargs for the operator carrying one site's weight.
    Empty at bf16, so an unquantized call is the shape it would have had with no precision plane.
    row_group is deliberately omitted: GEMV self-derives it (K022), and passing it here would be
    a second, possibly-disagreeing computation of the same value.

    force_header_first: attn_block_dp's Wqkv reorder (wqkv_head_major, below) slices individual
    output-feature rows out of Wq/Wk/Wv, which needs the fixed per-row stride header_first gives
    and row_group_planar does not (a row's header sits ROW_GROUP*payload away from its own
    payload). See attn-block-quantized-wqkv-assumes-header-first: the reorder used to inherit
    whatever layout the dump declared and silently mis-sliced a planar dump. header_first buys
    the planar width benefit nothing at the group sizes this model ships (quant.py's
    max_legal_vec_size docstring), so forcing it here is free at the shipped config.
    """
    spec = _spec(site)
    if not spec.quantized:
        return {}
    kw = dict(weight_dtype=spec.dtype, group_size=spec.group_size)
    if _BUILD_STATE["layout"] != "header_first" and not force_header_first:
        kw["layout"] = _BUILD_STATE["layout"]
    if _BUILD_STATE["scale_dtype"] != "f32":
        # Sizes the weight buffer's row and the kernel's header: a build that reads a bf16-scale
        # dump with the f32 stride lands every payload pointer n_groups*2 bytes late.
        kw["scale_dtype"] = _BUILD_STATE["scale_dtype"]
    return kw


_SITE_OF_SUFFIX = {"Wqkv": "qkv", "Wq": "qkv", "Wk": "qkv", "Wv": "qkv", "Wo": "attn_o",
                   "Wg": "mlp", "Wu": "mlp", "Wd": "mlp", "W_head": "head",
                   "kc": "kv", "vc": "kv"}


def _site_of(buffer_name):
    """Which census site a weight buffer belongs to, by its `L<n>_<key>` suffix."""
    return _SITE_OF_SUFFIX.get(buffer_name.rsplit("_", 1)[-1]
                               if buffer_name.startswith("L") else buffer_name)


def _stream_pad_rows(wqkv, op):
    """attn_block_dp strides its L3 Wqkv by the SHARED STREAM TILE, not by the packed row: one
    acquire is exactly one padded row (design.py's quant_tile_bytes). The fill is never read --
    mv_quant takes K from -DDIM_K, not from the tile's extent."""
    if op is None or op.weight_dtype == "bf16":
        return wqkv
    from iron.operators.attn_block_dp.design import quant_tile_bytes
    WB, TB, _ = quant_tile_bytes(op.D, op.HD, op.group_size, op.weight_dtype, op.scale_dtype,
                                 op.max_seq)
    assert wqkv.shape[1] == WB, f"packed row {wqkv.shape[1]} B, operator expects {WB} B"
    return np.pad(wqkv, ((0, 0), (0, TB - WB)))


def _pack(w, site):
    """Host-side pack of one weight under its site's spec, into the packer's wire format."""
    spec = _spec(site)
    if not spec.quantized:
        return bf16(w).reshape(-1)
    kw = {}
    if spec.dtype in precision.SYMMETRIC:
        kw["clip_search"] = spec.scale_kind in ("clip", "clip_full")
        if spec.scale_kind == "clip_full":
            kw["full_range"] = True
    else:
        kw["affine_zero_on_grid"] = spec.scale_kind == "zero_grid"
    layout = _BUILD_STATE["layout"]
    if layout != "header_first":
        if derive_row_group is None:
            raise ModuleNotFoundError(
                f"_BUILD_STATE layout={layout!r} but this IRON tree has no derive_row_group "
                f"(pre-6a347dc). Point IRON at a tree past the quant.py move."
            )
        K = w.shape[-1]
        kw["layout"] = layout
        # K026: the width is `widest_chunk`, not `min(64, group_size)`. Those agreed until
        # chunk_scales (IRON 836ac0d) let a chunk span two groups and doubled the reader's
        # width for int4; this site kept the old rule, so int4 g32 sbf16 at K=3840 packed
        # row_group=1 while GEMV read row_group=2 -- a silent layout mismatch on every
        # K=3840 site. int8 g64 is unaffected (both rules give 64).
        kw["row_group"] = derive_row_group([K], spec.group_size, spec.dtype,
                                           vec_size=widest_chunk(spec.group_size, spec.dtype),
                                           scale_dtype=_BUILD_STATE["scale_dtype"])
    if _BUILD_STATE["scale_dtype"] != "f32":
        kw["scale_dtype"] = _BUILD_STATE["scale_dtype"]
    return quantize_weight(w, spec.group_size, spec.dtype, **kw)


DECODE_PLACER_FLAGS_DEFAULT = "--cores-per-col 1"

# INSTRUMENT, not a feature. Alternates the per-head qk-norm between two IDENTICAL RMSNorm
# instances. RMSNorm has no design_key, so two instances are two DESIGNS: the 24 consecutive runs
# stop sharing one aiex.configure and become 24. Runs, bytes and output are unchanged, so it
# isolates the cost of a CHEAP configure (18 KB of views) the way share_designs isolated an
# expensive one. Predicted +644 configures/token; at the measured 61.9 us for a big configure that
# is +39.9 ms if the cost is flat, and ~0 if it tracks the view count.
SPLIT_QKNORM = os.environ.get("SPLIT_QKNORM", "0") == "1"

# INSTRUMENT, not a feature -- the sibling of SPLIT_QKNORM above, aimed at the other count.
# SPLIT_QKNORM isolated a CONFIGURE by turning one design into many; this chops swiglu_mlp_dp's gh
# drain group into k groups over the SAME drains in the SAME order, so bytes, shim tasks, BDs,
# configures, designs and output are all identical and the ONLY quantity that moves is the number
# of SYNC POINTS: +(k-1) per layer, +28*(k-1) per token. That is the last unrefuted candidate for
# the per-layer transport residual -- 0.135-0.159 ms/layer over 10 TaskGroup closes is 13.5-15.9 us
# each, so at L=28 a k=12 arm predicts +4.2 to +4.9 ms if the cost is per sync point, and ~0 if it
# is not. 1 (default) is byte for byte the unsplit path.
SPLIT_GH_DRAIN = int(os.environ.get("SPLIT_GH_DRAIN", "1"))

# Fold the FFN activation into the gate GEMV as a fused tile epilogue.
#
# DEFAULT OFF, because on THIS graph it is a wash and the counting is the lesson. Deleting op_act
# removes one design, but giving op_gate an epilogue makes it differ from op_up, which breaks the
# share_designs pair those two were in. Measured device-free: designs 18 -> 18, configures 534 ->
# 534, runs 1262 -> 1234. share_designs and epilogue-folding target the SAME single configure here
# and are not additive; whichever runs second buys nothing.
#
# Kept because the mechanism is correct and is the one the folds that DO pay need -- RoPE into the
# q/k GEMVs, the residual adds into the o/down GEMVs -- where the consumer is not half of a shared
# pair. Turn it on with FUSE_ACT=1.
FUSE_ACT = os.environ.get("FUSE_ACT", "0") == "1"

# iron/operators/_trace.py wires per-op hardware trace into EVERY design.py this file can reach
# (decode_layer_dp's fused arm and the older per-operator arms alike), but is a documented no-op at
# its own default -- `maybe_enable_trace` returns before touching `prog` when this is 0, so it never
# perturbs a production build. Read here anyway: it is still a real graph change once set, and a
# traced .mlir must not share this function's name with the same build untraced.
IRON_TRACE_SIZE = int(os.environ.get("IRON_TRACE_SIZE", "0"))

# Replace the MLP block's SIX designs with ONE data-parallel fused design: every core runs every
# stage on its own 1/N slice, N=8 (one core per column). Measured standalone at -29.3% against the
# same six designs with a contemporaneous alternated control -- 1770.5 -> 1251.9 us/layer. N=16 and
# N=32 are SLOWER, because fitting them inside the 16-channel ShimDMA budget needs a MemTile
# split/join whose small strided-gather fills cost more than the finer parallelism buys.
FUSE_MLP_DP = os.environ.get("FUSE_MLP_DP", "1") == "1"
MLP_DP_COLS = int(os.environ.get("MLP_DP_COLS", "4"))

# Wq/Wk/Wv concatenated into ONE [QD+2*KVD, D] weight and projected by ONE GEMV writing a single
# `qkv` buffer; q/k/v become byte slices of it. Three runs and two configures per layer become one
# of each, and the weight stream becomes one contiguous 8.39 MB read instead of three. Arithmetic
# on the rows is untouched -- each output row is its own dot product -- so this arm is expected to
# be bit-identical, which is the gate it is checked against.
FUSE_QKV_GEMV = os.environ.get("FUSE_QKV_GEMV", "1") == "1"

# One RoPE run over the 24 q+k head rows instead of two runs of 16 and 8. Needs FUSE_QKV_GEMV,
# because it is only expressible when q and k are adjacent in one buffer. Same angle row, same
# per-row kernel, so also expected bit-identical.
FUSE_ROPE_QK = os.environ.get("FUSE_ROPE_QK", "1") == "1"

# Fold `attn_scale` into the q-norm gain instead of running an ElementwiseMul over the whole
# [Hq, S] score matrix. RMSNorm's gain multiply and RoPE's rotation are both linear in q, and the
# scores GEMV is linear in q, so scaling n_qn by attn_scale scales `sc` by exactly the same factor
# -- one design, one configure and 28 runs of the token deleted outright.
#
# NOT bit-identical: it removes a bf16 rounding of the intermediate `sc` and adds one of the gain.
# Same class of change as FUSE_MLP_DP's summation order, and gated the same way (token parity).
#
# The three flags together are worth 0.230 MB/layer = 6.44 MB/token, measured off the shim BDs of
# one controlled pair (400.88 -> 400.42 MB/dispatch at depth 2). It attributes exactly: op_scale's
# read and write of `sc` plus its `attn_scale` operand is 3 x 64 KB, and the two GEMV invocations
# FUSE_QKV_GEMV deletes each stopped broadcasting `hn` to 8 columns, 2 x 8 x 2 KB. At the fabric
# wall that is 0.12 ms of the measured -6.19, so these are dispatch levers that happen to move a
# few bytes, not the other way round.
SCALE_IN_QNORM = os.environ.get("SCALE_IN_QNORM", "1") == "1"

# The whole QKV head -- pre-attn RMSNorm, the concatenated QKV GEMV, the 24 per-head qk-norms and
# the q+k RoPE -- as ONE data-parallel design: every core owns a contiguous row slice of Wqkv and
# runs every stage on it. Four configures per layer become one, and `hn` stops reaching DDR.
#
# DEFAULT ON 2026-09-07 (owner call). Measured on device, ABBA, separate single-context process
# per arm, 18 cells per arm: 57.078 -> 54.071 ms/token, -3.007 ms, -5.27%, 17.52 -> 18.49 tok/s.
# The two arms' clean modes are DISJOINT (=0's minimum 56.862 exceeds =1's maximum 56.681), so no
# statistic choice carries the result and no outlier filter is applied. Teacher-forced parity is
# identical to the =0 arm.
#
# -84 configures bought 35.8 us each against the independently measured 51-62 us flat per-configure
# cost, so this design hands back ~1.6 ms of its own saving as per-invocation work. The three flags
# above went the OTHER way, 73.7 us/configure, because the concatenated GEMV also improved
# transport. Fusion is the configure model PLUS a per-design work delta whose sign the configure
# count does not predict -- do not price the next group off the count alone.
#
# The previous attempt at this group (fuse/qkv-head) measured +28.2% SLOWER, and neither of its two
# defects was fusing: see the operator's design.py. Needs FUSE_QKV_GEMV for the concatenated weight.
FUSE_QKV_DP = os.environ.get("FUSE_QKV_DP", "1") == "1"
# Build the lm-head as its OWN graph instead of the last op of the fused one, and hand `xf` between
# them through the host. Measured 2026-09-09: inside the fused graph at >=6 Gemma-4 layers the
# lm-head's output comes back with whole runs unwritten (2169 elements in 16 runs at 6 layers,
# doubling at 7) and a few wildly wrong, while the SAME GEMV at the SAME shape is exact standalone
# (rel-L2 1.41e-05, zero unwritten). So the fault is an interaction with the surrounding design and
# not the operation. `xf` is D*2 = 7680 bytes, so the round trip is negligible against a step that
# is already dispatch-dominated. See the-gemma4-fault-is-in-the-lm-head-output-not-the-layers.
SPLIT_LM_HEAD = os.environ.get("SPLIT_LM_HEAD", "0") == "1"

# Split the LAYER STACK across N dispatches, threading the residual between them through the host.
# The arena limit this exists for is `aiex.npu.address_patch`'s I32 arg_plus (see
# check_arena_offsets_are_addressable): a single 48-layer Gemma-4-12B arena is 9.13 GiB, so no
# arrangement of one dispatch can address it, and the fix that needs no toolchain change is to make
# each dispatch's arena small enough that the question does not arise.
#
# Splitting also closes the guard's known GAP for free. The guard tests a buffer's START offset and
# cannot see a buffer that starts below 2^32 and STREAMS across it -- the partial case that still
# holes the 6-layer lm-head. When every arena is under 4 GiB no buffer can start or end past the
# boundary, so the sufficient condition holds by construction rather than by measurement.
#
# Cost, stated because it is real: each extra segment is one more hardware-context transition per
# token, and a decode step is already ~90% dispatch. Correctness first, per the build methodology --
# judge a move by whether it advances the single-hardware graph, not by whether it is faster today.
# 1 is the default and is byte-for-byte the unsplit build.
DECODE_SEGMENTS = int(os.environ.get("DECODE_SEGMENTS", "1"))
if DECODE_SEGMENTS < 1:
    raise SystemExit(f"DECODE_SEGMENTS={DECODE_SEGMENTS} must be >= 1")

# Fold the attention output projection (`a = Wo @ cx`) into the SwiGLU MLP data-parallel design:
# every core computes its own D/N row-slice of `a` from its own row-slice of Wo before doing
# anything else, all-gathers it through the same DRAM-scratch mechanism gh already uses, then
# proceeds exactly as FUSE_MLP_DP already does. Deletes op_o's own standalone design/configure/run
# from the per-layer runlist outright (6 designs/layer -> 5). Needs FUSE_MLP_DP (there is nothing
# to fold INTO otherwise). DEFAULT OFF: device-free only so far (places at 61.7% .text against
# FUSE_MLP_DP's own 49.8%; see iron/operators/swiglu_mlp_dp/design.py's FUSE_O module docstring for
# the TSI_O-vs-D_PER_CORE divisibility issue this works around with a 1-row Wo pad, not measured on
# device).
FUSE_MLP_O = os.environ.get("FUSE_MLP_O", "1") == "1"

# Weight ObjectFifo depth for the two fused designs. 2 is plain double-buffering; the
# layer body moves bytes at 29.3 GB/s against the 53.5 GB/s the same dispatch's lm_head
# GEMV achieves, and a core stalling on every weight tile is the shape that would explain
# it. An A/B axis, not a settled default.
WEIGHT_DEPTH = int(os.environ.get("WEIGHT_DEPTH", "2"))
# KV_ALLOC -- allocate the KV cache for a WIDE capacity while attention computes over a NARROW
# window, so window buckets can share ONE cache. Since the blocked layout landed this is
# nearly free to express: KVLayout owns every stride and buffer size, so widening the capacity is
# ONE argument to it plus the operators' own alloc_M/alloc_K. Default 0 = capacity is the window,
# byte for byte the pre-existing build.
KV_ALLOC = int(os.environ.get("KV_ALLOC", "0"))
# SLIDING_KV_CIRCULAR -- give layers with a declared `sliding_window` (spec.sliding_window) a
# per-geometry KV cache sized to the WINDOW instead of the build's max_seq, addressed circularly:
# kv_off = (pos % W) * head_dim, sm_mask = min(pos+1, W). Correct because softmax is
# order-independent and RoPE is applied at KV-append time against the ABSOLUTE position, so a
# rotated slot ordering downstream is not observable -- see g4-t3-sliding-window.md. `mask_bf16`
# (aie_kernels/aie2p/softmax.cc) already masks the SUFFIX [unmasked, total), which is exactly what
# circular warmup (pos < W) needs, so no kernel changes.
#
# HOST-HARNESS ONLY as of 2026-09-14: verify_llm_decode.py and eval_llm_perplexity.py compute the
# modulo; the Rust serving path's Artifact::kv_offs writes `pos * head_dim` uniformly for every
# slot in `kv_params` (this file's own comment on `kv_slots`, below) and does NOT know about this
# flag. A build with this on is NOT safe to serve from `npu generate` -- it will write past the
# smaller buffer the moment n_past exceeds W. Default 0 = today's graph, byte-for-byte unchanged.
SLIDING_KV_CIRCULAR = os.environ.get("SLIDING_KV_CIRCULAR", "0") == "1"
# Pin the persistent buffers (weights + KV cache) to the FRONT of the scratch arena so window
# buckets present ONE layout for everything that survives a bucket crossing. Without it the
# window-sized softmax scratch (sc/sw, Hq*S) sits ahead of them and shifts every later offset:
# measured, buckets at window 256 and 512 over one allocation disagreed on 304 of 313 offsets.
BUCKET_SCRATCH_ORDER = os.environ.get("BUCKET_SCRATCH_ORDER", "0") == "1"
# Weight tile ROWS for the fused MLP. Trades against WEIGHT_DEPTH at constant L1.
MLP_TILE_ROWS = int(os.environ.get("MLP_TILE_ROWS", "0"))
# Row-parallel down: each core keeps only its own FF slice of `gh` and the partials are
# summed over the hardware cascade, so the all-gathered FF-wide buffer disappears. Needs
# post_norm (it reuses that path's gh_scratch round trip to broadcast the result back).
MLP_ROW_PARALLEL = os.environ.get("MLP_ROW_PARALLEL", "0") == "1"
MLP_D_CHUNKS = int(os.environ.get("MLP_D_CHUNKS", "8"))

# Columns per softmax L1 acquire. 0 keeps the unchunked operator, whose in1/out tiles are the whole
# attention row -- four `memref<w x bf16>` buffers in one core's L1, which caps the context at
# w <= 8063 and trips the 16383-word aie.dma_bd length field at w ~ 32768. Set it and the softmax
# streams each row in `w // SOFTMAX_SEGMENT` pieces instead, so neither limit sees w.
#
# MEASURED 2026-09-16 at 1024, 48 layers, instance 8b326264833a: every S from 8192 to 262144
# builds, scratch 6.69 -> 10.69 GiB. The unchunked op places at none of them.
SOFTMAX_SEGMENT = int(os.environ.get("SOFTMAX_SEGMENT", "0"))

# Make attention's reduction follow the POSITION instead of the CAPACITY. Both halves compute the
# full window every token today -- `op_scores` is gemv(M=w, K=hd) and `op_ctx` is TMatVec(K=w) --
# so raising max_seq costs core time at every position, which is why the 2048 -> 6912 raise was a
# regression. Set it and both read the geometry's OWN `sm_mask` (already min(n_past+1, w), already
# written per dispatch by the Rust path for the softmax) and skip the work above it.
#
# EXACT, not approximate: the rows past the mask carry a softmax weight of exactly 0, and the
# softmax overwrites them with -inf before any exp2, so a dispatch at the full window is
# bit-identical to the build-constant path.
#
# It does NOT move bytes -- a shim BD length is a static field on the binary TXN target, so the
# fill still streams the whole window and the surplus is drained untouched. What it buys is core
# time, which is what the S raise actually cost. Two things it does not reach: the softmax and
# the elementwise scale still walk the full window, and the op_ctx GEMV FALLBACK (no TMV_CTX)
# reduces along K, an axis GEMV has no runtime knob for.
ATTN_RUNTIME_EXTENT = os.environ.get("ATTN_RUNTIME_EXTENT", "0") == "1"


def softmax_segment(w):
    """The per-acquire tile for a softmax over `w` columns, or None to keep the unchunked op.

    A window the unchunked op already reaches keeps it -- chunking costs two extra streams of the
    scores and buys nothing there, and it is what leaves the sliding geometry's 1024-wide softmax
    untouched while the global one is split. Divisibility is raised HERE rather than left to
    iron/operators/softmax/op.py, so the message names the window a caller actually chose.
    """
    if not SOFTMAX_SEGMENT or w <= SOFTMAX_SEGMENT:
        return None
    if w % SOFTMAX_SEGMENT:
        raise ValueError(
            f"SOFTMAX_SEGMENT={SOFTMAX_SEGMENT} does not divide the attention window {w}"
        )
    return SOFTMAX_SEGMENT

# K-SPLIT for reductions that do not fit L1 at ANY tiling. The B vector is double-buffered at
# 2*K*2 bytes and is independent of every tiling knob, so a large enough K has no legal
# (tsi, tso) -- Gemma-4-12B's down projection is K=15360 and needs 61440 B of 57344 usable before
# a weight or output byte is counted. The reduction is split over K and the partials summed, which
# needs no new operator. llm_decode_spec.k_chunks_for() decides the count and is the SAME model the
# weight dump uses, so the two cannot disagree about layout.
#
# NOT free in bf16: mv.cc rounds its f32 accumulator to bf16 once PER PARTIAL, so n chunks round n
# times instead of once. FORCE_K_SPLIT exists to price exactly that on a shape that does not need
# the split, by building it both ways.
FORCE_K_SPLIT = int(os.environ.get("FORCE_K_SPLIT", "0"))
# Run the FF-wide activation and gate*up multiply as this many equal slices. Elementwise, so the
# arithmetic is unchanged; at FF/n == d_model the multiply shares the d_model-wide Mul design.
FF_POINTWISE_CHUNKS = int(os.environ.get("FF_POINTWISE_CHUNKS", "1"))
# Quantized weight GEMVs that differ only in M take one tiling per family plus tiles_rtp, and IRON's
# merge_devices folds each family into one device. Rows are independent, so the arithmetic is unchanged.
MERGE_WEIGHT_GEMVS = os.environ.get("MERGE_WEIGHT_GEMVS", "0") == "1"
# Same-width Add, Mul and GELU become modes of IRON's Pointwise, merged onto one device.
POINTWISE_MODES = os.environ.get("POINTWISE_MODES", "0") == "1"
# Back-to-back runs on a merged device share one configure (IRON collapse_configures).
COLLAPSE_MERGED_CONFIGURES = os.environ.get("COLLAPSE_MERGED_CONFIGURES", "0") == "1"
# Same axis for the attention output projection. Gemma-4's GLOBAL layers have head_dim 512, so
# o_proj is K=8192 there and needs 2 chunks while its sliding layers at K=4096 need none -- a
# PER-LAYER split, which this per-spec version does not yet express (it needs q_dim_for(layer),
# a Gemma-4 spec axis). What is here covers every spec whose q_dim is uniform.
FORCE_O_SPLIT = int(os.environ.get("FORCE_O_SPLIT", "0"))


def packed_zero_rows(n_rows, K, group_size, weight_dtype):
    """`n_rows` all-zero rows in quantize_weight's on-wire layout, built WITHOUT unpacking.

    A pre-packed dump has no float domain left to pad in, but fuse_o still needs its Wo pad. A zero
    row is constructible directly: quantize_weight takes amax=0 to scale 1.0
    (`np.where(amax > 0, amax/qmax, 1.0)`) and q=0, so the row is [n_groups x float32(1.0)][zeros].
    Gated by an equality against the float path -- see the round-trip check in tests.
    """
    stride = row_stride_bytes(K, group_size, weight_dtype)
    n_groups = K // group_size
    out = np.zeros((n_rows, stride), np.uint8)
    ones = np.ones((n_rows, n_groups), np.float32)
    out[:, : n_groups * 4] = ones.view(np.uint8).reshape(n_rows, n_groups * 4)
    return out.reshape(-1).view(np.int8)

# The WHOLE decoder layer -- attention block AND SwiGLU MLP -- as ONE fused `aie.device`:
# iron/operators/decode_layer_dp. All 28 layers become a CONTIGUOUS run of the SAME op object in
# the runlist, so the fused-MLIR assembly collapses adjacent identical designs to ONE configure
# point (iron/common/compilation/sequence.py's needs_additional_reset()/fuse_mlir()). Token total
# is 3 configure POINTS: 1 (layer) + 1 (final RMSNorm) + 1 (final GEMV/lm-head), against today's
# 142 and attn_block_dp alone's 58 -- MEASURED (aiecc, 2026-09-09) in the emitted MLIR's
# `aiex.configure` blocks. 3 is ODD, so the same pass adds a mandatory 4th `reset_device` configure
# to keep the `--expand-load-pdis` two-slot alternation even across the token-boundary replay
# (needs_additional_reset's own docstring) -- so the real per-token count this arm pays is 4, not
# the 3 the op's own design.py docstring projects device-free without that pass in view.
#
# Eligibility is qkv_dp_why/mlp_dp_why's (the spec-shape rules those two arms already check) PLUS
# what is true only of the MERGED device: attn_block_dp's Hkv==COLS rule, SCALE_IN_QNORM (no
# separate scale stage), GROUPED_K+TMV_CTX (the variant attn_block_dp actually computes),
# FUSE_MLP_O (Wo's padding rides that flag), and bf16-only weights (plain kernel archive).
#
# ON by default since 2026-09-10. Device-gated: numerics bitwise identical over 2000 paired
# perplexity positions (max |dNLL| 0.000e+00), determinism 5/5 on both arms, served 53.1 -> 37.6
# ms/token with a TIGHTER tail (p99-mean 0.8 ms against 2.2). The cost is array footprint -- 12
# cores over 3 columns against 8 over 2, and --cores-per-col 1 is not available on this arm.
# An ineligible spec still falls back: decode_layer_why below names the rule it missed.
FUSE_DECODE_LAYER = os.environ.get("FUSE_DECODE_LAYER", "1") == "1"
# attn_block_dp ALONE, without decode_layer_dp's whole-layer requirements (no FUSE_MLP_O, no MLP
# shape check, no single-geometry constraint) -- so a spec whose layers disagree about attention
# geometry (Gemma-4-12B: sliding 256/8, global 512/1) can fuse the geometries that qualify and
# fall back to the unfused chain on the rest. Per geometry: see attn_block_why in build_graph,
# which reuses qkv_dp_why/_tmv_declined rather than re-deriving the same rules decode_layer_why
# already checks. Default OFF: device-free only so far.
FUSE_ATTN_BLOCK = os.environ.get("FUSE_ATTN_BLOCK", "0") == "1"
# Thread decode_layer_dp's window_parameter through: the AIE core reads its attention window from
# a per-dispatch ScratchpadParameter ("attn_window", int32) instead of baking N_KV_CHUNKS into the
# build. Only takes effect when decode_layer_dp itself is eligible (decode_layer_why is None below)
# -- there is nowhere else in this graph for it to attach. Default 0 = build-constant window,
# byte-for-byte the pre-existing graph and meta.json; params.txt (read further down) picks up the
# new parameter's real offset for free once this is on, so the meta writer never hardcodes one.
#
# Implied by WINDOW_RUNGS: a rung quantises the shim's FILL and relies on the core taking its own
# window at runtime, so building a rung set with a build-constant window compiles clean and the
# service refuses to load it at start ("window_rungs without scratchpad.window_param"). Found
# 2026-09-12 rebuilding the shipped rung-ladder artifact without knowing this. Making the implied
# flag explicit-only would leave the same trap for the next caller; deriving it removes the trap.
DYNAMIC_WINDOW = (os.environ.get("DYNAMIC_WINDOW", "0") == "1"
                   or bool(os.environ.get("WINDOW_RUNGS", "").strip()))
# ATTN_SPLIT -- process the attention window in segments of this many positions, carrying the
# softmax's running max/sum across them (split-K flash). sc/sw are then sized to a SEGMENT, so L1
# stops scaling with max_seq and the 4544-position window cap goes away: `attn_block_dp` places at
# max_seq=32768 with .text byte-identical to its 2048 build, because the segment loop is a runtime
# loop. 0 (default) is one segment, byte for byte the pre-split design.
#
# The cap moves onto the SPLIT, and it is 4542 by the same arithmetic that used to bound the window
# (65536 L1 minus 29196 of fixed terms, over the 8 B/position sc+sw cost). Must be a multiple of
# lcm(stream-tile rows, kv block, 64).
ATTN_SPLIT = int(os.environ.get("ATTN_SPLIT", "0"))
# WINDOW_RUNGS -- extra attention windows, comma-separated, served from THE SAME ELF as named
# control codes rather than as separate artifacts.
#
# WHY THIS EXISTS. The core already takes its window from a scratchpad parameter at 128-position
# granularity (DYNAMIC_WINDOW above), but the SHIM's KV fill size is a static BD field and cannot be
# made runtime without leaving the resident full-ELF dispatch model -- the static TXN target rejects
# every non-constant operand, address patches excepted. So the fill streams the whole built window
# every token and a drain discards the surplus, which is the entire measured regression against the
# bucketed model. A rung is a SECOND `decode_layer_dp` design at a narrower window over the SAME KV
# capacity, reached through its own named runtime sequence: aiecc emits one control code per
# `aie.runtime_sequence` and XRT resolves `main:<name>` against ONE registered hw_context, so the
# rungs cost neither a rebuild nor a context. Device-proven on a two-sequence module before this
# landed.
#
# The rungs quantise the FILL only; the core keeps its fine runtime window, so compute stays at
# 128-position granularity and only the streamed bytes round up to a rung.
#
# Needs decode_layer_dp to be eligible (there is no other design here holding a window) and every
# rung must be < max_seq and satisfy the same divisibility the top window does.
WINDOW_RUNGS = tuple(
    int(w) for w in os.environ.get("WINDOW_RUNGS", "").replace(" ", "").split(",") if w
)


def weight_bytes(arr):
    """Bytes for one weight buffer exactly as written into the .bin / device arena.

    A quantize_weight() packed array (np.int8, opaque on-wire bytes) must NOT be value-cast to
    bf16 like every other weight here -- that would renumber the packed bytes instead of copying
    them.
    """
    a = np.asarray(arr)
    if a.dtype == np.int8:
        return a.tobytes()
    return np.asarray(a, BF16).tobytes()


def load_weight_buffer(buf, arr):
    """Load one weight into its device buffer, mirroring weight_bytes()'s dtype split.

    `buf.data` is always a bfloat16-dtype view (iron.common.sequence's shared bf16-granule arena,
    FusedFullELFCallable.get_buffer) regardless of the argument's declared dtype, so a packed
    int8 array must be written via a raw byte view, never `np.asarray(arr, BF16)` (which would
    numerically reinterpret the packed byte VALUES as floats).
    """
    a = np.asarray(arr)
    if a.dtype == np.int8:
        dst = buf.data.view(np.uint8)
        assert dst.nbytes == a.nbytes, f"weight byte-size mismatch: buf {dst.nbytes} vs arr {a.nbytes}"
        dst[:] = a.view(np.uint8)
    else:
        with buf.overwrite() as _buf:
            _buf[:] = np.asarray(a, BF16).reshape(-1)



def sequence_name(sp, NL, S, placer_flags, decode_layer_active=False, T=None, tmv_declined=(),
                  tmv_chunked=(), attn_block_geoms=(), ff_chunks=1, weight_families=0,
                  pointwise_widths=0, scores_blocks=()):
    """Name the fused sequence after everything that changes its graph, not just the model.

    IRON keys the cached artifact by this name. Every knob below produces a DIFFERENT ELF, so
    without them two arms share one filename and a later run executes the earlier arm's binary --
    see isolate_build_dir() for what that cost twice on 2026-09-07. Isolating the build dir hides
    the collision; naming the arm removes it, and only naming it makes an arm's artifact
    identifiable after the fact.

    Suffixes are emitted only for NON-DEFAULT values, so the shipped default keeps the bare
    `<spec>_decode` name and its existing artifact stays valid. Same convention as TMatVec's
    `_ak{alloc_K}`.
    """
    base = f"{sp.name.replace('-','_').replace('.','_')}_decode"
    parts = []
    # `noctx` means the whole graph is on the transpose+GEMV context path; `noctx<hd>` means only
    # the named head_dims are, because TMatVec does not fit their L1. The two MUST NOT share a name:
    # they are different graphs over the same buffers, and IRON keys its artifact cache on this.
    if not TMV_CTX:
        parts.append("noctx")
    elif tmv_declined:
        parts.append("noctx" + "".join(f"_{h}" for h in sorted(tmv_declined)))
    # An output-chunked TMatVec is a different graph at the same shape, so it must not share a
    # cache key with the unchunked one -- same reason as noctx above.
    if tmv_chunked:
        parts.append("mc" + "".join(f"_{h}x{m}" for h, m in sorted(tmv_chunked)))
    if not GROUPED_K:
        parts.append("nogk")
    # The KV cache's block size (iron.common.kv_layout). T == S (or None, pre-this-task callers)
    # is the flat pre-blocking layout and keeps the bare name; T < S addresses the SAME cache
    # buffers completely differently, so it must not share a name with the flat build.
    if T is not None and T != S:
        parts.append(f"kvt{T}")
    # The SCORES GEMV's K block, per geometry, where it differs from that geometry's V block.
    # `kvt` above names the PHYSICAL cache layout; this names an access pattern over the same
    # bytes, so the two are different suffixes and a build can carry either, both or neither.
    # Empty on every geometry that stays flat, which is every shipped arm -- see
    # scores_block_size().
    if scores_blocks:
        parts.append("sckt" + "".join(f"_{h}x{t}" for h, t in sorted(scores_blocks)))
    if KV_ALLOC and KV_ALLOC != S:
        parts.append(f"ka{KV_ALLOC}")
    if GROUPED_V:
        parts.append("gv")
    if TMV_CTX and TMV_RPC != TMV_RPC_DEFAULT:
        parts.append(f"rpc{TMV_RPC}")
    # Three more graph-changing switches that predate this function's audit and were missed by it:
    # each has its own doc comment above proving it changes the per-layer runlist (configure count
    # and/or dispatch structure), the same class of change TMV_CTX/GROUPED_K above are named for.
    # Nested on their own gate, same shape as fuse_rope/scale_in_qnorm at their build site, so an
    # already-off parent doesn't also emit its child's suffix.
    if not FUSE_QKV_GEMV:
        parts.append("noqkvgemv")
    if FUSE_QKV_GEMV and not FUSE_ROPE_QK:
        parts.append("noropeqk")
    if sp.qk_norm and not SCALE_IN_QNORM:
        parts.append("noscaleqn")
    # One fragment per quantized site. scale_kind rides the name only when it is not the class
    # default: it moves weight VALUES at a fixed wire format, so two arms differing in it are the
    # same GRAPH and would otherwise collide as ARTIFACTS.
    for _site, _tag in (("mlp", ""), ("attn_o", "attn"), ("head", "head"), ("qkv", "qkv"),
                        ("kv", "kv")):
        _sp = PRECISION_PLAN.get(_site, precision.BF16_SPEC)
        if not _sp.quantized:
            continue
        _frag = f"{_tag}{_sp.dtype}g{_sp.group_size}"
        if _sp != precision.parse_spec(f"{_sp.dtype}/g{_sp.group_size}", _site):
            _frag += _sp.scale_kind
        # Unlike scale_kind, the scale WIDTH changes the graph: it sizes the weight row and picks
        # the kernel's header type, so two arms differing in it are different ELFs and must not
        # share a name. It rides every quantized fragment because the dump declares one width for
        # all of them.
        if _BUILD_STATE["scale_dtype"] != "f32":
            _frag += f"s{_BUILD_STATE['scale_dtype']}"
        parts.append(_frag)
    if SPLIT_QKNORM:
        parts.append("splitqk")
    # Suffix stays ON the default here, unlike the other switches: the shipped artifact was BUILT
    # and gated under this name, and aiecc is not byte-reproducible, so a rename would mean the
    # next rebuild produces a different ELF under a name nothing was ever gated against.
    if FUSE_MLP_DP and sp.mlp_dp_reason() is None:
        parts.append(f"mlpdp{MLP_DP_COLS}")
    if FUSE_MLP_O and sp.mlp_dp_reason() is None:
        parts.append("mlpo")
    # Same convention, its sibling arm: FUSE_QKV_DP defaulted ON 2026-09-07 (27a9411) and was
    # missing here entirely -- not just off-the-default-name, ABSENT, so a build before that
    # commit and a build after it shared this function's name unchanged. sp.qkv_dp_reason(COLS)
    # mirrors qkv_dp_why's own gate exactly (build_graph computes the same three-way check further
    # down); COLS is the module constant, not a build_graph local, so it is reachable here.
    if FUSE_QKV_DP and FUSE_QKV_GEMV and sp.qkv_dp_reason(COLS) is None:
        parts.append("qkvdp")
    # decode_layer_dp REPLACES the qkvdp/mlpdp/mlpo designs above outright (a different runlist,
    # not an additional flag on theirs), so it gets its own suffix rather than stacking onto
    # theirs -- two graphs sharing this name is exactly the isolate_build_dir() hazard this
    # function exists to prevent (see its docstring: "a later run executes the earlier arm's
    # binary"). Passed in rather than re-derived from the FUSE_*/QUANT_* globals here, because
    # build_graph already computed the one true eligibility check (decode_layer_why) and a second
    # copy of that logic is exactly the kind of drift this file's other suffixes warn about.
    if decode_layer_active:
        parts.append("declayer")
    # attn_block_dp used WITHOUT decode_layer_dp, per geometry -- a different graph over the same
    # buffers as the unfused chain, same collision this function's docstring warns about. Passed
    # in rather than re-derived, same reason decode_layer_active is: build_graph already computed
    # attn_block_why. Empty when no geometry qualifies, so an on-but-inert flag keeps the name.
    if attn_block_geoms:
        parts.append("ab" + "".join(f"_{h}" for h in sorted(attn_block_geoms)))
    # The window override changes the graph (rpc, the KV ring, every sliding design's max_seq),
    # so it has to reach the name or two arms share one cached artifact -- this function's own
    # docstring is about exactly that failure. Experiment knob; absent on every shipped arm.
    if os.environ.get("SLIDING_WINDOW_OVERRIDE"):
        parts.append(f"win{os.environ['SLIDING_WINDOW_OVERRIDE']}")
    if SPLIT_GH_DRAIN != 1:
        parts.append(f"sgh{SPLIT_GH_DRAIN}")
    if ATTN_SPLIT:
        parts.append(f"sp{ATTN_SPLIT}")
    if SCORES_ROWBATCH > 1:
        parts.append(f"rb{SCORES_ROWBATCH}")
    if WEIGHT_DEPTH != 2:
        parts.append(f"wd{WEIGHT_DEPTH}")
    if MLP_TILE_ROWS:
        parts.append(f"tr{MLP_TILE_ROWS}")
    if MLP_ROW_PARALLEL:
        parts.append(f"rp{MLP_D_CHUNKS}")
    if SOFTMAX_SEGMENT:
        parts.append(f"smseg{SOFTMAX_SEGMENT}")
    if ff_chunks > 1:
        parts.append(f"ffc{ff_chunks}")
    if weight_families:
        parts.append("wgm")
    if pointwise_widths:
        parts.append("pwm")
    if ATTN_RUNTIME_EXTENT:
        parts.append("rtext")
    if FUSE_ACT:
        parts.append("fuseact")
    # Flat, not nested under decode_layer_active: _trace.py wires into every design.py this file
    # can build, fused or not, so the suffix must apply on both paths.
    if IRON_TRACE_SIZE > 0:
        parts.append(f"trace{IRON_TRACE_SIZE}")
    if NL != sp.n_layers:
        parts.append(f"l{NL}")
    if S != 2048:
        parts.append(f"s{S}")
    if placer_flags != DECODE_PLACER_FLAGS_DEFAULT.split():
        parts.append("p" + hashlib.sha1(" ".join(placer_flags).encode()).hexdigest()[:6])
    return "_".join([base, *parts])


def isolate_build_dir(tag):
    """chdir into a private build dir, because IRON writes build/ intermediates under CWD.

    IRON keys cached operator artifacts by NAME, and the name encodes SHAPES but not the dataflow
    flags -- so two arms of this graph produce the same filenames. Any entry point that runs in a
    shared directory can therefore assemble an ELF partly from another arm's operator binaries. The
    result is not a crash and not noise: it is a deterministic, reproducible wrong answer that looks
    exactly like a numerical bug.

    MEASURED COST 2026-09-07, twice in one day. Once here (an A/B in a shared dir produced a
    different step-0 token for a FIXED graph, which is impossible), and once in a parallel session
    that spent ~3 hours on a full-depth decode returning the constant token 3972 at every step,
    deterministic across runs, because its runner never left the shared xdna-engine/build.

    build_llm_decode.sh already does `WORK=$(mktemp -d); cd "$WORK"` for exactly this reason. This
    gives the Python entry points the same protection instead of trusting the caller's cwd.

    Set DECODE_WORK=<dir> to use a specific directory (kept, not deleted) when you need the
    intermediates -- a byte census needs the fused MLIR, which is otherwise discarded.

    NOT ON RAM. `tempfile.mkdtemp()` honours TMPDIR and otherwise picks /tmp, which is a 16 GB
    tmpfs of 30 GB total on this class of box -- so the default landed every one of this
    function's six callers' intermediates in MEMORY, competing with the weight buffers the model
    must hold resident. `scripts/require_disk_backed.sh` has guarded exactly this for the shell
    build path since it landed; the Python entry points were the sibling that never got it, which
    is why `verify_llm_decode.py` was still announcing `build dir /tmp/verify-...` on 2026-09-08.
    Same rules as that file, deliberately: XDNA_SCRATCH names the disk root, ALLOW_TMPFS_BUILD=1
    overrides, and a fallback to RAM SAYS SO rather than happening silently.
    """
    import atexit
    import shutil
    import subprocess
    import tempfile

    def _fs_type(path):
        """Filesystem type, via the same `df -PT` the shell guard uses.

        Same command on purpose: two different ways of deciding "is this RAM" is two answers that
        can disagree, and this one has a sibling in require_disk_backed.sh that must agree with it.
        """
        try:
            out = subprocess.run(["df", "-PT", str(path)], capture_output=True, text=True,
                                 timeout=10).stdout.splitlines()
            return out[1].split()[1] if len(out) > 1 else ""
        except (OSError, IndexError, subprocess.SubprocessError):
            return ""

    explicit = os.environ.get("DECODE_WORK")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        fs = _fs_type(explicit)
        if fs in ("tmpfs", "ramfs") and os.environ.get("ALLOW_TMPFS_BUILD") != "1":
            raise SystemExit(
                f"[{tag}] DECODE_WORK={explicit} is on {fs}, which is RAM. A decode artifact is "
                f"~1.4 GB and competes with the weight buffers the model holds resident. Use a "
                f"disk-backed path (e.g. ${{XDNA_SCRATCH:-/mnt/data/xdna/scratch}}/{tag}), or set "
                f"ALLOW_TMPFS_BUILD=1 for a small probe build.")
        os.chdir(explicit)
        print(f"[{tag}] build dir {explicit} (DECODE_WORK, kept)", flush=True)
        return explicit

    root = os.environ.get("XDNA_SCRATCH", "/mnt/data/xdna/scratch")
    base, why = None, ""
    try:
        os.makedirs(root, exist_ok=True)
        if os.access(root, os.W_OK) and _fs_type(root) not in ("tmpfs", "ramfs"):
            base = root
        else:
            why = f"{root} is not writable or is itself RAM"
    except OSError as e:
        why = f"{root} unusable ({e})"
    if base is None:
        print(f"[{tag}] WARN: no disk-backed scratch -- {why}; intermediates go to RAM", flush=True)
    work = tempfile.mkdtemp(prefix=f"{tag}-", dir=base)
    atexit.register(shutil.rmtree, work, ignore_errors=True)
    os.chdir(work)
    print(f"[{tag}] build dir {work} (private, removed on exit; "
          f"set DECODE_WORK=<dir> to keep)", flush=True)
    return work


def repo_root():
    # gen_llm_decode.py -> decode_fused -> designs -> repo root (toolchain.lock lives there).
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generator_provenance():
    """Best-effort build provenance for meta.json: the git commit of THIS FILE's tree at build
    time, and whether designs/decode_fused was dirty against it.

    toolchain_provenance() answers "which toolchain built this"; this answers "which generator
    graph built this" -- a different axis. Two designs/decode_fused commits landed the same day
    this artifact shipped (2026-09-07 23:40) that changed the GRAPH (b7b0498, 198 -> 170
    configures), and nothing recorded which side of that change a given decode.elf was on --
    telling them apart took a manual byte-size diff against a rebuild. Returns {} rather than
    raising: provenance is a record, not a gate.
    """
    repo = repo_root()
    try:
        sha = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--", "designs/decode_fused"],
            capture_output=True, text=True, timeout=10,
        )
    except OSError:
        return {}
    if sha.returncode != 0 or not sha.stdout.strip():
        return {}
    return {"commit": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def iron_provenance():
    """Best-effort build provenance for meta.json: the git commit of the IRON tree that supplied
    the OPERATORS AND KERNELS, plus whether aie_kernels/ or iron/ was dirty against it.

    A third axis, and the one that was missing. `toolchain_provenance` answers which toolchain
    compiled this and `generator_provenance` which generator graph emitted it; neither says
    anything about IRON, where the kernel SOURCE lives. A kernel edit changes the arithmetic an
    artifact performs and leaves no trace in its name, its toolchain hash or its generator commit
    -- `MVQ_UNROLL` (mv_quant.cc's accumulator count) is exactly that: it moved a decode step
    321.71 -> 233.52 ms and is invisible to all three existing records. Resolved from the `iron`
    package actually on sys.path, so it describes the tree that was really used rather than a
    path someone believed was used. Returns {} rather than raising: provenance is a record, not a
    gate.
    """
    try:
        import iron  # noqa: PLC0415 -- resolved at call time, on purpose
        tree = os.path.dirname(os.path.dirname(os.path.abspath(iron.__file__)))
        sha = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", tree, "status", "--porcelain", "--",
                                "iron", "aie_kernels"],
                               capture_output=True, text=True, timeout=10)
    except (OSError, ImportError):
        return {}
    if sha.returncode != 0 or not sha.stdout.strip():
        return {}
    return {"tree": tree, "commit": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def toolchain_provenance():
    """Best-effort build provenance for meta.json: the toolchain.lock semantic hash this ELF was
    just compiled against, plus the instance dir the build used, if resolvable.

    Sources scripts/kernel_sandbox.sh's `current_toolchain_hash` (the canonical derivation --
    comment/blank-stripped toolchain.lock, sha256, first 12 hex; matches
    kernel_registry.rs::current_toolchain_hash and toolchain_up.sh's LOCKHASH) rather than
    reimplementing it a fifth time. Returns {} rather than raising: provenance is a record, not a
    gate, and a build must not fail because this is unresolvable.
    """
    repo = repo_root()
    sandbox = os.path.join(repo, "scripts", "kernel_sandbox.sh")
    if not os.path.isfile(sandbox):
        return {}
    try:
        out = subprocess.run(
            ["bash", "-c", f'source "{sandbox}" && current_toolchain_hash "$1"', "_", repo],
            capture_output=True, text=True, timeout=10,
        )
    except OSError:
        return {}
    h = out.stdout.strip()
    if out.returncode != 0 or not h:
        return {}
    prov = {"hash": h}
    inst = os.environ.get("MLIR_AIE_INSTANCE")
    if inst:
        prov["instance"] = os.path.basename(inst.rstrip("/"))
    # The COMPILER that ran, not the lock that named it. `hash` and `instance` both describe
    # toolchain.lock and are blind to AIECC_PATH, so an artifact built by another aiecc recorded
    # provenance indistinguishable from a pinned one -- see aiecc_require_pin in scripts/amd_paths.sh.
    aiecc = os.environ.get("AIECC_PATH")
    if aiecc and os.path.exists(aiecc):
        try:
            out = subprocess.run([aiecc, "--version"], capture_output=True, timeout=30).stdout.decode(errors="replace")
            for ln in out.splitlines():
                if "git SHA:" in ln:
                    prov["aiecc_sha"] = ln.split("git SHA:")[1].strip()
                    break
        except (OSError, subprocess.SubprocessError):
            pass
        prov["aiecc_path"] = aiecc
    return prov


def report_artifact_freshness(weights_dir):
    """Print-only freshness check for the sibling `decode/` artifact a `--weights` dir implies
    (`artifacts/<spec>/weights` -> `artifacts/<spec>/decode`).

    verify_llm_decode.py and bench_llm_decode.py always recompile the graph fresh via build_graph,
    so THIS run never dispatches stale bytes -- but the shipped Rust engine loads decode.elf/
    meta.json directly (npu-engine::LlmArtifact) and does not rebuild. A stale artifact sitting next
    to fresh weights is exactly the silent-wrong-token hole this closes: the decode ELF that
    shipped 2026-09-03 returned a wrong token at one margin step after the 2026-09-04 re-pin, with
    nothing to say so. Never raises and never affects the caller's exit code -- this is a report
    about a DIFFERENT consumer, not a gate on the graph this process just verified.
    """
    art_dir = os.path.join(os.path.dirname(os.path.normpath(weights_dir)), "decode")
    meta_path = os.path.join(art_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return
    meta = json.load(open(meta_path))
    built = meta.get("toolchain", {}).get("hash")
    tag = "[freshness]"
    if not built:
        print(f"{tag} {meta_path}: no toolchain provenance recorded (built before this check "
              f"existed) -- the Rust-loaded artifact's freshness is UNVERIFIED", file=sys.stderr)
        return
    cur = toolchain_provenance().get("hash")
    if not cur:
        print(f"{tag} {meta_path}: built against {built}, but the current toolchain.lock could not "
              f"be resolved here -- the Rust-loaded artifact's freshness is UNVERIFIED", file=sys.stderr)
    elif cur != built:
        print(f"{tag} STALE: {meta_path} was built against toolchain {built}, current toolchain.lock "
              f"is {cur} -- rebuild with scripts/build_llm_decode.sh before trusting what the Rust "
              f"engine would load from {art_dir}", file=sys.stderr)
    else:
        print(f"{tag} {meta_path}: OK (toolchain {cur})", file=sys.stderr)

    # Same shape, the other provenance axis: which designs/decode_fused commit built this graph.
    # Older artifacts (built before generator_provenance() existed) have no "generator" key --
    # report that as unverified rather than silently skipping, since that is the exact hole this
    # closes (see installed-decode-artifact-predates-head, filed after a manual byte-size diff
    # was the only way to notice a graph-changing commit had landed after the shipped ELF).
    gen = meta.get("generator", {}).get("commit")
    if not gen:
        print(f"{tag} {meta_path}: no generator provenance recorded (built before this check "
              f"existed) -- staleness against designs/decode_fused HEAD is UNVERIFIED", file=sys.stderr)
        return
    curgen = generator_provenance().get("commit")
    if not curgen:
        print(f"{tag} {meta_path}: built from generator {gen[:12]}, but the current commit could "
              f"not be resolved here -- staleness is UNVERIFIED", file=sys.stderr)
    elif curgen != gen:
        print(f"{tag} {meta_path} (arm {meta.get('sequence_name', '?')}): built from "
              f"designs/decode_fused @ {gen[:12]}, current HEAD is {curgen[:12]} -- a commit in "
              f"between may change the graph rather than a default; diff before trusting what the "
              f"Rust engine would load from {art_dir}", file=sys.stderr)
    else:
        print(f"{tag} {meta_path}: OK (generator {curgen[:12]})", file=sys.stderr)


def gemv(M, K, ctx, **kw):
    """GEMV tiled as large as both the design asserts AND L1 allow."""
    tsi, tso = gemv_tile_output(M, K, cols=COLS)
    return GEMV(M=M, K=K, num_aie_columns=COLS, tile_size_input=tsi,
                tile_size_output=tso, context=ctx, **kw)



# A runtime buffer's offset is patched into its BD by `aiex.npu.address_patch`, whose `arg_plus`
# operand mlir-aie declares as I32 (AIEX.td) and emits through a uint32_t path
# (AIETargetNPU.cpp::appendAddressPatch -> TxnEncoding.h::txn_append_address_patch). Anything at or
# past 2^32 therefore WRAPS, and the BD writes to the wrong place -- silently, with no diagnostic at
# any layer, on a design that builds and runs.
#
# Measured 2026-09-09 on Gemma-4-12B: at 12 layers the residual buffer at offset 3,897,195,520 is
# written and the next one at 4,379,395,584 reads back exactly zero; at 48 layers int4 the boundary
# falls between 4,216,907,776 and 4,376,916,480. Both bracket 2^32, the second to within 160 MB. It
# cost this chain a long hunt through arithmetic that was never wrong.
#
# The hardware and the driver are NOT the limit -- aie-rt's patch_op_t declares `u64 argplus`
# (xaiegbl.h) and the 12-word TXN op reserves the high word right after it. This is mlir-aie
# narrowing a field both ends of it carry at 64 bits, so it is fixable upstream; until it is, refuse
# here rather than emit a design that lies.
ADDRESS_PATCH_MAX_OFFSET = 1 << 32

_ARGPLUS64 = None


def toolchain_carries_argplus64():
    """Does the toolchain that will COMPILE this design carry the 64-bit arg_plus widening?

    PROBED, not read off a source file or a pin string, and that choice is the lesson of
    2026-09-09: the shared instance spent an hour with a hand-patched `src/` whose binaries had
    been rebuilt out from under it, so source and binaries disagreed and a check reading either
    one alone would have been confidently wrong. The only authority is the binary that runs.

    Feeds `aie-translate` an `address_patch` with an i64 arg_plus. The narrowed dialect rejects it
    at verification ("operand #0 must be 32-bit signless integer"); the widened one accepts it.

    UNKNOWN COUNTS AS NARROW. Guessing wrong in that direction refuses a design that would have
    been fine, which is a build error someone reads. Guessing wrong the other way emits a design
    that builds, runs and writes to the wrong address in silence -- the failure this whole guard
    exists for.
    """
    global _ARGPLUS64
    if _ARGPLUS64 is not None:
        return _ARGPLUS64
    inst = os.environ.get("MLIR_AIE_INSTANCE")
    exe = os.path.join(inst, "bin", "aie-translate") if inst else None
    if not exe or not os.path.exists(exe):
        _ARGPLUS64 = False
        return _ARGPLUS64
    probe = ("module { aie.device(npu2) { aie.runtime_sequence(%a0: memref<8xi32>) {\n"
             "  %off = arith.constant 5000000000 : i64\n"
             "  aiex.npu.address_patch(%off : i64) {addr = 74560 : ui32, arg_idx = 0 : i32}\n"
             "} } }\n")
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as fh:
            fh.write(probe)
            path = fh.name
        try:
            r = subprocess.run([exe, "--aie-npu-to-binary", "-aie-output-binary=false", path],
                               capture_output=True, text=True, timeout=60)
            _ARGPLUS64 = r.returncode == 0
        finally:
            os.unlink(path)
    except Exception:
        _ARGPLUS64 = False
    return _ARGPLUS64


def runlist_buffer_names(entries):
    """Every buffer an OperatorSequence runlist slice touches, as bare names.

    Operands are strings, and `split_over_k` writes BYTE-SLICED ones (`L0_Wd[0:1234]`) whose base
    buffer is what the layout actually names -- so the bracket is stripped here rather than at each
    call site. Non-string entries (the operator object in position 0) are skipped.
    """
    out = set()
    for e in entries:
        for arg in e[1:]:
            if isinstance(arg, str):
                out.add(arg.split("[", 1)[0])
    return out


def check_arena_offsets_are_addressable(seq, names):
    """Refuse a design whose buffers cannot be addressed by a 32-bit arg_plus.

    SKIPPED ENTIRELY when the toolchain carries the 64-bit widening -- the cap is a property of the
    COMPILER, not of the hardware (aie-rt declares `u64 argplus` and the TXN op has the high word),
    so a toolchain that emits both words has no boundary to violate. Probed, never assumed: see
    toolchain_carries_argplus64().

    CATCHES the total failure: a buffer whose START is at or past 2^32 gets a wrapped BD offset and
    is never written -- measured as exactly-zero residual buffers from layer 8 (12 layers, bf16) and
    layer 26 (48 layers, int4).

    DOES NOT catch the PARTIAL case, and that case is genuinely UNRESOLVED rather than merely
    uncaught. Two measurements on the same mechanism point opposite ways:

      * 5 layers: `W_head` runs 2.41 -> 4.12 GiB, crosses 2^32, and the design is CLEAN on device.
      * 6 layers: every buffer STARTS below 2^32, this check passes, and the lm-head still comes
        back with 2169 elements unwritten in 16 runs.

    So "a transfer crossing the boundary is safe" and "crossing it is fatal" are each refuted by one
    of the two, and no rule here separates them. An `off + len` test would reject the 5-layer design
    that works; the START test passes the 6-layer design that does not. Both readings are recorded
    because picking one would be a guess wearing a measurement's clothes. Treat a pass as NECESSARY,
    NOT SUFFICIENT, and prefer DECODE_SEGMENTS, which removes the question by construction: with
    every arena under 4 GiB no buffer can start or span past the boundary at all.
    """
    if toolchain_carries_argplus64():
        return
    bad = []
    for n in names:
        try:
            arena, off, ln = seq.get_layout_for_buffer(n)
        except Exception:
            continue
        # START offset only, because that is the value the BD is patched with. This deliberately
        # does NOT test `off + len`: the 5-layer W_head spans 2.41 -> 4.12 GiB, across 2^32, and is
        # clean on device, so an end-of-buffer test would reject a working configuration. That is a
        # reason to keep the test narrow, NOT evidence that crossing is safe -- the 6-layer lm-head
        # crosses too and loses 2169 elements. See the docstring: the partial case is open.
        if off >= ADDRESS_PATCH_MAX_OFFSET:
            bad.append((n, arena, off, ln))
    if not bad:
        return
    if os.environ.get("ALLOW_UNADDRESSABLE_OFFSETS") == "1":
        # Escape hatch, now NARROWER than it was: a fixed toolchain no longer needs it, because
        # toolchain_carries_argplus64() detects that case and returns above. What is left is the
        # case where the probe could not run (no MLIR_AIE_INSTANCE) but the operator knows the
        # toolchain is fixed. Anything else and this is an override of a measured failure, so it
        # says so at the volume the risk deserves.
        print(f"[gen] WARNING: {len(bad)} buffer(s) at or past 4 GiB, allowed by "
              f"ALLOW_UNADDRESSABLE_OFFSETS=1 (first {bad[0][0]!r} at {bad[0][2]:,}). The probe "
              f"says this toolchain NARROWS arg_plus to 32 bits, so these BDs are expected to be "
              f"patched with a wrapped offset and write to the wrong place SILENTLY. Verify the "
              f"emitted npu_insts DDR_PATCH high words before trusting any result from this build.",
              file=sys.stderr)
        return
    bad.sort(key=lambda b: b[2])
    first = bad[0]
    raise ValueError(
        f"{len(bad)} buffer(s) sit at or past the 4 GiB that aiex.npu.address_patch's I32 arg_plus "
        f"can address, so their BDs would be patched with a WRAPPED offset and write to the wrong "
        f"place -- silently. First: {first[0]!r} in arena {first[1]} at offset {first[2]:,} "
        f"(+{first[3]:,} bytes) = {(first[2] + first[3]) / 2**30:.3f} GiB. "
        f"Split this graph so each dispatch's arena stays under 4 GiB, or narrow the weights."
    )


# THE SCORES GEMV'S K CACHE IS A SEPARATE DECISION FROM THE CTX TMatVec'S V CACHE, and
# KV_BLOCK_ELIGIBLE above ANDs them into one build-wide bit. They are different ops on
# different buffers: gemv/design.py's blocked branch builds its A taps from
# `M, cols, n_matrices, block_size, a_row_width` alone and never references TMatVec, `m_chunk`
# or op_ctx. The collision the build-wide bit protects against is entirely inside
# tmatvec/design.py, whose blocked and m_chunk arms both want all four access-pattern dims --
# a real constraint on the V side and none at all on the K side.
#
# Gemma-4's global geometry is where that coupling costs: its ctx MUST stay output-chunked
# (at head_dim 512 the W term `batch_group*K*2` is 8 MB against 64 KB of L1, which no
# rows_per_chunk reaches), so `_tmv_chunked` is non-empty, so the whole build stays flat --
# and its scores GEMV then re-delivers the K cache 4x per invocation because the flat A run
# has no wrap-legal split at S=262144.
def scores_block_size(hd, hkv, w, kva, num_batches, batch_group, flat):
    """`block_size` for THIS geometry's scores GEMV, or `flat` (the V-side block) when blocking
    buys nothing.

    `flat` is the geometry's SHARED block `T_g`, not the window: the two coincide on every arm
    that ships (KVA_g == w makes T_g == w), and diverge only under KV_ALLOC, where the V side
    must shrink to divide the window's per-column share while the K side has no such constraint.
    Returning `w` there would hand the two appends different layouts for no reason and trip
    append_layouts_coincide.

    Blocked ONLY where blocking wins a delivery, which is asked of the operator rather than
    re-derived here: `group_reuse_n_vec` is gemv/design.py's own coalescing verdict, and the
    answer differs between the two layouts exactly when the flat A run is too long for the
    BD wrap field. Every geometry it does not win one for keeps the flat tap byte-for-byte --
    measured: gemma4-12b at S=6912 and its sliding w=1024 both read 'no change', so the
    shipped arms' ELFs are untouched by this being on.
    """
    if not (GROUPED_K and SCORES_KV_BLOCK):
        return flat
    # K and V share one `kv_off` slot and one host write, so at hkv > 1 they must be the SAME
    # block -- see append_layouts_coincide. Decline here rather than pick a divergent T and let
    # construction raise: a chooser that returns an unbuildable answer is not a chooser.
    if hkv > 1 and flat != w:
        return flat
    Tc = derive_block_size(hd, hkv, S=kva, n_cols=COLS,
                           addr_gran_elems=precision.kv_addr_gran_elems(PRECISION_PLAN))
    # Both divisibility rules the blocked GEMV asserts, checked HERE so a geometry that
    # cannot take the tiling falls back to flat instead of failing the build: `alloc_M % BLK`
    # (my_matvec) and `(M // cols) % BLK` (the blocked-tap assert).
    if Tc >= w or w % Tc or kva % Tc or (w // COLS) % Tc:
        return flat
    alloc = None if kva == w else kva
    flat_nvec = group_reuse_n_vec(w, COLS, num_batches, batch_group, hd, alloc, None)
    blocked = group_reuse_n_vec(w, COLS, num_batches, batch_group, hd, alloc, Tc)
    return Tc if blocked > flat_nvec else flat


def append_layouts_coincide(hkv, hd, w, T_k, T_v):
    """Do the K and V appends put every (head, position) at the SAME element offset?

    They share one `kv_off` scratchpad slot and one host write, so a K/V block split is only
    expressible today where the two layouts are the same ADDRESSING -- which happens whenever
    `hkv == 1`, because `block_stride = hkv*T*HD` then equals `T*HD` and the block and
    within-block terms recombine to exactly `pos*HD` for any T. Checked over the positions
    that straddle a block boundary rather than asserted from that argument, so this fails on
    the geometry it is wrong for instead of on the next reader's confidence.
    """
    a, b = KVLayout(Hkv=hkv, S=w, HD=hd, T=T_k), KVLayout(Hkv=hkv, S=w, HD=hd, T=T_v)
    probe = sorted({0, 1, T_k - 1, T_k, T_k + 1, T_v - 1, T_v, T_v + 1, w - 1} & set(range(w)))
    return all(a.offset(h, q) == b.offset(h, q) for h in range(hkv) for q in probe)


def gemv_tiling(M, K, **kw):
    """The (tile_size_input, tile_size_output) gemv() builds this shape with."""
    wdt = kw.get("weight_dtype", "bf16")
    # gemv_tile_output's default budget assumes a bf16 A row (K*2 B). A quantized row is narrower
    # -- 1.125*K at int8 g32, K at int8 g64 -- and the bf16 model overstates it by up to 1.78x,
    # which refuses tile_size_input values a quantized design actually fits (4371511). Row-group
    # planar makes this load-bearing, not just tighter: a planar block cannot be cut, so
    # tile_size_input must be a MULTIPLE of the derived row_group, and the overstated bf16 budget
    # was rejecting exactly the tsi values row_group_planar needs.
    a_row_bytes = row_stride_bytes(K, kw["group_size"], wdt) if wdt != "bf16" else None
    # n_vec, from the operator's own verdict rather than from `batch_group`: the two differ
    # exactly when the tap does not coalesce, and the design then runs ungrouped at n_vec=1 with
    # today's tiling. Asking group_reuse_n_vec keeps the arms that decline reuse -- every shipped
    # one -- tiled byte-identically, and shrinks the tile only where reuse is really on.
    #
    # UNITS: design.py's `a_row_width` is ELEMENTS for bf16 and BYTES for a quantized row, which
    # is the opposite convention to gemv_tile_output's `a_row_bytes`. Reuse needs batch_group>1,
    # which the operator asserts is bf16-only, so the quantized branch can only ever return 1.
    n_vec = group_reuse_n_vec(M, COLS, kw.get("num_batches", 1), kw.get("batch_group", 1),
                              K if wdt == "bf16" else a_row_bytes,
                              kw.get("alloc_M"), kw.get("block_size"))
    tsi, tso = gemv_tile_output(M, K, a_row_bytes=a_row_bytes, n_vec=n_vec)
    if kw.get("layout") == "row_group_planar":
        # The free search picks tsi for the LARGEST legal C tile, which can be SMALLER than the
        # row_group needs -- a planar block cannot be cut, so tsi must be a MULTIPLE of it
        # (K022), but a smaller tsi frees L1 budget quadratically (gemv_tile_output's own
        # docstring) and can win on tile size anyway. Measured at the lm-head shape (M=262144,
        # K=3840, row_group=4): free search returns tsi=1 (8192-elt C tile) over tsi=4's legal
        # 2048-elt one, because 1 has the bigger tile -- and GEMV.__post_init__'s own
        # derive_row_group then refuses it. Pin tsi to the row_group only when the free answer
        # would violate it; every site whose free answer already clears this (mlp/attn_o/qkv, as
        # built and shipped) takes the same tsi as today, unchanged.
        rg = derive_row_group([K], kw["group_size"], wdt,
                              vec_size=widest_chunk(kw["group_size"], wdt),
                              scale_dtype=_BUILD_STATE["scale_dtype"])
        if tsi % rg:
            tsi, tso = gemv_tile_output(M, K, a_row_bytes=a_row_bytes, tsi=rg, n_vec=n_vec)
    return tsi, tso


def gemv(M, K, ctx, **kw):
    """GEMV tiled as large as both the design asserts AND L1 allow."""
    tsi, tso = gemv_tiling(M, K, **kw)
    wdt = kw.get("weight_dtype", "bf16")
    g = kw.get("group_size", 0)
    if g and g < 64:
        # The width a chunk may span, from the one function that owns it. This used to pin
        # kernel_vector_size = group_size on the premise that "the dequant chunk must not
        # straddle a quant group" -- true until chunk_scales (IRON 836ac0d) made mv_quant.cc
        # build the scale vector as two half-broadcasts. GEMV's own guard already allows
        # kvs == 2*group_size; this was the last site still enforcing the old rule, and it
        # capped int4 g32 at 32 lanes on a 64-lane core (0.438 bundles/element against
        # int8 g64's 0.250) no matter what widest_chunk derived upstream.
        kw["kernel_vector_size"] = widest_chunk(g, wdt)
    return GEMV(M=M, K=K, num_aie_columns=COLS, tile_size_input=tsi,
                tile_size_output=tso, context=ctx, **kw)


def unify_weight_gemvs(rl):
    """Retile every family of quantized weight GEMVs that differ only in M, with tiles_rtp.

    A family is tiled the way gemv() tiles its gcd shape (the gcd of the members' per-core rows),
    so the one tiling divides every member and fits L1 for all of them. Families with a single M
    are left alone. Returns the rewritten runlist and (K, tsi, tso, Ms) per retiled family.
    """
    import dataclasses
    from math import gcd
    fams = {}
    for op in {id(e[0]): e[0] for e in rl}.values():
        if not (isinstance(op, GEMV) and op.weight_dtype != "bf16" and op.num_batches == 1
                and op.alloc_M is None and op.vector_size_parameter is None and not op.tiles_rtp):
            continue
        key = op.design_key().split("|")
        del key[1:6]  # cols, M, K, tsi, tso: keep cols and K, drop M and the tiling
        fams.setdefault((op.num_aie_columns, op.K, *key), []).append(op)
    new, report = {}, []
    for (cols, K, *_), ops in fams.items():
        Ms = sorted({op.M for op in ops})
        if len(Ms) < 2:
            continue
        rows = 0
        for M in Ms:
            rows = gcd(rows, M // cols)
        o = ops[0]
        tsi, tso = gemv_tiling(rows * cols, K, weight_dtype=o.weight_dtype,
                               group_size=o.group_size, layout=o.layout)
        for op in ops:
            new[id(op)] = dataclasses.replace(op, tile_size_input=tsi, tile_size_output=tso,
                                              tiles_rtp=True)
        report.append((K, tsi, tso, Ms))
    return [(new.get(id(op), op), *bufs) for op, *bufs in rl], report


def pointwise_modes(rl):
    """Replace Add, Mul and GELU that share (size, tile, columns) with Pointwise modes.

    Only widths carrying at least two of the three ops change, since one mode alone merges with
    nothing. GELU's (in, out) entries gain the input again as the ignored second operand. Returns
    the rewritten runlist and (size, tile_size, modes) per rewritten width.
    """
    from iron.operators.pointwise.op import Pointwise
    mode_of = {ElementwiseAdd: "add", ElementwiseMul: "mul", GELU: "gelu"}
    widths = {}
    for op in {id(e[0]): e[0] for e in rl}.values():
        if type(op) in mode_of and getattr(op, "num_channels", 1) == 1:
            widths.setdefault((op.size, op.tile_size, op.num_aie_columns), []).append(op)
    new, report = {}, []
    for (size, tile, cols), ops in widths.items():
        modes = sorted({mode_of[type(op)] for op in ops})
        if len(modes) < 2:
            continue
        for op in ops:
            new[id(op)] = Pointwise(size=size, tile_size=tile, mode=mode_of[type(op)],
                                    num_aie_columns=cols, context=op.context)
        report.append((size, tile, modes))
    out = []
    for op, *bufs in rl:
        if id(op) in new and isinstance(op, GELU):
            bufs = [bufs[0], *bufs]
        out.append((new.get(id(op), op), *bufs))
    return out, report


def _swiglu_default_tile_rows_gu():
    """swiglu_mlp_dp's own default Wg/Wu row batch, read off the design module.

    Hardcoding 6 here would be a second copy of a constant the operator owns, and the two would
    drift the first time either moved.
    """
    from iron.operators.swiglu_mlp_dp import design as _swiglu_design
    return _swiglu_design.TSI_GU


def build_graph(spec_name, weights_dir, layers=None, max_seq=2048, precision_plan=None):
    """Construct the fused decode graph + its weight dict for a spec.

    Shared by the generator CLI and verify_llm_decode.py so the harness drives the SAME graph the
    artifact was built from, rather than a re-typed copy that can drift from it.
    Returns (spec, fused, weights, meta_dims).

    `precision_plan` overrides the env-resolved plan for this call only, so one process can build
    several precision arms and hold them resident -- which is what an interleaved A/B needs, and
    the env cannot express twice in one process. The swap is scoped and restored, because the
    module-level helpers (`_spec`, `_quant_kw`, `_pack`) read the global by design: they are also
    called from the CLI path, where there is exactly one plan.
    """
    global PRECISION_PLAN
    if precision_plan is not None:
        _saved_plan = PRECISION_PLAN
        PRECISION_PLAN = dict(precision_plan)
        try:
            return build_graph(spec_name, weights_dir, layers, max_seq)
        finally:
            PRECISION_PLAN = _saved_plan
    sp = SPECS[spec_name]
    sp.check(cols=COLS, tsi=TSI)
    sp.check_seq(max_seq)
    NL = layers if layers is not None else sp.n_layers
    S = max_seq
    D, FF, HD = sp.d_model, sp.ffn, sp.head_dim
    Hq, Hkv, QD, KVD, VOCAB = sp.n_q_heads, sp.n_kv_heads, sp.q_dim, sp.kv_dim, sp.vocab

    # KV-cache layout: [S/T, Hkv, T, HD], block-major -- see iron.common.kv_layout, the single
    # owner of this addressing. T == S (one block) reproduces the pre-existing flat [Hkv, S, HD]
    # layout byte-for-byte; T < S makes head_stride and block_stride INDEPENDENT of S, which is
    # what lets a wide S address at all -- flat [Hkv,S,HD]'s per-head stride is S*HD elements,
    # and that lands in the shim's 20-bit BD step field (32-bit address granules), capping S at
    # ~8191 for head_dim=128 regardless of anything else in the design.
    #
    # The cache CAPACITY (what is allocated and addressable) and the attention WINDOW (how many
    # positions a dispatch reads) are separable: a narrow window over a wide allocation costs no
    # extra bytes above the coalescing threshold, and measured -0.388 ms rather than the 6.06 ms
    # penalty a naive read-the-whole-allocation model predicts.
    #
    # T is DERIVED, not chosen -- derive_block_size picks the largest T whose strides fit the
    # NARROWER of the shim (20-bit) and mem-tile (17-bit) step fields, because the mem-tile bound
    # is what a later staging step needs and re-deriving T when that lands would mean re-checking
    # every stride again. For Qwen3's shape (Hkv=8, HD=128) that is 128: at T=256 the block stride
    # is 262144 elements = 131072 granules, ONE over the mem-tile field's 131071; at T=128 it is
    # 65536 granules, comfortably under both fields.
    #
    # That field bound is not the only constraint on T: the blocked GEMV below also needs each of
    # the COLS columns' share of S to be a whole number of T-blocks, and the field bound alone
    # knows nothing about S or COLS. The two collide on single-KV-head geometries, where a small
    # Hkv lets the field bound keep doubling T past S//COLS before it ever binds -- Gemma3-270M
    # (Hkv=1, HD=256) is exactly that case (field bound alone: T=512; S//COLS=256; unbuildable).
    # Qwen3 (Hkv=8) never hits this second bound -- its field-derived T=128 already divides
    # S//COLS=256, incidentally, not because the field bound knows about columns. So pass S/COLS
    # in and let derive_block_size enforce both.
    #
    # Only activated for the arms that can actually ADDRESS a blocked cache today: gemv's
    # group_reuse coalesced path (GROUPED_K, batch_group>1) and tmatvec's one-head-per-column path
    # (TMV_CTX). Any other combination stays on the flat layout -- not a regression (identical to
    # every arm's behaviour before this task), a capability gate matching qkv_dp_why/mlp_dp_why's
    # own convention just above. Both arms read/write the SAME buffers, so this must be ONE
    # decision reaching every site, never re-evaluated per site -- a stale T on any one of them
    # would silently disagree with the layout the others wrote.
    # GEOMETRIES, and the TMatVec verdict per geometry. Both are pure functions of the spec, and
    # they are derived HERE -- above the KV-block decision -- because that decision has to know
    # whether EVERY geometry can take the tmatvec path, not just whether the flag is on.
    geoms = []
    for l in range(NL):
        gk = (sp.head_dim_for(l), sp.n_kv_heads_for(l), sp.has_v_proj(l))
        if gk not in geoms:
            geoms.append(gk)

    # TMV_CTX IS A PER-GEOMETRY CAPABILITY, NOT A BUILD-WIDE ONE. It used to be refused outright
    # when any geometry could not take it, which cost Gemma-4 the tmatvec path on all 48 layers
    # because 8 of them cannot: at head_dim 512 with batch_group 16 the W term alone
    # (batch_group*K*2) is the entire L1, and no rows_per_chunk touches it. The other 40 fit at
    # rows_per_chunk 32. The refusal was never about buildability -- it was about the artifact NAME,
    # since sequence_name() read the global flag and a mixed graph would have collided in the build
    # cache with an all-tmatvec one. sequence_name() now carries the declining head_dims, so the
    # name describes the graph and the decline can be per geometry.
    #
    # ONE OWNER for the verdict: this dict, consulted by attn_ops below and by the KV-block gate
    # just under it. Computing it twice is how a stale T silently disagrees with the layout the
    # other sites wrote -- the failure the KV_BLOCK_ELIGIBLE comment already warns about.
    tmv_rpc = {}
    if TMV_CTX:
        from iron.operators.tmatvec.design import (check_l1_fits,
                                                   largest_fitting_m_chunk)
        for _hd, _hkv, _ in geoms:
            _gqa, _r = Hq // _hkv, TMV_RPC
            while _r > 1 and (S % _r or check_l1_fits(_hd, S, _gqa, _r) is not None):
                _r //= 2
            if check_l1_fits(_hd, S, _gqa, _r) is None:
                tmv_rpc[_hd] = (_r, None)
                continue
            # Unchunked does not fit -- try the OUTPUT axis before declining; see m_chunk's own
            # comment in tmatvec/design.py for why it is the only knob that moves this floor.
            # WIDEST chunk that fits, not the first: design.py's headroom constant under-counts
            # (the data region is per-tiling), so model margin reads optimistic. Measured here,
            # m_chunk=256 leaves 6656 B of real L1 against m_chunk=128's 1024.
            _best = None
            _rr = TMV_RPC
            while _rr >= 1:
                if S % _rr == 0:
                    _mc = largest_fitting_m_chunk(_hd, S, _gqa, _rr)
                    if _mc and (_best is None or _mc > _best[1]):
                        _best = (_rr, _mc)
                _rr //= 2
            tmv_rpc[_hd] = _best
        for _hd, _v in sorted(tmv_rpc.items()):
            if _v is None:
                print(f"[gen] TMV_CTX declined at head_dim={_hd}: TMatVec does not fit L1 at any "
                      f"rows_per_chunk or m_chunk; that geometry keeps the transpose+GEMV path")
                continue
            _r, _mc = _v
            if _mc is not None:
                print(f"[gen] TMatVec at head_dim={_hd}: m_chunk={_mc} rows_per_chunk={_r} "
                      f"(output-chunked; unchunked does not fit L1 at any rows_per_chunk)")
            elif _r != TMV_RPC:
                print(f"[gen] TMatVec rows_per_chunk {TMV_RPC} -> {_r} (L1 fit at head_dim={_hd})")

    # The blocked cache stays a BUILD-WIDE decision even though the tmatvec verdict is not: both
    # arms read and write the same buffers, so a geometry on the fallback path would be addressing a
    # layout it cannot express. Blocking therefore needs EVERY geometry on the tmatvec path.
    _tmv_declined = tuple(sorted(h for h, v in tmv_rpc.items() if v is None))
    _tmv_chunked = tuple(sorted((h, v[1]) for h, v in tmv_rpc.items()
                                if v is not None and v[1] is not None))
    # ...and no geometry may be OUTPUT-CHUNKED: blocked A and m_chunk both want all four of
    # TMatVec's access-pattern dims, which the operator asserts against. This gate has to agree, or
    # enabling the global geometry flips the cache to a layout that cannot be built.
    KV_BLOCK_ELIGIBLE = GROUPED_K and TMV_CTX and not _tmv_declined and not _tmv_chunked
    _kv_block_env = os.environ.get("KV_BLOCK_T")
    if _kv_block_env is not None:
        T = int(_kv_block_env)
        if T != S:
            assert KV_BLOCK_ELIGIBLE, (
                f"KV_BLOCK_T={T} forces blocking but GROUPED_K={GROUPED_K}/TMV_CTX={TMV_CTX} "
                f"do not support it (see the blocked-tap NotImplementedError in gemv/tmatvec "
                f"design.py) -- set GQA_GROUPED_K=1 TMV_CTX=1 or KV_BLOCK_T={S}"
            )
    else:
        # addr_gran_elems is dtype-dependent -- a 4-byte granule over the cache's element width
        # -- and derive_block_size's own default is bf16's answer to a question it does not know
        # it is asking. precision.kv_addr_gran_elems owns that conversion.
        T = (derive_block_size(HD, Hkv, S=S, n_cols=COLS,
                               addr_gran_elems=precision.kv_addr_gran_elems(PRECISION_PLAN))
             if KV_BLOCK_ELIGIBLE else S)
    if T != S:
        assert S % T == 0, (
            f"T={T} does not divide S ({S}) -- pick an S that is a multiple of T"
        )
        assert (S // COLS) % T == 0, (
            f"blocked GEMV needs each of the {COLS} columns' share of S ({S // COLS}) to be a "
            f"whole number of blocks (T={T})"
        )
    # The descriptor is built on the CAPACITY, not the window: kv_off and the strides are
    # S-independent under blocking, so this only sizes the cache -- and every site that asks
    # kv_layout (buffer sizing, the append tap, the host's kv_off) then follows with no further
    # edit. Before the seam fix this same change had to be spelled out at five sites.
    KVA = KV_ALLOC or S
    if KVA < S:
        raise ValueError(f"KV_ALLOC={KVA} < max_seq={S}: it is the capacity, not a window")
    kv_layout = KVLayout(Hkv=Hkv, S=KVA, HD=HD, T=T)
    print(f"[gen] KV cache layout: T={T}"
          + (" (flat [Hkv,S,HD])" if T == S else
             f" (blocked [S/T,Hkv,T,HD], head_stride={kv_layout.head_stride}, "
             f"block_stride={kv_layout.block_stride} elements)"))

    def npy(name):
        # mmap_mode + copy=False, and BOTH halves matter. The dump is f32 on disk and the tied
        # embedding is the whole table: 3.75 GiB for Gemma-4-12B (262144x3840). Reading that into
        # anonymous memory, then copying it again because astype() copies even when the dtype
        # ALREADY MATCHES, is 7.5 GiB of resident memory for one tensor -- which is what
        # OOM-killed the 12B build, and why an 8-layer build died too: the embedding is fixed cost
        # and depth does not touch it. Every consumer below is read-only and builds a new array
        # (quantize_weight writes into its own `out`, bf16()/np.pad() allocate), so a read-only
        # view is sufficient. Page cache is evictable; anonymous memory is not.
        return np.load(os.path.join(weights_dir, f"{name}.npy"),
                       mmap_mode="r").astype(np.float32, copy=False)

    def npy_raw(name):
        """Load without casting. A packed weight is np.int8 ON-WIRE BYTES, not values -- widening
        it to float32 renumbers the payload instead of copying it, and does so silently."""
        return np.load(os.path.join(weights_dir, f"{name}.npy"), mmap_mode="r")

    # THE DUMP DECLARES ITSELF; this reads the declaration rather than agreeing with it via matching
    # env vars. The packer and the matvec kernel share an on-wire row layout, and nothing else in
    # the build checks it -- so a disagreement about dtype or group size is silent numerical
    # garbage, not a load error. A dump too large to write at f32 (a 12B is 43 GB f32, 6 GB at int4
    # g64) can only arrive pre-packed, which is why this path exists at all.
    _qmf_path = os.path.join(weights_dir, "quant.json")
    _qmf = (json.load(open(_qmf_path)) if os.path.isfile(_qmf_path)
            else {"dtype": "bf16", "group_size": 0, "packed": []})
    PACKED = set(_qmf.get("packed", []))
    # The tied head/embedding packs into its OWN sidecar name (dump_llm_weights.py's
    # _pack_head_chunked), never the bare key -- unlike mlp/attn_o/qkv the base f32 tensor always
    # stays on disk too (the host embedding-gather reads it), so membership has to be checked by
    # this exact name rather than assumed from PACKED being non-empty.
    _head_key = f"{sp.weight_prefix}embed_tokens.weight"
    _head_packed = f"{_head_key}.headpack" in PACKED
    # Layout is not a per-site plan choice (every quantized site shares one on-wire row shape),
    # so unlike dtype/group_size below there is nothing to conflict-check against -- the dump's
    # declaration is simply adopted, same as PACKED itself. Old dumps have no "layout" key and
    # default to header_first, byte-identical to every build before this axis existed.
    _dump_layout = _qmf.get("layout", "header_first")
    if _dump_layout not in ("header_first", "row_group_planar"):
        raise SystemExit(f"{_qmf_path}: unknown layout {_dump_layout!r}")
    if _dump_layout == "row_group_planar" and derive_row_group is None:
        raise SystemExit(
            f"{_qmf_path}: dump is row_group_planar but this IRON tree has no derive_row_group "
            f"(pre-6a347dc). Point IRON at a tree past the quant.py move.")
    _BUILD_STATE["layout"] = _dump_layout
    # Same adoption rule as layout: one on-wire scale width for the whole dump, taken from the
    # manifest rather than declared here. Old dumps have no key and are f32, which is what every
    # build before this axis wrote.
    _dump_scale_dtype = _qmf.get("scale_dtype", "f32")
    if _dump_scale_dtype not in ("f32", "bf16"):
        raise SystemExit(f"{_qmf_path}: unknown scale_dtype {_dump_scale_dtype!r}")
    _BUILD_STATE["scale_dtype"] = _dump_scale_dtype
    if PACKED and _qmf.get("dtype", "bf16") == "bf16":
        raise SystemExit(f"{_qmf_path}: lists {len(PACKED)} packed tensors but dtype is bf16")
    if PACKED:
        # A packed dump is the AUTHORITY for the weight format, because the bytes are already on
        # disk in it -- the plan can no longer choose. A plan that asks for something ELSE at a
        # packed site is a contradiction and fails loudly here; a site left at bf16 simply adopts
        # the manifest's format. The scale_kind is the dump's own: it moved weight VALUES at pack
        # time, so re-declaring it here would claim a calibration this build did not do.
        _mdt, _mgs = _qmf["dtype"], int(_qmf["group_size"])
        _dump_spec = precision.Spec(dtype=_mdt, group_size=_mgs)
        for _site in ("mlp", "attn_o", "qkv"):
            _cur = PRECISION_PLAN.get(_site, precision.BF16_SPEC)
            if _cur.quantized and (_cur.dtype, _cur.group_size) != (_mdt, _mgs):
                raise SystemExit(
                    f"the precision plan asks for {_cur} at site {_site!r}, but {_qmf_path} is "
                    f"already packed at {_mdt} g{_mgs} and the build cannot re-choose the format. "
                    f"Drop {_site!r} from the plan to take the dump's, or re-dump at the format "
                    f"you want.")
            PRECISION_PLAN[_site] = _dump_spec
        if _head_packed:
            _cur = PRECISION_PLAN.get("head", precision.BF16_SPEC)
            if _cur.quantized and (_cur.dtype, _cur.group_size) != (_mdt, _mgs):
                raise SystemExit(
                    f"the precision plan asks for {_cur} at site 'head', but {_qmf_path} carries "
                    f"a packed head sidecar at {_mdt} g{_mgs} and the build cannot re-choose the "
                    f"format. Drop 'head' from the plan to take the dump's, or re-dump at the "
                    f"format you want.")
            PRECISION_PLAN["head"] = _dump_spec
        print(f"[gen] packed dump is the authority: {_mdt} g{_mgs} at mlp/attn_o/qkv"
             + ("+head" if _head_packed else ""))

    def load_norm(name):
        # Gemma-3 stores RMSNorm gain as w with the kernel computing x_hat*(1+w); Qwen3 stores it
        # already absolute. IRON's weighted RMSNorm always does x_hat*w', so Gemma folds the +1 here.
        w = npy(name)
        return bf16(1.0 + w) if sp.norm_gain == "one_plus_w" else bf16(w)

    # AIE_DEVICE=npu2 pins the target instead of probing the runtime for it. The probe OPENS
    # /dev/accel, which makes an otherwise device-free build queue behind whatever is running on
    # the single-tenant NPU -- tonight that was ~16 min of another session's eval for a 3.5 min
    # build. Pinning is opt-in, so the default still verifies you are building for the device you
    # actually have; set it only when you know the target and want the build off the device lock.
    if os.environ.get("AIE_DEVICE"):
        import aie.utils as _aie_utils
        from aie.iron.device import from_name as _from_name
        # n_cols=None means the device's full width; COLS above already assumes all 8.
        _aie_utils.set_current_device(_from_name(os.environ["AIE_DEVICE"], n_cols=None))

    ctx = AIEContext()

    # ---- op vocabulary: created ONCE, reused across every layer (same dims per layer) ----
    # 1 column, and this is FORCED by the operator's semantics, not a placement budget.
    # RMSNorm's `tile_size` is the NORMALISED WIDTH, not a work split: its own test reads
    # `rows = input_length // tile_size, cols = tile_size`, and design_weighted.py carries one
    # weight ObjectFifo of `tile_size` elements shared across every column. num_aie_columns splits
    # ROWS. A decode step normalises ONE vector of d_model, so rows == 1 and there is nothing to
    # spread. Setting tile_size = D // 4 to buy width would compute four independent 256-wide
    # normalisations instead of one 1024-wide one -- a different function, not a faster one. The
    # buffer-size gate caught it first ("L0_n_in.bin is 2048 bytes, layout declares 512"), which is
    # luck: the sizes happened to disagree. Had D//4 divided evenly into the dumped weight it would
    # have run and been quietly wrong.
    op_norm = RMSNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D,
                      weighted=True, epsilon=sp.eps, context=ctx)
    # Wo weight-stream dtype axis (see QUANT_ATTN_DTYPE above). bf16 (default) is byte-for-byte the
    # pre-existing path.
    attn_quant_kw = _quant_kw("attn_o")

    # ---- attention op vocabulary, keyed on the layer's ATTENTION GEOMETRY ----
    # Everything below depends on (head_dim, n_kv_heads), and Gemma-4-12B does not have one pair:
    # its sliding layers are 256/8 and its global layers 512/1. So these cannot be built once for
    # the stack the way the d_model- and ffn-shaped ops above and below can.
    #
    # A memoized factory rather than per-layer construction, because the uniform case has to stay
    # exactly what it was: every shipped spec has ONE geometry, so the cache holds one entry, the
    # ops are the same objects every layer references, and the artifact is byte-identical to the
    # pre-refactor build. The non-uniform case then costs one more entry rather than a rewrite.
    #
    # What is NOT in here is as load-bearing as what is. op_norm, op_scale, op_softmax and the whole
    # MLP vocabulary are shaped by d_model, ffn and n_q_heads, none of which varies per layer, so
    # hoisting them in would key them on something they do not depend on and multiply designs for
    # nothing.

    # ---- which fused arms this MODEL can use ----
    # Whether a fused arm applies is the OPERATOR's rule, not a choice here -- the same shape as
    # fuse_act further down, which already asks the GEMV instead of assuming. The env flag can only
    # turn an arm OFF (for A/B); it can no longer turn one ON for a spec the operator does not
    # cover. It used to, and a Qwen-shaped default then met Gemma-3 as a NotImplementedError three
    # frames down -- a capability gap reported as a crash, and only after the previous gap was
    # cleared, so the four of them surfaced one build at a time.
    #
    # The qkv arm is decided PER GEOMETRY: its rule is `d_model % head_dim`, so two head_dims can
    # genuinely disagree about it. The mlp arm's rules (sandwich norms, activation) touch no
    # attention geometry, so it stays one verdict for the build.
    # Split in two because attn_block_dp REPLACES qkv_head_dp and so shares the SHAPE rules but
    # not the A/B switch: FUSE_QKV_DP is not one of its preconditions, FUSE_ATTN_BLOCK is.
    # FUSE_QKV_GEMV is shared -- both arms read one concatenated Wqkv.
    qkv_shape_why = ({g: "needs FUSE_QKV_GEMV=1 for the concatenated Wqkv" for g in geoms}
                     if not FUSE_QKV_GEMV else
                     {g: sp.qkv_dp_reason(COLS, head_dim=g[0]) for g in geoms})
    qkv_dp_why = {g: "FUSE_QKV_DP=0" for g in geoms} if not FUSE_QKV_DP else qkv_shape_why
    mlp_dp_why = "FUSE_MLP_DP=0" if not FUSE_MLP_DP else sp.mlp_dp_reason()
    if mlp_dp_why is None:
        from iron.operators.swiglu_mlp_dp.op import SwiGLUMLPDataParallel

        _unaccepted = operator_rejects(SwiGLUMLPDataParallel, _quant_kw("mlp"))
        if _unaccepted:
            mlp_dp_why = (
                f"swiglu_mlp_dp takes no {'/'.join(_unaccepted)} parameter, so it cannot read the "
                f"{_BUILD_STATE['layout']}/{_BUILD_STATE['scale_dtype']} dump this build packs"
            )
        # The tiling this build will actually ask for, checked HERE rather than left to fire as an
        # AssertionError inside design.py. On the unchunked, non-row-parallel path a Wd row batch is
        # TSI_GU//R, so TSI_GU must be a whole number of R=FF/D; MLP_TILE_ROWS=0 means the
        # operator's own module default. Gemma-4 (R=4) meets the default TSI_GU=6 here: the gate
        # stayed shut on the _unaccepted clause above until swiglu_mlp_dp grew layout/scale_dtype,
        # and the first build after it opened died 250 lines inside the design function.
        elif FF % D != 0:
            # design.py asserts this outright ("this design assumes FF is a whole multiple of D").
            # Gemma-3-270M is D=640 FF=2048, so R would be 3.2 -- the operator has no form for it.
            mlp_dp_why = (
                f"swiglu_mlp_dp needs FF ({FF}) to be a whole multiple of D ({D}); this spec's "
                f"ratio is {FF / D:.4g}"
            )
        elif os.environ.get("MLP_ROW_PARALLEL", "0") != "1":
            _tsi_gu = MLP_TILE_ROWS or _swiglu_default_tile_rows_gu()
            if _tsi_gu % (FF // D) != 0:
                mlp_dp_why = (
                    f"MLP_TILE_ROWS={_tsi_gu} is not a multiple of R=FF/D={FF // D}, which "
                    f"swiglu_mlp_dp's unchunked down projection requires (design.py's TSI_D). "
                    f"Set MLP_TILE_ROWS to a multiple of {FF // D}, or use MLP_ROW_PARALLEL=1, "
                    f"whose local matvec has no relationship to R"
                )

    # Per-GEOMETRY eligibility for attn_block_dp used WITHOUT decode_layer_dp. Same clauses
    # decode_layer_why checks below for the one geometry it requires, reused via qkv_shape_why/
    # _tmv_declined rather than re-derived, so the two verdicts cannot drift apart -- but with no
    # opinion on the MLP half (no mlp_dp_why, no FUSE_MLP_O) and no single-geometry requirement, so
    # a spec whose layers disagree about attention geometry can fuse some and fall back on others.
    #
    # v_norm no longer refuses here: attn_block_dp/op.py now takes v_norm (step 5 gets the same
    # weighted-RMSNorm op as qk-norm), so the only remaining v_norm exposure is op_qkv_dp's own
    # NotImplementedError below (fused QKV head still has no value-norm stage).
    def _attn_block_why(g):
        hd, hkv, has_v = g
        return ("FUSE_ATTN_BLOCK=0" if not FUSE_ATTN_BLOCK else
                qkv_shape_why[g] if qkv_shape_why[g] else
                f"needs Hkv ({hkv}) == COLS ({COLS})" if hkv != COLS else
                "attn_block_dp always fuses K and V; this geometry has no v_proj" if not has_v else
                "needs SCALE_IN_QNORM=1 (attn_block_dp has no separate scale stage)"
                if not (SCALE_IN_QNORM and sp.qk_norm) else
                "needs GQA_GROUPED_K=1 and TMV_CTX=1 (attn_block_dp computes exactly that "
                "variant internally)" if not (GROUPED_K and TMV_CTX and hd not in _tmv_declined)
                else None)

    attn_block_why = {g: _attn_block_why(g) for g in geoms}
    # head_dims that actually qualify, for sequence_name()'s suffix -- computed once here, same
    # discipline as _tmv_declined/_tmv_chunked above, rather than re-derived at the call site.
    _attn_block_fused = tuple(sorted(g[0] for g in geoms if attn_block_why[g] is None))
    # Filled by attn_ops (below) as each geometry is built, then handed to sequence_name -- the
    # SAME value the op was constructed with, not a re-derivation of the gate.
    _scores_blocks = []

    # HOISTED ABOVE THE UNFUSED ATTENTION OPERATORS, and the move is load-bearing rather than
    # tidy-up. When the fused layer wins, op_rep_k/op_rep_v/op_scores/op_softmax/op_trv/op_ctx are
    # constructed and then never reach a runlist -- every use of them is inside the `else` arm
    # below. Constructing them anyway means their CONSTRAINTS still gate the build, and TMatVec's
    # in particular is a window cap the fused arm does not have: its W buffer is batch_group*K, so
    # at K=32768 it is 131072 B against a 64 KB L1 and the build dies on a dead operator. Measured
    # 2026-09-11 -- that is exactly what blocked the first 32k decode build, AFTER split-K had
    # already placed attn_block_dp at the same window.
    #
    # WHICH ARM CARRIES THE LAYER: the precision check below is conditional on it, and every
    # clause here is a spec/flag question that needs no operator. Eligibility is the union of
    # qkv_dp_why/mlp_dp_why (the spec-shape rules attn_block_dp and swiglu_mlp_dp already check)
    # plus what is true only of the MERGED device: attn_block_dp's own Hkv==COLS rule, and no
    # sandwich norms (the op has no post-attn/post-ffn norm slot).
    # ONE geometry. decode_layer_dp bakes HD/Hq/Hkv into a single design and the runlist
    # substitutes ONE op per layer (that is also what a window rung rewrites), so a spec whose
    # layers disagree about head_dim has no single design to substitute. Refuse by name rather
    # than build the first geometry's design and run every layer through it.
    _geom1 = geoms[0] if len(geoms) == 1 else None
    decode_layer_why = ("FUSE_DECODE_LAYER=0" if not FUSE_DECODE_LAYER else
                        # FIRST, ahead of the QD clause below: fuse_o is now a real, checked
                        # DecodeLayerDataParallel field, and False raises there (no embeddable
                        # O-projection operator, and the shim/L1 budget forecloses one anyway --
                        # see decode_layer_dp.op.__post_init__). QD not dividing D only matters
                        # because fuse_o=True is the only buildable arm; a FUSE_MLP_O=0 caller was
                        # never reaching it. Falls back to attn_block_dp + a standalone op_o gemv
                        # + swiglu_mlp_dp(fuse_o=False), same as fused-operators-cannot-read-our-
                        # quantized-weight-layout's sibling case.
                        "FUSE_MLP_O=0" if not FUSE_MLP_O else
                        # BEFORE the geometry count, because it does not depend on it.
                        # swiglu_mlp_dp's fuse_o derives R_CX = QD//D, so a model whose QD is not
                        # a whole multiple of D is refused at ANY geometry count. Checked across
                        # EVERY geometry rather than the single one, which is None exactly when
                        # this used to go unreported: the count clause below short-circuited
                        # first and the real failure surfaced as a raise three frames down in
                        # another operator, naming a variable instead of a model. A session
                        # ordered a day's work behind the count clause believing it was the
                        # blocker.
                        "fuse_o needs QD to be a whole multiple of D ({}); {} has {}".format(
                            D, sp.name,
                            ", ".join(f"QD={Hq * _hd} (remainder {(Hq * _hd) % D})"
                                      for _hd, _, _ in geoms if (Hq * _hd) % D))
                        if any((Hq * _hd) % D for _hd, _, _ in geoms) else
                        f"needs ONE attention geometry; {sp.name} has {len(geoms)}: {geoms}"
                        if _geom1 is None else
                        qkv_dp_why[_geom1] if qkv_dp_why[_geom1] else
                        mlp_dp_why if mlp_dp_why else
                        f"needs Hkv ({Hkv}) == COLS ({COLS})" if Hkv != COLS else
                        "needs SCALE_IN_QNORM=1 (attn_block_dp has no separate scale stage)"
                        if not (SCALE_IN_QNORM and sp.qk_norm) else
                        "needs GQA_GROUPED_K=1 and TMV_CTX=1 (attn_block_dp computes exactly "
                        "that variant internally)"
                        if not (GROUPED_K and TMV_CTX and not _tmv_declined) else
                        # The weight FORMAT is not a clause here. The MLP half forwards its
                        # dtype to swiglu_mlp_dp, and a format the attention half cannot carry is
                        # a REFUSAL (P003), not a reason to quietly drop to the unfused arm --
                        # which is what silently unfusing a whole decoder layer used to be.
                        None)
    # Sandwich norms pass the post-FFN gain as the fused design's own argument, and under fuse_o
    # get_arg_spec wants BOTH post-norm gains packed as one 2*D buffer [n_pa | n_pff] while the
    # generator holds them as two D buffers. Declining fuse_o is not a fallback here, it is the
    # only expressible arm: the unfused runlist below already carries n_pff correctly.
    fuse_o_why = ("sandwich norms need the post-norm gains packed as one 2*D buffer"
                  if sp.sandwich_norms else None)
    fuse_o = FUSE_MLP_O and mlp_dp_why is None and fuse_o_why is None
    if FUSE_MLP_O and mlp_dp_why is None and fuse_o_why:
        print(f"[gen] fused arm mlp_o: OFF -- {fuse_o_why}")
    for g in geoms:
        why, tag = qkv_dp_why[g], "" if len(geoms) == 1 else f" [head_dim={g[0]}, kv_heads={g[1]}, v_proj={g[2]}]"
        print(f"[gen] fused arm qkv_head_dp{tag}: {'OFF -- ' + why if why else 'on'}")
    for g in geoms:
        why = attn_block_why[g]
        tag = "" if len(geoms) == 1 else f" [head_dim={g[0]}, kv_heads={g[1]}]"
        print(f"[gen] fused arm attn_block_dp{tag}: {'OFF -- ' + why if why else 'on'}")
    print(f"[gen] fused arm swiglu_mlp_dp: {'OFF -- ' + mlp_dp_why if mlp_dp_why else 'on'}")
    if fuse_o:
        if _spec("attn_o") != _spec("mlp"):
            # Under fuse_o, Wo rides the MLP design's single weight ObjectFifo, and one fifo
            # carries one wire format. So Wo's dtype is QUANT_MLP_DTYPE's, not its own axis --
            # QUANT_ATTN_DTYPE would silently mean nothing here rather than a little.
            raise NotImplementedError(
                "FUSE_MLP_O folds Wo into swiglu_mlp_dp's shared weight channel, and one fifo "
                f"carries one wire format, so Wo must take the mlp site's format "
                f"({_spec('mlp')}); attn_o is {_spec('attn_o')}. Set them equal, or FUSE_MLP_O=0 to "
                "quantize Wo independently"
            )
        if len(geoms) > 1:
            # op_mlp_dp is built ONCE with QD baked in, because Wo rides its weight channel. Two
            # q_dims cannot share it, and the failure would be a silent stride error rather than a
            # crash -- so refuse here rather than build the wrong thing.
            qds = sorted({Hq * hd for hd, _, _ in geoms})
            raise NotImplementedError(
                f"FUSE_MLP_O folds Wo into one swiglu_mlp_dp design carrying a single QD, but "
                f"{sp.name} has {len(geoms)} attention geometries {geoms} and therefore the "
                f"q_dims {qds}. Set FUSE_MLP_O=0, or give the operator a per-layer QD.")
    qkv_quant_kw = _quant_kw("qkv")
    # THE PRECISION PLAN IS CHECKED HERE, not at the top of the file: which combinations are
    # buildable depends on the fused arms decided just above (which weights share an ObjectFifo,
    # which operator declares which buffer, how many shim channels are spent). Refusing here is
    # what turns "undefined symbol" and "weight byte-size mismatch: buf 6291456 vs arr 1671168"
    # into a named rule.
    _pdtypes, _pkind, _pfull = precision.packer_capability()
    precision_ctx = precision.GraphContext(
        fused_layer=decode_layer_why is None and FUSE_DECODE_LAYER,
        fuse_o=fuse_o, fused_qkv_gemv=bool(FUSE_QKV_GEMV),
        fused_qkv_dp=qkv_dp_why is None,
        d_model=D, ffn=FF, q_dim=QD, head_dim=HD, attn_cols=COLS,
        packer_dtypes=_pdtypes, packer_takes_scale_kind=_pkind,
        packer_takes_full_range=_pfull)
    precision.check(PRECISION_PLAN, precision_ctx)
    print(f"[gen] precision [{PRECISION_PROV}]")
    for _line in precision.describe(PRECISION_PLAN, sp.name).splitlines()[1:]:
        print(f"[gen] {_line}")

    # Wqkv's own dtype axis. The concatenated [Wq|Wk|Wv] GEMV has its own weight ObjectFifo, so
    # it takes a format independently -- which the fused layer's attention half does NOT, because
    # attn_block_dp streams Wqkv, K and V down one fifo per core (P002/P003 above).
    op_qkv = gemv(QD + 2 * KVD, D, ctx, **_quant_kw("qkv")) if FUSE_QKV_GEMV else None
    op_q = gemv(QD, D, ctx)
    op_kv = gemv(KVD, D, ctx)
    op_o = None if fuse_o else gemv(D, QD, ctx, **_quant_kw("attn_o"))
    # RoPE over q and k together (24 head rows) needs them adjacent, which only the fused qkv
    # buffer gives; angle_rows=1 is unchanged, so every row still reads the same single angle row.
    fuse_rope = FUSE_QKV_GEMV and FUSE_ROPE_QK

    # One (scratchpad slot, head_dim) pair per DISTINCT attention geometry, appended by the factory
    # as it builds each one. The host reads this as a list (Artifact::kv_offs) and writes
    # `pos * head_dim` to each slot. It has to be derived beside the StridedCopy ops that consume
    # the slot, not restated at the meta site, because the pairing IS the contract: which slot a
    # layer's KV-append reads and which head_dim scales it are the same decision.
    kv_slots = []
    # One (scratchpad slot, window) pair per DISTINCT geometry that gets its OWN softmax/mask --
    # same contract as kv_slots, appended beside the Softmax op that consumes it. Empty (today's
    # default) means every geometry shares the ONE build-wide `sm_mask` built outside attn_ops;
    # SLIDING_KV_CIRCULAR is what makes this list non-empty.
    mask_slots = []
    # Per-GEOMETRY record, kept separate from kv_slots (whose 2-tuple shape other call sites
    # already destructure -- extending it would break them). One entry per distinct geometry,
    # carrying everything a host loop needs to drive it: which kv_off slot, its head_dim, this
    # geometry's own capacity, and which sm_mask slot pairs with it. kv_slots and mask_slots
    # cannot be zipped positionally for this -- mask_slots is keyed by DISTINCT WINDOW, so two
    # geometries sharing one window (every artifact before this flag) collapse to ONE mask_slots
    # entry while kv_slots still has two.
    geom_slots = []
    _attn_cache = {}
    # Softmax/scale keyed by WINDOW, not by the full geometry key: two geometries sharing a window
    # (today, always -- both at S) must share the SAME design object, exactly as the pre-existing
    # single build-wide op_softmax did, or gemma4-12b's default build silently gains a second
    # configure block it did not have. Only SLIDING_KV_CIRCULAR narrowing one geometry's w away
    # from S makes this cache produce more than one entry.
    _win_cache = {}

    def attn_ops(hd, hkv, has_v):
        """The ops shaped by one (head_dim, n_kv_heads, has_v_proj) triple.

        `has_v` is a third key component and not derived from the other two on purpose. Under
        Gemma-4 the layers that lack a v_proj are exactly the ones with the other geometry, so a
        two-part key would work by coincidence here -- and silently mis-key the first model where
        attention_k_eq_v and the geometry split do not coincide.
        """
        if (hd, hkv, has_v) in _attn_cache:
            return _attn_cache[(hd, hkv, has_v)]
        qd, kvd, gqa = Hq * hd, hkv * hd, Hq // hkv
        # This geometry's own attention capacity. The GLOBAL geometry (identified by matching
        # spec.global_head_dim/global_n_kv_heads, not by an explicit flag -- attn_ops is keyed
        # purely on (hd, hkv, has_v)) always stays at the build's max_seq; a declared
        # sliding_window narrows every OTHER geometry only under SLIDING_KV_CIRCULAR.
        is_global_geom = hd == sp.global_head_dim and hkv == sp.global_n_kv_heads
        w = (S if (is_global_geom or not SLIDING_KV_CIRCULAR or sp.sliding_window is None)
             else sp.sliding_window)
        # EXPERIMENT KNOB, 2026-09-18, not a shipped default. attn_block_dp pads each quantized
        # Wqkv row up to `rpc` cache rows, and `rpc` must both cover the row and DIVIDE the
        # window -- so 1024 forces rpc=8 (1.90x) where any multiple of 5 reaches rpc=5 (1.19x).
        # This narrows the sliding window, so it is a QUALITY change and must not be defaulted on.
        if not is_global_geom and os.environ.get("SLIDING_WINDOW_OVERRIDE"):
            w = int(os.environ["SLIDING_WINDOW_OVERRIDE"])
        # KV_ALLOC widens the CAPACITY of a geometry whose window can grow; a CIRCULAR sliding
        # geometry's window never grows -- it wraps at `w` by construction -- so its capacity IS
        # `w` and widening it would allocate 256x the cache those layers can ever address. `w != S`
        # is exactly the circular condition (see `w`'s own derivation above).
        KVA_g = w if w != S else (KV_ALLOC or w)
        # T (the KVLayout block size) is derived once, globally, against S -- today always S
        # itself (flat, "one block"). A geometry whose own capacity is w < T needs its OWN flat
        # block size, or the GEMV/TMatVec block_size%alloc_M==0 check fails (T does not divide a
        # smaller alloc). Keeping the same "one block" convention this build already uses for S
        # means T_g is simply w, not a blocking scheme of its own.
        T_g = min(T, w)
        # A WINDOWED geometry over a WIDER capacity makes the GEMV genuinely
        # blocked (alloc_M=KVA_g != block_size), and the blocked tap then also needs the block to
        # divide each COLUMN'S SHARE OF THE WINDOW -- `(M//cols) % _BLK` in gemv/design.py -- not
        # just the capacity, AND one block's run must have a wrap-legal split, which is why this
        # asks gemv for `split_run` rather than re-deriving the 10-bit field. At KVA_g == w the
        # GEMV is unblocked and none of these bind, so this loop cannot fire on a shipped arm.
        if KVA_g != w:
            # tmatvec composes m_chunk with block_size only when one block is one K-chunk, so a
            # CHUNKED ctx over a wider capacity has exactly one legal block size: its own
            # rows_per_chunk. Anything else re-raises the four-dims refusal from inside the
            # operator, which is a worse place to learn it.
            _rpc_mc = tmv_rpc.get(hd)
            if _rpc_mc is not None and _rpc_mc[1] is not None:
                T_g = _rpc_mc[0]
            while T_g > 1 and ((w // COLS) % T_g or KVA_g % T_g
                               or gemv_split_run(T_g * hd) is None):
                T_g //= 2
        # SCORES BATCHING, derived here because BOTH the block-size gate below and the op itself
        # must be built from the same pair. The gate asks the operator what n_vec this shape gets,
        # and n_vec is a function of (num_batches, batch_group) -- computing it from `Hq, gqa`
        # while the op is built at `MAX_GROUP_REUSE, MAX_GROUP_REUSE` would be a gate answering
        # about a different graph than the one that ships (see a-name-and-a-graph-need-one-
        # predicate). GQA's own group_reuse gate (gemv/design.py) DECLINES batch_group >
        # MAX_GROUP_REUSE (a measured shim-BD ceiling) and falls back to a stride-0 outer BD that
        # re-reads the whole matrix once per query head -- measured 14.71x on Gemma-4's global
        # layers (hkv=1, gqa=Hq=16), 270.01 MB/token against 18.35 MB unique. Below the ceiling
        # (sliding, gqa=2) this is unreachable and op_scores is unchanged.
        #
        # The fix stays inside group_reuse instead of raising the ceiling (a known dead end --
        # batch_group=16 makes aiecc's B_L3L1_0 exceed 16 blocks, see MAX_GROUP_REUSE's own
        # comment): build op_scores at batch_group=MAX_GROUP_REUSE (n_matrices=1, matching the
        # ONE real K matrix) and call it scores_groups times over MAX_GROUP_REUSE-head slices --
        # the same "one configure, N runs over slices" idiom down_runlist/o_runlist already use.
        scores_group_fix = GROUPED_K and gqa > MAX_GROUP_REUSE
        if scores_group_fix:
            assert hkv == 1, (
                f"scores group-reuse fallback assumes ONE real K matrix per geometry (hkv=1); "
                f"got hkv={hkv} (hd={hd}) -- the per-slice K addressing is unimplemented for "
                f"hkv>1"
            )
            assert gqa % MAX_GROUP_REUSE == 0, (
                f"scores group-reuse fallback slices gqa into fixed {MAX_GROUP_REUSE}-head "
                f"calls; gqa={gqa} (hd={hd}) is not a multiple of it"
            )
        scores_groups = gqa // MAX_GROUP_REUSE if scores_group_fix else 1
        scores_nb = MAX_GROUP_REUSE if scores_group_fix else Hq
        scores_bg = MAX_GROUP_REUSE if scores_group_fix else (gqa if GROUPED_K else 1)
        # The K cache's own block size for the SCORES GEMV. `T_g` (the V/physical one) is what
        # kv_layout, the host's kv_off and the prefill pairing all describe and it is unchanged;
        # this is a re-TAPING of the same bytes, not a relayout -- append_layouts_coincide below
        # is what holds that true rather than this comment.
        T_k = scores_block_size(hd, hkv, w, KVA_g, scores_nb, scores_bg, T_g)
        if T_k != T_g:
            if not append_layouts_coincide(hkv, hd, w, T_k, T_g):
                raise NotImplementedError(
                    f"scores K block T={T_k} and V block T={T_g} at (head_dim={hd}, "
                    f"n_kv_heads={hkv}) address the SAME (head, position) differently, and both "
                    f"appends share one `kv_off` slot and one host write. Needs a second kv_off "
                    f"scratchpad slot (gen_llm_decode's kv_slots, meta.json's kv_windows and "
                    f"npu_decode.rs's per-geometry write) before a split can be built here -- or "
                    f"SCORES_KV_BLOCK=0 to keep this geometry flat.")
            for _op_name, _why in (("op_attn_block", attn_block_why[(hd, hkv, has_v)]),
                                   ("op_qkv_dp", qkv_dp_why[(hd, hkv, has_v)])):
                if _why is None:
                    raise NotImplementedError(
                        f"scores K block T={T_k} differs from the V block T={T_g}, but {_op_name} "
                        f"appends BOTH caches itself through one `kv_block_size` and cannot "
                        f"express two. Set SCORES_KV_BLOCK=0, or give that operator a per-cache "
                        f"block size.")
            print(f"[gen] scores K cache BLOCKED at head_dim={hd}: T={T_k} (V cache stays flat "
                  f"at {T_g}, same bytes) -- A deliveries per invocation {scores_bg} -> 1")
            _scores_blocks.append((hd, T_k))
        if w not in _win_cache:
            # First geometry at this window keeps the bare name "sm_mask" -- same convention as
            # kv_slots's bare "kv_off", and for the same reason (baked into the design, host's
            # pre-list fallback reads that spelling).
            mask_slot = "sm_mask" if not mask_slots else f"sm_mask{len(mask_slots)}"
            mask_slots.append((mask_slot, w))
            win_softmax = Softmax(rows=Hq, cols=w, num_aie_columns=sp.softmax_cols(COLS),
                                  num_channels=1, rtp_vector_size=w,
                                  vector_size_parameter=mask_slot,
                                  segment=softmax_segment(w), context=ctx)
            win_scale = ElementwiseMul(size=Hq * w, tile_size=w // COLS, num_aie_columns=COLS,
                                       context=ctx)
            _win_cache[w] = (win_softmax, win_scale, mask_slot)
        op_softmax, op_scale, mask_slot = _win_cache[w]
        # Both attention reductions read the SAME per-geometry width the softmax already masks
        # with, so a geometry cannot disagree with itself about where its positions end.
        rtp_extent = {"vector_size_parameter": mask_slot} if ATTN_RUNTIME_EXTENT else {}
        dp_why = qkv_dp_why[(hd, hkv, has_v)]
        op_qk_norm = RMSNorm(size=hd, num_aie_columns=1, num_channels=1, tile_size=hd,
                             weighted=True, epsilon=sp.eps, context=ctx) if sp.qk_norm else None
        op_qk_norm_b = (RMSNorm(size=hd, num_aie_columns=1, num_channels=1, tile_size=hd,
                                weighted=True, epsilon=sp.eps, context=ctx)
                        if (sp.qk_norm and SPLIT_QKNORM) else op_qk_norm)
        # QKV projection: one GEMV over the concatenated weight, or the three separate ones. op_kv
        # is built in both arms because share_designs pairs Wk with Wv only in the unfused one.
        #
        # The v_norm raise below is gated on `attn_block_why[g] is not None` (op_qkv_dp will
        # actually reach the runlist) as well as `dp_why is None` (it would otherwise be
        # constructed at all): attn_block_dp replaces op_qkv_dp wholesale for a geometry it
        # covers, so op_qkv_dp is built but dead there -- same "constructed but never reaches the
        # runlist" shape as op_rep_k/op_scores/etc. above, and its own capability gaps must not
        # gate a build that never routes data through it.
        # attention_k_eq_v: a layer with no v_proj concatenates TWO parts, not three, so both the
        # GEMV shape and the qkv buffer layout are per-geometry. V is then derived from k rather
        # than projected -- see the runlist, where v_norm reads the k slice.
        kv_parts = 2 if has_v else 1
        op_qkv = gemv(qd + kv_parts * kvd, D, ctx, **_quant_kw("qkv")) if FUSE_QKV_GEMV else None
        op_q = gemv(qd, D, ctx, **_quant_kw("qkv"))
        op_kv = gemv(kvd, D, ctx, **_quant_kw("qkv"))
        # o_proj, split over K on the same terms as the down projection. Under fuse_o there is no
        # standalone op_o at all -- Wo rides the MLP design's weight channel -- so the split is
        # moot. k_chunks_for reads q_dim, so the chunk COUNT is per-geometry too, and it reaches
        # the weight loop through this namespace rather than as a build-wide constant.
        o_chunks = 1 if fuse_o else (FORCE_O_SPLIT or k_chunks_for(D, qd, COLS))
        op_o = None if fuse_o else gemv(D, qd // o_chunks, ctx, **_quant_kw("attn_o"))
        op_rope_qk = RoPE(rows=Hq + hkv, cols=hd, angle_rows=1, context=ctx) if fuse_rope else None
        op_qkv_dp = None
        if dp_why is None:
            from iron.operators.qkv_head_dp.op import QKVHeadDataParallel
            # No force_header_first: unlike attn_block_dp, this design reads Wq/Wk/Wv as
            # contiguous stock rows (no per-head reorder), so it takes the dump's own layout.
            op_qkv_dp = QKVHeadDataParallel(D=D, HD=hd, Hq=Hq, Hkv=hkv, max_seq=KVA,
                                            num_aie_columns=sp.qkv_dp_cols(COLS, n_kv_heads=hkv),
                                            epsilon=sp.eps,
                                            tile_size_input=TSI, context=ctx,
                                            weight_depth=WEIGHT_DEPTH, **_quant_kw("qkv"))
        # Gemma-4 applies a GAINLESS RMSNorm to the value path of every layer. with_scale=False in
        # the reference removes the learned gain, not the normalisation, so there is no weight
        # tensor anywhere in the checkpoint -- which is why nothing could ever have failed on its
        # absence.
        #
        # It reuses the QK-NORM OBJECT rather than a `weighted=False` one of its own, fed a
        # ones-filled gain. The operator has that axis and using it costs a configure: designs are
        # shared by object IDENTITY (RMSNorm declares no design_key), and the v-norm entries sit
        # immediately before the qk-norm entries in the runlist, so one object makes them ONE
        # contiguous same-design block instead of two. Bit-identical, not approximately: the
        # weighted path is the same gainless normalise followed by a multiply, and bf16 1.0 is an
        # exact multiplicative identity. Costs one 512 B buffer per geometry, shared by every layer.
        op_v_norm = op_qk_norm if sp.v_norm else None
        if sp.v_norm and dp_why is None and attn_block_why[(hd, hkv, has_v)] is not None:
            # Fires only when op_qkv_dp is the arm that actually runs (attn_block_dp declined
            # this geometry). The fused head drains k and v straight into the caches, so `v`
            # never exists as a buffer this graph can normalise -- the norm would be silently
            # skipped rather than rejected, on every layer. Unlike weight_dtype, this gap is
            # real: qkv_head_dp has no value-norm stage (attn_block_dp's design.py grew one,
            # qkv_head_dp/design.py never did) -- confirmed by exercising this raise directly
            # (FUSE_ATTN_BLOCK=0 FUSE_QKV_DP=1 on gemma4-12b, which sets v_norm=True).
            raise NotImplementedError(
                "spec sets v_norm but the fused QKV head is on, and QKVHeadDataParallel appends v "
                "to the cache itself -- the value norm would be silently dropped. Set "
                "FUSE_QKV_DP=0, or give the operator a value-norm stage.")
        op_rope_q = RoPE(rows=Hq, cols=hd, angle_rows=1, context=ctx)
        op_rope_k = RoPE(rows=hkv, cols=hd, angle_rows=1, context=ctx)
        # KV append: deep-C scratchpad offset "kv_off" (element units = n_past*hd), constant ELF.
        # head_dim rides the output STRIDE as well as the sizes, so a second geometry is a second
        # StridedCopy design and not a re-parameterisation of this one.
        #
        # The FIRST geometry keeps the bare name "kv_off": it is baked into the design, and the
        # host's pre-list fallback reads that spelling.
        slot = "kv_off" if not kv_slots else f"kv_off{len(kv_slots)}"
        kv_slots.append((slot, hd))
        # CAPACITY, not window -- the host wraps `pos % capacity` and reads the cache at
        # `kv_block`, and under KV_ALLOC a global geometry's capacity exceeds its window while a
        # circular sliding one's does not. Both are per geometry; emitting `w` here was the same
        # confusion the buffer sizing carried, one layer further out (at the host boundary).
        geom_slots.append((slot, hd, KVA_g, mask_slot, T_g, hkv))
        # The per-head stride comes from the cache's OWN layout, so K and V each write the one
        # they are read through. `KVLayout(S=w, T=w).head_stride` is `w*hd`, the literal this
        # replaces, so every flat geometry is unchanged.
        def append_op(T_blk):
            # Size and stride from the CAPACITY, not the window. kc/vc are
            # allocated at KVA_g (bufsz uses kv_layout.total_elems); only this declaration said
            # `w`, which is why KV_ALLOC>w made one operator declare L0_kc at hkv*w*hd and the
            # scores GEMV declare it at hkv*KVA*hd. At KVA_g == w both spellings coincide.
            kvl = KVLayout(Hkv=hkv, S=KVA_g, HD=hd, T=T_blk)
            return StridedCopy(
                input_sizes=(hkv, hd), input_strides=(hd, 1), input_offset=0,
                output_sizes=(1, hkv, hd), output_offset=0,
                output_strides=(0, kvl.head_stride, 1),
                input_buffer_size=hkv * hd, output_buffer_size=kvl.total_elems,
                num_aie_channels=1, output_offset_parameter=slot, context=ctx)
        op_sck = append_op(T_k)
        # V stays [S][hd]. A transposed append would delete op_trv, but a SINGLE-token transposed
        # write is 1024 isolated bf16 elements (h*hd*S + d*S + p) and the shim address generator
        # steps in 4-byte granules: the BD silently halves the innermost dimension (measured on the
        # emitted descriptor -- d0_size 64 for 128 elements, d0_stride 1023, i.e. 64 granules of two
        # ADJACENT elements). The runtime offset has the same granule floor, so an odd `p` truncates
        # down. The working shape is a PAIR write on an even offset, whose staging cannot itself be
        # a DMA.
        op_scv = append_op(T_g)
        # GQA broadcast. Correctness-first; the byte-free form is a batch-stride-0 GEMV read of the
        # kv head (0 ops, 0 bytes) -- at Hq=16 x 28 layers this Repeat plus the V transpose are 41%
        # of the per-token DDR budget, so it is the first optimisation after parity, not an
        # afterthought.
        #
        # S below is deliberately ONE value shared by kc/vc/kr/vr/vt/sc/sw, op_scores, op_rep_k/v,
        # op_trv AND op_ctx -- not the op_ctx-excluded 4-of-5 split
        # llm-decode-attention-pads-to-full-window.md scoped out device-free. That split needs
        # op_trv to write a bucket-wide `vt` while op_ctx reads it at full max_seq width, and
        # symmetrically op_rep_k/v to read a bucket-wide prefix of a kc/vc row whose true stride is
        # max_seq*hd (hkv=8 here, not a degenerate single-row case where prefix == whole buffer).
        # Neither holds with today's operators: Repeat's input TensorAccessPattern ties its row
        # stride directly to `cols` (repeat/design.py: strides=[0, cols, cols_split, 1]), and
        # Transpose's output stride is tied to its own `M` (transpose/design.py: taps_out_L1L3
        # strides derive from M) -- neither exposes a stride independent of its own declared size,
        # so "read/write a narrower window of a wider-strided buffer" is new IRON capability, not a
        # generator change. Bucketing S UNIFORMLY (this build already takes it as `max_seq`) is the
        # route that needs none.
        # Constructed only when the runlist will use them -- the same predicate its two
        # `*([] if ... else [...])` sites spell. Under KV_ALLOC these size by the WINDOW while
        # every live consumer sizes by the CAPACITY, and a constructed-but-dead op still reaches
        # the arena's argument collection, so it wins the buffer length and the append then
        # declares 32x what the arena provides.
        op_rep_k = (Repeat(rows=hkv, cols=w * hd, repeat=gqa, transfer_size=hd, context=ctx)
                    if not GROUPED_K else None)
        op_rep_v = (Repeat(rows=hkv, cols=w * hd, repeat=gqa, transfer_size=hd, context=ctx)
                    if not (GROUPED_V or tmv_rpc.get(hd) is not None) else None)
        # Built from the SAME (scores_nb, scores_bg, T_k) the block-size gate was answered on --
        # see their derivation above.
        op_scores = gemv(w, hd, ctx, num_batches=scores_nb, batch_group=scores_bg,
                         block_size=T_k, alloc_M=None if KVA_g == w else KVA_g, **rtp_extent)
        # num_batches=Hq, not Hq separate invocations. transpose/design.py's L3 tensors already hold
        # "num_batches contiguous (M,N) matrices stacked along the row dimension", which is EXACTLY
        # what vr/vt are; calling it per head issued 448 configure+run pairs per token to do 28 ops'
        # work.
        #
        # It is NOT a speed fix and must not be quoted as one. Isolated on device against the same
        # placer: -0.01 ms/token, 0.0%. Dropping 420 dispatches per token is worth nothing
        # measurable, because these are mode selections inside ONE hardware context. Kept because it
        # is correct, free, and 2.2 MB smaller in the ELF -- not because it is faster.
        # 4, not COLS: Transpose splits N across columns as `N // num_columns // n`, and at N=hd=128
        # with n=32 that is 4 tiles, so 8 columns divides to ZERO. Its __post_init__ does not catch
        # it -- it checks M*N % (m*n*cols*channels), which 8 satisfies -- and the failure surfaces
        # deep in taplib as "All sizes must be >= 1, but got [8, 0, 256, 32]", naming no operator
        # and no parameter. 4 is the real ceiling at this n; raising it needs n=16.
        # GQA broadcast as an ACCESS PATTERN instead of a materialised copy. gqa query heads attend
        # to one kv head; with batch_group the consumer reads that head directly and the Repeat that
        # duplicated it into DDR disappears. Opt-in per side because k and v are not the same edit
        # (k: Repeat feeds the GEMV; v: Repeat feeds a Transpose that feeds the GEMV), so they must
        # be gated separately to stay attributable in an A/B ladder.
        op_trv = Transpose(M=w, N=hd, num_aie_columns=4, num_channels=1, m=256, n=32, s=8,
                           num_batches=Hq, batch_group=gqa if GROUPED_V else 1, context=ctx)
        # CONTEXT STEP. op_trv exists only because gemv reduces ALONG a row: it wants [hd][S] and
        # the cache is [S][hd], so the whole cache is rearranged every token -- 16.777 MB/layer
        # measured, at 0% compute. TMatVec reduces DOWN the rows instead and reads `vc` as it is
        # stored, so the transpose has nothing left to do. One kv head per column
        # (n_matrices == cols == hkv), so each column streams its own head ONCE and applies both
        # query heads' softmax rows out of L1 -- the stride-0 group re-read goes too.
        # rows_per_chunk=64 puts the L1 A tile at 16 KB double-buffered.
        # The verdict and its rows_per_chunk come from `tmv_rpc`, derived once above -- not
        # recomputed here. A second copy of the L1 model is exactly how the two would drift.
        _tmv = tmv_rpc.get(hd)
        uses_tmv = _tmv is not None
        if uses_tmv:
            rpc, mc = _tmv
            op_ctx = TMatVec(M=hd, K=w, num_aie_columns=hkv, num_batches=Hq, batch_group=gqa,
                                 alloc_K=None if KVA_g == w else KVA_g, block_size=T_g,
                             rows_per_chunk=rpc, m_chunk=mc, context=ctx, **rtp_extent)
        else:
            # The fallback reduces along K, and GEMV's runtime extent is its M. Left at the
            # build-time window: narrowing it needs TMV_CTX, which this geometry declined.
            op_ctx = gemv(hd, w, ctx, num_batches=Hq)
        # attn_block_dp AS A WHOLE, for this geometry alone -- one memoized object per (hd, hkv,
        # has_v), same trap as every other op above: a fresh object per LAYER would defeat
        # unique_designs' id()-keyed collapse and build one configure per layer instead of one per
        # geometry. kv_offset_parameter/mask_parameter must match this geometry's own slot names
        # (`slot`/`mask_slot`, assigned above), not the operator's single-geometry defaults, or the
        # host writes a scratchpad parameter this design never reads.
        op_attn_block = None
        if attn_block_why[(hd, hkv, has_v)] is None:
            from iron.operators.attn_block_dp.op import AttnBlockDataParallel
            op_attn_block = AttnBlockDataParallel(
                D=D, HD=hd, Hq=Hq, Hkv=hkv, max_seq=w, num_aie_columns=hkv, epsilon=sp.eps,
                tile_size_input=TSI, context=ctx, weight_depth=WEIGHT_DEPTH,
                wqkv_head_major=True, kv_offset_parameter=slot, mask_parameter=mask_slot,
                kv_alloc=None if KVA_g == w else KVA_g, kv_block_size=None if T_g == w else T_g,
                v_norm=sp.v_norm, **_quant_kw("qkv", force_header_first=True))
        g = SimpleNamespace(
            hd=hd, hkv=hkv, qd=qd, kvd=kvd, gqa=gqa, kv_slot=slot, o_chunks=o_chunks,
            op_qk_norm=op_qk_norm, op_qk_norm_b=op_qk_norm_b, op_qkv=op_qkv, op_q=op_q,
            op_kv=op_kv, op_o=op_o, op_rope_qk=op_rope_qk, op_qkv_dp=op_qkv_dp,
            op_rope_q=op_rope_q, op_rope_k=op_rope_k, op_sck=op_sck, op_scv=op_scv,
            op_rep_k=op_rep_k, op_rep_v=op_rep_v, op_scores=op_scores, op_trv=op_trv,
            op_ctx=op_ctx, op_v_norm=op_v_norm, has_v=has_v, kv_parts=kv_parts,
            uses_tmv_ctx=uses_tmv, scores_groups=scores_groups, scores_block=T_k,
            op_softmax=op_softmax, op_scale=op_scale, mask_slot=mask_slot, window=w,
            capacity=KVA_g, kv_block=T_g,
            op_attn_block=op_attn_block, circular=(w != S))
        _attn_cache[(hd, hkv, has_v)] = g
        return g

    # Built up front, in layer order, rather than lazily from the loop: construction order is then
    # what it was before this was a factory, and a geometry that cannot be built fails here instead
    # of 20 layers into the runlist.
    for gk in geoms:
        attn_ops(*gk)
    # Folded into n_qn when SCALE_IN_QNORM; built anyway so the A/B arm stays reachable.
    scale_in_qnorm = SCALE_IN_QNORM and sp.qk_norm
    op_rep_k = op_rep_v = op_scores = op_scale = op_softmax = op_trv = op_ctx = None
    if decode_layer_why is not None:
        op_rep_k = Repeat(rows=Hkv, cols=S * HD, repeat=sp.gqa_group, transfer_size=HD, context=ctx)
        op_rep_v = Repeat(rows=Hkv, cols=S * HD, repeat=sp.gqa_group, transfer_size=HD, context=ctx)
        _rtp_extent = {"vector_size_parameter": "sm_mask"} if ATTN_RUNTIME_EXTENT else {}
        op_scores = gemv(S, HD, ctx, num_batches=Hq,
                         batch_group=sp.gqa_group if GROUPED_K else 1, block_size=T,
                         alloc_M=None if KVA == S else KVA, **_rtp_extent)
        # Folded into n_qn when SCALE_IN_QNORM; built anyway so the A/B arm stays reachable.
        op_scale = (None if scale_in_qnorm else
                    ElementwiseMul(size=Hq * S, tile_size=S // COLS, num_aie_columns=COLS,
                                   context=ctx))
        # Not COLS: with fewer q heads than columns each core gets less than one tile and the
        # op computes nothing (IRON raises). Gemma-3's 4 heads run at 4 columns.
        op_softmax = Softmax(rows=Hq, cols=S, num_aie_columns=sp.softmax_cols(COLS),
                             num_channels=1, rtp_vector_size=S,
                             vector_size_parameter="sm_mask",
                             segment=softmax_segment(S), context=ctx)
        # num_batches=Hq, not Hq separate invocations. transpose/design.py's L3 tensors already hold
        # "num_batches contiguous (M,N) matrices stacked along the row dimension", which is EXACTLY what
        # vr/vt are; calling it per head issued 448 configure+run pairs per token to do 28 ops' work.
        #
        # It is NOT a speed fix and must not be quoted as one. Isolated on device against the same
        # placer: -0.01 ms/token, 0.0%. Dropping 420 dispatches per token is worth nothing measurable,
        # because these are mode selections inside ONE hardware context. Kept because it is correct,
        # free, and 2.2 MB smaller in the ELF -- not because it is faster.
        # 4, not COLS: Transpose splits N across columns as `N // num_columns // n`, and at N=HD=128
        # with n=32 that is 4 tiles, so 8 columns divides to ZERO. Its __post_init__ does not catch it
        # -- it checks M*N % (m*n*cols*channels), which 8 satisfies -- and the failure surfaces deep in
        # taplib as "All sizes must be >= 1, but got [8, 0, 256, 32]", naming no operator and no
        # parameter. 4 is the real ceiling at this n; raising it needs n=16.
        # GQA broadcast as an ACCESS PATTERN instead of a materialised copy. gqa_group query heads
        # attend to one kv head; with batch_group the consumer reads that head directly and the Repeat
        # that duplicated it into DDR disappears. Opt-in per side because k and v are not the same edit
        # (k: Repeat feeds the GEMV; v: Repeat feeds a Transpose that feeds the GEMV), so they must be
        # gated separately to stay attributable in an A/B ladder.
        op_trv = Transpose(M=S, N=HD, num_aie_columns=4, num_channels=1, m=256, n=32, s=8,
                           num_batches=Hq, batch_group=sp.gqa_group if GROUPED_V else 1, context=ctx)
        # CONTEXT STEP. op_trv exists only because gemv reduces ALONG a row: it wants [HD][S] and the
        # cache is [S][HD], so the whole cache is rearranged every token -- 16.777 MB/layer measured, at
        # 0% compute. TMatVec reduces DOWN the rows instead and reads `vc` as it is stored, so the
        # transpose has nothing left to do. One kv head per column (n_matrices == cols == Hkv), so each
        # column streams its own head ONCE and applies both query heads' softmax rows out of L1 -- the
        # stride-0 group re-read goes too. rows_per_chunk=64 puts the L1 A tile at 16 KB double-buffered.
        _tmv = tmv_rpc.get(HD) if TMV_CTX else None
        if _tmv is not None:
            # tmv_rpc is the ONE OWNER of this verdict and it already ran the two-dimensional
            # (rows_per_chunk, m_chunk) search per geometry. Re-deriving it here with an
            # rows_per_chunk-only loop is the duplicate that dict's own comment warns about: it
            # cannot reach m_chunk, so it halved to 1 and constructed anyway, and TMatVec threw
            # `W (batch_group*K) 131072 B ... shrinking rows_per_chunk cannot help` at S=32768
            # while BOTH geometries had already fitted above at m_chunk=256.
            rpc, mc = _tmv
            op_ctx = TMatVec(M=HD, K=S, num_aie_columns=Hkv, num_batches=Hq,
                             batch_group=sp.gqa_group, alloc_K=None if KVA == S else KVA,
                             rows_per_chunk=rpc, m_chunk=mc, context=ctx, block_size=T,
                             **_rtp_extent)
        else:
            op_ctx = gemv(HD, S, ctx, num_batches=Hq)
    # MLP weight-stream dtype axis (Wg/Wu/Wd -- "MLP weights" in the byte breakdown, the largest
    # single weight class). bf16 (default) is byte-for-byte the pre-existing path; QUANT_MLP_DTYPE
    # is an engineering-check toggle (see its definition above), not a quality-validated default.
    mlp_quant_kw = _quant_kw("mlp")
    # The activation runs on the gate projection's output, immediately after it and before anything
    # else reads `g`, so folding it into that GEMV's epilogue preserves the order exactly.
    # Two things can veto the fold, and both are the operator's own rules rather than choices here:
    # the epilogue walks the C tile 32 lanes at a time, and GEMV refuses an epilogue on a quantized
    # weight stream (untested combination, not a hardware conflict). gemv() picks tile_size_output
    # itself, so ask it rather than assuming FF // COLS.
    _gate_tso = gemv_tile_output(FF, D, cols=COLS)[1]
    fuse_act = (
        FUSE_ACT
        and not _spec("mlp").quantized
        and _gate_tso % 32 == 0
    )
    op_gate = gemv(FF, D, ctx, **mlp_quant_kw,
                   **(dict(epilogue=sp.act) if fuse_act else {}))
    op_up = gemv(FF, D, ctx, **mlp_quant_kw)
    op_act = None
    op_mlp_dp = None
    if mlp_dp_why is None:
        from iron.operators.swiglu_mlp_dp.op import SwiGLUMLPDataParallel
        # act/post_norm carry the two clauses mlp_dp_reason() used to refuse on. Both default to
        # silu/False, so a spec that never needed them (qwen3) builds the identical design.
        op_mlp_dp = SwiGLUMLPDataParallel(D=D, FF=FF, num_aie_columns=MLP_DP_COLS,
                                          epsilon=sp.eps,
                                          QD=QD if fuse_o else None, fuse_o=fuse_o,
                                          act=sp.act, post_norm=sp.sandwich_norms,
                                          context=ctx, weight_depth=WEIGHT_DEPTH,
                                          tile_rows_gu=MLP_TILE_ROWS,
                                          **(dict(row_parallel_down=True, d_chunks=MLP_D_CHUNKS)
                                             if MLP_ROW_PARALLEL else {}),
                                          **mlp_quant_kw)
    # The whole decoder layer (attention + MLP) as ONE fused device -- see FUSE_DECODE_LAYER above.
    op_decode_layer = None
    rung_ops = {}
    if decode_layer_why is None:
        from iron.operators.decode_layer_dp.op import DecodeLayerDataParallel

        def _decode_layer(window):
            return DecodeLayerDataParallel(
                D=D, FF=FF, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=window, attn_cols=Hkv,
                mlp_cols=MLP_DP_COLS,
                eps_attn=sp.eps, eps_mlp=sp.eps, tile_size_input=TSI, context=ctx,
                weight_depth=WEIGHT_DEPTH, wqkv_head_major=True,
                # max_seq stays the WINDOW the attention math iterates; these two carry the
                # capacity and the blocked storage, the same split gemv/tmatvec already have. Both
                # None on the unwidened, unblocked default, which is byte-identical to before they
                # existed. Compared against THIS op's own window, not against the top one: a rung
                # is precisely the case where capacity and window differ, and comparing to `S`
                # would hand a rung `kv_alloc=None` and silently shrink its cache to its window.
                kv_alloc=None if KVA == window else KVA,
                kv_block_size=None if T == S else T,
                # Passed as a kwarg ONLY when the flag is on. Handing it through unconditionally --
                # even as None -- is a TypeError against any IRON whose decode_layer_dp predates
                # the field, and the default IRON_DIR (wt-iron-integ) is exactly that. Measured
                # 2026-09-10: it broke every decode build on the default path, DYNAMIC_WINDOW=0
                # included, because an unknown kwarg fails at the call and never reaches the flag
                # test inside.
                # Conditional for the SAME reason window_parameter is: an unknown kwarg is a
                # TypeError at the call against any IRON whose decode_layer_dp predates the field,
                # and it never reaches the flag test inside.
                **({"split_gh": SPLIT_GH_DRAIN} if SPLIT_GH_DRAIN != 1 else {}),
                **({"attn_split": ATTN_SPLIT} if ATTN_SPLIT else {}),
                **({"window_parameter": "attn_window"} if DYNAMIC_WINDOW else {}),
                # Same discipline again. The MLP half's four weights (Wo, Wg, Wu, Wd) share one
                # fifo and one format, which P002 has already enforced.
                **_quant_kw("mlp"))

        op_decode_layer = _decode_layer(S)
        # A rung is the SAME design at a narrower window over the SAME capacity, so `kv_alloc`
        # doing the capacity/window split is what makes the rungs share one arena byte for byte:
        # `kc`/`vc` are sized from KVA, not from the window. Rejected loudly rather than clamped --
        # a rung wider than the top window would stream MORE than the design it is meant to
        # undercut, and a duplicate would silently emit two control codes doing the same thing.
        for _w in WINDOW_RUNGS:
            if _w >= S:
                raise SystemExit(f"WINDOW_RUNGS: rung {_w} is not narrower than max_seq {S}")
            if _w in rung_ops:
                raise SystemExit(f"WINDOW_RUNGS: rung {_w} listed twice")
            rung_ops[_w] = _decode_layer(_w)
    elif WINDOW_RUNGS:
        raise SystemExit(
            f"WINDOW_RUNGS={','.join(map(str, WINDOW_RUNGS))} needs decode_layer_dp, which is "
            f"OFF here: {decode_layer_why}"
        )
    print(f"[gen] fused arm decode_layer_dp: "
          f"{'OFF -- ' + decode_layer_why if decode_layer_why else 'on'}")

    # Passed to sequence_name() as computed here, so the suffix and the graph share one predicate.
    ff_chunks = FF_POINTWISE_CHUNKS if op_mlp_dp is None and op_decode_layer is None else 1
    if ff_chunks > 1 and (FF % ff_chunks or (FF // ff_chunks) % COLS):
        raise SystemExit(f"FF_POINTWISE_CHUNKS={ff_chunks}: FF={FF} must split into slices that "
                         f"are whole multiples of COLS={COLS}")
    ffw = FF // ff_chunks
    if not fuse_act:
        if sp.act == "silu":
            op_act = SiLU(size=ffw, num_aie_columns=COLS, tile_size=ffw // COLS, context=ctx)
        else:
            op_act = GELU(size=ffw, num_aie_columns=COLS, num_channels=1, tile_size=ffw // COLS, context=ctx)
    op_mul_ffn = ElementwiseMul(size=ffw, tile_size=ffw // COLS, num_aie_columns=COLS, context=ctx)
    # Down projection, split over K when it does not fit L1 (or when FORCE_K_SPLIT prices it).
    # One GEMV per chunk at K=FF/n plus n-1 adds; the chunk GEMVs are the SAME op object, so the
    # split costs one design and n runs, not n designs.
    down_chunks = FORCE_K_SPLIT or k_chunks_for(D, FF, COLS)
    if down_chunks > 1:
        assert FF % down_chunks == 0, f"FF={FF} not divisible by {down_chunks} K chunks"
        op_down = gemv(D, FF // down_chunks, ctx, **mlp_quant_kw)
    else:
        op_down = gemv(D, FF, ctx, **mlp_quant_kw)
    op_add = ElementwiseAdd(size=D, tile_size=D // COLS, num_aie_columns=COLS, context=ctx)
    # Gemma-4's trained per-layer scalar, applied to the block output after BOTH residual adds.
    # It cannot fold anywhere: it scales the residual stream itself, so the next layer's norm sees
    # it and every later layer compounds it. One D-wide multiply per layer is the honest form.
    op_lscale = ((op_mul_ffn if ff_chunks > 1 and ffw == D else
                  ElementwiseMul(size=D, tile_size=D // COLS, num_aie_columns=COLS, context=ctx))
                 if sp.layer_scalar else None)

    def ff_slices(buf):
        if ff_chunks == 1:
            return [buf]
        return [f"{buf}[{i * ffw * 2}:{(i + 1) * ffw * 2}]" for i in range(ff_chunks)]

    def split_over_k(op, w, xin, out, n, k_elems, tag):
        """A reduction over K as one GEMV, or as `n` partial GEMVs plus a summation tree.

        Chunk i reads `xin[i*CW:(i+1)*CW]` -- a BYTE slice of the buffer that already holds the
        input, so the split moves no data and adds no buffer on the input side. Partials fold
        PAIRWISE, not linearly: mv.cc rounds its f32 accumulator to bf16 once per partial, so a
        pairwise fold makes the rounding depth log2(n) instead of n-1.

        Used by BOTH the down projection and o_proj. One helper rather than two, because the
        weight-naming it implies (`<w>k<i>`) has to match what the weight loop writes, and two
        copies of a naming scheme is the seam this file has already paid for.
        """
        if n == 1:
            return [(op, w, xin, out)]
        cw = (k_elems // n) * 2                          # chunk width in BYTES, the slice unit
        steps = [(op, f"{w}k{i}", f"{xin}[{i * cw}:{(i + 1) * cw}]", f"{tag}p{i}")
                 for i in range(n)]
        level, r = [f"{tag}p{i}" for i in range(n)], 0
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level) - 1, 2):
                # the final add writes `out`, so everything downstream is untouched by the split
                o = out if len(level) == 2 else f"{tag}s{r}_{i // 2}"
                steps.append((op_add, level[i], level[i + 1], o))
                nxt.append(o)
            if len(level) % 2:
                nxt.append(level[-1])
            level, r = nxt, r + 1
        return steps

    def down_runlist(p):
        return split_over_k(op_down, p + "Wd", p + "gh", p + "d", down_chunks, FF, p + "d")

    def o_runlist(p, g):
        return split_over_k(g.op_o, p + "Wo", p + "cx", p + "a", g.o_chunks, g.qd, p + "a")

    def scores_runlist(p, g, ref_q):
        """(g.op_scores, K-buffer, q-slice, sc-slice) tuples.

        One call, byte-identical to the pre-fix single call, when g.scores_groups==1 (every
        geometry whose gqa fits MAX_GROUP_REUSE). Otherwise g.scores_groups calls of the SAME op
        -- built at batch_group=MAX_GROUP_REUSE in attn_ops -- each over a MAX_GROUP_REUSE-head
        slice of q and the matching slice of sc, all reading the one real K matrix at its
        unsliced offset. `ref_q` may itself already be a byte slice of a wider qkv buffer
        (FUSE_QKV_GEMV); Q always starts at byte 0 of whichever buffer it names, so slicing off
        that base reaches the same bytes a further bracket on `ref_q` would.
        """
        a = p + ("kc" if GROUPED_K else "kr")
        if g.scores_groups == 1:
            return [(g.op_scores, a, ref_q, p + "sc")]
        base, n = ref_q.split("[", 1)[0], MAX_GROUP_REUSE
        return [
            (g.op_scores, a, f"{base}[{i*n*g.hd*2}:{(i+1)*n*g.hd*2}]",
             f"{p}sc[{i*n*g.window*2}:{(i+1)*n*g.window*2}]")
            for i in range(g.scores_groups)
        ]
    # W_head weight-stream dtype axis (see QUANT_HEAD_DTYPE above -- READ THE TIED-EMBEDDING NOTE
    # before turning this on).
    head_quant_kw = _quant_kw("head")
    op_head = gemv(VOCAB, D, ctx, **head_quant_kw)

    weights, bufsz, cache_names, rl = {}, {}, [], []
    if sp.v_norm:
        # The gainless v-norm's gain, one per head_dim and shared by EVERY layer -- a true constant,
        # unlike the per-layer learned gains beside it, so it is registered once here rather than in
        # the layer loop.
        # Only the geometries the UNFUSED arm carries: attn_block_dp holds the gain as a
        # compile-time L1 constant, so a fused geometry has no L3 buffer here to look up.
        for _hd in sorted({gk[0] for gk in geoms if attn_block_why[gk] is not None}):
            weights[f"ones_h{_hd}"] = np.ones(_hd, dtype=BF16)

    def _dequant_wd_from_kchunks(hf):
        """Wd for the PLAIN fused path (gh_chunks=1, row_parallel_down=False -- swiglu_mlp_dp
        wants it as ONE buffer, same as Wg/Wu). The dump only ever writes a kchunked Wd
        (down_chunks-wide, for op_down's unfused GEMV), so reconstruct the flat [D, FF] float
        matrix the same way _pack_wd_row_parallel does for its own re-chunked shape, but hand it
        back unchunked for the caller's existing `_pack(w, "mlp")`. See
        fused-operators-cannot-read-our-quantized-weight-layout."""
        chunk_hf = [f"{hf}.kchunk{i}" for i in range(down_chunks)]
        if not all(n in PACKED for n in chunk_hf):
            raise SystemExit(f"{hf}: {down_chunks}-way kchunk dump is partial (need all of "
                              f"{chunk_hf})")
        if dequantize_weight_chunked is None:
            raise SystemExit("the fused swiglu_mlp_dp path needs "
                              "iron.common.quant.dequantize_weight_chunked (post-6a347dc IRON "
                              "tree) to read a kchunked Wd dump")
        packed = np.concatenate([np.asarray(npy_raw(n)) for n in chunk_hf])
        spec = _spec("mlp")
        # layout/row_group default to header_first/ROW_GROUP_DEFAULT inside dequantize_weight,
        # which silently misreads a row_group_planar dump's [payload][scales] arrangement as
        # interleaved [header][payload] -- caught by the RuntimeWarnings (invalid value in
        # divide/multiply/cast) a first version of this function produced against the shipped
        # int4g32 planar dump before this was added. op_mlp_dp already derived the real values
        # (its own __post_init__), so ask it rather than re-deriving them here.
        return dequantize_weight_chunked(packed, D, FF, spec.group_size, spec.dtype,
                                         n_chunks=down_chunks,
                                         scale_dtype=_BUILD_STATE["scale_dtype"],
                                         layout=op_mlp_dp.layout, row_group=op_mlp_dp.row_group)

    def _pack_wd_row_parallel(hf, n_chunks):
        """row_parallel_down's Wd: ONE buffer of `n_chunks` independently-quantized column-shard
        blocks (design.py's ROW_PARALLEL_DOWN). This dump has no unchunked Wd to slice -- only
        the unfused path's `down_chunks`-wide kchunks -- so reconstruct the float weight from
        those and re-chunk at n_chunks. See fused-operators-cannot-read-our-quantized-weight-layout.
        """
        chunk_hf = [f"{hf}.kchunk{i}" for i in range(down_chunks)]
        if not all(n in PACKED for n in chunk_hf):
            raise SystemExit(f"{hf}: no {down_chunks}-way kchunk dump to rebuild a row-parallel "
                              f"Wd from (row_parallel_down needs a header_first int4/int8 dump)")
        if dequantize_weight_chunked is None:
            raise SystemExit("row_parallel_down needs iron.common.quant.dequantize_weight_chunked "
                              "(post-6a347dc IRON tree)")
        packed = np.concatenate([np.asarray(npy_raw(n)) for n in chunk_hf])
        spec = _spec("mlp")
        w = dequantize_weight_chunked(packed, D, FF, spec.group_size, spec.dtype,
                                      n_chunks=down_chunks,
                                      scale_dtype=_BUILD_STATE["scale_dtype"])
        return np.concatenate([_pack(np.ascontiguousarray(part), "mlp")
                               for part in np.split(w, n_chunks, axis=1)])

    cur = "x"
    # (runlist index, residual buffer entering this layer) per layer, so the stack can be cut into
    # segments AFTER it is built. Recorded rather than reconstructed: the residual chain is
    # x -> x1 -> ... -> xNL and a cut is only sound at a layer boundary, which is exactly here.
    layer_marks = []

    for l in range(NL):
        layer_marks.append((len(rl), cur))
        # The layer's attention geometry. Uniform for every shipped spec, so this is the same
        # namespace object every iteration and the ops are shared exactly as they were when they
        # were module-level; Gemma-4-12B is where it starts returning two.
        g = attn_ops(sp.head_dim_for(l), sp.n_kv_heads_for(l), sp.has_v_proj(l))
        p = f"L{l}_"
        nm = sp.norm_weight_names(l)
        for key, tensor in nm.items():
            w = load_norm(tensor)
            if key == "n_qn" and scale_in_qnorm:
                # scores = (K @ RoPE(rms(q_raw) * n_qn)) * attn_scale, and both RoPE and the
                # scores GEMV are linear in q -- so the constant rides on the gain instead of on
                # a whole [Hq, S] elementwise pass. Scaled in f32 before the bf16 round.
                w = bf16(np.asarray(w, np.float32) * sp.attn_scale)
            weights[p + key] = w
        mlp_keys = {"Wg", "Wu", "Wd"}
        qkv_keys = ("Wq", "Wk", "Wv")
        qkv_parts = []   # filled in Wq, Wk, Wv order below -- the row order op_qkv assumes
        for key, tensor in (("Wq", "self_attn.q_proj"), ("Wk", "self_attn.k_proj"),
                            ("Wv", "self_attn.v_proj"), ("Wo", "self_attn.o_proj"),
                            ("Wg", "mlp.gate_proj"), ("Wu", "mlp.up_proj"), ("Wd", "mlp.down_proj")):
            if key == "Wv" and not g.has_v:
                continue     # attention_k_eq_v: no v_proj tensor exists for this layer
            hf = f"{sp.weight_prefix}layers.{l}.{tensor}.weight"
            if key == "Wd" and mlp_dp_why is None and MLP_ROW_PARALLEL:
                # swiglu_mlp_dp's row_parallel_down path owns its own K-split (the hardware
                # cascade, not down_chunks) and wants ONE buffer of MLP_D_CHUNKS column-shard
                # blocks -- see _pack_wd_row_parallel.
                weights[p + key] = _pack_wd_row_parallel(hf, MLP_D_CHUNKS)
                continue
            # down_chunks only applies to op_down's unfused GEMV (mlp_dp_why is not None); the
            # plain fused swiglu_mlp_dp wants Wd as ONE buffer like Wg/Wu, below.
            _wd_chunks = down_chunks if (key == "Wd" and mlp_dp_why is not None) else 1
            if {"Wd": _wd_chunks, "Wo": g.o_chunks}.get(key, 1) > 1:
                # Each chunk is its own contiguous tensor. A pre-chunked dump names them
                # `<tensor>.kchunkN` and we take those bytes as-is; otherwise the split happens
                # here, along K, BEFORE quantizing -- so each chunk carries its own per-group
                # scales, exactly as the kernel reads it. Splitting AFTER packing would cut
                # through a group.
                nch = {"Wd": _wd_chunks, "Wo": g.o_chunks}[key]
                chunk_hf = [f"{hf}.kchunk{i}" for i in range(nch)]
                if all(n in PACKED for n in chunk_hf):
                    for i, n in enumerate(chunk_hf):
                        weights[f"{p}{key}k{i}"] = np.asarray(npy_raw(n))
                else:
                    wd = npy(hf)
                    for i, part in enumerate(np.split(wd, nch, axis=1)):
                        part = np.ascontiguousarray(part)
                        weights[f"{p}{key}k{i}"] = (
                            _pack(part, _site_of(key)))
                continue
            if hf in PACKED:
                # Already on the wire; npy_raw, never npy -- widening these bytes to f32 renumbers
                # the payload instead of copying it, and does so silently.
                if key in qkv_keys and not _spec("qkv").quantized:
                    raise SystemExit(
                        f"{hf} is packed but the precision plan leaves qkv at bf16, so op_qkv/op_q/op_kv would "
                        f"consume the packed bytes as bf16 values. This should be unreachable when "
                        f"the dump's quant.json set the axis; re-dump with --quant-leaves excluding "
                        f"q_proj,k_proj,v_proj if that is what you meant.")
                wp = np.asarray(npy_raw(hf))
                if key in qkv_keys and FUSE_QKV_GEMV:
                    qkv_parts.append(wp)
                    continue
                if key == "Wo" and fuse_o:
                    # The pad cannot be done in the float domain here -- there is no float domain
                    # left. Zero rows are built directly in the wire format instead; see
                    # packed_zero_rows for why a zero row is exactly [f32(1.0) x n_groups][zeros].
                    wp = np.concatenate([wp, packed_zero_rows(
                        op_mlp_dp._wo_rows_padded - D, g.qd,
                        _spec("mlp").group_size, _spec("mlp").dtype)])
                weights[p + key] = wp
                continue
            w = (_dequant_wd_from_kchunks(hf)
                 if key == "Wd" and mlp_dp_why is None and f"{hf}.kchunk0" in PACKED
                 else npy(hf))  # [M, K], f32
            if key in mlp_keys:
                weights[p + key] = _pack(w, "mlp")
            elif key == "Wo" and fuse_o:
                # Pad FIRST, then quantize: the pad rows must be a whole number of groups in the
                # same wire format as the rest of the channel. Zero rows quantize to amax=0 ->
                # scale 1.0, q=0, so their contribution stays exactly zero.
                # swiglu_mlp_dp's fuse_o tiles Wo's D output rows in TSI_O=3-row groups shared
                # byte-identically with Wg/Wu/Wd's weight channel; D/MLP_DP_COLS is never a
                # multiple of 3 (D is a power of two), so every core reads one row PAST its own
                # slice and the last core's read would run off the end of Wo -- padded here with
                # `_wo_rows_padded - D` zero rows so that read stays in bounds. Their computed
                # contribution is exactly zero and is never drained (see design.py's FUSE_O
                # module docstring for the full derivation).
                pad_rows = op_mlp_dp._wo_rows_padded - D
                w_padded = np.pad(w, ((0, pad_rows), (0, 0)))
                weights[p + key] = _pack(w_padded, "mlp")
            elif key == "Wo" and _spec("attn_o").quantized:
                # AFTER the fuse_o branch, not before it: fused Wo needs the pad, and testing the
                # dtype first would send a quantized+fused Wo down the unpadded path.
                weights[p + key] = _pack(w, "attn_o")
            elif key in qkv_keys and FUSE_QKV_GEMV:
                qkv_parts.append(_pack(w, "qkv"))         # row-major, so concatenation IS stacking
            elif key in qkv_keys and _spec("qkv").quantized:
                weights[p + key] = _pack(w, "qkv")
            else:
                weights[p + key] = bf16(w).reshape(-1)
        if FUSE_QKV_GEMV:
            want = 1 + g.kv_parts
            assert len(qkv_parts) == want, (
                f"L{l}: expected {want} concat parts ({'Wq, Wk, Wv' if g.has_v else 'Wq, Wk'}); "
                f"got {len(qkv_parts)}")
            if ((op_decode_layer is not None and op_decode_layer.wqkv_head_major
                 or g.op_attn_block is not None and g.op_attn_block.wqkv_head_major)
                    and g.has_v):
                # attn_block_dp wqkv_head_major: one contiguous run of (gqa+2) hd-row
                # blocks per core. Per GEOMETRY -- gqa and the row height are g.hd/g.hkv,
                # not the spec-wide pair, or the reorder shuffles row fragments.
                gqa_ = Hq // g.hkv
                row_w = precision.wire_row_units(_spec("qkv"), D,
                                                 _BUILD_STATE["scale_dtype"])
                # This reorder slices individual output-feature ROWS out of Wq/Wk/Wv at a fixed
                # row_w stride, which only addresses header_first bytes correctly -- under
                # row_group_planar a row's header sits ROW_GROUP*payload away from its own payload
                # (iron/common/quant.py::row_offsets), so the slice below would cut across block
                # boundaries. Un-planarize first (a pure byte permutation, no requantization) so
                # every consumer of this reorder gets header_first rows regardless of the dump's
                # own declared layout -- see attn-block-quantized-wqkv-assumes-header-first.
                if _BUILD_STATE["layout"] == "row_group_planar":
                    if _planar_to_rows is None:
                        raise ModuleNotFoundError(
                            "dump is row_group_planar but this IRON tree predates "
                            "_planar_to_rows (pre-6a347dc)")
                    _qspec = _spec("qkv")
                    _rg = derive_row_group([D], _qspec.group_size, _qspec.dtype,
                                           vec_size=widest_chunk(_qspec.group_size, _qspec.dtype),
                                           scale_dtype=_BUILD_STATE["scale_dtype"])
                    # This branch is gated on g.has_v (above), so qkv_parts is always [Wq, Wk, Wv].
                    _sizes = (Hq * g.hd, g.hkv * g.hd, g.hkv * g.hd)
                    # _planar_to_rows returns np.uint8 (quant.py's own internal byte-view
                    # convention); every OTHER packer exit point normalizes to np.int8 before
                    # returning, which is the dtype weight_bytes() tests for "already packed" --
                    # left as uint8, it fails that test and gets bf16-cast, doubling the size and
                    # renumbering the bytes as floats. view(), not astype(): same bits, signed label.
                    qkv_parts = [
                        _planar_to_rows(a, m, D, row_w, _qspec.dtype, _rg).view(np.int8).reshape(-1)
                        for a, m in zip(qkv_parts, _sizes)
                    ]
                wq2, wk2, wv2 = (a.reshape(-1, row_w) for a in qkv_parts)
                parts = []
                for c in range(g.hkv):
                    parts += [wq2[(gqa_ * c + gi) * g.hd:(gqa_ * c + gi + 1) * g.hd]
                              for gi in range(gqa_)]
                    parts += [wk2[c * g.hd:(c + 1) * g.hd], wv2[c * g.hd:(c + 1) * g.hd]]
                weights[p + "Wqkv"] = _stream_pad_rows(
                    np.concatenate(parts, axis=0), g.op_attn_block).reshape(-1)
            else:
                weights[p + "Wqkv"] = np.concatenate(qkv_parts)
        # Size is layout-independent (T < S rearranges the same elements) but PER GEOMETRY, and
        # the axis is the CAPACITY, not the window: Gemma-4's global layers are hkv=1/hd=512
        # against sliding 8/256, and under KV_ALLOC a global geometry's capacity exceeds its
        # window while a CIRCULAR sliding one's does not. Both come from the geometry itself --
        # `g.capacity`/`g.kv_block` are the values its operators were built with, not a second
        # derivation of them, which is how this line used to size a global cache at `g.window`
        # and hand the append a buffer a quarter the size it declares.
        _kvl = KVLayout(Hkv=g.hkv, S=g.capacity, HD=g.hd, T=g.kv_block)
        weights[p + "kc"] = np.zeros(_kvl.total_elems, BF16)
        weights[p + "vc"] = np.zeros(_kvl.total_elems, BF16)
        cache_names += [p + "kc", p + "vc"]
        ang = "rope_global" if sp.is_global(l) else "rope_local"

        # q/k/v are byte slices of ONE `qkv` buffer in the fused arm -- op_qkv writes all three in
        # one pass, and q|k adjacency is what lets a single RoPE cover both. Declared with an
        # explicit size because a parent that is only ever referenced sliced has no arg spec to
        # take its length from (iron/common/sequence.py: calculate_buffer_layout).
        if g.op_qkv_dp is not None:
            # The fused head appends k and v to the caches itself, so neither ever becomes an L3
            # buffer and only `q` survives as an intermediate.
            ref_q = p + "q"
            bufsz[ref_q] = g.qd * 2
        elif FUSE_QKV_GEMV:
            qkvb, kb, vb = p + "qkv", g.qd * 2, (g.qd + g.kvd) * 2
            ref_q, ref_k = f"{qkvb}[0:{kb}]", f"{qkvb}[{kb}:{vb}]"
            ref_qk = f"{qkvb}[0:{vb}]"
            if g.has_v:
                ref_v = f"{qkvb}[{vb}:{vb + g.kvd * 2}]"
                vhb, vho = qkvb, vb
            else:
                # No v_proj: the concatenation ends at k, and `v` is its own buffer that v_norm
                # writes from the k slice. It cannot alias the k slice -- k is normed and rotated
                # in place afterwards, and V must be the RAW projection.
                ref_v, vhb, vho = p + "v", p + "v", 0
                bufsz[p + "v"] = g.kvd * 2
            # per-head norm slice base + byte offset, for q, k and v alike
            qhb, qho, khb, kho = qkvb, 0, qkvb, kb
            bufsz[qkvb] = (g.qd + g.kv_parts * g.kvd) * 2
        else:
            ref_q, ref_k, ref_v = p + "q", p + "k", p + "v"
            qhb, qho, khb, kho, vhb, vho = p + "q", 0, p + "k", 0, p + "v", 0
            bufsz.update({p + "q": g.qd * 2, p + "k": g.kvd * 2, p + "v": g.kvd * 2})
        # kr/vr/vt are the GQA-broadcast and V-transpose intermediates, and each exists ONLY in the
        # arm whose op writes it. Declaring them unconditionally allocated them anyway: an entry in
        # `buffer_sizes` that no runlist op references still lands in the scratch arena, because
        # calculate_buffer_layout appends every explicit buffer not already placed
        # (iron/common/sequence.py, the `explicit_buf not in scratch_args` branch). At the shipped
        # defaults (GROUPED_K=1, TMV_CTX=1) all three are dead, and at Hq*S*HD*2 = 8 MiB each over
        # 28 layers that is 672 MiB of arena that nothing reads -- 33.8% of the 1.99 GiB scratch,
        # and it reconciles exactly: 1.9898 GiB total minus 1.32904 GiB of named buffers = 0.661.
        bufsz.update({
            # CAPACITY, not window -- the same axis `weights[p+"kc"]` is sized on. These two
            # disagreed under KV_ALLOC: this said window and every operator said capacity, and
            # because bufsz is EXPLICIT it wins the arena layout, so the append then declared 32x
            # what the arena provided. One geometry owns both numbers; read them off it.
            p + "kc": g.hkv * g.capacity * g.hd * 2,
            p + "vc": g.hkv * g.capacity * g.hd * 2,
            p + "sc": Hq * g.window * 2, p + "sw": Hq * g.window * 2,
            p + "cx": g.qd * 2,
            p + "g": FF * 2, p + "u": FF * 2, p + "gh": FF * 2, p + "d": D * 2,
            # every partial and fold-level buffer the K-split introduces; the names come FROM
            # down_runlist so the two cannot drift apart
            **({} if down_chunks == 1 else
               {step[-1]: D * 2 for step in down_runlist(p)}),
            **({} if (fuse_o or g.o_chunks == 1) else
               {step[-1]: D * 2 for step in o_runlist(p, g)}),
            p + "hn": D * 2, p + "hf": D * 2,
        })
        if not GROUPED_K:
            bufsz[p + "kr"] = Hq * g.window * g.hd * 2
        if not (GROUPED_V or g.uses_tmv_ctx):
            bufsz[p + "vr"] = Hq * g.window * g.hd * 2
        if not g.uses_tmv_ctx:
            bufsz[p + "vt"] = Hq * g.window * g.hd * 2
        # `a` is purely internal to op_mlp_dp's own fuse_o path (never an L3 buffer -- see
        # design.py) once folded; only declare it when something outside that design still reads
        # or writes it.
        if not fuse_o:
            bufsz[p + "a"] = D * 2
        nxt = f"x{l+1}"
        if op_decode_layer is not None:
            # The whole layer -- attention AND MLP, including Wo -- is one design. q/k/v/qkv, sc,
            # sw, hn, hf, g, u, d and a are all core-local to attn_block_dp/swiglu_mlp_dp and never
            # become L3 buffers, so none of them gets a bufsz entry here (contrast the unfused arms
            # below, which still over-declare hn/hf/g/u/d unconditionally). `norms` packs n_in |
            # n_qn | n_kn (n_qn already carries attn_scale -- SCALE_IN_QNORM is a decode_layer_dp
            # eligibility precondition); `n_pf` stays separate, the MLP half's own argument.
            weights[p + "norms"] = np.concatenate(
                [weights.pop(p + "n_in"), weights.pop(p + "n_qn"), weights.pop(p + "n_kn")])
            bufsz[p + "kc"] = kv_layout.total_elems * 2
            bufsz[p + "vc"] = kv_layout.total_elems * 2
            bufsz[p + "cx"] = QD * 2
            rl.append((op_decode_layer, cur, p + "norms", p + "Wqkv", ang, p + "kc", p + "vc",
                       p + "cx", p + "n_pf", p + "Wo", p + "Wg", p + "Wu", p + "Wd",
                       "mlp_gh", "mlp_a_scratch", nxt))
        else:
            if g.op_attn_block is not None:
                # One device replaces norm/QKV/qk-norm/RoPE/KV-append/scores/softmax/ctx below; see
                # attn_block_why for the per-geometry eligibility this reuses. o_runlist (Wo) and
                # the MLP half are unaffected -- attn_block_dp stops at `cx`, same as op_ctx does.
                attn_rl = [(g.op_attn_block, cur, p + "n_in", p + "Wqkv", p + "n_qn", p + "n_kn",
                            ang, p + "kc", p + "vc", p + "cx")]
            else:
                qk = proj = rope = vnorm = []
                if sp.qk_norm and g.op_qkv_dp is None:
                    hq = [f"{qhb}[{qho + h*g.hd*2}:{qho + (h+1)*g.hd*2}]" for h in range(Hq)]
                    hk = [f"{khb}[{kho + h*g.hd*2}:{kho + (h+1)*g.hd*2}]" for h in range(g.hkv)]
                    qk = [*[((g.op_qk_norm if h % 2 == 0 else g.op_qk_norm_b),
                             hq[h], p + "n_qn", hq[h]) for h in range(Hq)],
                          *[((g.op_qk_norm if h % 2 == 0 else g.op_qk_norm_b),
                             hk[h], p + "n_kn", hk[h]) for h in range(g.hkv)]]
                if g.op_qkv_dp is None:
                    proj = ([(g.op_qkv, p + "Wqkv", p + "hn", p + "qkv")] if FUSE_QKV_GEMV else
                            [(g.op_q, p + "Wq", p + "hn", ref_q),
                             (g.op_kv, p + "Wk", p + "hn", ref_k),
                             *([(g.op_kv, p + "Wv", p + "hn", ref_v)] if g.has_v else [])])
                    rope = ([(g.op_rope_qk, ref_qk, ang, ref_qk)] if fuse_rope else
                            [(g.op_rope_q, ref_q, ang, ref_q),
                             (g.op_rope_k, ref_k, ang, ref_k)])
                    if g.op_v_norm is not None:
                        # Per kv head over head_dim, NOT rotated -- RoPE is a q/k-only step. Three args:
                        # this is the qk-norm design, so it takes a gain, and `ones` is what makes it
                        # gainless (see the construction site).
                        #
                        # SOURCE, and this is the whole of attention_k_eq_v: where the layer has a v_proj
                        # this is in place on v, but where it does not, V is the RAW k_proj output and the
                        # norm READS the k slice and WRITES the v buffer. That out-of-place form is also
                        # the copy, so k_eq_v needs no copy operator at all.
                        #
                        # ORDER is load-bearing in the second case and free in the first, so it is placed
                        # for the second: BEFORE the qk-norm and RoPE entries, which mutate k in place.
                        hv = [f"{vhb}[{vho + h*g.hd*2}:{vho + (h+1)*g.hd*2}]" for h in range(g.hkv)]
                        src = ([f"{khb}[{kho + h*g.hd*2}:{kho + (h+1)*g.hd*2}]" for h in range(g.hkv)]
                               if not g.has_v else hv)
                        vnorm = [(g.op_v_norm, a, f"ones_h{g.hd}", b) for a, b in zip(src, hv)]
                # The fused head replaces the norm, the projection, every qk-norm and the RoPE with one
                # design; `hn` lives and dies in L1 instead of round-tripping DDR between four of them.
                # The fused head absorbs the KV append too: k and v are drained straight into the caches
                # at `kv_off` instead of into buffers a StridedCopy then re-reads and re-writes. The caches
                # were their only consumer, so the intermediate had no reader -- it existed because the
                # append was a separate operator. Two runs and one more configure per layer.
                head = ([(g.op_qkv_dp, cur, p + "n_in", p + "Wqkv", p + "n_qn", p + "n_kn", ang,
                          ref_q, p + "kc", p + "vc")]
                        if g.op_qkv_dp is not None else
                        [(op_norm, cur, p + "n_in", p + "hn"), *proj, *vnorm, *qk, *rope,
                         (g.op_sck, ref_k, p + "kc"), (g.op_scv, ref_v, p + "vc")])
                attn_rl = [
                    *head,
                    *([] if GROUPED_K else [(g.op_rep_k, p + "kc", p + "kr")]),
                    # TMV_CTX subsumes the v-side grouping: TMatVec reads vc per kv head itself, so a
                    # Repeat would materialise a `vr` nothing consumes.
                    *([] if (GROUPED_V or g.uses_tmv_ctx) else [(g.op_rep_v, p + "vc", p + "vr")]),
                    *scores_runlist(p, g, ref_q),
                    *([] if scale_in_qnorm else [(g.op_scale, p + "sc", "attn_scale", p + "sc")]),
                    (g.op_softmax, p + "sc", p + "sw"),
                    *([] if g.uses_tmv_ctx else
                      [(g.op_trv, p + ("vc" if GROUPED_V else "vr"), p + "vt")]),
                    (g.op_ctx, p + ("vc" if g.uses_tmv_ctx else "vt"), p + "sw", p + "cx"),
                ]
            rl += [*attn_rl, *([] if fuse_o else o_runlist(p, g))]
            if sp.sandwich_norms:
                rl.append((op_norm, p + "a", p + "n_pa", p + "a"))
            if op_mlp_dp is not None:
                # cur + a -> x1 -> norm -> gate/up -> silu -> mul -> down -> +x1, all inside one design.
                # x1/hf/g/u/gh/d never reach DDR; `mlp_gh` is the all-gather round-trip buffer and is
                # shared across layers because the sequence runs them one at a time. FUSE_MLP_O folds
                # `a = Wo @ cx` in too: `cx`/`Wo` replace `a` as the design's own inputs, and
                # `mlp_a_scratch` is a's own all-gather round-trip buffer, the same idiom as mlp_gh's.
                # post_norm adds ONE argument before `nxt` (get_arg_spec's post_norms_spec):
                # [n_pff] at D here, [n_pa | n_pff] at 2*D under fuse_o -- so the fused design
                # applies the post-FFN norm itself and the standalone op_norm below must not.
                if fuse_o:
                    # sandwich norms cannot reach here: fuse_o declines them at its derivation,
                    # where op_o and the arg specs are shaped to match.
                    rl.append((op_mlp_dp, cur, p + "cx", p + "n_pf", p + "Wo", p + "Wg", p + "Wu",
                               p + "Wd", "mlp_gh", "mlp_a_scratch", nxt))
                else:
                    rl.append((op_mlp_dp, cur, p + "a", p + "n_pf", p + "Wg", p + "Wu", p + "Wd",
                               "mlp_gh", *([p + "n_pff"] if sp.sandwich_norms else []), nxt))
            else:
                rl += [
                    (op_add, cur, p + "a", p + "x1"),
                    (op_norm, p + "x1", p + "n_pf", p + "hf"),
                    (op_gate, p + "Wg", p + "hf", p + "g"),
                    (op_up, p + "Wu", p + "hf", p + "u"),
                    *([] if op_act is None else [(op_act, g_, g_) for g_ in ff_slices(p + "g")]),
                    *[(op_mul_ffn, g_, u_, gh_) for g_, u_, gh_ in
                      zip(ff_slices(p + "g"), ff_slices(p + "u"), ff_slices(p + "gh"))],
                    *down_runlist(p),
                ]
            # `d` is the unfused chain's own output buffer and does not exist in the fused arm,
            # which carries this norm internally (see the post_norm argument above).
            if sp.sandwich_norms and op_mlp_dp is None:
                rl.append((op_norm, p + "d", p + "n_pff", p + "d"))
            if op_mlp_dp is None:
                rl.append((op_add, p + "x1", p + "d", nxt))
        bufsz[p + "x1"] = D * 2
        if op_lscale is not None:
            # `hidden_states *= self.layer_scalar` is the LAST statement of the reference decoder
            # layer, after both residual adds, so it goes here and not inside either arm above.
            # In place on `nxt`: it is written by whichever arm ran and nothing has read it yet.
            #
            # Broadcast to D rather than passed as an RTP because the operator multiplies two
            # buffers elementwise, and a D-wide constant is 7680 B per layer against a decode step
            # that already streams hundreds of MB. Its VALUE is per layer (0.053 at layer 0, 0.048
            # at 47), so this cannot be one shared buffer.
            weights[p + "ls"] = np.full(D, float(npy(sp.layer_scalar_name(l))[0]), BF16)
            rl.append((op_lscale, nxt, p + "ls", nxt))
        cur = nxt

    if op_mlp_dp is not None or op_decode_layer is not None:
        bufsz["mlp_gh"] = FF * 2   # one buffer, reused by every layer -- they run one at a time
        if fuse_o or op_decode_layer is not None:
            bufsz["mlp_a_scratch"] = D * 2   # a's own all-gather round-trip buffer, same idiom

    weights["n_final"] = load_norm(f"{sp.weight_prefix}norm.weight")
    # tied: also the host's embedding-gather table
    embed_f32 = npy(f"{sp.weight_prefix}embed_tokens.weight")
    # Quantizing W_head narrows the DEVICE lm-head stream, but W_head is TIED, so the host also
    # gathers embed[token] out of it. Rather than teach the host to dequantise -- which would
    # quantise the embedding INPUT too, a second quality change for no extra speed -- the exact
    # bf16 table is emitted alongside as a HOST-ONLY blob. It is written outside the `weights`
    # dict on purpose: the device loader takes its buffer set from `weights`/`wnames`, so a side
    # file costs 311 MB of disk and ZERO device arena, and the host only ever faults in the one
    # 2 KB row it gathers.
    embed_blob, host_embed = "W_head", None
    if _spec("head").quantized:
        # Pre-packed by the dump when available (_head_packed) -- read the bytes straight off
        # disk rather than re-quantizing here, which is what OOM'd on Gemma-4-12B's 262144x3840
        # table (dump_llm_weights.py's _pack_head_chunked does the same math in row chunks).
        weights["W_head"] = npy_raw(f"{_head_key}.headpack") if _head_packed else \
            _pack(embed_f32, "head")
        embed_blob = "W_embed"
        host_embed = bf16(embed_f32).reshape(-1)
    else:
        weights["W_head"] = bf16(embed_f32).reshape(-1)
    if not scale_in_qnorm:
        weights["attn_scale"] = np.full(Hq * S, sp.attn_scale, BF16)
    rl += [(op_norm, cur, "n_final", "xf")]
    if not SPLIT_LM_HEAD:
        rl += [(op_head, "W_head", "xf", "logits")]
    bufsz["xf"] = D * 2
    bufsz["logits"] = VOCAB * 2

    if os.environ.get("DUMP_OPS"):
        from collections import Counter
        c = Counter(type(e[0]).__name__ for e in rl)
        print(f"# {sp.name}: runlist {len(rl)} entries over NL={NL} "
              f"({(len(rl)-2)//NL}/layer + 2 tail)")
        for k, v in c.most_common():
            print(f"  {k:16} {v:5}")
        raise SystemExit(0)

    # Declare the angle buffers the BUILT layers actually read, not the ones the spec could
    # produce at full depth. The two differ under truncation: Gemma-4 is global on layers
    # where (l+1)%sw_pattern == 0, so `--layers 5` is sliding-only and declaring
    # `rope_global` there makes calculate_buffer_layout refuse the design -- "Input argument
    # rope_global not found in runlist buffers" -- because no op consumes it. Same rule as
    # the per-layer `ang` selection above, read over the range that was emitted.
    angs = {"rope_global" if sp.is_global(l) else "rope_local" for l in range(NL)}
    inputs = ["x"] + [n for n in ("rope_global", "rope_local") if n in angs]
    # cores-per-col=1 spreads each operator's workers one per column instead of stacking them four
    # deep in two columns, which is what the default column-major SequentialPlacer does. Every op
    # here has <= 8 workers, so one per column fits the 8-column array. Overridable because this is
    # a placement experiment, not a settled default.
    #
    # THIS is the whole measured win: -11.14 ms/token, -7.1%, isolated on device with the transpose
    # batching held constant and DDR bytes identical at 3105.99 MB in every arm.
    # decode_layer_dp needs 4 ROWS of placement (its own verified layout is columns 0-2, rows
    # 2-5 -- 12 cores across 3 columns), which --cores-per-col 1 forecloses outright: aiecc
    # reports "cores-per-col=1 leaves 8 of this device's 32 compute tiles placeable". The
    # single-row spread's measured win is specific to the unfused per-op designs' <=8-worker
    # shape and does not transfer, so this arm's default is the placer's OWN default (no
    # restriction) instead of DECODE_PLACER_FLAGS_DEFAULT -- still overridable via the env var.
    placer_default = "" if op_decode_layer is not None else DECODE_PLACER_FLAGS_DEFAULT
    placer_flags = os.environ.get("DECODE_PLACER_FLAGS", placer_default).split()
    # Two designs where one would do: gate/up are the same GEMV shape and adjacent, as are the two
    # KV StridedCopys. Each duplicate pair costs an extra aiex.configure PER LAYER -- 56 per token
    # against a measured ~40 us each. SHARE_DESIGNS=0 restores the unshared build for an A/B.
    share = os.environ.get("SHARE_DESIGNS", "1") == "1"
    # Under SPLIT_LM_HEAD the stack's output is `xf`, not the logits. Declaring it as an OUTPUT
    # rather than leaving it a scratch intermediate is load-bearing twice over: the second graph
    # needs it, and the output arena is the one that gets synced back -- a scratch read of `xf`
    # returns zeros at 12 layers while the device plainly computed from it.
    head_name = "logits" if not SPLIT_LM_HEAD else "xf"

    weight_families = []
    if MERGE_WEIGHT_GEMVS:
        rl, weight_families = unify_weight_gemvs(rl)
        for K_, tsi_, tso_, Ms_ in weight_families:
            print(f"[gen] weight GEMV family K={K_}: tsi {tsi_} tso {tso_} tiles_rtp over M={Ms_}")
    pointwise_widths = []
    if POINTWISE_MODES:
        rl, pointwise_widths = pointwise_modes(rl)
        for size_, tile_, modes_ in pointwise_widths:
            print(f"[gen] pointwise width {size_} (tile {tile_}): modes {modes_}")
    merge = bool(weight_families or pointwise_widths)

    if DECODE_SEGMENTS > NL:
        raise SystemExit(f"DECODE_SEGMENTS={DECODE_SEGMENTS} exceeds the {NL} layers there are to "
                         f"split; a segment boundary only exists at a layer boundary")
    # Cut points in LAYER index. Contiguous and near-equal, remainder to the earliest segments, so
    # 48 over 3 is 16/16/16 and 48 over 5 is 10/10/10/9/9. The last segment also carries the tail
    # (final norm, and the lm-head unless SPLIT_LM_HEAD moved it out), which is why an equal layer
    # split still leaves the last arena the largest.
    q, r = divmod(NL, DECODE_SEGMENTS)
    cuts, _a = [], 0
    for i in range(DECODE_SEGMENTS):
        _b = _a + q + (1 if i < r else 0)
        cuts.append((_a, _b))
        _a = _b

    segments = []
    for si, (la, lb) in enumerate(cuts):
        first, last = si == 0, si == len(cuts) - 1
        lo = layer_marks[la][0]
        hi = layer_marks[lb][0] if lb < NL else len(rl)
        entries = rl[lo:hi] if not last else rl[lo:]
        seg_in = layer_marks[la][1]
        seg_out = head_name if last else layer_marks[lb][1]
        # Which buffers this slice touches, read off the entries themselves rather than rebuilt from
        # a name convention -- `split_over_k` emits BYTE-SLICED operands (`Wd[0:1234]`), and a
        # convention-based list would silently drop or duplicate them.
        refs = runlist_buffer_names(entries)
        # UNSPLIT IS VERBATIM. At one segment the arg lists must be the objects the pre-segmentation
        # build passed, not a filtered reconstruction of them -- `bufsz` carries entries no op reads
        # (`logits` under SPLIT_LM_HEAD), and dropping them would change the default artifact while
        # looking like a refactor. The filtered path exists only where there is a seam to fit.
        one = len(cuts) == 1
        seg_inputs = inputs if one else [seg_in] + [n for n in inputs if n != "x" and n in refs]
        seg_bufsz = bufsz if one else {n: v for n, v in bufsz.items() if n in refs}
        # The residual chain x -> x1 -> ... -> xNL is sized implicitly in the unsplit build, because
        # every link is produced and consumed inside one graph. A cut turns one link into a declared
        # arg on both sides of the seam, and a declared arg needs a size -- the same reason the
        # lm-head split passes `{"xf": D * 2}` rather than letting it be inferred. `x` itself stays
        # out: it is the model input and the unsplit build does not size it either.
        if not first:
            seg_bufsz[seg_in] = D * 2
        if not last:
            seg_bufsz[seg_out] = D * 2
        _sn = sequence_name(sp, NL, S, placer_flags, tmv_declined=_tmv_declined,
                            tmv_chunked=_tmv_chunked, attn_block_geoms=_attn_block_fused,
                            decode_layer_active=op_decode_layer is not None, T=T,
                            ff_chunks=ff_chunks, weight_families=len(weight_families),
                            pointwise_widths=len(pointwise_widths),
                            scores_blocks=tuple(_scores_blocks))
        name = _sn if len(cuts) == 1 else f"{_sn}_seg{si}of{len(cuts)}"
        # A rung is THIS runlist with the layer design substituted. Only for an unsplit
        # stack: a rung rewrites one runlist, and a segmented stack has one per segment
        # with a host seam between them, so "the runlist" is not a single object.
        _extra = ({f"sequence_w{w}": [((rung if op is op_decode_layer else op), *bufs)
                                      for op, *bufs in entries]
                   for w, rung in sorted(rung_ops.items())} if (one and rung_ops) else {})
        if _extra:
            print(f"# {sp.name}: window rungs {sorted(rung_ops)} + top {S}, "
                  f"{len(_extra) + 1} named control codes in one ELF")
        seq = OperatorSequence(name, entries,
                               input_args=seg_inputs, output_args=[seg_out],
                               buffer_sizes=seg_bufsz, context=ctx, extra_flags=placer_flags,
                               share_designs=share,
                               **({"merge_devices": True} if merge else {}),
                               **({"collapse_configures": True}
                                  if merge and COLLAPSE_MERGED_CONFIGURES else {}),
                               **({"extra_runlists": _extra} if _extra else {}),
                               **({"scratch_order": list(weights.keys())}
                                  if BUCKET_SCRATCH_ORDER else {}))
        seq.compile()
        # Per-segment weight set, refs-filtered even when unsplit: under SPLIT_LM_HEAD `weights`
        # still holds W_head but no op in this graph reads it, so a consumer that loads by this list
        # asks for a buffer the arena does not have. The addressability CHECK keeps taking the full
        # list when unsplit, because that is what it took before and it tolerates a missing name.
        seg_weights = sorted(n for n in weights if n in refs)
        seg_caches = cache_names if one else [n for n in cache_names if n in refs]
        check_arena_offsets_are_addressable(
            seq, [*seg_inputs, seg_out, *(weights if one else seg_weights), *seg_caches])
        # PER-SEGMENT KV SLOTS. `kv_off`/`kv_off1`/... are named per GEOMETRY in first-appearance
        # order and baked into that geometry's StridedCopy, so a segment's scratchpad declares only
        # the slots its own layers' head_dims use. Gemma-4 is the case: sliding layers are hd 256
        # and global ones hd 512, so a segment holding no global layer has no `kv_off1` and writing
        # one raises "ParameterScratchpad: unknown parameter". Derived from head_dim_for(), the same
        # source the names were assigned from, rather than by probing the scratchpad -- a probe
        # would silently skip a slot that SHOULD have been there.
        seg_hds = {sp.head_dim_for(l) for l in range(la, lb)}
        seg_kv_slots = [(n, hd) for n, hd in kv_slots if hd in seg_hds]
        seg_geom_slots = [t for t in geom_slots if t[1] in seg_hds]
        # Same reasoning as seg_kv_slots, one axis over: a segment whose layers are all one
        # geometry has no reason to declare the OTHER geometry's mask slot.
        seg_ws = {sp.sliding_window if (SLIDING_KV_CIRCULAR and sp.sliding_window is not None
                                        and not sp.is_global(l)) else S
                 for l in range(la, lb)}
        seg_mask_slots = [(n, ww) for n, ww in mask_slots if ww in seg_ws]
        segments.append(dict(seq=seq, layers=(la, lb), inlet=seg_in, outlet=seg_out,
                             weights=seg_weights, caches=seg_caches, inputs=seg_inputs,
                             kv_slots=seg_kv_slots, mask_slots=seg_mask_slots,
                             geom_slots=seg_geom_slots))
        if len(cuts) > 1:
            print(f"[gen] segment {si}: layers {la}..{lb - 1}, {seg_in} -> {seg_out}, "
                  f"{len(seg_weights)} weights, arena {seq.buffer_sizes[2] / 2**30:.3f} GiB",
                  file=sys.stderr)
    fused = segments[0]["seq"]

    # The lm-head as its own graph: one op, `xf` in, logits out, its own W_head. No arena sharing --
    # `xf` is 7680 bytes and crosses through the host, which costs one small copy per token against a
    # step that is already dispatch-dominated. Arena sharing via scratch_order is the faster form and
    # the mechanism exists (gen_llm_prefill.py::decode_arena_plan); this is the correctness fix, and
    # the two are independent.
    head = None
    if SPLIT_LM_HEAD:
        head_rl = [(op_head, "W_head", "xf", "logits")]
        head = OperatorSequence(
            f"{sequence_name(sp, NL, S, placer_flags, tmv_declined=_tmv_declined, tmv_chunked=_tmv_chunked)}_lmhead", head_rl,
                                input_args=["xf"], output_args=["logits"],
                                buffer_sizes={"xf": D * 2, "logits": VOCAB * 2},
                                context=ctx, extra_flags=placer_flags, share_designs=share)
        head.compile()
    return sp, fused, weights, dict(NL=NL, S=S, T=T, inputs=inputs, cache_names=cache_names,
                                        decode_layer_active=op_decode_layer is not None,
                                        # getattr, not attribute access: a spec whose fused
                                        # layer did not build has no such attribute.
                                        # Only when the parameter was actually WIRED. The
                                        # operator carries a granule either way, so reporting
                                        # it unconditionally makes the host write a scratchpad
                                        # parameter the ELF never declared.
                                        window_granule=(getattr(op_decode_layer,
                                                                "window_granule", None)
                                                        if DYNAMIC_WINDOW else None),
                                        window_rungs={f"sequence_w{w}": w for w in sorted(rung_ops)},
                                    head=head, split_lm_head=SPLIT_LM_HEAD,
                                    segments=segments, layer_marks=layer_marks,
                                    embed_blob=embed_blob, host_embed=host_embed,
                                    kv_slots=kv_slots, mask_slots=mask_slots,
                                    geom_slots=geom_slots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS), help="model spec name")
    ap.add_argument("--weights", required=True, help="dir of dumped .npy weights (see dump_llm_weights.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (bring-up)")
    # S IS THREE THINGS AND THEY ARE NOT THE SAME NUMBER. It is the KV CAPACITY allocated, the
    # WINDOW attention reads (sliding layers read sp.sliding_window, not S, under
    # SLIDING_KV_CIRCULAR), and the REDUCTION LENGTH that sizes L1 tiling. They coincide only on a
    # non-windowed geometry. `w` at the attn_ops construction site is the window; `KVA`/`alloc_K`
    # is the capacity; a site that wants either MUST take it rather than reach for S.
    # Passing S where the window belongs sizes a sliding layer's L1 for positions it can never
    # reach, which is what held this model's context ceiling at 6912. See
    # a-name-that-means-three-things-fails-at-the-site-without-the-comment.
    ap.add_argument("--max-seq", type=int, default=2048, help="KV-cache padded capacity S")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    NL, S, T, inputs, cache_names = md["NL"], md["S"], md["T"], md["inputs"], md["cache_names"]
    # meta.json describes ONE elf and ONE layout. A segmented stack is N of each plus the seam
    # order between them, and none of that has a field here yet -- so writing the artifact anyway
    # would emit segment 0 under the full model's name: a Gemma-4 artifact that loads, runs, and
    # silently decodes the first sixteen layers. Refuse instead. verify_llm_decode.py drives the
    # segments in-process and is the gate until this format grows the fields.
    if len(md["segments"]) > 1:
        raise SystemExit(
            f"DECODE_SEGMENTS={len(md['segments'])}: this writer emits a single-ELF artifact and "
            f"would record only segment 0 ({md['segments'][0]['layers'][1]} of {NL} layers) as the "
            f"whole model. Gate a segmented stack through verify_llm_decode.py, which drives every "
            f"segment, until meta.json carries the per-segment ELF list and seam order.")
    embed_blob, host_embed = md["embed_blob"], md["host_embed"]
    decode_layer_active = md["decode_layer_active"]
    window_granule = md["window_granule"]
    window_rungs = md["window_rungs"]
    dynamic_window = DYNAMIC_WINDOW and decode_layer_active
    D, HD, Hq, Hkv, VOCAB = sp.d_model, sp.head_dim, sp.n_q_heads, sp.n_kv_heads, sp.vocab
    FF = sp.ffn
    elf = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    wnames = list(weights.keys())
    # Build the layout over what the graph DECLARES, never a hand-written list. The literal this
    # replaces omitted `rope_global` -- a declared input -- so every emitted meta.json described a
    # buffer set the ELF did not have, and a consumer placing buffers by layout could not find it.
    # IRON already computed the answer: `subbuffer_layout` covers every input, output and scratch
    # arg, and `calculate_buffer_layout` raises if a declared arg is missing from the runlist.
    lay = {n: fused.get_layout_for_buffer(n) for n in [*inputs, "logits", *wnames]}

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
        b = weight_bytes(arr)
        # The seam a precision plan crosses: the HOST packs a weight and an OPERATOR declares the
        # buffer it lands in, and neither side can see the other's units. Each is correct alone;
        # a disagreement exists only between them, which is why it survives every type check and
        # surfaces as a load-time size error against an artifact that built clean.
        if n_ not in lay:
            raise SystemExit(f"[gen] {n_}: no layout entry; the design declares no such buffer")
        declared = lay[n_][2]          # (buf_type, offset_bytes, length_bytes)
        if len(b) != declared:
            raise SystemExit(
                f"[gen] {n_}: packed {len(b)} B, the graph declares {declared} B. The precision "
                f"plan ({PRECISION_PLAN.get(_site_of(n_), precision.BF16_SPEC)} at site "
                f"{_site_of(n_)!r}) is not the format the operator holding this buffer was built "
                "for -- see precision.py P003.")
        write_blob(os.path.join(bdir, f"{n_}.bin"), b)
    if embed_blob != "W_head":
        # Host-only, deliberately not in `wnames`: see the tied-embedding note at its build site.
        write_blob(os.path.join(bdir, f"{embed_blob}.bin"), weight_bytes(host_embed))
    open(os.path.join(a.out, "decode.elf"), "wb").write(elf)

    meta = {
        "spec": sp.name, "elf": "decode.elf", "kernel_name": "main:sequence",
        # The resolved OperatorSequence name (sequence_name()'s return, plus IRON's own
        # `_shared` suffix) -- the same string IRON keys the build cache by. Two arms can build
        # identical dims/weight_quant and still be different graphs (TMV_CTX, FUSE_MLP_DP,
        # WEIGHT_DEPTH, ... are not otherwise recorded here); this is the one field that
        # disambiguates them without re-deriving the flag values from ELF size.
        "sequence_name": fused.name,
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": inputs, "weights": wnames, "output": "logits",
        # Which blob the HOST gathers embed[token] from. Always bf16 [vocab, d_model]; it is
        # W_head itself unless the lm-head was quantised, in which case W_head is packed and this
        # names the bf16 sidecar. Absent in older artifacts -- consumers default to "W_head".
        "embed_blob": embed_blob,
        # `kv_param` is the single-slot form every artifact before this carried, kept so an older
        # consumer still loads; `kv_params` is the list the host prefers.
        # window_param mirrors kv_param/mask_param -- the POINTER into scratchpad.params, not the
        # entry itself. That entry ("attn_window": {byte_offset, kind, dtype}) needs no special
        # case here: it is already in scratchpad_params, read generically off params.txt above
        # like every other declared ScratchpadParameter, so its offset is never a literal.
        # `kv_param` is the single-slot form every artifact before this carried, kept so an
        # older consumer still loads; `kv_params` is the per-geometry list the host prefers.
        # window_param mirrors both -- the POINTER into scratchpad.params, not the entry
        # itself, which is already in scratchpad_params read generically off params.txt.
        "scratchpad": {"params": scratchpad_params, "kv_param": "kv_off",
                       "mask_param": "sm_mask",
                       "kv_params": [{"param": n, "head_dim": hd} for n, hd in md["kv_slots"]],
                       # mask_params mirrors kv_params one axis over: one entry per DISTINCT
                       # window this build actually declared a Softmax for. A single entry here
                       # (today's default, and every artifact before SLIDING_KV_CIRCULAR) means
                       # every geometry shares "sm_mask" -- byte-identical to before this existed.
                       "mask_params": [{"param": n, "window": ww} for n, ww in md["mask_slots"]],
                       # Per-GEOMETRY join of the two lists above -- see geom_slots's own comment
                       # at its declaration. Not consumed by the Rust host as of 2026-09-14 (see
                       # SLIDING_KV_CIRCULAR's own doc); present so a future consumer has the
                       # pairing without re-deriving it, and so this artifact is self-describing.
                       "kv_windows": [{"kv_param": n, "head_dim": hd, "window": cap,
                                       "mask_param": mn, "kv_block": blk, "kv_heads": khv}
                                      for n, hd, cap, mn, blk, khv in md["geom_slots"]],
                       "head_dim": HD, "kv_heads": Hkv,
                       **({"window_param": "attn_window"} if dynamic_window else {})},
        "dims": {"layers": NL, "d_model": D, "q_heads": Hq, "kv_heads": Hkv, "head_dim": HD,
                 "ffn": FF, "vocab": VOCAB, "S": S, "kv_block": T,
                 # Wqkv's ROW ORDER, stated because another generator reads this buffer out of the
                 # shared arena and cannot see the flag that produced it. Prefill went on slicing
                 # the stock [Wq|Wk|Wv] for a week after this became head-major, which is a
                 # plausible wrong answer and never an error. Absent in older artifacts -- a
                 # consumer reads that as the stock order, which is what those artifacts hold.
                 "wqkv_head_major": decode_layer_active,
                 "sliding_window": sp.sliding_window, "sw_pattern": sp.sw_pattern,
                 # Declared BEFORE this existed too (sliding_window/sw_pattern above), but never
                 # wired to anything on-device -- this is the one field a host harness needs to
                 # tell "declared and honoured" apart from "declared and ignored" without grepping
                 # env vars the artifact itself does not otherwise record.
                 "sliding_kv_circular": SLIDING_KV_CIRCULAR,
                 # The runtime attn_window value's required granularity -- lcm(stream-tile rows,
                 # kv block), computed once at op construction (decode_layer_dp/op.py's
                 # window_granule). Ships explicitly so the host never re-derives it from
                 # kv_block: the two coincide (128) at this model's shape but would not at a
                 # different head_dim or tile_size_input, and a host that derived it anyway would
                 # be right here and silently wrong on the next model.
                 **({"window_granule": int(window_granule)} if dynamic_window else {})},
        # The named control codes this ELF carries BESIDES `main:sequence`, each a decode_layer_dp
        # design at a narrower attention window over the SAME KV capacity and the SAME arena. A
        # host binds one xrt::ext::kernel per entry against the ONE hw_context this ELF registers
        # and dispatches whichever rung covers n_past; `main:sequence` at dims.S stays the
        # fallback and is what a consumer that ignores this field keeps using. Absent when no
        # rungs were built, which is every artifact before this existed.
        **({"window_rungs": dict(sorted(window_rungs.items(), key=lambda kv: kv[1]))}
           if window_rungs else {}),
        # Per-token host protocol (the ELF is constant; only these change):
        #   x        = embed[token], scaled by sqrt(d_model) iff embed_scale == "sqrt_d_model"
        #   rope_*   = precomputed [S,HD] angle tables; the row for n_past is used
        #   kv_off   = the RUNTIME half of iron.common.kv_layout.KVLayout(kv_heads, S, head_dim,
        #              kv_block).kv_off(n_past) -- element units, addr kind, raw. At kv_block == S
        #              (dims.kv_block == dims.S) this is exactly `n_past * head_dim`, the
        #              pre-blocking formula; the host must compute the general form (block *
        #              block_stride + within_block * head_dim) whenever kv_block < S. See
        #              kv_layout.py -- the single owner of this arithmetic -- not this comment.
        #   sm_mask  = n_past + 1          (core kind, causal width; host writes it <<2)
        "host_protocol": {"embed_scale": sp.embed_scale, "attn_scale": float(sp.attn_scale),
                          "act": sp.act, "norm_gain": sp.norm_gain, "eps": sp.eps,
                          "rope_theta_global": sp.rope_theta_global,
                          "rope_theta_local": sp.rope_theta_local,
                          # The checkpoint's partial_rotary_factor, on the GLOBAL row only -- the
                          # host resolves it against that buffer's own width. Null on every model
                          # whose rope_type is "default", which is every one but Gemma-4's global
                          # layers.
                          "rope_partial_rotary": sp.rope_partial_rotary,
                          # final_logit_softcapping: tanh(logits/c)*c, applied by the host after
                          # the logits cross back. Null unless the checkpoint sets it.
                          "logit_softcap": sp.logit_softcap},
        "layer_types": ["global" if sp.is_global(l) else "sliding" for l in range(NL)],
        "cache_buffers": cache_names,
        # `plan` is the whole per-site truth and `projected_mb_per_token` is what it was priced
        # at; the flat keys beside them are the shape npu_decode.rs::provenance_extras reads.
        # scale_kind rides here rather than in the design name whenever it is the class default:
        # it moves weight VALUES only, so two such arms share one compiled design and differ
        # solely in the bytes loaded into it.
        "weight_quant": {
            "plan": {k: str(v) for k, v in sorted(PRECISION_PLAN.items())},
            "plan_source": PRECISION_PROV,
            # None, not another spec's number, when precision.py carries no census for sp.name
            # yet -- see precision.py's CENSUS and the gemma4-12b defect it fixes.
            "projected_mb_per_token": (
                round(precision.token_mb(PRECISION_PLAN, sp.name)["total"], 2)
                if sp.name in precision.CENSUS else None),
            "mlp_dtype": _spec("mlp").dtype, "mlp_group_size": _spec("mlp").group_size or 128,
            "attn_dtype": _spec("attn_o").dtype,
            "attn_group_size": _spec("attn_o").group_size or 128,
            "qkv_dtype": _spec("qkv").dtype,
            "qkv_group_size": _spec("qkv").group_size or 128,
            "head_dtype": _spec("head").dtype, "head_group_size": _spec("head").group_size or 128,
            "clip_search": any(v.scale_kind in ("clip", "clip_full")
                              for v in PRECISION_PLAN.values()),
            "full_range": any(v.scale_kind == "clip_full" for v in PRECISION_PLAN.values()),
        },
    }
    prov = toolchain_provenance()
    if prov:
        meta["toolchain"] = prov
    else:
        print("[build] WARNING: could not record toolchain provenance in meta.json "
              "(no toolchain.lock / kernel_sandbox.sh resolvable) -- this artifact will read as "
              "unstamped to any freshness check", file=sys.stderr)
    iprov = iron_provenance()
    if iprov:
        meta["iron"] = iprov
        if iprov["dirty"]:
            print(f"[build] WARNING: IRON was DIRTY at {iprov['commit'][:12]} -- meta.json's iron "
                  "commit will not reproduce this build's kernels", file=sys.stderr)
    else:
        print("[build] WARNING: could not record IRON provenance in meta.json -- a kernel change "
              "in this artifact will be invisible to any later check", file=sys.stderr)
    gprov = generator_provenance()
    if gprov:
        meta["generator"] = gprov
        if gprov["dirty"]:
            print(f"[build] WARNING: designs/decode_fused was DIRTY at {gprov['commit'][:12]} -- "
                  "meta.json's generator commit will not reproduce this build", file=sys.stderr)
    else:
        print("[build] WARNING: could not record generator provenance in meta.json "
              "(no git checkout resolvable) -- staleness against designs/decode_fused HEAD will "
              "be invisible to any future check", file=sys.stderr)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\nwrote {NL}-layer {sp.name} decode ELF ({len(elf)}B, scratch {scr/1e6:.1f}MB) to {a.out}")


if __name__ == "__main__":
    try:
        main()
    except precision.PrecisionRefusal as exc:
        # A refused plan is a diagnostic, not a crash: it names a rule and the source that owns
        # it, and a traceback through the generator adds nothing to either.
        sys.exit(f"\n[gen] precision plan REFUSED\n{exc}\n")
