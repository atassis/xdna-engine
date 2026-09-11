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
                             gemv_fits, gemv_tile_output, k_chunks_for)

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

import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext  # noqa: E402
from iron.common.kv_layout import KVLayout, derive_block_size  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemv.op import GEMV  # noqa: E402
from iron.operators.gemv.design import MAX_GROUP_REUSE  # noqa: E402
from iron.common.quant import quantize_weight, row_stride_bytes  # noqa: E402
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


def _quant_kw(site):
    """`weight_dtype`/`group_size` kwargs for the operator carrying one site's weight. Empty at
    bf16, so an unquantized call is the shape it would have had with no precision plane."""
    spec = _spec(site)
    return {} if not spec.quantized else dict(weight_dtype=spec.dtype,
                                              group_size=spec.group_size)


_SITE_OF_SUFFIX = {"Wqkv": "qkv", "Wq": "qkv", "Wk": "qkv", "Wv": "qkv", "Wo": "attn_o",
                   "Wg": "mlp", "Wu": "mlp", "Wd": "mlp", "W_head": "head",
                   "kc": "kv", "vc": "kv"}


def _site_of(buffer_name):
    """Which census site a weight buffer belongs to, by its `L<n>_<key>` suffix."""
    return _SITE_OF_SUFFIX.get(buffer_name.rsplit("_", 1)[-1]
                               if buffer_name.startswith("L") else buffer_name)


def _pack(w, site):
    """Host-side pack of one weight under its site's spec, into the packer's wire format."""
    spec = _spec(site)
    if not spec.quantized:
        return bf16(w).reshape(-1)
    kw = {}
    if spec.dtype in precision.SYMMETRIC:
        kw["clip_search"] = spec.scale_kind == "clip"
    else:
        kw["affine_zero_on_grid"] = spec.scale_kind == "zero_grid"
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
# Pin the persistent buffers (weights + KV cache) to the FRONT of the scratch arena so window
# buckets present ONE layout for everything that survives a bucket crossing. Without it the
# window-sized softmax scratch (sc/sw, Hq*S) sits ahead of them and shifts every later offset:
# measured, buckets at window 256 and 512 over one allocation disagreed on 304 of 313 offsets.
BUCKET_SCRATCH_ORDER = os.environ.get("BUCKET_SCRATCH_ORDER", "0") == "1"
# Weight tile ROWS for the fused MLP. Trades against WEIGHT_DEPTH at constant L1.
MLP_TILE_ROWS = int(os.environ.get("MLP_TILE_ROWS", "0"))

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
# Thread decode_layer_dp's window_parameter through: the AIE core reads its attention window from
# a per-dispatch ScratchpadParameter ("attn_window", int32) instead of baking N_KV_CHUNKS into the
# build. Only takes effect when decode_layer_dp itself is eligible (decode_layer_why is None below)
# -- there is nowhere else in this graph for it to attach. Default 0 = build-constant window,
# byte-for-byte the pre-existing graph and meta.json; params.txt (read further down) picks up the
# new parameter's real offset for free once this is on, so the meta writer never hardcodes one.
DYNAMIC_WINDOW = os.environ.get("DYNAMIC_WINDOW", "0") == "1"
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



def sequence_name(sp, NL, S, placer_flags, decode_layer_active=False, T=None, tmv_declined=()):
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
    if not GROUPED_K:
        parts.append("nogk")
    # The KV cache's block size (iron.common.kv_layout). T == S (or None, pre-this-task callers)
    # is the flat pre-blocking layout and keeps the bare name; T < S addresses the SAME cache
    # buffers completely differently, so it must not share a name with the flat build.
    if T is not None and T != S:
        parts.append(f"kvt{T}")
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
    if SPLIT_GH_DRAIN != 1:
        parts.append(f"sgh{SPLIT_GH_DRAIN}")
    if ATTN_SPLIT:
        parts.append(f"sp{ATTN_SPLIT}")
    if WEIGHT_DEPTH != 2:
        parts.append(f"wd{WEIGHT_DEPTH}")
    if MLP_TILE_ROWS:
        parts.append(f"tr{MLP_TILE_ROWS}")
    if FUSE_ACT:
        parts.append("fuseact")
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
                f"disk-backed path (e.g. ${{XDNA_SCRATCH:-/mnt/data/xdna-scratch}}/{tag}), or set "
                f"ALLOW_TMPFS_BUILD=1 for a small probe build.")
        os.chdir(explicit)
        print(f"[{tag}] build dir {explicit} (DECODE_WORK, kept)", flush=True)
        return explicit

    root = os.environ.get("XDNA_SCRATCH", "/mnt/data/xdna-scratch")
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
def gemv(M, K, ctx, **kw):
    """GEMV tiled as large as both the design asserts AND L1 allow."""
    tsi, tso = gemv_tile_output(M, K)
    g = kw.get("group_size", 0)
    if g and g < 64:
        # mv_quant.cc's dequant chunk must not straddle a quant group, and GEMV asserts
        # group_size % kernel_vector_size == 0. 64 is the default and the only width the shipped
        # groups (>=128) ever needed; a 32-wide group needs 32.
        kw["kernel_vector_size"] = g
    return GEMV(M=M, K=K, num_aie_columns=COLS, tile_size_input=tsi,
                tile_size_output=tso, context=ctx, **kw)


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
        from iron.operators.tmatvec.design import check_l1_fits
        for _hd, _hkv, _ in geoms:
            _gqa, _r = Hq // _hkv, TMV_RPC
            while _r > 1 and (S % _r or check_l1_fits(_hd, S, _gqa, _r) is not None):
                _r //= 2
            tmv_rpc[_hd] = None if check_l1_fits(_hd, S, _gqa, _r) is not None else _r
        for _hd, _r in sorted(tmv_rpc.items()):
            if _r is None:
                print(f"[gen] TMV_CTX declined at head_dim={_hd}: TMatVec does not fit L1 at any "
                      f"rows_per_chunk; that geometry keeps the transpose+GEMV context path")
            elif _r != TMV_RPC:
                print(f"[gen] TMatVec rows_per_chunk {TMV_RPC} -> {_r} (L1 fit at head_dim={_hd})")

    # The blocked cache stays a BUILD-WIDE decision even though the tmatvec verdict is not: both
    # arms read and write the same buffers, so a geometry on the fallback path would be addressing a
    # layout it cannot express. Blocking therefore needs EVERY geometry on the tmatvec path.
    _tmv_declined = tuple(sorted(h for h, r in tmv_rpc.items() if r is None))
    KV_BLOCK_ELIGIBLE = GROUPED_K and TMV_CTX and not _tmv_declined
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
        print(f"[gen] packed dump is the authority: {_mdt} g{_mgs} at mlp/attn_o/qkv")

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
    if not FUSE_QKV_DP:
        qkv_dp_why = {g: "FUSE_QKV_DP=0" for g in geoms}
    elif not FUSE_QKV_GEMV:
        qkv_dp_why = {g: "needs FUSE_QKV_GEMV=1 for the concatenated Wqkv" for g in geoms}
    else:
        qkv_dp_why = {g: sp.qkv_dp_reason(COLS, head_dim=g[0]) for g in geoms}
    mlp_dp_why = "FUSE_MLP_DP=0" if not FUSE_MLP_DP else sp.mlp_dp_reason()

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
                        f"needs ONE attention geometry; {sp.name} has {len(geoms)}: {geoms}"
                        if _geom1 is None else
                        qkv_dp_why[_geom1] if qkv_dp_why[_geom1] else
                        mlp_dp_why if mlp_dp_why else
                        "needs FUSE_MLP_O=1 (Wo's padding is wired through that flag via "
                        "op_mlp_dp._wo_rows_padded, and decode_layer_dp always fuses Wo)"
                        if not FUSE_MLP_O else
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
    fuse_o = FUSE_MLP_O and mlp_dp_why is None
    for g in geoms:
        why, tag = qkv_dp_why[g], "" if len(geoms) == 1 else f" [head_dim={g[0]}, kv_heads={g[1]}, v_proj={g[2]}]"
        print(f"[gen] fused arm qkv_head_dp{tag}: {'OFF -- ' + why if why else 'on'}")
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
    _pdtypes, _pkind = precision.packer_capability()
    precision_ctx = precision.GraphContext(
        fused_layer=decode_layer_why is None and FUSE_DECODE_LAYER,
        fuse_o=fuse_o, fused_qkv_gemv=bool(FUSE_QKV_GEMV),
        fused_qkv_dp=qkv_dp_why is None,
        d_model=D, ffn=FF, q_dim=QD, head_dim=HD, attn_cols=COLS,
        packer_dtypes=_pdtypes, packer_takes_scale_kind=_pkind)
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
    _attn_cache = {}

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
        dp_why = qkv_dp_why[(hd, hkv, has_v)]
        op_qk_norm = RMSNorm(size=hd, num_aie_columns=1, num_channels=1, tile_size=hd,
                             weighted=True, epsilon=sp.eps, context=ctx) if sp.qk_norm else None
        op_qk_norm_b = (RMSNorm(size=hd, num_aie_columns=1, num_channels=1, tile_size=hd,
                                weighted=True, epsilon=sp.eps, context=ctx)
                        if (sp.qk_norm and SPLIT_QKNORM) else op_qk_norm)
        # QKV projection: one GEMV over the concatenated weight, or the three separate ones. op_kv
        # is built in both arms because share_designs pairs Wk with Wv only in the unfused one.
        if _spec("qkv").quantized and dp_why is None:
            raise NotImplementedError(
                f"the precision plan sets qkv to {_spec('qkv')} but the fused QKV head is on, and "
                "QKVHeadDataParallel has no weight_dtype axis -- it would consume packed bytes as "
                "bf16 values. Set FUSE_QKV_DP=0, or add the axis to the operator.")
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
            op_qkv_dp = QKVHeadDataParallel(D=D, HD=hd, Hq=Hq, Hkv=hkv, max_seq=KVA,
                                            num_aie_columns=sp.qkv_dp_cols(COLS, n_kv_heads=hkv),
                                            epsilon=sp.eps,
                                            tile_size_input=TSI, context=ctx,
                                            weight_depth=WEIGHT_DEPTH)
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
        if sp.v_norm and dp_why is None:
            # The fused head drains k and v straight into the caches, so `v` never exists as a
            # buffer this graph can normalise -- the norm would be silently skipped rather than
            # rejected, on every layer.
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
        sc = dict(input_sizes=(hkv, hd), input_strides=(hd, 1), input_offset=0,
                  output_sizes=(1, hkv, hd), output_strides=(0, S * hd, 1), output_offset=0,
                  input_buffer_size=hkv * hd, output_buffer_size=hkv * S * hd, num_aie_channels=1)
        op_sck = StridedCopy(**sc, output_offset_parameter=slot, context=ctx)
        # V stays [S][hd]. A transposed append would delete op_trv, but a SINGLE-token transposed
        # write is 1024 isolated bf16 elements (h*hd*S + d*S + p) and the shim address generator
        # steps in 4-byte granules: the BD silently halves the innermost dimension (measured on the
        # emitted descriptor -- d0_size 64 for 128 elements, d0_stride 1023, i.e. 64 granules of two
        # ADJACENT elements). The runtime offset has the same granule floor, so an odd `p` truncates
        # down. The working shape is a PAIR write on an even offset, whose staging cannot itself be
        # a DMA.
        op_scv = StridedCopy(**sc, output_offset_parameter=slot, context=ctx)
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
        op_rep_k = Repeat(rows=hkv, cols=S * hd, repeat=gqa, transfer_size=hd, context=ctx)
        op_rep_v = Repeat(rows=hkv, cols=S * hd, repeat=gqa, transfer_size=hd, context=ctx)
        # GQA's own group_reuse gate (gemv/design.py) DECLINES batch_group > MAX_GROUP_REUSE (a
        # measured shim-BD ceiling) and falls back to a stride-0 outer BD that re-reads the whole
        # matrix once per query head -- measured 14.71x on Gemma-4's global layers (hkv=1,
        # gqa=Hq=16), 270.01 MB/token against 18.35 MB unique. Below the ceiling (sliding, gqa=2)
        # this is unreachable and op_scores is unchanged.
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
            scores_groups = gqa // MAX_GROUP_REUSE
            op_scores = gemv(S, hd, ctx, num_batches=MAX_GROUP_REUSE, batch_group=MAX_GROUP_REUSE,
                                 block_size=T, alloc_M=None if KVA == S else KVA)
        else:
            scores_groups = 1
            op_scores = gemv(S, hd, ctx, num_batches=Hq, batch_group=gqa if GROUPED_K else 1,
                                 block_size=T, alloc_M=None if KVA == S else KVA)
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
        op_trv = Transpose(M=S, N=hd, num_aie_columns=4, num_channels=1, m=256, n=32, s=8,
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
        rpc = tmv_rpc.get(hd)
        uses_tmv = rpc is not None
        if uses_tmv:
            op_ctx = TMatVec(M=hd, K=S, num_aie_columns=hkv, num_batches=Hq, batch_group=gqa,
                                 alloc_K=None if KVA == S else KVA, block_size=T,
                             rows_per_chunk=rpc, context=ctx)
        else:
            op_ctx = gemv(hd, S, ctx, num_batches=Hq)
        g = SimpleNamespace(
            hd=hd, hkv=hkv, qd=qd, kvd=kvd, gqa=gqa, kv_slot=slot, o_chunks=o_chunks,
            op_qk_norm=op_qk_norm, op_qk_norm_b=op_qk_norm_b, op_qkv=op_qkv, op_q=op_q,
            op_kv=op_kv, op_o=op_o, op_rope_qk=op_rope_qk, op_qkv_dp=op_qkv_dp,
            op_rope_q=op_rope_q, op_rope_k=op_rope_k, op_sck=op_sck, op_scv=op_scv,
            op_rep_k=op_rep_k, op_rep_v=op_rep_v, op_scores=op_scores, op_trv=op_trv,
            op_ctx=op_ctx, op_v_norm=op_v_norm, has_v=has_v, kv_parts=kv_parts,
            uses_tmv_ctx=uses_tmv, scores_groups=scores_groups)
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
        op_scores = gemv(S, HD, ctx, num_batches=Hq,
                         batch_group=sp.gqa_group if GROUPED_K else 1, block_size=T,
                         alloc_M=None if KVA == S else KVA)
        # Folded into n_qn when SCALE_IN_QNORM; built anyway so the A/B arm stays reachable.
        op_scale = (None if scale_in_qnorm else
                    ElementwiseMul(size=Hq * S, tile_size=S // COLS, num_aie_columns=COLS,
                                   context=ctx))
        # Not COLS: with fewer q heads than columns each core gets less than one tile and the
        # op computes nothing (IRON raises). Gemma-3's 4 heads run at 4 columns.
        op_softmax = Softmax(rows=Hq, cols=S, num_aie_columns=sp.softmax_cols(COLS),
                             num_channels=1, rtp_vector_size=S,
                             vector_size_parameter="sm_mask", context=ctx)
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
        if TMV_CTX:
            # TMV_RPC is a CAP, not the value: the largest chunk that fits L1 depends on head_dim, and
            # 64 is right for Qwen3's HD=128 and too big for Gemma-3's 256. check_l1_fits is the
            # operator's own arithmetic, so ask it rather than carrying a second copy of the L1 model
            # here -- or an env constant that was correct for one model and silently wrong for the next.
            from iron.operators.tmatvec.design import check_l1_fits
            rpc = TMV_RPC
            while rpc > 1 and (S % rpc or check_l1_fits(HD, S, sp.gqa_group, rpc) is not None):
                rpc //= 2
            if rpc != TMV_RPC:
                print(f"[gen] TMatVec rows_per_chunk {TMV_RPC} -> {rpc} (L1 fit at head_dim={HD})")
            op_ctx = TMatVec(M=HD, K=S, num_aie_columns=Hkv, num_batches=Hq,
                             batch_group=sp.gqa_group, alloc_K=None if KVA == S else KVA,
                             rows_per_chunk=rpc, context=ctx, block_size=T)
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
        op_mlp_dp = SwiGLUMLPDataParallel(D=D, FF=FF, num_aie_columns=MLP_DP_COLS,
                                          epsilon=sp.eps,
                                          QD=QD if fuse_o else None, fuse_o=fuse_o,
                                          context=ctx, weight_depth=WEIGHT_DEPTH,
                                          tile_rows_gu=MLP_TILE_ROWS,
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

    if not fuse_act:
        if sp.act == "silu":
            op_act = SiLU(size=FF, num_aie_columns=COLS, tile_size=FF // COLS, context=ctx)
        else:
            op_act = GELU(size=FF, num_aie_columns=COLS, num_channels=1, tile_size=FF // COLS, context=ctx)
    op_mul_ffn = ElementwiseMul(size=FF, tile_size=FF // COLS, num_aie_columns=COLS, context=ctx)
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
    op_lscale = (ElementwiseMul(size=D, tile_size=D // COLS, num_aie_columns=COLS, context=ctx)
                 if sp.layer_scalar else None)

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
             f"{p}sc[{i*n*S*2}:{(i+1)*n*S*2}]")
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
        for _hd in sorted({gk[0] for gk in geoms}):
            weights[f"ones_h{_hd}"] = np.ones(_hd, dtype=BF16)
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
            if {"Wd": down_chunks, "Wo": g.o_chunks}.get(key, 1) > 1:
                # Each chunk is its own contiguous tensor. A pre-chunked dump names them
                # `<tensor>.kchunkN` and we take those bytes as-is; otherwise the split happens
                # here, along K, BEFORE quantizing -- so each chunk carries its own per-group
                # scales, exactly as the kernel reads it. Splitting AFTER packing would cut
                # through a group.
                nch = {"Wd": down_chunks, "Wo": g.o_chunks}[key]
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
            w = npy(hf)  # [M, K], f32
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
            if (op_decode_layer is not None and op_decode_layer.wqkv_head_major
                    and g.has_v):
                # attn_block_dp wqkv_head_major: one contiguous run of (gqa+2) hd-row
                # blocks per core. Per GEOMETRY -- gqa and the row height are g.hd/g.hkv,
                # not the spec-wide pair, or the reorder shuffles row fragments.
                gqa_ = Hq // g.hkv
                row_w = precision.wire_row_units(_spec("qkv"), D)
                wq2, wk2, wv2 = (a.reshape(-1, row_w) for a in qkv_parts)
                parts = []
                for c in range(g.hkv):
                    parts += [wq2[(gqa_ * c + gi) * g.hd:(gqa_ * c + gi + 1) * g.hd]
                              for gi in range(gqa_)]
                    parts += [wk2[c * g.hd:(c + 1) * g.hd], wv2[c * g.hd:(c + 1) * g.hd]]
                weights[p + "Wqkv"] = np.concatenate(parts, axis=0).reshape(-1)
            else:
                weights[p + "Wqkv"] = np.concatenate(qkv_parts)
        # Size is layout-independent (T < S rearranges the same elements), but it is
        # PER GEOMETRY: Gemma-4 global layers are hkv=1/hd=512 against sliding 8/256.
        _kvl = KVLayout(Hkv=g.hkv, S=S, HD=g.hd, T=T)
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
            p + "kc": g.hkv * S * g.hd * 2, p + "vc": g.hkv * S * g.hd * 2,
            p + "sc": Hq * S * 2, p + "sw": Hq * S * 2,
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
            bufsz[p + "kr"] = Hq * S * g.hd * 2
        if not (GROUPED_V or g.uses_tmv_ctx):
            bufsz[p + "vr"] = Hq * S * g.hd * 2
        if not g.uses_tmv_ctx:
            bufsz[p + "vt"] = Hq * S * g.hd * 2
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
            rl += [
                *head,
                *([] if GROUPED_K else [(g.op_rep_k, p + "kc", p + "kr")]),
                # TMV_CTX subsumes the v-side grouping: TMatVec reads vc per kv head itself, so a
                # Repeat would materialise a `vr` nothing consumes.
                *([] if (GROUPED_V or g.uses_tmv_ctx) else [(g.op_rep_v, p + "vc", p + "vr")]),
                *scores_runlist(p, g, ref_q),
                *([] if scale_in_qnorm else [(op_scale, p + "sc", "attn_scale", p + "sc")]),
                (op_softmax, p + "sc", p + "sw"),
                *([] if g.uses_tmv_ctx else
                  [(g.op_trv, p + ("vc" if GROUPED_V else "vr"), p + "vt")]),
                (g.op_ctx, p + ("vc" if g.uses_tmv_ctx else "vt"), p + "sw", p + "cx"),
                *([] if fuse_o else o_runlist(p, g)),
            ]
            if sp.sandwich_norms:
                rl.append((op_norm, p + "a", p + "n_pa", p + "a"))
            if op_mlp_dp is not None:
                # cur + a -> x1 -> norm -> gate/up -> silu -> mul -> down -> +x1, all inside one design.
                # x1/hf/g/u/gh/d never reach DDR; `mlp_gh` is the all-gather round-trip buffer and is
                # shared across layers because the sequence runs them one at a time. FUSE_MLP_O folds
                # `a = Wo @ cx` in too: `cx`/`Wo` replace `a` as the design's own inputs, and
                # `mlp_a_scratch` is a's own all-gather round-trip buffer, the same idiom as mlp_gh's.
                if fuse_o:
                    rl.append((op_mlp_dp, cur, p + "cx", p + "n_pf", p + "Wo", p + "Wg", p + "Wu",
                               p + "Wd", "mlp_gh", "mlp_a_scratch", nxt))
                else:
                    rl.append((op_mlp_dp, cur, p + "a", p + "n_pf", p + "Wg", p + "Wu", p + "Wd",
                               "mlp_gh", nxt))
            else:
                rl += [
                    (op_add, cur, p + "a", p + "x1"),
                    (op_norm, p + "x1", p + "n_pf", p + "hf"),
                    (op_gate, p + "Wg", p + "hf", p + "g"),
                    (op_up, p + "Wu", p + "hf", p + "u"),
                    *([] if op_act is None else [(op_act, p + "g", p + "g")]),
                    (op_mul_ffn, p + "g", p + "u", p + "gh"),
                    *down_runlist(p),
                ]
            if sp.sandwich_norms:
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
        weights["W_head"] = _pack(embed_f32, "head")
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
                            decode_layer_active=op_decode_layer is not None, T=T)
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
        segments.append(dict(seq=seq, layers=(la, lb), inlet=seg_in, outlet=seg_out,
                             weights=seg_weights, caches=seg_caches, inputs=seg_inputs,
                             kv_slots=seg_kv_slots))
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
            f"{sequence_name(sp, NL, S, placer_flags, tmv_declined=_tmv_declined)}_lmhead", head_rl,
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
                                    kv_slots=kv_slots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS), help="model spec name")
    ap.add_argument("--weights", required=True, help="dir of dumped .npy weights (see dump_llm_weights.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (bring-up)")
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
            "clip_search": any(v.scale_kind == "clip" for v in PRECISION_PLAN.values()),
        },
    }
    prov = toolchain_provenance()
    if prov:
        meta["toolchain"] = prov
    else:
        print("[build] WARNING: could not record toolchain provenance in meta.json "
              "(no toolchain.lock / kernel_sandbox.sh resolvable) -- this artifact will read as "
              "unstamped to any freshness check", file=sys.stderr)
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
