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

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import SPECS  # noqa: E402

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
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemv.op import GEMV  # noqa: E402
from iron.operators.gemv.quant import quantize_weight  # noqa: E402
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


# Engineering-check MLP weight quantization axis (Wg/Wu/Wd -- the "MLP weights" byte class), gated
# by env vars so build/verify/bench need no CLI plumbing to A/B it, matching DECODE_PLACER_FLAGS'
# convention. QUANT_MLP_DTYPE="bf16" (default) is a no-op: every GEMV byte-for-byte unchanged.
# NOT a quality claim: this axis is validated as a byte-stream + determinism engineering check on
# Qwen3-0.6B, not a token-quality gate (tests/refs/qwen3-0.6b/bf16_oracle.json is 1 prompt / 8
# free-running tokens with knife-edge logit margins -- too small to see quantization damage).
# Accepted values: "bf16" (default, no-op), the SYMMETRIC "int4"/"int8" (w = q*s), and the
# AFFINE "int4a"/"int8a" (w = q*s + m, a bf16 scale and a bf16 min per group -- GGUF Q4_1's
# shape, and what FastFlowLM's shipped codec stores). Affine costs the same bytes as symmetric
# at the same nominal width once the f32 scale is dropped, and is measured better everywhere;
# see iron/operators/gemv/quant.py for the layout and the byte arithmetic.
_QUANT_DTYPES = ("bf16", "int4", "int8", "int4a", "int8a")
QUANT_MLP_DTYPE = os.environ.get("QUANT_MLP_DTYPE", "bf16")
QUANT_MLP_GROUP = int(os.environ.get("QUANT_MLP_GROUP", "128"))

# Same axis, same GEMV(weight_dtype=...) mechanism, applied to Wo (attention output projection,
# "Wo" -- the "attention weights" byte class) instead of the MLP. Independent env vars so an A/B
# can quantize Wo without touching Wg/Wu/Wd, and vice versa. Same caveat as QUANT_MLP_DTYPE: an
# engineering-check byte-stream axis, not a validated model default.
QUANT_ATTN_DTYPE = os.environ.get("QUANT_ATTN_DTYPE", "bf16")
QUANT_ATTN_GROUP = int(os.environ.get("QUANT_ATTN_GROUP", "128"))

# Same axis again, applied to W_head, the FINAL lm-head GEMV's weight.
#
# W_head IS THE TIED EMBEDDING TABLE, not an independent lm-head weight -- built below from
# `model.embed_tokens.weight` (Qwen3 ties them), and rust/npu-engine's NpuDecodeStep gathers the
# next step's `embed[token]` out of a bf16 [vocab, d_model] blob. Quantizing splits the tensor
# across its two consumers: the device GEMV reads the packed W_head, while `meta.json`'s
# `embed_blob` points the host gather at a bf16 W_embed sidecar emitted beside it (npu_decode.rs
# resolves that field and size-gates whichever blob it names). The sidecar costs 311 MB of disk
# and ZERO device arena -- the host faults in one 2 KB row per token -- and it holds the embedding
# INPUT at full width, so this axis moves the lm-head projection alone.
QUANT_HEAD_DTYPE = os.environ.get("QUANT_HEAD_DTYPE", "bf16")
QUANT_HEAD_GROUP = int(os.environ.get("QUANT_HEAD_GROUP", "128"))

# Scale-selection method shared by every quantized weight class above. Default takes each group's
# scale from its absmax; 1 grid-searches the clip ratio minimising that group's reconstruction MSE.
# Host-side only -- same wire format, same kernel -- so it A/Bs against a shipped artifact.
QUANT_CLIP_SEARCH = os.environ.get("QUANT_CLIP_SEARCH", "0") != "0"

for _n, _v in (("QUANT_MLP_DTYPE", QUANT_MLP_DTYPE), ("QUANT_ATTN_DTYPE", QUANT_ATTN_DTYPE),
               ("QUANT_HEAD_DTYPE", QUANT_HEAD_DTYPE)):
    # Fail here rather than at link. An unknown value reaches design.py as part of a kernel
    # symbol name (matvec_vectorized_<dtype>_bf16) and an archive name, so a typo currently
    # surfaces as "undefined symbol" after a full compile.
    if _v not in _QUANT_DTYPES:
        raise SystemExit(f"{_n}={_v!r} is not one of {_QUANT_DTYPES}")

DECODE_PLACER_FLAGS_DEFAULT = "--cores-per-col 1"

# INSTRUMENT, not a feature. Alternates the per-head qk-norm between two IDENTICAL RMSNorm
# instances. RMSNorm has no design_key, so two instances are two DESIGNS: the 24 consecutive runs
# stop sharing one aiex.configure and become 24. Runs, bytes and output are unchanged, so it
# isolates the cost of a CHEAP configure (18 KB of views) the way share_designs isolated an
# expensive one. Predicted +644 configures/token; at the measured 61.9 us for a big configure that
# is +39.9 ms if the cost is flat, and ~0 if it tracks the view count.
SPLIT_QKNORM = os.environ.get("SPLIT_QKNORM", "0") == "1"

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
# Weight tile ROWS for the fused MLP. Trades against WEIGHT_DEPTH at constant L1.
MLP_TILE_ROWS = int(os.environ.get("MLP_TILE_ROWS", "0"))

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
# OFF by default: device-free it PLACES (aiecc, 2026-09-09) and its shim-BD census is lighter than
# the five designs it replaces, but nothing has run it on hardware -- neither placement nor a byte
# census can see a wrong answer or a runtime-sequence deadlock.
FUSE_DECODE_LAYER = os.environ.get("FUSE_DECODE_LAYER", "0") == "1"


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



def sequence_name(sp, NL, S, placer_flags, decode_layer_active=False):
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
    if not TMV_CTX:
        parts.append("noctx")
    if not GROUPED_K:
        parts.append("nogk")
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
    if QUANT_MLP_DTYPE != "bf16":
        parts.append(f"{QUANT_MLP_DTYPE}g{QUANT_MLP_GROUP}")
    if QUANT_ATTN_DTYPE != "bf16":
        parts.append(f"attn{QUANT_ATTN_DTYPE}g{QUANT_ATTN_GROUP}")
    if QUANT_HEAD_DTYPE != "bf16":
        parts.append(f"head{QUANT_HEAD_DTYPE}g{QUANT_HEAD_GROUP}")
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
    """
    import atexit
    import shutil
    import tempfile

    explicit = os.environ.get("DECODE_WORK")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        os.chdir(explicit)
        print(f"[{tag}] build dir {explicit} (DECODE_WORK, kept)", flush=True)
        return explicit
    work = tempfile.mkdtemp(prefix=f"{tag}-")
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


L1_BYTES = 65536      # AIE2P core local memory (getLocalMemorySize(), AIETargetModel.h)
L1_RESERVE = 8192     # stack + the allocator's own slack; measured headroom, not a guess (see below)
C_TILE_GRANULE = 8    # tile_size_output must be a multiple of this (16 bytes of bf16); see gemv_tile_output


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
    # Search every m_input and keep the largest legal C TILE, not the largest m_input. Preferring
    # m_input first is a trap: at the lm-head shape it accepts (4, 16) -- legal, correct, and 1187
    # tiles per column -- while (2, 9496) exists one step down and is 593x fewer tiles. A smaller
    # m_input frees L1 budget quadratically faster than it costs, because A is m_input x K while C
    # is just m_output.
    best_pair = (0, 0)
    for cand_tsi in ([tsi] if tsi is not None else (4, 2, 1)):
        if per_col % cand_tsi:
            continue
        budget = L1_BYTES - L1_RESERVE - 2 * (cand_tsi * K * 2) - 2 * (K * 2)
        if budget <= 0:
            continue
        cap = budget // 4                  # C is double-buffered, 2 bytes per element
        # THIRD constraint, and nothing in the toolchain checks it: the C tile must be a multiple of
        # C_TILE_GRANULE elements. MEASURED 2026-09-03 with a standalone one-GEMV repro at the
        # lm-head shape -- M=151936 K=1024 with tso=4748 (4748 % 8 == 4) returns a PERMUTATION of the
        # right answer: values correct (sorted rel-L2 7.9e-3) in wrong positions (rel-L2 1.40). The
        # SAME M and K with tso=9496 is correct at 2.8e-3, and 1024/512/256 are correct at M=8192.
        # Every passing tile is a multiple of 8; the one failing tile is not. It is silent -- the
        # build succeeds and the argmax is simply wrong -- so it has to be refused here.
        best = max((d for d in range(cand_tsi, per_col + 1, cand_tsi)
                    if per_col % d == 0 and d <= cap and d % C_TILE_GRANULE == 0), default=0)
        if best > best_pair[1]:
            best_pair = (cand_tsi, best)
    if best_pair[1]:
        return best_pair
    raise ValueError(f"GEMV M={M} K={K}: no (tile_size_input, tile_size_output) fits L1 "
                     f"({L1_BYTES} B) with M//cols={per_col} and tile_size_output a multiple of "
                     f"{C_TILE_GRANULE}")


def gemv(M, K, ctx, **kw):
    """GEMV tiled as large as both the design asserts AND L1 allow."""
    tsi, tso = gemv_tile_output(M, K)
    return GEMV(M=M, K=K, num_aie_columns=COLS, tile_size_input=tsi,
                tile_size_output=tso, context=ctx, **kw)


def build_graph(spec_name, weights_dir, layers=None, max_seq=2048):
    """Construct the fused decode graph + its weight dict for a spec.

    Shared by the generator CLI and verify_llm_decode.py so the harness drives the SAME graph the
    artifact was built from, rather than a re-typed copy that can drift from it.
    Returns (spec, fused, weights, meta_dims).
    """
    sp = SPECS[spec_name]
    sp.check(cols=COLS, tsi=TSI)
    sp.check_seq(max_seq)
    NL = layers if layers is not None else sp.n_layers
    S = max_seq
    D, FF, HD = sp.d_model, sp.ffn, sp.head_dim
    Hq, Hkv, QD, KVD, VOCAB = sp.n_q_heads, sp.n_kv_heads, sp.q_dim, sp.kv_dim, sp.vocab

    def npy(name):
        return np.load(os.path.join(weights_dir, f"{name}.npy")).astype(np.float32)

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
    op_qk_norm = RMSNorm(size=HD, num_aie_columns=1, num_channels=1, tile_size=HD,
                         weighted=True, epsilon=sp.eps, context=ctx) if sp.qk_norm else None
    op_qk_norm_b = (RMSNorm(size=HD, num_aie_columns=1, num_channels=1, tile_size=HD,
                            weighted=True, epsilon=sp.eps, context=ctx)
                    if (sp.qk_norm and SPLIT_QKNORM) else op_qk_norm)
    # QKV projection: one GEMV over the concatenated weight, or the three separate ones. op_kv is
    # built in both arms because share_designs pairs Wk with Wv only in the unfused one.
    # Wo weight-stream dtype axis (see QUANT_ATTN_DTYPE above). bf16 (default) is byte-for-byte the
    # pre-existing path.
    attn_quant_kw = (dict(weight_dtype=QUANT_ATTN_DTYPE, group_size=QUANT_ATTN_GROUP)
                     if QUANT_ATTN_DTYPE != "bf16" else {})
    # ---- which fused arms this MODEL can use ----
    # Whether a fused arm applies is the OPERATOR's rule, not a choice here -- the same shape as
    # fuse_act further down, which already asks the GEMV instead of assuming. The env flag can only
    # turn an arm OFF (for A/B); it can no longer turn one ON for a spec the operator does not
    # cover. It used to, and a Qwen-shaped default then met Gemma-3 as a NotImplementedError three
    # frames down -- a capability gap reported as a crash, and only after the previous gap was
    # cleared, so the four of them surfaced one build at a time.
    qkv_dp_why = ("FUSE_QKV_DP=0" if not FUSE_QKV_DP else
                  "needs FUSE_QKV_GEMV=1 for the concatenated Wqkv" if not FUSE_QKV_GEMV else
                  sp.qkv_dp_reason(COLS))
    mlp_dp_why = "FUSE_MLP_DP=0" if not FUSE_MLP_DP else sp.mlp_dp_reason()
    fuse_o = FUSE_MLP_O and mlp_dp_why is None
    for arm, why in (("qkv_head_dp", qkv_dp_why), ("swiglu_mlp_dp", mlp_dp_why)):
        print(f"[gen] fused arm {arm}: {'OFF -- ' + why if why else 'on'}")
    if fuse_o:
        if QUANT_ATTN_DTYPE != "bf16":
            # Under fuse_o, Wo rides the MLP design's single weight ObjectFifo, and one fifo
            # carries one wire format. So Wo's dtype is QUANT_MLP_DTYPE's, not its own axis --
            # QUANT_ATTN_DTYPE would silently mean nothing here rather than a little.
            raise NotImplementedError(
                "FUSE_MLP_O folds Wo into swiglu_mlp_dp's shared weight channel, so Wo takes "
                f"QUANT_MLP_DTYPE ({QUANT_MLP_DTYPE!r}), not QUANT_ATTN_DTYPE "
                f"({QUANT_ATTN_DTYPE!r}); set FUSE_MLP_O=0 to quantize Wo independently"
            )
    op_qkv = gemv(QD + 2 * KVD, D, ctx) if FUSE_QKV_GEMV else None
    op_q = gemv(QD, D, ctx)
    op_kv = gemv(KVD, D, ctx)
    op_o = None if fuse_o else gemv(D, QD, ctx, **attn_quant_kw)
    # RoPE over q and k together (24 head rows) needs them adjacent, which only the fused qkv
    # buffer gives; angle_rows=1 is unchanged, so every row still reads the same single angle row.
    fuse_rope = FUSE_QKV_GEMV and FUSE_ROPE_QK
    op_rope_qk = RoPE(rows=Hq + Hkv, cols=HD, angle_rows=1, context=ctx) if fuse_rope else None
    op_qkv_dp = None
    if qkv_dp_why is None:
        from iron.operators.qkv_head_dp.op import QKVHeadDataParallel
        op_qkv_dp = QKVHeadDataParallel(D=D, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S,
                                        num_aie_columns=sp.qkv_dp_cols(COLS), epsilon=sp.eps,
                                        tile_size_input=TSI, context=ctx,
                                        weight_depth=WEIGHT_DEPTH)
    op_rope_q = RoPE(rows=Hq, cols=HD, angle_rows=1, context=ctx)
    op_rope_k = RoPE(rows=Hkv, cols=HD, angle_rows=1, context=ctx)
    # KV append: deep-C scratchpad offset "kv_off" (element units = n_past*HD), constant ELF.
    sc = dict(input_sizes=(Hkv, HD), input_strides=(HD, 1), input_offset=0,
              output_sizes=(1, Hkv, HD), output_strides=(0, S * HD, 1), output_offset=0,
              input_buffer_size=Hkv * HD, output_buffer_size=Hkv * S * HD, num_aie_channels=1)
    op_sck = StridedCopy(**sc, output_offset_parameter="kv_off", context=ctx)
    # V stays [S][HD]. A transposed append would delete op_trv, but a SINGLE-token transposed write
    # is 1024 isolated bf16 elements (h*HD*S + d*S + p) and the shim address generator steps in
    # 4-byte granules: the BD silently halves the innermost dimension (measured on the emitted
    # descriptor -- d0_size 64 for 128 elements, d0_stride 1023, i.e. 64 granules of two ADJACENT
    # elements). The runtime offset has the same granule floor, so an odd `p` truncates down.
    # The working shape is a PAIR write on an even offset, whose staging cannot itself be a DMA.
    op_scv = StridedCopy(**sc, output_offset_parameter="kv_off", context=ctx)
    # GQA broadcast. Correctness-first; the byte-free form is a batch-stride-0 GEMV read of the kv
    # head (0 ops, 0 bytes) -- at Hq=16 x 28 layers this Repeat plus the V transpose are 41% of the
    # per-token DDR budget, so it is the first optimisation after parity, not an afterthought.
    #
    # S below is deliberately ONE value shared by kc/vc/kr/vr/vt/sc/sw, op_scores, op_rep_k/v, op_trv
    # AND op_ctx -- not the op_ctx-excluded 4-of-5 split llm-decode-attention-pads-to-full-window.md
    # scoped out device-free. That split needs op_trv to write a bucket-wide `vt` while op_ctx reads
    # it at full max_seq width, and symmetrically op_rep_k/v to read a bucket-wide prefix of a kc/vc
    # row whose true stride is max_seq*HD (Hkv=8 here, not a degenerate single-row case where prefix
    # == whole buffer). Neither holds with today's operators: Repeat's input TensorAccessPattern
    # ties its row stride directly to `cols` (repeat/design.py: strides=[0, cols, cols_split, 1]),
    # and Transpose's output stride is tied to its own `M` (transpose/design.py: taps_out_L1L3
    # strides derive from M) -- neither exposes a stride independent of its own declared size, so
    # "read/write a narrower window of a wider-strided buffer" is new IRON capability, not a
    # generator change. Bucketing S UNIFORMLY (this build already takes it as `max_seq`) is the
    # route that needs none.
    op_rep_k = Repeat(rows=Hkv, cols=S * HD, repeat=sp.gqa_group, transfer_size=HD, context=ctx)
    op_rep_v = Repeat(rows=Hkv, cols=S * HD, repeat=sp.gqa_group, transfer_size=HD, context=ctx)
    op_scores = gemv(S, HD, ctx, num_batches=Hq,
                     batch_group=sp.gqa_group if GROUPED_K else 1)
    # Folded into n_qn when SCALE_IN_QNORM; built anyway so the A/B arm stays reachable.
    scale_in_qnorm = SCALE_IN_QNORM and sp.qk_norm
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
                         batch_group=sp.gqa_group,
                         rows_per_chunk=rpc, context=ctx)
    else:
        op_ctx = gemv(HD, S, ctx, num_batches=Hq)
    # MLP weight-stream dtype axis (Wg/Wu/Wd -- "MLP weights" in the byte breakdown, the largest
    # single weight class). bf16 (default) is byte-for-byte the pre-existing path; QUANT_MLP_DTYPE
    # is an engineering-check toggle (see its definition above), not a quality-validated default.
    mlp_quant_kw = (dict(weight_dtype=QUANT_MLP_DTYPE, group_size=QUANT_MLP_GROUP)
                    if QUANT_MLP_DTYPE != "bf16" else {})
    # The activation runs on the gate projection's output, immediately after it and before anything
    # else reads `g`, so folding it into that GEMV's epilogue preserves the order exactly.
    # Two things can veto the fold, and both are the operator's own rules rather than choices here:
    # the epilogue walks the C tile 32 lanes at a time, and GEMV refuses an epilogue on a quantized
    # weight stream (untested combination, not a hardware conflict). gemv() picks tile_size_output
    # itself, so ask it rather than assuming FF // COLS.
    _gate_tso = gemv_tile_output(FF, D)[1]
    fuse_act = (
        FUSE_ACT
        and QUANT_MLP_DTYPE == "bf16"
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
    # Eligibility is the union of qkv_dp_why/mlp_dp_why (the spec-shape rules attn_block_dp and
    # swiglu_mlp_dp already check) plus what is true only of the MERGED device: attn_block_dp's own
    # Hkv==COLS rule, no sandwich norms (the op has no post-attn/post-ffn norm slot), and no
    # quantized weight stream (its kernel archive is plain bf16 mv.cc, built once, not per weight
    # dtype -- see get_kernel_artifacts in iron/operators/decode_layer_dp/op.py).
    op_decode_layer = None
    decode_layer_why = ("FUSE_DECODE_LAYER=0" if not FUSE_DECODE_LAYER else
                        qkv_dp_why if qkv_dp_why else
                        mlp_dp_why if mlp_dp_why else
                        "needs FUSE_MLP_O=1 (Wo's padding is wired through that flag via "
                        "op_mlp_dp._wo_rows_padded, and decode_layer_dp always fuses Wo)"
                        if not FUSE_MLP_O else
                        f"needs Hkv ({Hkv}) == COLS ({COLS})" if Hkv != COLS else
                        "needs SCALE_IN_QNORM=1 (attn_block_dp has no separate scale stage)"
                        if not (SCALE_IN_QNORM and sp.qk_norm) else
                        "needs GQA_GROUPED_K=1 and TMV_CTX=1 (attn_block_dp computes exactly "
                        "that variant internally)" if not (GROUPED_K and TMV_CTX) else
                        "needs QUANT_MLP_DTYPE=bf16 and QUANT_ATTN_DTYPE=bf16 (plain-bf16 kernel "
                        "archive, no quantized-weight variant)"
                        if QUANT_MLP_DTYPE != "bf16" or QUANT_ATTN_DTYPE != "bf16" else None)
    if decode_layer_why is None:
        from iron.operators.decode_layer_dp.op import DecodeLayerDataParallel
        op_decode_layer = DecodeLayerDataParallel(
            D=D, FF=FF, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S, attn_cols=Hkv, mlp_cols=MLP_DP_COLS,
            eps_attn=sp.eps, eps_mlp=sp.eps, tile_size_input=TSI, context=ctx,
            weight_depth=WEIGHT_DEPTH, wqkv_head_major=True)
    print(f"[gen] fused arm decode_layer_dp: "
          f"{'OFF -- ' + decode_layer_why if decode_layer_why else 'on'}")

    if not fuse_act:
        if sp.act == "silu":
            op_act = SiLU(size=FF, num_aie_columns=COLS, tile_size=FF // COLS, context=ctx)
        else:
            op_act = GELU(size=FF, num_aie_columns=COLS, num_channels=1, tile_size=FF // COLS, context=ctx)
    op_mul_ffn = ElementwiseMul(size=FF, tile_size=FF // COLS, num_aie_columns=COLS, context=ctx)
    op_down = gemv(D, FF, ctx, **mlp_quant_kw)
    op_add = ElementwiseAdd(size=D, tile_size=D // COLS, num_aie_columns=COLS, context=ctx)
    # W_head weight-stream dtype axis (see QUANT_HEAD_DTYPE above -- READ THE TIED-EMBEDDING NOTE
    # before turning this on).
    head_quant_kw = (dict(weight_dtype=QUANT_HEAD_DTYPE, group_size=QUANT_HEAD_GROUP)
                     if QUANT_HEAD_DTYPE != "bf16" else {})
    op_head = gemv(VOCAB, D, ctx, **head_quant_kw)

    weights, bufsz, cache_names, rl = {}, {}, [], []
    cur = "x"

    for l in range(NL):
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
            w = npy(f"model.layers.{l}.{tensor}.weight")  # [M, K], f32
            if key in mlp_keys and QUANT_MLP_DTYPE != "bf16":
                weights[p + key] = quantize_weight(w, QUANT_MLP_GROUP, QUANT_MLP_DTYPE, clip_search=QUANT_CLIP_SEARCH)
            elif key == "Wo" and QUANT_ATTN_DTYPE != "bf16":
                weights[p + key] = quantize_weight(w, QUANT_ATTN_GROUP, QUANT_ATTN_DTYPE, clip_search=QUANT_CLIP_SEARCH)
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
                weights[p + key] = (
                    quantize_weight(w_padded, QUANT_MLP_GROUP, QUANT_MLP_DTYPE, clip_search=QUANT_CLIP_SEARCH)
                    if QUANT_MLP_DTYPE != "bf16" else bf16(w_padded).reshape(-1)
                )
            elif key in qkv_keys and FUSE_QKV_GEMV:
                qkv_parts.append(bf16(w).reshape(-1))     # row-major, so concatenation IS stacking
            else:
                weights[p + key] = bf16(w).reshape(-1)
        if FUSE_QKV_GEMV:
            assert len(qkv_parts) == 3, f"expected Wq, Wk, Wv; got {len(qkv_parts)}"
            if op_decode_layer is not None and op_decode_layer.wqkv_head_major:
                # attn_block_dp/design.py's wqkv_head_major=True core c reads ONE contiguous run
                # of (gqa+2) HD-row blocks: its gqa query heads, then its own k head, then its own
                # v head -- a build-time numpy reorder of the stock [Wq|Wk|Wv] rows, same bytes.
                gqa = Hq // Hkv
                wq2, wk2, wv2 = (a.reshape(-1, D) for a in qkv_parts)
                parts = []
                for c in range(Hkv):
                    parts += [wq2[(gqa * c + g) * HD:(gqa * c + g + 1) * HD] for g in range(gqa)]
                    parts += [wk2[c * HD:(c + 1) * HD], wv2[c * HD:(c + 1) * HD]]
                weights[p + "Wqkv"] = np.concatenate(parts, axis=0).reshape(-1)
            else:
                weights[p + "Wqkv"] = np.concatenate(qkv_parts)
        weights[p + "kc"] = np.zeros(Hkv * S * HD, BF16)
        weights[p + "vc"] = np.zeros(Hkv * S * HD, BF16)
        cache_names += [p + "kc", p + "vc"]
        ang = "rope_global" if sp.is_global(l) else "rope_local"

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
            bufsz[p + "kc"] = Hkv * S * HD * 2
            bufsz[p + "vc"] = Hkv * S * HD * 2
            bufsz[p + "cx"] = QD * 2
            rl.append((op_decode_layer, cur, p + "norms", p + "Wqkv", ang, p + "kc", p + "vc",
                       p + "cx", p + "n_pf", p + "Wo", p + "Wg", p + "Wu", p + "Wd",
                       "mlp_gh", "mlp_a_scratch", nxt))
        else:
            # q/k/v are byte slices of ONE `qkv` buffer in the fused arm -- op_qkv writes all three
            # in one pass, and q|k adjacency is what lets a single RoPE cover both. Declared with an
            # explicit size because a parent that is only ever referenced sliced has no arg spec to
            # take its length from (iron/common/sequence.py: calculate_buffer_layout).
            if op_qkv_dp is not None:
                # The fused head appends k and v to the caches itself, so neither ever becomes an L3
                # buffer and only `q` survives as an intermediate.
                ref_q = p + "q"
                bufsz[ref_q] = QD * 2
            elif FUSE_QKV_GEMV:
                qkvb, kb, vb = p + "qkv", QD * 2, (QD + KVD) * 2
                ref_q, ref_k = f"{qkvb}[0:{kb}]", f"{qkvb}[{kb}:{vb}]"
                ref_v, ref_qk = f"{qkvb}[{vb}:{vb + KVD * 2}]", f"{qkvb}[0:{vb}]"
                qhb, qho, khb, kho = qkvb, 0, qkvb, kb    # per-head qk-norm slice base + byte offset
                bufsz[qkvb] = (QD + 2 * KVD) * 2
            else:
                ref_q, ref_k, ref_v = p + "q", p + "k", p + "v"
                qhb, qho, khb, kho = p + "q", 0, p + "k", 0
                bufsz.update({p + "q": QD * 2, p + "k": KVD * 2, p + "v": KVD * 2})
            # kr/vr/vt are the GQA-broadcast and V-transpose intermediates, and each exists ONLY in the
            # arm whose op writes it. Declaring them unconditionally allocated them anyway: an entry in
            # `buffer_sizes` that no runlist op references still lands in the scratch arena, because
            # calculate_buffer_layout appends every explicit buffer not already placed
            # (iron/common/sequence.py, the `explicit_buf not in scratch_args` branch). At the shipped
            # defaults (GROUPED_K=1, TMV_CTX=1) all three are dead, and at Hq*S*HD*2 = 8 MiB each over
            # 28 layers that is 672 MiB of arena that nothing reads -- 33.8% of the 1.99 GiB scratch,
            # and it reconciles exactly: 1.9898 GiB total minus 1.32904 GiB of named buffers = 0.661.
            bufsz.update({
                p + "kc": Hkv * S * HD * 2, p + "vc": Hkv * S * HD * 2,
                p + "sc": Hq * S * 2, p + "sw": Hq * S * 2,
                p + "cx": QD * 2,
                p + "g": FF * 2, p + "u": FF * 2, p + "gh": FF * 2, p + "d": D * 2,
                p + "hn": D * 2, p + "hf": D * 2,
            })
            if not GROUPED_K:
                bufsz[p + "kr"] = Hq * S * HD * 2
            if not (GROUPED_V or TMV_CTX):
                bufsz[p + "vr"] = Hq * S * HD * 2
            if not TMV_CTX:
                bufsz[p + "vt"] = Hq * S * HD * 2
            # `a` is purely internal to op_mlp_dp's own fuse_o path (never an L3 buffer -- see
            # design.py) once folded; only declare it when something outside that design still reads
            # or writes it.
            if not fuse_o:
                bufsz[p + "a"] = D * 2
            # The unfused arms only. With the fused head, the per-head norms, the projection and the
            # RoPE are all inside one design and none of these runlist entries exists.
            qk = proj = rope = []
            if sp.qk_norm and op_qkv_dp is None:
                hq = [f"{qhb}[{qho + h*HD*2}:{qho + (h+1)*HD*2}]" for h in range(Hq)]
                hk = [f"{khb}[{kho + h*HD*2}:{kho + (h+1)*HD*2}]" for h in range(Hkv)]
                qk = [*[((op_qk_norm if h % 2 == 0 else op_qk_norm_b),
                         hq[h], p + "n_qn", hq[h]) for h in range(Hq)],
                      *[((op_qk_norm if h % 2 == 0 else op_qk_norm_b),
                         hk[h], p + "n_kn", hk[h]) for h in range(Hkv)]]
            if op_qkv_dp is None:
                proj = ([(op_qkv, p + "Wqkv", p + "hn", p + "qkv")] if FUSE_QKV_GEMV else
                        [(op_q, p + "Wq", p + "hn", ref_q),
                         (op_kv, p + "Wk", p + "hn", ref_k),
                         (op_kv, p + "Wv", p + "hn", ref_v)])
                rope = ([(op_rope_qk, ref_qk, ang, ref_qk)] if fuse_rope else
                        [(op_rope_q, ref_q, ang, ref_q),
                         (op_rope_k, ref_k, ang, ref_k)])
            # The fused head replaces the norm, the projection, every qk-norm and the RoPE with one
            # design; `hn` lives and dies in L1 instead of round-tripping DDR between four of them.
            # The fused head absorbs the KV append too: k and v are drained straight into the caches
            # at `kv_off` instead of into buffers a StridedCopy then re-reads and re-writes. The caches
            # were their only consumer, so the intermediate had no reader -- it existed because the
            # append was a separate operator. Two runs and one more configure per layer.
            head = ([(op_qkv_dp, cur, p + "n_in", p + "Wqkv", p + "n_qn", p + "n_kn", ang,
                      ref_q, p + "kc", p + "vc")]
                    if op_qkv_dp is not None else
                    [(op_norm, cur, p + "n_in", p + "hn"), *proj, *qk, *rope,
                     (op_sck, ref_k, p + "kc"), (op_scv, ref_v, p + "vc")])
            rl += [
                *head,
                *([] if GROUPED_K else [(op_rep_k, p + "kc", p + "kr")]),
                # TMV_CTX subsumes the v-side grouping: TMatVec reads vc per kv head itself, so a
                # Repeat would materialise a `vr` nothing consumes.
                *([] if (GROUPED_V or TMV_CTX) else [(op_rep_v, p + "vc", p + "vr")]),
                (op_scores, p + ("kc" if GROUPED_K else "kr"), ref_q, p + "sc"),
                *([] if scale_in_qnorm else [(op_scale, p + "sc", "attn_scale", p + "sc")]),
                (op_softmax, p + "sc", p + "sw"),
                *([] if TMV_CTX else [(op_trv, p + ("vc" if GROUPED_V else "vr"), p + "vt")]),
                (op_ctx, p + ("vc" if TMV_CTX else "vt"), p + "sw", p + "cx"),
                *([] if fuse_o else [(op_o, p + "Wo", p + "cx", p + "a")]),
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
                    (op_down, p + "Wd", p + "gh", p + "d"),
                ]
            if sp.sandwich_norms:
                rl.append((op_norm, p + "d", p + "n_pff", p + "d"))
            if op_mlp_dp is None:
                rl.append((op_add, p + "x1", p + "d", nxt))
        bufsz[p + "x1"] = D * 2
        cur = nxt

    if op_mlp_dp is not None or op_decode_layer is not None:
        bufsz["mlp_gh"] = FF * 2   # one buffer, reused by every layer -- they run one at a time
        if fuse_o or op_decode_layer is not None:
            bufsz["mlp_a_scratch"] = D * 2   # a's own all-gather round-trip buffer, same idiom

    weights["n_final"] = load_norm("model.norm.weight")
    embed_f32 = npy("model.embed_tokens.weight")   # tied: also the host's embedding-gather table
    # Quantizing W_head narrows the DEVICE lm-head stream, but W_head is TIED, so the host also
    # gathers embed[token] out of it. Rather than teach the host to dequantise -- which would
    # quantise the embedding INPUT too, a second quality change for no extra speed -- the exact
    # bf16 table is emitted alongside as a HOST-ONLY blob. It is written outside the `weights`
    # dict on purpose: the device loader takes its buffer set from `weights`/`wnames`, so a side
    # file costs 311 MB of disk and ZERO device arena, and the host only ever faults in the one
    # 2 KB row it gathers.
    embed_blob, host_embed = "W_head", None
    if QUANT_HEAD_DTYPE != "bf16":
        weights["W_head"] = quantize_weight(embed_f32, QUANT_HEAD_GROUP, QUANT_HEAD_DTYPE, clip_search=QUANT_CLIP_SEARCH)
        embed_blob = "W_embed"
        host_embed = bf16(embed_f32).reshape(-1)
    else:
        weights["W_head"] = bf16(embed_f32).reshape(-1)
    if not scale_in_qnorm:
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
        raise SystemExit(0)

    inputs = ["x", "rope_global"] + (["rope_local"] if sp.rope_theta_local is not None else [])
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
    fused = OperatorSequence(sequence_name(sp, NL, S, placer_flags,
                                           decode_layer_active=op_decode_layer is not None), rl,
                              input_args=inputs, output_args=["logits"],
                              buffer_sizes=bufsz, context=ctx, extra_flags=placer_flags,
                              share_designs=share)
    fused.compile()
    return sp, fused, weights, dict(NL=NL, S=S, inputs=inputs, cache_names=cache_names,
                                    embed_blob=embed_blob, host_embed=host_embed)


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
    NL, S, inputs, cache_names = md["NL"], md["S"], md["inputs"], md["cache_names"]
    embed_blob, host_embed = md["embed_blob"], md["host_embed"]
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
        open(os.path.join(bdir, f"{n_}.bin"), "wb").write(weight_bytes(arr))
    if embed_blob != "W_head":
        # Host-only, deliberately not in `wnames`: see the tied-embedding note at its build site.
        open(os.path.join(bdir, f"{embed_blob}.bin"), "wb").write(weight_bytes(host_embed))
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
        # Engineering-check axis (see QUANT_MLP_DTYPE above), not a validated model default.
        # clip_search rides here rather than in the design name: it moves weight VALUES only, so
        # two arms share one compiled design and differ solely in the bytes loaded into it.
        "weight_quant": {"mlp_dtype": QUANT_MLP_DTYPE, "mlp_group_size": QUANT_MLP_GROUP,
                         "attn_dtype": QUANT_ATTN_DTYPE, "attn_group_size": QUANT_ATTN_GROUP,
                         "head_dtype": QUANT_HEAD_DTYPE, "head_group_size": QUANT_HEAD_GROUP,
                         "clip_search": QUANT_CLIP_SEARCH},
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
    main()
