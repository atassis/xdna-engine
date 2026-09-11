#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The decode precision plane: one declarative per-site map, checked before anything is built.

A PLAN names a weight format per SITE, where a site is a byte class of the decode token as the
shim-BD census measures it -- not a tensor, not an operator, not a kernel. The census is the
ranking that decides what is worth quantizing, so it is the vocabulary the plan is written in:

    {"mlp": "int8a/g128", "head": "int8a/g128", "qkv": "bf16", "attn_o": "bf16", "kv": "bf16"}

Two things use one plan. `gen_llm_decode.py` builds the artifact from it, and
`hostlab/plan_arms.py` turns the same plan into a host-side quality arm, so the arm whose
perplexity was measured is the arm that gets built.

WHY A PLANE AND NOT MORE FLAGS. A precision change alters buffer BYTES, so unlike a control-stream
change it cannot be one of N named variants in one ELF -- every arm is its own artifact. There is
therefore no runtime fallback and no way to discover a bad combination late and route around it:
the combination has to be refused before the build. The rules below are the refusals. Each carries
the source that owns the constraint, and `check()` raises naming it.

WHAT THIS MODULE DOES NOT OWN. The on-wire byte layout of a quantized row belongs to
`iron/operators/gemv/quant.py`, which is the kernel's contract. This module owns which site gets
which format and whether that combination can be built at all. `wire_bytes_per_element()` below
duplicates the packer's arithmetic on purpose and `check_packer_agreement()` cross-examines it, so
a divergence between what we price and what we ship is a test failure rather than a wrong census.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Tuple

BF16 = "bf16"
SYMMETRIC = ("int4", "int8")
AFFINE = ("int4a", "int8a")
DTYPES = (BF16,) + SYMMETRIC + AFFINE

_PAYLOAD_BITS = {"int4": 4, "int8": 8, "int4a": 4, "int8a": 8}
_HEADER_BYTES = {"int4": 4, "int8": 4, "int4a": 4, "int8a": 4}

SCALE_KINDS = {
    BF16: ("none",),
    # Symmetric: how the per-group scale is chosen. `clip` grid-searches the clip ratio that
    # minimises the group's reconstruction MSE; same wire format, host-side only.
    "int4": ("absmax", "clip"),
    "int8": ("absmax", "clip"),
    # Affine: where the offset puts the grid. `zero_grid` constrains the reconstruction grid to
    # contain exact zero; `free_min` fits the range. Which wins is per weight class and measured.
    "int4a": ("zero_grid", "free_min"),
    "int8a": ("zero_grid", "free_min"),
}


@dataclass(frozen=True)
class Rule:
    id: str
    claim: str
    owner: str


RULES = {r.id: r for r in (
    Rule("P001", "a site's dtype must be one this tree's packer implements, with the signature "
                 "the generator calls",
         "iron/operators/gemv/quant.py::quantize_weight"),
    Rule("P002", "every weight on one ObjectFifo carries one dtype for the fifo's lifetime",
         "iron/operators/attn_block_dp/design.py (STREAM channel), "
         "iron/operators/swiglu_mlp_dp/design.py (shared weight fifo)"),
    Rule("P003", "the operator that DECLARES a weight buffer must carry a weight_dtype axis",
         "iron/operators/{gemv,swiglu_mlp_dp}/op.py have one; "
         "{decode_layer_dp,attn_block_dp,qkv_head_dp} declare buffers without one"),
    Rule("P004", "giving a byte class its own dtype costs it its own shim channels, and the "
                 "device has 16 per direction",
         "iron/common/utils.py::get_shim_dma_limit, "
         "iron/operators/decode_layer_dp/op.py (the merged 14 in / 12 out check)"),
    Rule("P005", "a non-bf16 KV cache must derive its block size on its own address granule",
         "iron/common/kv_layout.py::derive_block_size (addr_gran_elems defaults to bf16's 2)"),
    Rule("P006", "the KV cache's quantization axis is per POSITION, which 4 bits cannot carry",
         "iron/operators/gemv/quant.py (scale per row per group; for kc a row is a position)"),
    Rule("P007", "K must be a whole number of groups and the packed row 4-byte aligned",
         "iron/operators/gemv/quant.py::row_stride_bytes"),
    Rule("P008", "a scale_kind must belong to its dtype's family",
         "this module's SCALE_KINDS"),
    Rule("P009", "quantizing the head splits the tied embedding and needs the bf16 sidecar",
         "gen_llm_decode.py (W_head is model.embed_tokens.weight; meta.json embed_blob)"),
)}


class PrecisionRefusal(ValueError):
    """A plan that cannot be built. Carries the rule that refused it."""

    def __init__(self, rule_id: str, detail: str):
        self.rule = RULES[rule_id]
        super().__init__(f"{rule_id}: {self.rule.claim}\n  {detail}\n  owner: {self.rule.owner}")


@dataclass(frozen=True)
class Site:
    key: str
    kind: str                  # "weight" reads once per token; "cache" grows with the window
    mb_per_token: float
    hostlab_class: Optional[str]
    note: str


# MEASURED 2026-09-11 off the shim BDs of the shipped rung-ladder build's own fused MLIR
# (artifacts/qwen3-0.6b, generator 58c4db4, window rung 4096), scripts/decode_ddr_bytes.py.
# These are a Qwen3-0.6B property at one window, not a constant: `token_mb()` reports against them
# and `scripts/precision_census.py` re-derives them from a build.
CENSUS_TOKEN_MB = 1664.09
CENSUS_WINDOW = 4096
CENSUS_DATE = "2026-09-11"

SITES = {s.key: s for s in (
    Site("mlp", "weight", 528.49, "mlp",
         "Wg/Wu/Wd, three per layer; the largest single class"),
    Site("kv", "cache", 469.88, None,
         "K and V read at the full padded window, so this row moves with CENSUS_WINDOW"),
    Site("head", "weight", 311.16, "head",
         "W_head, read once per token; it IS the tied embedding table"),
    Site("qkv", "weight", 234.88, "qkv",
         "Wqkv, one concatenated [Wq|Wk|Wv] per layer"),
    Site("attn_o", "weight", 118.36, "attn_o",
         "Wo, the attention output projection"),
)}

# Everything the census does not attribute to a site: cx, the KV append, the two all-gathers,
# norms, angles. 0.08% of the token, and no format touches it.
CENSUS_UNSITED_MB = round(CENSUS_TOKEN_MB - sum(s.mb_per_token for s in SITES.values()), 2)


@dataclass(frozen=True)
class Spec:
    dtype: str = BF16
    group_size: int = 0
    scale_kind: str = "none"

    def __str__(self) -> str:
        return self.dtype if self.dtype == BF16 else \
            f"{self.dtype}/g{self.group_size}/{self.scale_kind}"

    @property
    def quantized(self) -> bool:
        return self.dtype != BF16

    @property
    def affine(self) -> bool:
        return self.dtype in AFFINE


BF16_SPEC = Spec()

# Where the AFFINE offset goes, per class, from the host ladder: the MLP and the head want the
# grid constrained to contain exact zero; q/k/v want the free min and invert the ranking. The
# symmetric family has no offset to place, so it always takes its own default.
_AFFINE_OFFSET_BY_CLASS = {"qkv": "free_min"}


def parse_spec(text: str, site_key: str = "") -> Spec:
    """`bf16`, `int8`, `int4a/g64`, or `int4a/g64/free_min`."""
    parts = [p for p in str(text).strip().split("/") if p]
    if not parts:
        raise PrecisionRefusal("P008", f"{site_key or 'spec'}: empty")
    dtype = parts[0]
    if dtype not in DTYPES:
        raise PrecisionRefusal("P001", f"{site_key or 'spec'}: {dtype!r} is not one of {DTYPES}")
    if dtype == BF16:
        if len(parts) > 1:
            raise PrecisionRefusal("P008", f"{site_key or 'spec'}: bf16 takes no group or scale")
        return BF16_SPEC
    group = int(parts[1].lstrip("gG")) if len(parts) > 1 else 128
    default = SCALE_KINDS[dtype][0]
    if dtype in AFFINE:
        default = _AFFINE_OFFSET_BY_CLASS.get(site_key, default)
    kind = parts[2] if len(parts) > 2 else default
    if kind not in SCALE_KINDS[dtype]:
        raise PrecisionRefusal(
            "P008", f"{site_key or 'spec'}: scale_kind {kind!r} is not valid for {dtype} "
                    f"(expected one of {SCALE_KINDS[dtype]})")
    return Spec(dtype, group, kind)


def parse_plan(text: str) -> Dict[str, Spec]:
    """A JSON object of site -> spec string, or a path to one. Unnamed sites stay bf16."""
    if os.path.exists(text):
        with open(text) as fh:
            text = fh.read()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PrecisionRefusal("P008", f"plan is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PrecisionRefusal("P008", f"plan must be an object of site -> spec, got {type(raw)}")
    unknown = set(raw) - set(SITES)
    if unknown:
        raise PrecisionRefusal(
            "P008", f"unknown site(s) {sorted(unknown)}; the census names {sorted(SITES)}")
    plan = {k: BF16_SPEC for k in SITES}
    for key, val in raw.items():
        plan[key] = parse_spec(val, key) if isinstance(val, str) else Spec(**val)
    return plan


# Named points. Three questions are independent here and a preset answers only the first two:
# what it COSTS IN BYTES, what it costs in QUALITY, and what its KERNEL costs. On the third the
# record is blunt and it does not follow the other two -- measured on the unfused graph at an
# identical dispatch count, symmetric int4 is at parity with bf16 (0.99x), symmetric int8 is
# 2.10x SLOWER (its dequant loop), and BOTH affine forms add a constant +47.3/+48.3 ms from
# `group_sums_of_b`'s per-call cost.
# So the best-measured arm is the symmetric int4 one, not the best-quality or fewest-bytes one.
# `reach` is which arm each can be BUILT on, a fourth and separate question.
PRESETS = {
    "bf16": ({}, "fused"),
    # SYMMETRIC, and deliberately: the affine forms are better on quality at the same bytes and
    # carry a measured constant that dwarfs the difference. Wo rides the MLP's fifo under fuse_o,
    # so the two move together by construction.
    "mlp-int4-sym": ({"mlp": "int4/g128", "attn_o": "int4/g128"}, "fused"),
    "mlp-int8-sym": ({"mlp": "int8/g128", "attn_o": "int8/g128"}, "fused"),
    "mlp-int8": ({"mlp": "int8a/g128", "attn_o": "int8a/g128"}, "fused"),
    "mlp-head-int8": ({"mlp": "int8a/g128", "attn_o": "int8a/g128", "head": "int8a/g128"},
                      "fused"),
    # The whole weight stream at int8, group 128. Host-measured at +0.02% perplexity
    # [-0.2, +0.3] over 3000 paired positions -- the only format measured on this model whose
    # generation also holds.
    "all-int8": ({"mlp": "int8a/g128", "attn_o": "int8a/g128", "head": "int8a/g128",
                  "qkv": "int8a/g128"}, "no-fusion"),
    # The arm the device has run end to end: +7.50% perplexity [+4.05, +11.06] t=4.35 on 2000
    # paired positions.
    "mlp-int4": ({"mlp": "int4a/g128/zero_grid", "attn_o": "int4a/g128/zero_grid"}, "fused"),
}


def plan_from_env(env: Optional[Mapping[str, str]] = None) -> Tuple[Dict[str, Spec], str]:
    """Resolve the plan for this build. Returns (plan, provenance)."""
    env = os.environ if env is None else env
    legacy = {k: v for k, v in (
        ("mlp", env.get("QUANT_MLP_DTYPE")), ("attn_o", env.get("QUANT_ATTN_DTYPE")),
        ("head", env.get("QUANT_HEAD_DTYPE"))) if v}
    spec_text = env.get("PRECISION")
    if spec_text and legacy:
        raise PrecisionRefusal(
            "P008", f"PRECISION and the QUANT_*_DTYPE vars both set ({sorted(legacy)}); "
                    "the plan is the single source of truth -- drop one")
    if spec_text:
        if spec_text in PRESETS:
            return (parse_plan(json.dumps(PRESETS[spec_text][0])),
                    f"PRECISION={spec_text} (preset)")
        return parse_plan(spec_text), f"PRECISION={spec_text[:60]}"
    if not legacy:
        return {k: BF16_SPEC for k in SITES}, "default (bf16 everywhere)"
    groups = {"mlp": "QUANT_MLP_GROUP", "attn_o": "QUANT_ATTN_GROUP", "head": "QUANT_HEAD_GROUP"}
    clip = env.get("QUANT_CLIP_SEARCH", "0") != "0"
    plan = {k: BF16_SPEC for k in SITES}
    for site, dtype in legacy.items():
        if dtype == BF16:
            continue
        group = int(env.get(groups[site], "128"))
        if dtype in SYMMETRIC:
            kind = "clip" if clip else "absmax"
        else:
            kind = "zero_grid"
        plan[site] = parse_spec(f"{dtype}/g{group}/{kind}", site)
    return plan, f"legacy QUANT_* env ({','.join(sorted(legacy))})"


# The payload's vector-load width in bytes, per dtype, at the kernel's VEC_SIZE=64. int4 packs
# two nibbles per byte so it loads half as wide. Mirrors iron/operators/gemv/quant.py's own
# `_LOAD_BYTES`; used only by the fallback below.
_LOAD_BYTES = {"int4": 32, "int4a": 32, "int8": 64, "int8a": 64}


def wire_row_units(spec: Spec, K: int) -> int:
    """Wire units in one weight ROW of width K, refusing an illegal row.

    UNITS, not bytes: bf16 elements for an unquantized weight and packed bytes for a quantized
    one, because those are the units each array is actually indexed in. Returning bytes for both
    is the bytes-vs-elements seam -- it reads correct, and it silently halves the row count of
    every bf16 reshape.

    THE PACKER OWNS THIS NUMBER and this defers to it whenever it can be imported. The
    arithmetic below is a FALLBACK for pricing a plan against a tree that cannot build it, and
    it is deliberately not the authority: a packed row is not simply header+payload, because the
    header is padded so the payload clears its own vector-load width. A plane that re-derived
    the unpadded form would price every arm slightly wrong and disagree with what shipped.
    `test_the_plane_agrees_with_the_packer` is what keeps the fallback honest.
    """
    if not spec.quantized:
        return K
    try:
        from iron.operators.gemv.quant import row_stride_bytes
    except ImportError:
        pass
    else:
        try:
            return row_stride_bytes(K, spec.group_size, spec.dtype)
        except ValueError as exc:
            raise PrecisionRefusal("P007", f"{spec} at K={K}: {exc}") from exc
    if K % spec.group_size:
        raise PrecisionRefusal(
            "P007", f"K={K} is not a whole number of groups (group_size={spec.group_size})")
    load = _LOAD_BYTES[spec.dtype]
    header = -(-(_HEADER_BYTES[spec.dtype] * (K // spec.group_size)) // load) * load
    stride = header + K * _PAYLOAD_BITS[spec.dtype] // 8
    if stride % load:
        raise PrecisionRefusal(
            "P007", f"packed row stride {stride} B does not clear the {load} B load width at "
                    f"K={K} group_size={spec.group_size} {spec.dtype}")
    return stride


def wire_bytes_per_element(spec: Spec, K: int) -> float:
    """Bytes on the wire per weight element, at row width K.

    Takes K because the packer owns the answer and the packer takes K. The rate happens to be
    K-independent for the formats shipped here -- the row is header+payload and n_groups scales
    with K -- but that is a property of the current layout, not of the question, and it stopped
    being true for one afternoon when the header was padded.
    """
    return 2.0 if not spec.quantized else wire_row_units(spec, K) / K


@dataclass(frozen=True)
class GraphContext:
    """The graph facts the rules are conditional on. Every field is something the plan cannot
    change and the generator already knows before it builds anything."""
    fused_layer: bool            # decode_layer_dp carries the layer
    fuse_o: bool                 # Wo rides swiglu_mlp_dp's weight fifo
    fused_qkv_gemv: bool         # one concatenated Wqkv GEMV
    fused_qkv_dp: bool           # qkv_head_dp carries Wqkv, norms, RoPE and the KV append
    d_model: int
    ffn: int
    q_dim: int
    head_dim: int
    attn_cols: int = 8
    shim_in_used: int = 14
    shim_out_used: int = 12
    shim_limit: int = 16
    kv_dedicated_channels: bool = False
    packer_dtypes: Tuple[str, ...] = DTYPES
    packer_takes_scale_kind: bool = True


# The row width each site's weight is quantized along. A group runs along K, so K is what has to
# divide; for the caches a "row" is one cached position and K is the head dimension.
def site_k(site_key: str, ctx: GraphContext) -> Tuple[int, ...]:
    return {
        "mlp": (ctx.d_model, ctx.ffn),        # Wg/Wu along D, Wd along FF
        "attn_o": (ctx.q_dim,),
        "qkv": (ctx.d_model,),
        "head": (ctx.d_model,),
        "kv": (ctx.head_dim,),
    }[site_key]


QWEN3_06B = GraphContext(
    fused_layer=True, fuse_o=True, fused_qkv_gemv=True, fused_qkv_dp=True,
    d_model=1024, ffn=3072, q_dim=2048, head_dim=128,
)


def dedicated_channel_cost(ctx: GraphContext) -> Dict[str, int]:
    """What it costs to give the KV cache a dtype of its own.

    K and V share one fifo while they share a dtype, so the cost is one input channel per
    attention column for the pair, and one output channel per column for the append.
    """
    need_in = need_out = ctx.attn_cols
    return {"in_need": need_in, "out_need": need_out,
            "in_margin": ctx.shim_limit - ctx.shim_in_used,
            "out_margin": ctx.shim_limit - ctx.shim_out_used,
            "in_over": max(0, ctx.shim_in_used + need_in - ctx.shim_limit),
            "out_over": max(0, ctx.shim_out_used + need_out - ctx.shim_limit)}


def check(plan: Mapping[str, Spec], ctx: GraphContext) -> None:
    """Refuse a plan this tree cannot build, naming the rule. Silence means buildable."""
    def get(key):
        return plan.get(key, BF16_SPEC)

    for key, spec in sorted(plan.items()):
        if not spec.quantized:
            continue
        if spec.dtype not in ctx.packer_dtypes:
            raise PrecisionRefusal(
                "P001", f"{key}={spec}: the packer on this PYTHONPATH implements "
                        f"{sorted(ctx.packer_dtypes)}. Point IRON at a checkout that has "
                        f"{spec.dtype}, or pick a dtype it has")
        if spec.scale_kind not in ("absmax", "zero_grid") and not ctx.packer_takes_scale_kind:
            raise PrecisionRefusal(
                "P001", f"{key}={spec}: the packer on this PYTHONPATH has no scale-selection "
                        f"argument, so scale_kind={spec.scale_kind!r} cannot be honoured")
        for k in site_k(key, ctx):
            wire_row_units(spec, k)                  # P007

    # P002 -- attn_block_dp collapses Wqkv, K and V onto ONE input fifo per core, which is the
    # only reason the fused layer fits the channel budget at all.
    wire = lambda k: (get(k).dtype, get(k).group_size)   # noqa: E731 -- what a fifo can see
    if ctx.fused_layer and wire("qkv") != wire("kv"):
        c = dedicated_channel_cost(ctx)
        raise PrecisionRefusal(
            "P002", f"the fused layer streams Wqkv, K and V down one ObjectFifo per core, so they "
                    f"take one dtype: qkv={get('qkv')} kv={get('kv')}. Splitting them needs "
                    f"{c['in_need']} dedicated input and {c['out_need']} output channels against "
                    f"margins of {c['in_margin']} and {c['out_margin']} -- over by "
                    f"{c['in_over']} and {c['out_over']}. Make them equal, or build the unfused "
                    "arm (FUSE_DECODE_LAYER=0)")
    # P002 -- under fuse_o, Wo rides the MLP's shared weight fifo.
    if ctx.fuse_o and wire("attn_o") != wire("mlp"):
        raise PrecisionRefusal(
            "P002", f"FUSE_MLP_O folds Wo into the MLP's weight fifo, so attn_o takes the MLP's "
                    f"dtype: mlp={get('mlp')} attn_o={get('attn_o')}. Make them equal, or set "
                    "FUSE_MLP_O=0 to give Wo its own channel")

    # P004 -- the escape from P002, priced. The route past it is MemTile staging for the K
    # read, which trades shim channels for an L2 hop and is a topology change.
    if ctx.kv_dedicated_channels:
        c = dedicated_channel_cost(ctx)
        if c["in_over"] or c["out_over"]:
            raise PrecisionRefusal(
                "P004", f"dedicated KV channels need {c['in_need']} in / {c['out_need']} out "
                        f"against margins of {c['in_margin']} / {c['out_margin']} -- over by "
                        f"{c['in_over']} / {c['out_over']} of {ctx.shim_limit} each")

    # P006 -- one scale per row, and for the cache a row is one cached position.
    if get("kv").quantized and _PAYLOAD_BITS[get("kv").dtype] < 8:
        raise PrecisionRefusal(
            "P006", f"kv={get('kv')}: the packer scales per row per group and a kc row is a "
                    "POSITION, so this is per-token scaling. Published practice wants "
                    "per-channel for K, whose outliers are channel-consistent; per-token is "
                    "usually acceptable at 8 bits and is not at 4")

    # P003 -- the operator that DECLARES a buffer must be able to size it in packed bytes.
    # Named per carrier, because "which operator holds Wqkv" is four different answers depending
    # on the fused arms and only one of them has the axis.
    if get("qkv").quantized or get("kv").quantized:
        carrier = ("attn_block_dp (inside the fused layer)" if ctx.fused_layer else
                   "qkv_head_dp" if ctx.fused_qkv_dp else
                   None if ctx.fused_qkv_gemv else
                   "three separate GEMVs over one concatenated Wqkv blob")
        if get("kv").quantized and carrier is None:
            carrier = "whichever operator reads kc/vc"
        if carrier is not None:
            raise PrecisionRefusal(
                "P003", f"qkv={get('qkv')} kv={get('kv')}: {carrier} declares these buffers and "
                        "has no weight_dtype axis, so they would be sized in bf16 elements "
                        "against packed bytes -- an artifact that builds clean and fails its "
                        "own load-time size check. Reach the axis with FUSE_DECODE_LAYER=0 "
                        "FUSE_QKV_DP=0 FUSE_QKV_GEMV=1, which puts Wqkv on a plain GEMV")


def kv_addr_gran_elems(plan: Mapping[str, Spec]) -> int:
    """The address granule, in ELEMENTS, that `kv_layout.derive_block_size` must be given.

    P005. The granule is 4 bytes on this target; the element width is the KV cache's dtype. The
    function's own default is 2, which is bf16's answer to this question and nothing else's.
    """
    spec = plan.get("kv", BF16_SPEC)
    elem_bytes = 2 if not spec.quantized else _PAYLOAD_BITS[spec.dtype] // 8
    if elem_bytes < 1:
        raise PrecisionRefusal(
            "P005", f"kv={spec}: a sub-byte cache element has no address granule in elements")
    return 4 // elem_bytes


def token_mb(plan: Mapping[str, Spec]) -> Dict[str, float]:
    """Projected MB/token per site under `plan`, against the census baseline."""
    out = {key: site.mb_per_token
           * wire_bytes_per_element(plan.get(key, BF16_SPEC), site_k(key, QWEN3_06B)[0]) / 2
           for key, site in SITES.items()}
    out["unsited"] = CENSUS_UNSITED_MB
    out["total"] = sum(out.values())
    return out


# `t = bytes/54.71 GB/s + 67.3 us per command`, fitted on a GEMV stream-length sweep
#. A precision arm
# moves the byte term only: it changes no command count and the fitted leftover is measured
# byte-independent across a 352 MB span, so the marginal rate is the whole prediction.
MARGINAL_US_PER_MB = 1e6 / 54.71e3


def predicted_ms_delta(plan: Mapping[str, Spec]) -> float:
    """Predicted change in ms/token from the byte cut alone. Negative is faster.

    This is a TRANSPORT prediction and nothing else. It is blind to what the dequant costs on the
    core, which is the term that once made an int4 arm 11.7x slower than bf16 at fewer bytes.
    """
    return (token_mb(plan)["total"] - CENSUS_TOKEN_MB) * MARGINAL_US_PER_MB / 1e3


def packer_capability() -> Tuple[Tuple[str, ...], bool]:
    """What the packer on this PYTHONPATH can actually do.

    The build's own API gate checks that `quantize_weight` EXISTS. It has a signature too, and a
    tree carrying the symbol without the arguments accepts a documented dtype and then dies in the
    packer with a TypeError naming a keyword.
    """
    try:
        import inspect
        from iron.operators.gemv.quant import quantize_weight
    except ImportError:
        return (BF16,), False
    params = inspect.signature(quantize_weight).parameters
    takes_kind = "clip_search" in params or "scale_kind" in params or \
        any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    try:
        from iron.operators.gemv.quant import is_affine  # affine-capable packers export this
        dtypes = DTYPES
    except ImportError:
        dtypes = (BF16,) + SYMMETRIC
    return dtypes, takes_kind


def describe(plan: Mapping[str, Spec]) -> str:
    mb = token_mb(plan)
    lines = [f"precision plan (census {CENSUS_DATE}, window {CENSUS_WINDOW}, "
             f"{CENSUS_TOKEN_MB:.2f} MB/token bf16)"]
    for key, site in sorted(SITES.items(), key=lambda kv: -kv[1].mb_per_token):
        spec = plan.get(key, BF16_SPEC)
        lines.append(f"  {key:7} {str(spec):22} {site.mb_per_token:8.2f} -> {mb[key]:8.2f} MB"
                     f"  ({100 * mb[key] / site.mb_per_token:5.1f}%)")
    lines.append(f"  {'TOTAL':7} {'':22} {CENSUS_TOKEN_MB:8.2f} -> {mb['total']:8.2f} MB")
    lines.append(f"  transport prediction: {predicted_ms_delta(plan):+.2f} ms/token "
                 f"(byte term only; the dequant's core cost is not in this number)")
    return "\n".join(lines)


def suffix(plan: Mapping[str, Spec]) -> str:
    """Artifact-name fragment. Two plans that differ in any way must not share a build."""
    parts = [f"{k}{plan[k].dtype}g{plan[k].group_size}"
             + ("" if plan[k].scale_kind in ("absmax", "zero_grid") else plan[k].scale_kind)
             for k in sorted(SITES) if plan.get(k, BF16_SPEC).quantized]
    return "_".join(parts)


def resolved_context(**over) -> GraphContext:
    """QWEN3_06B with the packer capability of whatever IRON is on this PYTHONPATH."""
    dtypes, kind = packer_capability()
    return replace(QWEN3_06B, packer_dtypes=dtypes, packer_takes_scale_kind=kind, **over)


def _main(argv):
    if argv and argv[0] in ("--list", "-l"):
        print(f"{'preset':16} {'arm':10} {'MB/token':>9} {'ms':>7}  plan")
        for name, (raw, reach) in PRESETS.items():
            plan = parse_plan(json.dumps(raw))
            print(f"{name:16} {reach:10} {token_mb(plan)['total']:9.1f} "
                  f"{predicted_ms_delta(plan):+7.2f}  "
                  f"{ {k: str(v) for k, v in plan.items() if v.quantized} or 'bf16'}")
        return 0
    if argv and argv[0] in PRESETS:
        plan, prov = parse_plan(json.dumps(PRESETS[argv[0]][0])), f"preset {argv[0]}"
    elif argv:
        plan, prov = parse_plan(argv[0]), "argv"
    else:
        plan, prov = plan_from_env()
    # The arm P003's own message recommends: FUSE_DECODE_LAYER=0 FUSE_QKV_DP=0 FUSE_QKV_GEMV=1.
    # Turning off only the layer leaves qkv_head_dp holding Wqkv, which has no axis either.
    ctx = resolved_context(**({"fused_layer": False, "fuse_o": False, "fused_qkv_dp": False}
                              if "--unfused" in argv else {}))
    print(f"[{prov}]  packer: dtypes={list(ctx.packer_dtypes)} "
          f"scale-selection={ctx.packer_takes_scale_kind}")
    print(describe(plan))
    try:
        check(plan, ctx)
    except PrecisionRefusal as exc:
        print(f"\nREFUSED\n{exc}")
        return 2
    print("\ncheck: OK")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
