#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decoder-LLM specs for the fused decode rail -- a MODEL is DATA, not a generator.

`gen_llm_decode.py` builds one fused decode ELF from any spec here. The axes below are exactly the
ones two real checkpoints disagreed on; each was read off the checkpoint's own `modeling_*.py`, not
inferred from a sibling model, because three of them are same-name-different-meaning traps:

  * `norm_gain`      Gemma3RMSNorm returns `x_hat * (1 + w)`, Qwen3RMSNorm returns `x_hat * w`.
                     Storing the wrong one is a silent ~1%/layer scale drift, not a crash --
                     the exact failure mode that cost the Gemma bring-up its 3/8 token parity.
  * `sandwich_norms` Gemma normalises the attention and FFN OUTPUTS before the residual add
                     (4 norms/layer); Qwen3 is plain pre-norm (2/layer).
  * `pre_ffn_norm`   the weight NAME moves with that. Gemma's pre-FFN norm is
                     `pre_feedforward_layernorm`; Qwen3's is `post_attention_layernorm`, which in
                     Gemma is a DIFFERENT tensor (the attention-output sandwich norm).

`attn_scale` is likewise not universal: Gemma-3 divides by `query_pre_attn_scalar**0.5` (256 -> 1/16),
Qwen3 by `head_dim**0.5`. And Gemma scales the embedding by `sqrt(d_model)` on the way in while Qwen3
does not (`embed_scale`), which is a HOST-side per-token step, recorded here so the two sides agree.
"""
from dataclasses import dataclass

# ---- GEMV L1 fit -- ONE model, shared by the generator and the weight dump ----
# This lives here rather than in gen_llm_decode.py because the WEIGHT DUMP has to make the same
# decision: whether a GEMV fits L1 decides whether its K is split, and a K-split changes how many
# tensors the dump writes and what they are named. Two copies of an L1 model is a seam with no
# owner, which is the class that has already cost this tree a build.

L1_BYTES = 65536      # AIE2P core local memory (getLocalMemorySize(), AIETargetModel.h)
L1_RESERVE = 8192     # stack + the allocator's own slack; measured headroom, not a guess (see below)
C_TILE_GRANULE = 8    # tile_size_output must be a multiple of this (16 bytes of bf16); see gemv_tile_output


def gemv_tile_output(M, K, cols=8, tsi=None):
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


def gemv_fits(M, K, cols=8):
    """Does ANY legal (tile_size_input, tile_size_output) exist for this GEMV within L1?

    A predicate over gemv_tile_output rather than a second budget calculation -- a duplicated fit
    model is exactly what moving this here was meant to delete.
    """
    try:
        gemv_tile_output(M, K, cols=cols)
        return True
    except ValueError:
        return False


def k_chunks_for(M, K, cols=8):
    """Fewest power-of-two chunks of K whose GEMV fits L1. 1 when the shape already fits.

    The B vector is double-buffered at 2*K*2 bytes and is INDEPENDENT of every tiling knob, so a
    large enough K does not fit at ANY (tsi, tso): Gemma-4-12B's down projection needs 61440 B of
    57344 usable before a single weight or output byte is counted. Splitting the reduction over K
    and summing the partials is the fix and needs no new operator. It is NOT free in bf16 -- each
    partial is rounded to bf16 before the sum.
    """
    n = 1
    while n <= 64:
        if K % n == 0 and gemv_fits(M, K // n, cols):
            return n
        n *= 2
    raise ValueError(f"GEMV M={M} K={K}: no power-of-two K split up to 64 fits L1")


@dataclass(frozen=True)
class LlmSpec:
    name: str
    d_model: int
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    eps: float
    act: str                    # "gelu_tanh" | "silu"  -> the gated-FFN activation
    norm_gain: str              # "one_plus_w" | "w"    -> how the checkpoint stores RMSNorm weights
    sandwich_norms: bool        # normalise attn/FFN OUTPUT before the residual add (Gemma-3)
    qk_norm: bool               # per-head RMSNorm over head_dim, between projection and RoPE
    embed_scale: str            # "sqrt_d_model" | "none" -> host applies this to embed[token]
    rope_theta_global: float
    rope_theta_local: float | None    # None = single-theta RoPE (no local/global split)
    sliding_window: int | None
    sw_pattern: int | None            # every Nth layer (1-based) is GLOBAL; None = all global
    query_pre_attn_scalar: float | None   # None => scale = head_dim ** -0.5

    # ---- per-layer attention geometry (Gemma-4) ----
    # Gemma-4-12B is not uniform: its SLIDING layers are head_dim 256 / 8 kv heads and its GLOBAL
    # layers are head_dim 512 / 1 kv head, selected by the same is_global() rule the theta split
    # already uses. Both are stated correctly in that checkpoint's config.json (global_head_dim,
    # num_global_key_value_heads) -- an older note in this file claimed otherwise and was stale.
    # None on both means "uniform", which is every spec here today.
    #
    # THESE FIELDS ARE DATA ONLY AND A NON-UNIFORM SPEC CANNOT BUILD YET. See check(): the host
    # protocol assumes one head_dim per artifact in two places, so a build would be silently wrong
    # on the global layers rather than failing. The refusal is deliberately landed BEFORE the
    # capability.
    global_head_dim: int | None = None
    global_n_kv_heads: int | None = None

    # ---- Gemma-4 axes, DATA ONLY: every one of these is refused by name in check() ----
    # Each was read off the checkpoint or `transformers/models/gemma4_unified/`, never inferred from
    # Gemma-3 by family resemblance -- which is the trap that put `norm_gain="one_plus_w"` in the
    # stranded literal, wrong on all 193 norm tensors. Facts and citations:
    # [[gemma4-norm-gain-is-w-not-one-plus-w]].
    #
    # They are fields BEFORE they are capabilities on purpose. A spec that cannot say what a model
    # needs cannot refuse it by name either, and "gemma4-12b is not supported" is a far worse
    # failure than a list of five things that are missing.

    # attention scaling is a FIXED 1.0, not head_dim**-0.5 and not a query_pre_attn_scalar.
    # Gemma4UnifiedTextAttention sets `self.scaling = 1.0` and passes it explicitly, so the
    # eager-attention default is never taken. Encoding it as query_pre_attn_scalar=1.0 would
    # compute the right number by coincidence and read as a config value the checkpoint does not
    # have.
    attn_scale_fixed: float | None = None
    # attention_k_eq_v: global layers have NO v_proj and V is the RAW k_proj output -- captured
    # before k_norm and before RoPE, because the Python binds value_states to key_states and then
    # rebinds key_states.
    v_from_k_on_global: bool = False
    # A GAINLESS RMSNorm (with_scale=False) on the value path of EVERY layer. with_scale=False
    # removes the learned gain, NOT the normalisation -- and because it removes the gain there is
    # no weight tensor anywhere in the checkpoint, so nothing can fail on its absence. This flag
    # exists so the absence is a declaration instead of an oversight.
    v_norm: bool = False
    # Fraction of head_dim rotated on the global layers (0.25 = 64 of 256 frequency pairs at
    # global_head_dim=512), with the rope_type the config names for them.
    rope_partial_rotary: float | None = None
    rope_type_global: str | None = None
    # A trained per-layer scalar applied to the block output AFTER both residual adds. A
    # register_buffer, so it is in the checkpoint but not in config.json; 0.053 at layer 0 and
    # 0.048 at 47, so treating it as 1.0 is worst at the ends of the stack.
    layer_scalar: bool = False
    # tanh softcap at the LM head (logits/c -> tanh -> *c). An LM-head axis, not an attention one:
    # eager_attention_forward takes a softcap argument and the decoder layer passes none.
    logit_softcap: float | None = None
    # Tensor-name prefix in the dump. Gemma-4-12B is a MULTIMODAL checkpoint -- vision and audio
    # embedders sit alongside the text stack -- so its text tensors are under
    # `model.language_model.`, not `model.`.
    weight_prefix: str = "model."

    # ---- derived ----
    @property
    def q_dim(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_group(self) -> int:
        return self.n_q_heads // self.n_kv_heads

    @property
    def attn_scale(self) -> float:
        if self.attn_scale_fixed is not None:
            return self.attn_scale_fixed
        qpas = self.query_pre_attn_scalar
        return (qpas ** -0.5) if qpas is not None else (self.head_dim ** -0.5)

    def is_global(self, layer_idx: int) -> bool:
        """Gemma-3 alternates local/global attention; a single-theta model is global everywhere."""
        if self.sw_pattern is None:
            return True
        return (layer_idx + 1) % self.sw_pattern == 0

    def geometry_is_uniform(self) -> bool:
        """True when every layer shares one head_dim and one n_kv_heads."""
        return self.global_head_dim is None and self.global_n_kv_heads is None

    def head_dim_for(self, layer_idx: int) -> int:
        if self.global_head_dim is not None and self.is_global(layer_idx):
            return self.global_head_dim
        return self.head_dim

    def n_kv_heads_for(self, layer_idx: int) -> int:
        if self.global_n_kv_heads is not None and self.is_global(layer_idx):
            return self.global_n_kv_heads
        return self.n_kv_heads

    def q_dim_for(self, layer_idx: int) -> int:
        return self.n_q_heads * self.head_dim_for(layer_idx)

    def kv_dim_for(self, layer_idx: int) -> int:
        return self.n_kv_heads_for(layer_idx) * self.head_dim_for(layer_idx)

    def unimplemented(self) -> list[str]:
        """The model features this spec declares that the build cannot express yet, each named.

        One list rather than a refusal per feature, because the useful question at porting time is
        "what is still missing", and answering it one exception at a time is how the Gemma-3
        bring-up met four capability gaps as four separate crashes on four separate builds.

        Each entry says what breaks, not just what is absent -- a reader deciding whether to
        implement it needs the failure, and the failure is the argument for the refusal.
        """
        gaps = []
        if not self.geometry_is_uniform():
            gaps.append(
                f"per-layer attention geometry (head_dim {self.head_dim}/{self.global_head_dim}, "
                f"n_kv_heads {self.n_kv_heads}/{self.global_n_kv_heads}): the GENERATOR handles it "
                f"(the op vocabulary is keyed on the pair), but meta.json still carries "
                f"dims.head_dim as one scalar and the host's rope cross-check is gated on "
                f"kv_offs.len()==1, so it does nothing in exactly this case")
        if self.v_norm:
            gaps.append(
                "v_norm: a GAINLESS RMSNorm on the value path of every layer. There is no weight "
                "tensor for it anywhere in the checkpoint (with_scale=False), so its absence "
                "cannot raise a missing-key error -- skipping it is a silent numerical change")
        if self.v_from_k_on_global:
            gaps.append(
                "attention_k_eq_v: global layers have no v_proj and V is the RAW k_proj output, "
                "taken before k_norm and before RoPE. The graph currently derives V from its own "
                "projection on every layer")
        if self.rope_partial_rotary is not None:
            gaps.append(
                f"partial rotary {self.rope_partial_rotary} (rope_type "
                f"{self.rope_type_global!r}): only that fraction of head_dim is rotated on the "
                f"global layers")
        if self.logit_softcap is not None:
            gaps.append(f"logit softcap {self.logit_softcap}: tanh(logits/c)*c at the LM head")
        return gaps

    def softmax_cols(self, cap: int) -> int:
        """num_aie_columns for the attention Softmax: the widest split of the q heads that fits.

        The Softmax is built with rows = n_q_heads, and iron/operators/softmax/op.py requires
        rows >= num_aie_columns*num_channels and rows % num_aie_columns == 0 -- a head count under
        the split leaves a core with less than one tile, so the op silently computes nothing rather
        than failing. Taking the largest divisor of n_q_heads at or below the cap satisfies both:
        Qwen3's 16 heads give 8, Gemma-3's 4 give 4.
        """
        return max(d for d in range(1, cap + 1) if self.n_q_heads % d == 0)

    def qkv_dp_cols(self, cap: int, n_kv_heads: int | None = None) -> int:
        """num_aie_columns for QKVHeadDataParallel: every core owns a whole number of head rows.

        `n_kv_heads` overrides the spec's own value for a per-layer geometry; omitted, it is the
        uniform one, so a caller that has no geometry to name gets the pre-existing answer.
        """
        kvh = self.n_kv_heads if n_kv_heads is None else n_kv_heads
        heads = self.n_q_heads + 2 * kvh
        return max(d for d in range(1, cap + 1) if heads % d == 0)

    def qkv_dp_reason(self, cap: int, head_dim: int | None = None) -> str | None:
        """Why QKVHeadDataParallel does not cover this spec, or None when it does.

        Every rule here is the OPERATOR's (iron/operators/qkv_head_dp/op.py), read off its
        __post_init__ rather than guessed: it applies a per-head qk-norm, and `cur`/`n_in` ride an
        HD-wide misc channel so d_model must be a whole number of head_dim chunks. Gemma-3-270M
        fails the second at 640/256 = 2.5, which no column count can fix -- worth stating, because
        the column rule looks like the whole story and is not.
        """
        if not self.qk_norm:
            return "the op applies a per-head qk-norm and this spec has none"
        hd = self.head_dim if head_dim is None else head_dim
        if self.d_model % hd:
            return (f"d_model={self.d_model} is not a whole number of head_dim={hd} "
                    f"chunks -- cur/n_in ride the HD-wide misc channel")
        return None

    def mlp_dp_reason(self) -> str | None:
        """Why SwiGLUMLPDataParallel does not cover this spec, or None when it does."""
        if self.sandwich_norms:
            return "the fused block has no sandwich norms (this spec normalises the FFN output)"
        if self.act != "silu":
            return f"the fused block is SwiGLU; this spec's activation is {self.act!r}"
        return None

    def layer_scalar_name(self, layer: int) -> str:
        """The per-layer scalar tensor. A register_buffer, so config.json never mentions it."""
        return f"{self.weight_prefix}layers.{layer}.layer_scalar"

    def norm_weight_names(self, layer: int) -> dict:
        """Per-layer RMSNorm tensor names. The pre-FFN norm's NAME differs between the two families."""
        p = f"{self.weight_prefix}layers.{layer}."
        names = {
            "n_in": p + "input_layernorm.weight",
            "n_pf": p + ("pre_feedforward_layernorm" if self.sandwich_norms
                         else "post_attention_layernorm") + ".weight",
        }
        if self.qk_norm:
            names["n_qn"] = p + "self_attn.q_norm.weight"
            names["n_kn"] = p + "self_attn.k_norm.weight"
        if self.sandwich_norms:
            names["n_pa"] = p + "post_attention_layernorm.weight"
            names["n_pff"] = p + "post_feedforward_layernorm.weight"
        return names

    def check(self, cols: int = 8, tsi: int = 4) -> None:
        """Every GEMV tiling constraint, taken from the design's own asserts rather than restated.

        `iron/operators/gemv/design.py` requires, for a GEMV of output length M over `cols` columns
        with input tile `m_input` and output tile `m_output`:
            M % cols == 0;  m_output <= M//cols;  (M//cols) % m_output == 0;  same for m_input.
        We take m_output = M//cols (the largest legal tile), so what a spec must satisfy is
        `M % cols == 0` and `(M//cols) % tsi == 0`. GEMV additionally needs `K % 64 == 0`
        (kernel_vector_size); the K side is every dim that feeds a projection.

        NOTE this is why the tracked gen_gemma_decode.py cannot build as written: its op_kv passes
        tile_size_output=head_dim//2=128 while M//cols is 32, violating `m_output <= M//cols`. The
        device run that gated Gemma used a scratchpad diag copy, not that file.
        """
        gaps = self.unimplemented()
        if gaps:
            raise ValueError(
                f"{self.name}: {len(gaps)} model feature(s) this build cannot express yet:\n" +
                "".join(f"  - {g}\n" for g in gaps) +
                "This refusal is deliberately landed BEFORE the capabilities. Every one of these "
                "fails SILENTLY if implemented halfway -- a clean build that is wrong on a subset "
                "of layers, which teacher-forced parity over a handful of tokens can miss.")

        for label, m in (("q_dim", self.q_dim), ("kv_dim", self.kv_dim), ("d_model", self.d_model),
                         ("head_dim", self.head_dim), ("ffn", self.ffn), ("vocab", self.vocab)):
            if m % cols:
                raise ValueError(f"{self.name}: GEMV M={label}={m} not divisible by cols={cols}")
            if (m // cols) % tsi:
                raise ValueError(f"{self.name}: GEMV {label}: (M//cols)={m//cols} not a multiple of "
                                 f"tile_size_input={tsi}")
        for label, k in (("d_model", self.d_model), ("q_dim", self.q_dim),
                         ("head_dim", self.head_dim), ("ffn", self.ffn)):
            if k % 64:
                raise ValueError(f"{self.name}: GEMV K={label}={k} not a multiple of "
                                 f"kernel_vector_size=64")
        if self.head_dim % 32:
            raise ValueError(f"{self.name}: head_dim={self.head_dim} % 32 != 0 (Transpose n=32)")
        # No n_q_heads % 16 rule here any more. It cited iron/operators/softmax/op.py, which had
        # rejected `rows % 16` with no stated derivation and has since dropped it: the real
        # requirements are rows >= num_aie_columns*num_channels and rows % num_aie_columns == 0,
        # both of which softmax_cols() satisfies by construction. The stale copy outlived its
        # source and was the ONLY thing failing gemma3-270m, whose 4 heads run at 4 columns.
        if self.softmax_cols(cols) * 1 > self.n_q_heads:
            raise ValueError(f"{self.name}: Softmax rows=n_q_heads={self.n_q_heads} cannot be "
                             f"split across {cols} columns")
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError(f"{self.name}: n_q_heads={self.n_q_heads} not a multiple of "
                             f"n_kv_heads={self.n_kv_heads}")
        if self.act not in ("gelu_tanh", "silu"):
            raise ValueError(f"{self.name}: unknown act {self.act!r}")
        if self.norm_gain not in ("one_plus_w", "w"):
            raise ValueError(f"{self.name}: unknown norm_gain {self.norm_gain!r}")

    def check_seq(self, S: int) -> None:
        """Constraints that depend on the KV capacity, so they cannot be checked on the spec alone."""
        if S % 256:
            raise ValueError(f"{self.name}: max_seq={S} % 256 != 0 (Transpose m=256)")
        if S * self.head_dim <= 1023:
            raise ValueError(f"{self.name}: Repeat cols=S*head_dim={S*self.head_dim} must exceed 1023")
        if S % 8 or (S // 8) % 4:
            raise ValueError(f"{self.name}: scores GEMV M=S={S} must satisfy S%8==0 and (S//8)%4==0")
        if S % 64:
            raise ValueError(f"{self.name}: context GEMV K=S={S} must be a multiple of 64")


# Gemma-3 270M -- the checkpoint the rail was brought up on (8/8 greedy token parity on device,
# 2026-07-19). Dims from unsloth/gemma-3-270m-it config.json; mirrors rust/npu-gemma GEMMA3_270M.
GEMMA3_270M = LlmSpec(
    name="gemma3-270m", d_model=640, n_layers=18, n_q_heads=4, n_kv_heads=1, head_dim=256,
    ffn=2048, vocab=262144, eps=1e-6, act="gelu_tanh", norm_gain="one_plus_w",
    sandwich_norms=True, qk_norm=True, embed_scale="sqrt_d_model",
    rope_theta_global=1_000_000.0, rope_theta_local=10_000.0,
    sliding_window=512, sw_pattern=6, query_pre_attn_scalar=256.0,
)

# Qwen3-0.6B -- dims from Qwen/Qwen3-0.6B config.json; conventions read off transformers
# models/qwen3/modeling_qwen3.py (Qwen3RMSNorm returns `weight * x_hat`; Qwen3Attention sets
# scaling = head_dim**-0.5; Qwen3DecoderLayer carries exactly two norms; Qwen3Model feeds
# inputs_embeds through unscaled). sliding_window is null in the config and use_sliding_window
# is false, so every layer is global and RoPE is single-theta.
QWEN3_0_6B = LlmSpec(
    name="qwen3-0.6b", d_model=1024, n_layers=28, n_q_heads=16, n_kv_heads=8, head_dim=128,
    ffn=3072, vocab=151936, eps=1e-6, act="silu", norm_gain="w",
    sandwich_norms=False, qk_norm=True, embed_scale="none",
    rope_theta_global=1_000_000.0, rope_theta_local=None,
    sliding_window=None, sw_pattern=None, query_pre_attn_scalar=None,
)

# Gemma-4-12B-IT (unsloth/gemma-4-12b-it, "unsloth_fixed": true). Every axis below is VERIFIED
# against the packed dump and the checkpoint header by scripts/check_llm_spec_against_dump.py --
# nothing here is typed from config.json alone, and the axis facts live on
# [[gemma4-norm-gain-is-w-not-one-plus-w]].
#
# Two corrections to the earlier stranded literal are baked in here. norm_gain is "w", NOT
# "one_plus_w": Gemma4UnifiedRMSNorm inits the weight to ones and multiplies directly, and the
# trained weights (means 6-20, max 604) are only sane under a direct-multiply gain. That one would
# have been wrong on all 193 norm tensors and invisible to every build gate, because load_norm
# folds the gain into the weight buffer. And attn_scale is a fixed 1.0 rather than the
# head_dim**-0.5 the spec would otherwise compute -- 1/16 on sliding layers and 1/22.6 on global,
# which would have built and run quietly wrong.
#
# NOT BUILDABLE: check() refuses it, naming six features. That is the point of landing it.
GEMMA4_12B = LlmSpec(
    name="gemma4-12b", d_model=3840, n_layers=48, n_q_heads=16, n_kv_heads=8, head_dim=256,
    ffn=15360, vocab=262144, eps=1e-6, act="gelu_tanh", norm_gain="w",
    sandwich_norms=True, qk_norm=True, embed_scale="sqrt_d_model",
    rope_theta_global=1_000_000.0, rope_theta_local=10_000.0,
    sliding_window=1024, sw_pattern=6, query_pre_attn_scalar=None,
    global_head_dim=512, global_n_kv_heads=1,
    attn_scale_fixed=1.0, v_from_k_on_global=True, v_norm=True,
    rope_partial_rotary=0.25, rope_type_global="proportional",
    layer_scalar=True, logit_softcap=30.0, weight_prefix="model.language_model.",
)

SPECS = {s.name: s for s in (GEMMA3_270M, QWEN3_0_6B, GEMMA4_12B)}
