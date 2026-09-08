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
        qpas = self.query_pre_attn_scalar
        return (qpas ** -0.5) if qpas is not None else (self.head_dim ** -0.5)

    def is_global(self, layer_idx: int) -> bool:
        """Gemma-3 alternates local/global attention; a single-theta model is global everywhere."""
        if self.sw_pattern is None:
            return True
        return (layer_idx + 1) % self.sw_pattern == 0

    def softmax_cols(self, cap: int) -> int:
        """num_aie_columns for the attention Softmax: the widest split of the q heads that fits.

        The Softmax is built with rows = n_q_heads, and iron/operators/softmax/op.py requires
        rows >= num_aie_columns*num_channels and rows % num_aie_columns == 0 -- a head count under
        the split leaves a core with less than one tile, so the op silently computes nothing rather
        than failing. Taking the largest divisor of n_q_heads at or below the cap satisfies both:
        Qwen3's 16 heads give 8, Gemma-3's 4 give 4.
        """
        return max(d for d in range(1, cap + 1) if self.n_q_heads % d == 0)

    def qkv_dp_cols(self, cap: int) -> int:
        """num_aie_columns for QKVHeadDataParallel: every core owns a whole number of head rows."""
        heads = self.n_q_heads + 2 * self.n_kv_heads
        return max(d for d in range(1, cap + 1) if heads % d == 0)

    def qkv_dp_reason(self, cap: int) -> str | None:
        """Why QKVHeadDataParallel does not cover this spec, or None when it does.

        Every rule here is the OPERATOR's (iron/operators/qkv_head_dp/op.py), read off its
        __post_init__ rather than guessed: it applies a per-head qk-norm, and `cur`/`n_in` ride an
        HD-wide misc channel so d_model must be a whole number of head_dim chunks. Gemma-3-270M
        fails the second at 640/256 = 2.5, which no column count can fix -- worth stating, because
        the column rule looks like the whole story and is not.
        """
        if not self.qk_norm:
            return "the op applies a per-head qk-norm and this spec has none"
        if self.d_model % self.head_dim:
            return (f"d_model={self.d_model} is not a whole number of head_dim={self.head_dim} "
                    f"chunks -- cur/n_in ride the HD-wide misc channel")
        return None

    def mlp_dp_reason(self) -> str | None:
        """Why SwiGLUMLPDataParallel does not cover this spec, or None when it does."""
        if self.sandwich_norms:
            return "the fused block has no sandwich norms (this spec normalises the FFN output)"
        if self.act != "silu":
            return f"the fused block is SwiGLU; this spec's activation is {self.act!r}"
        return None

    def norm_weight_names(self, layer: int) -> dict:
        """Per-layer RMSNorm tensor names. The pre-FFN norm's NAME differs between the two families."""
        p = f"model.layers.{layer}."
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

    # ---- batched prefill (GEMM, not GEMV) ----
    #
    # The lm-head stays a GEMV at batch=1 even during prefill (only the LAST prefill position needs
    # logits) -- it is deliberately not one of the ops below, not an oversight.
    #
    # API shape: `tile_n` is ONE global default, but a caller passes `tile_n_overrides={op: tile_n}`
    # for the ops it does not cover. This is necessary because the ops below do not share an N: `ctx`
    # has Nout=head_dim, which for Qwen3 (128) and Gemma-3 (256) is far narrower than d_model/ffn, so
    # a single tile_n that fits the wide ops (qkv/gate/up/down) generally does not divide the narrow
    # one. The alternative -- one tile_n per whole spec -- would make `check_prefill` reject shapes a
    # real per-op build can still legally tile, which is exactly the K007 failure mode this file's
    # `check()` docstring already warns about (gen_gemma_decode.py's op_kv). So every failure below
    # names the offending number AND, for the Nout%... case, computes and prints a tile_n that WOULD
    # work for that op, rather than making the caller guess.
    def _check_prefill_tiles_and_batch(self, batch: int, tile_m: int, tile_k: int, bfp16: bool) -> None:
        """The batch- and tile-level constraints that do not depend on any one projection's shape.

        `n_aie_rows=4` is hardcoded in `iron/operators/gemm/design.py::my_matmul`; `op.py`'s
        `GEMM.__post_init__` states it back as `min_M = tile_m * num_aie_rows` and asserts
        `M % min_M == 0` -- M here is the token batch (A is token-major: `C[batch,Nout] = A[batch,K]
        @ B[K,Nout]`, B stored `[Nout,K]` and read `b_col_maj`).

        The kernel's own tile_m/tile_k rule is NARROWER than what `op.py` checks. `op.py` only
        asserts `tile_m >= min_tile_m` (8 for the bfp16-emulation path, 4 for plain bf16) and
        `tile_k >= 8` -- a caller can satisfy that and still fail `mm.cc`'s
        `static_assert(m % (2*r) == 0)` / `static_assert(k % s == 0)`, where `(r,s,t)` is `(8,8,8)`
        under `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` (the operator's own default) and `(4,8,8)`
        without it. `s=8` in both paths, so `tile_k % 8 == 0` is the real (and, here, coincidentally
        matching) rule; `r` differs, so the real tile_m rule is `tile_m % 16 == 0` (bfp16) or
        `tile_m % 8 == 0` (plain bf16) -- both stricter than op.py's `>=` check.
        """
        min_M = tile_m * 4
        if batch % min_M:
            raise ValueError(f"{self.name}: prefill batch={batch} not a multiple of "
                              f"tile_m({tile_m})*n_aie_rows(4)={min_M} (gemm/design.py hardcodes "
                              f"n_aie_rows=4; op.py's own min_M)")
        m_mod = 16 if bfp16 else 8
        if tile_m % m_mod:
            raise ValueError(f"{self.name}: prefill tile_m={tile_m} not a multiple of {m_mod} -- "
                              f"mm.cc static_assert(m % (2*r) == 0), r={m_mod // 2} on the "
                              f"{'bfp16-emulation' if bfp16 else 'plain-bf16'} path; op.py's own "
                              f"check only requires tile_m >= {m_mod // 2}, which is weaker and "
                              f"would silently pass e.g. tile_m={m_mod // 2}")
        if tile_k % 8:
            raise ValueError(f"{self.name}: prefill tile_k={tile_k} not a multiple of 8 -- "
                              f"mm.cc static_assert(k % s == 0), s=8 on both the bfp16-emulation "
                              f"and plain-bf16 paths; op.py's own check only requires tile_k >= 8, "
                              f"which is weaker and would silently pass e.g. tile_k=12")

    @staticmethod
    def _largest_valid_tile_n(Nout: int, cols: int) -> int | None:
        """Largest tile_n satisfying both `Nout % (tile_n*cols) == 0` and the kernel's own
        `tile_n % 16 == 0` (mm.cc: `n % (2*t) == 0`, t=8 on both compute paths) -- used only to
        print a working suggestion in a raised error, never to silently pick one."""
        if Nout % cols:
            return None
        per_col = Nout // cols
        candidates = (d for d in range(16, per_col + 1, 16) if per_col % d == 0)
        return max(candidates, default=None)

    def _check_prefill_ops(self, ops, tile_m: int, tile_k: int, tile_n: int, cols: int,
                            bfp16: bool, overrides: dict) -> None:
        for label, K, Nout in ops:
            if K % tile_k:
                raise ValueError(f"{self.name}: prefill {label} K={K} not divisible by "
                                  f"tile_k={tile_k} (op.py: K % tile_k == 0)")
            eff_tile_n = overrides.get(label, tile_n)
            if eff_tile_n % 16:
                raise ValueError(f"{self.name}: prefill {label} tile_n={eff_tile_n} not a "
                                  f"multiple of 16 -- mm.cc static_assert(n % (2*t) == 0), t=8 on "
                                  f"both compute paths; op.py's own check only requires "
                                  f"tile_n >= 8, which is weaker")
            min_N = eff_tile_n * cols
            if Nout % min_N:
                fix = self._largest_valid_tile_n(Nout, cols)
                hint = (f"; tile_n_overrides={{{label!r}: {fix}}} would satisfy it" if fix
                        else f"; no tile_n multiple of 16 divides Nout={Nout} at cols={cols}")
                raise ValueError(f"{self.name}: prefill {label} Nout={Nout} not divisible by "
                                  f"tile_n({eff_tile_n})*cols({cols})={min_N} (op.py: "
                                  f"N % (tile_n*num_aie_columns) == 0){hint}")
            # L1 capacity: nothing in the toolchain checks this for GEMM. Per-core L1 is 64KB
            # (getLocalMemorySize()). The GEMM worker's A[tile_m,tile_k]/B[tile_k,tile_n]/
            # C[tile_m,tile_n] buffers are all bf16 (2B) and all double-buffered (ObjectFifo
            # default depth=2; design.py passes depths=None on the A/B split()/forward() calls, so
            # they inherit it, and explicitly passes depths=[fifo_depth]*n_aie_rows on C's join(),
            # same value) -- hence the 2*2=4 byte-per-element factor below. Plus the per-core
            # stack_size=0xD00 reservation design.py passes to each Worker(...).
            l1_bytes = 4 * (tile_m * tile_k + tile_k * eff_tile_n + tile_m * eff_tile_n) + 0xD00
            if l1_bytes > 65536:
                raise ValueError(f"{self.name}: prefill {label} L1 footprint {l1_bytes}B exceeds "
                                  f"64KB (getLocalMemorySize()) at tile_m={tile_m}, tile_k={tile_k}, "
                                  f"tile_n={eff_tile_n} -- 4*(A+B+C double-buffered bf16 tiles) + "
                                  f"stack_size=0xD00 (design.py); unchecked anywhere else in the "
                                  f"toolchain for GEMM")

    def check_prefill(self, batch: int, tile_m: int = 64, tile_k: int = 64, tile_n: int = 64,
                       cols: int = 8, bfp16: bool = True, tile_n_overrides: dict | None = None
                       ) -> None:
        """Every batched-prefill (GEMM) tiling constraint for the projections whose shape depends
        only on the spec, not on the runtime KV window: qkv, o, gate, up, down. `scores`/`ctx` need
        the KV window S and are checked by `check_prefill_seq` instead (mirrors `check`/`check_seq`
        above). The lm-head is NOT here -- see the module note at the top of this section.

        Pass `tile_n_overrides={"o": 16, ...}` for any op whose Nout does not divide
        `tile_n*cols`; the raised error names a tile_n that would work.
        """
        self._check_prefill_tiles_and_batch(batch, tile_m, tile_k, bfp16)
        ops = (
            ("qkv", self.d_model, self.q_dim + 2 * self.kv_dim),
            ("o", self.q_dim, self.d_model),
            ("gate", self.d_model, self.ffn),
            ("up", self.d_model, self.ffn),
            ("down", self.ffn, self.d_model),
        )
        self._check_prefill_ops(ops, tile_m, tile_k, tile_n, cols, bfp16, tile_n_overrides or {})

    def check_prefill_seq(self, batch: int, S: int, tile_m: int = 64, tile_k: int = 64,
                           tile_n: int = 64, cols: int = 8, bfp16: bool = True,
                           tile_n_overrides: dict | None = None) -> None:
        """The two per-q-head prefill GEMMs that depend on the KV window S: `scores`
        (K=head_dim, Nout=S) and `ctx` (K=S, Nout=head_dim). Separate from `check_prefill` for the
        same reason `check_seq` is separate from `check`: S is a runtime choice, not a spec field.

        `ctx`'s Nout=head_dim is usually far narrower than the other ops' Nout (128 for Qwen3,
        256 for Gemma-3) and almost always needs a `tile_n_overrides={"ctx": ...}` entry -- this
        does NOT also run `check_prefill`'s batch-independent ops; call both when both apply.
        """
        self._check_prefill_tiles_and_batch(batch, tile_m, tile_k, bfp16)
        ops = (
            ("scores", self.head_dim, S),
            ("ctx", S, self.head_dim),
        )
        self._check_prefill_ops(ops, tile_m, tile_k, tile_n, cols, bfp16, tile_n_overrides or {})

    def legal_prefill_batches(self, cap: int, tile_m: int = 64, tile_k: int = 64,
                               tile_n: int = 64, cols: int = 8, bfp16: bool = True,
                               tile_n_overrides: dict | None = None) -> list[int]:
        """Every prefill batch size <= cap this spec can legally build `check_prefill` at, for a
        given tile/column config -- so a caller asks "what M can I use?" once instead of guessing
        a batch and catching a `ValueError`.

        batch enters `check_prefill` only through `batch % (tile_m*4) == 0`; every other
        constraint there is batch-independent. So this is either every multiple of `tile_m*4` up
        to `cap`, or none (if the tile/column config itself is illegal for this spec) -- never
        checks `check_prefill_seq`'s S-dependent ops, since S is not a spec property.
        """
        step = tile_m * 4
        try:
            self.check_prefill(step, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, cols=cols,
                                bfp16=bfp16, tile_n_overrides=tile_n_overrides)
        except ValueError:
            return []
        return list(range(step, cap + 1, step))


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

SPECS = {s.name: s for s in (GEMMA3_270M, QWEN3_0_6B)}
