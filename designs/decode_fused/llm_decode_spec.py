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
from itertools import product

# ---------------------------------------------------------------------------------------------
# GEMM tiling constraints -- ONE implementation, two consumers.
#
# `check_prefill*` below raises on the first violation, at the point the shape is picked (K007).
# The tile sweep (sweep_gemm_tiles.py) needs the SAME verdict as data instead: it enumerates a
# candidate grid and has to count how many candidates each rule kills, and why. Two copies of these
# rules would drift -- and the rules themselves are the expensive kind, because half of them are
# NOT checked anywhere in the toolchain (K008) and the other half are checked more weakly by
# `iron/operators/gemm/op.py` than by the kernel that actually runs.
# ---------------------------------------------------------------------------------------------

GEMM_AIE_ROWS = 4            # hardcoded n_aie_rows in iron/operators/gemm/design.py::my_matmul
GEMM_FIFO_DEPTH = 2          # design.py: fifo_depth = 2 for A, B, and (unless prio_accuracy) C
GEMM_WORKER_STACK = 0xD00    # design.py passes stack_size=0xD00 to every GEMM Worker
GEMM_COLS = (1, 2, 4, 8)     # design.py's own --n-aie-cols domain; npu2 is a 4x8 array
L1_BYTES = 65536             # getLocalMemorySize(), AIE2/AIE2P core tile
MEMTILE_BYTES = 0x80000      # getMemTileSize(), AIETargetModel.h -- 512 KB per column


@dataclass(frozen=True)
class TilingRejection:
    """Why one (shape, tiling) candidate is illegal. `code` is stable so a census can count it."""
    code: str
    detail: str


def gemm_mac_dims(bfp16: bool) -> tuple[int, int, int]:
    """mm.cc's (r, s, t) for the bf16 path, which decides the tile GRANULARITY rules.

    `aie_kernels/aie2p/mm.cc` selects the microkernel by `#ifdef
    AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`: (8,8,8) with the emulation on (IRON GEMM's default),
    (4,8,8) without -- for bf16->bf16 and bf16->f32 alike. Each instantiation static_asserts
    `m % (2*r) == 0`, `k % s == 0`, `n % (2*t) == 0`. `op.py` only checks `tile_* >= r-ish`, which
    is weaker in every case and silently admits tiles the C++ rejects.
    """
    return (8, 8, 8) if bfp16 else (4, 8, 8)


def gemm_l1_bytes(tile_m: int, tile_k: int, tile_n: int, *, prio_accuracy: bool = False,
                  elem_in: int = 2, elem_out: int = 2) -> int:
    """Per-core L1 the GEMM worker holds. Nothing in the toolchain checks this for GEMM.

    A[tile_m,tile_k] and B[tile_k,tile_n] ride ObjectFifos at the default depth 2 (design.py passes
    `depths=None` on the A split()/B forward()). C is the one that moves with `prio_accuracy`:

      prio_accuracy=False   C_L1L2 carries C_l1_ty (dtype_out) at `depths=[fifo_depth]*rows` = 2.
      prio_accuracy=True    design.py sets `fifo_depth_out = 1` AND allocates a SEPARATE
                            `acc_buffer` of C_l1_ty_internal = f32, so the core holds one bf16
                            transfer tile plus one f32 accumulator -- 6 bytes/element, not 4.

    The old formula here was `4 * (A + B + C)`, which is the prio_accuracy=False case only; under
    prio_accuracy it understates the footprint by `2 * tile_m * tile_n` bytes.
    """
    c_bytes = (elem_out + 4) if prio_accuracy else (elem_out * GEMM_FIFO_DEPTH)
    return (GEMM_FIFO_DEPTH * elem_in * tile_m * tile_k
            + GEMM_FIFO_DEPTH * elem_in * tile_k * tile_n
            + c_bytes * tile_m * tile_n
            + GEMM_WORKER_STACK)


def gemm_memtile_bytes(tile_m: int, tile_k: int, tile_n: int, cols: int, *,
                       elem_in: int = 2, elem_out: int = 2) -> int:
    """L2 the busiest MemTile column holds, from design.py's three L3<->L2 ObjectFifos.

    A_L3L2 is `mem_tile_m_A * tile_k` at depth 2, B_L3L2 is `tile_k * tile_n` at depth 2, and
    C_L2L3 is `tile_m * n_aie_rows * tile_n` at depth 2 -- C dominates, and it is the term that
    grows fastest as the tile widens. A_L3L2 only lands on `min(cols, 4)` columns (Tile(2*i, 1)
    when cols == 8), but B and C are per column, so the worst column carries all three.

    A ceiling, not a placement: objectFIFO puts a buffer on ONE MemTile, so this is the capacity
    the widest tile has to fit in, and aiecc reports a miss as "'aie.tile' op Basic sequential
    allocation also failed" -- naming a tile and not a size.
    """
    a_tiles_per_shim = GEMM_AIE_ROWS // cols if cols < GEMM_AIE_ROWS else 1
    a_l2 = GEMM_FIFO_DEPTH * elem_in * (tile_m * a_tiles_per_shim) * tile_k
    b_l2 = GEMM_FIFO_DEPTH * elem_in * tile_k * tile_n
    c_l2 = GEMM_FIFO_DEPTH * elem_out * (tile_m * GEMM_AIE_ROWS) * tile_n
    return a_l2 + b_l2 + c_l2


def largest_valid_tile_n(Nout: int, cols: int, bfp16: bool = True) -> int | None:
    """Largest tile_n satisfying `Nout % (tile_n*cols) == 0` and mm.cc's `tile_n % (2*t) == 0`.

    Used only to print a working suggestion in a raised error, never to silently pick one.
    """
    if Nout % cols:
        return None
    step = 2 * gemm_mac_dims(bfp16)[2]
    per_col = Nout // cols
    return max((d for d in range(step, per_col + 1, step) if per_col % d == 0), default=None)


def gemm_tile_rejection(tile_m: int, tile_k: int, tile_n: int, cols: int, *,
                        bfp16: bool = True) -> TilingRejection | None:
    """The tile-granularity rules, which depend on neither the batch nor the projection shape."""
    if cols not in GEMM_COLS:
        return TilingRejection("cols", f"num_aie_columns={cols} is not one of {GEMM_COLS} "
                                       f"(gemm/design.py's own --n-aie-cols domain; npu2 has 8)")
    r, s, t = gemm_mac_dims(bfp16)
    path = "bfp16-emulation" if bfp16 else "plain-bf16"
    if tile_m % (2 * r):
        return TilingRejection("tile_m", f"tile_m={tile_m} not a multiple of {2 * r} -- mm.cc "
                                         f"static_assert(m % (2*r) == 0), r={r} on the {path} "
                                         f"path; op.py's own check only requires tile_m >= {r}, "
                                         f"which is weaker and would silently pass e.g. "
                                         f"tile_m={r}")
    if tile_k % s:
        return TilingRejection("tile_k", f"tile_k={tile_k} not a multiple of {s} -- mm.cc "
                                         f"static_assert(k % s == 0), s={s} on both the "
                                         f"bfp16-emulation and plain-bf16 paths; op.py's own "
                                         f"check only requires tile_k >= 8, which is weaker and "
                                         f"would silently pass e.g. tile_k=12")
    if tile_n % (2 * t):
        return TilingRejection("tile_n", f"tile_n={tile_n} not a multiple of {2 * t} -- mm.cc "
                                         f"static_assert(n % (2*t) == 0), t={t} on both compute "
                                         f"paths; op.py's own check only requires tile_n >= 8, "
                                         f"which is weaker")
    return None


def gemm_batch_rejection(batch: int, tile_m: int) -> TilingRejection | None:
    """`M % (tile_m * n_aie_rows) == 0` -- the only rule the token batch enters."""
    min_M = tile_m * GEMM_AIE_ROWS
    if batch % min_M:
        return TilingRejection("batch", f"batch={batch} not a multiple of "
                                        f"tile_m({tile_m})*n_aie_rows({GEMM_AIE_ROWS})={min_M} "
                                        f"(gemm/design.py hardcodes n_aie_rows={GEMM_AIE_ROWS}; "
                                        f"op.py's own min_M)")
    return None


def gemm_batch_tile_rejection(batch: int, tile_m: int, tile_k: int, tile_n: int, cols: int, *,
                              bfp16: bool = True) -> TilingRejection | None:
    """Tile granularity, then the batch modulus."""
    return (gemm_tile_rejection(tile_m, tile_k, tile_n, cols, bfp16=bfp16)
            or gemm_batch_rejection(batch, tile_m))


def gemm_shape_rejection(K: int, Nout: int, tile_m: int, tile_k: int, tile_n: int, cols: int, *,
                         bfp16: bool = True, prio_accuracy: bool = False,
                         check_memtile: bool = True) -> TilingRejection | None:
    """The per-projection rules: K/N divisibility, then the two capacity budgets."""
    if K % tile_k:
        return TilingRejection("K", f"K={K} not divisible by tile_k={tile_k} "
                                    f"(op.py: K % tile_k == 0)")
    min_N = tile_n * cols
    if Nout % min_N:
        fix = largest_valid_tile_n(Nout, cols, bfp16)
        hint = (f"; tile_n={fix} would satisfy it" if fix
                else f"; no tile_n multiple of {2 * gemm_mac_dims(bfp16)[2]} divides "
                     f"Nout={Nout} at cols={cols}")
        return TilingRejection("N", f"Nout={Nout} not divisible by tile_n({tile_n})*cols({cols})"
                                    f"={min_N} (op.py: N % (tile_n*num_aie_columns) == 0){hint}")
    l1 = gemm_l1_bytes(tile_m, tile_k, tile_n, prio_accuracy=prio_accuracy)
    if l1 > L1_BYTES:
        return TilingRejection("l1", f"L1 footprint {l1}B exceeds {L1_BYTES} (64KB, "
                                     f"getLocalMemorySize()) at tile_m={tile_m}, tile_k={tile_k}, "
                                     f"tile_n={tile_n}, prio_accuracy={prio_accuracy} -- "
                                     f"double-buffered A+B+C tiles plus stack_size=0xD00 "
                                     f"(design.py); unchecked anywhere else in the toolchain")
    if check_memtile:
        l2 = gemm_memtile_bytes(tile_m, tile_k, tile_n, cols)
        if l2 > MEMTILE_BYTES:
            return TilingRejection("memtile", f"MemTile footprint {l2}B exceeds {MEMTILE_BYTES} "
                                              f"(512KB, getMemTileSize()) at tile_m={tile_m}, "
                                              f"tile_k={tile_k}, tile_n={tile_n}, cols={cols} -- "
                                              f"A/B/C L3<->L2 ObjectFifos at depth "
                                              f"{GEMM_FIFO_DEPTH}, C the dominant term")
    return None


def gemm_tiling_rejection(M: int, K: int, N: int, tile_m: int, tile_k: int, tile_n: int,
                          cols: int, *, bfp16: bool = True, prio_accuracy: bool = False,
                          check_memtile: bool = True) -> TilingRejection | None:
    """One verdict for one (shape, tiling) candidate. `None` means legal."""
    rej = gemm_batch_tile_rejection(M, tile_m, tile_k, tile_n, cols, bfp16=bfp16)
    return rej or gemm_shape_rejection(K, N, tile_m, tile_k, tile_n, cols, bfp16=bfp16,
                                       prio_accuracy=prio_accuracy, check_memtile=check_memtile)


def gemm_tile_grid(*, bfp16: bool = True, tile_ms=None, tile_ks=None, tile_ns=None,
                   colss=None) -> list[tuple[int, int, int, int]]:
    """The candidate grid to enumerate, DERIVED from the kernel's granularity rather than listed.

    Defaults walk every tile the microkernel admits up to 256 (`2*r`, `s`, `2*t` steps); the caller
    narrows it. The point of enumerating the whole thing is the census: "how many candidates are
    legal" is only a useful number against a grid whose bounds come from the hardware.
    """
    r, s, t = gemm_mac_dims(bfp16)
    tile_ms = tile_ms or list(range(2 * r, 257, 2 * r))
    tile_ks = tile_ks or list(range(s, 257, s))
    tile_ns = tile_ns or list(range(2 * t, 257, 2 * t))
    colss = colss or list(GEMM_COLS)
    return [(tm, tk, tn, c) for tm, tk, tn, c in product(tile_ms, tile_ks, tile_ns, colss)]


def gemm_tile_census(M: int, K: int, N: int, *, bfp16: bool = True, prio_accuracy: bool = False,
                     check_memtile: bool = True, grid=None, **grid_kw):
    """`(legal, rejected)` for one shape over a candidate grid.

    `legal` is a list of `(tile_m, tile_k, tile_n, cols)`; `rejected` maps a rejection CODE to the
    list of `(candidate, detail)` it killed, so a caller can print how much freedom the shape
    actually has and which rule took the rest.
    """
    legal, rejected = [], {}
    for cand in (grid if grid is not None else gemm_tile_grid(bfp16=bfp16, **grid_kw)):
        rej = gemm_tiling_rejection(M, K, N, *cand, bfp16=bfp16, prio_accuracy=prio_accuracy,
                                    check_memtile=check_memtile)
        if rej is None:
            legal.append(cand)
        else:
            rejected.setdefault(rej.code, []).append((cand, rej.detail))
    return legal, rejected

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
        # EMPTY as of 2026-09-08, and kept as a list rather than deleted: it is where the next
        # model's gaps get named, and the shape of the refusal is the part worth keeping.
        #
        # The three that closed, and what closes them:
        #   per-layer geometry -- the generator keys the op vocabulary on (head_dim, n_kv_heads)
        #     and emits scratchpad.kv_params; the host writes one kv_off per distinct head_dim,
        #     takes each RoPE row width from its own buffer, and LlmArtifact::load refuses an
        #     artifact whose angle-row widths and declared head_dims disagree as sets.
        #   partial rotary -- host-side: rope_type "proportional" zeroes the inverse frequency past
        #     int(f * head_dim // 2) pairs, keeping the full head_dim width, and divides the
        #     exponent by head_dim rather than by the rotated width (which is what distinguishes it
        #     from ordinary partial rotary). Gated against transformers' own rotary embedding.
        #   logit softcap -- host-side: tanh(logits/c)*c after readback, no dispatch.
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

    def has_v_proj(self, layer: int) -> bool:
        """False where attention_k_eq_v applies: the layer has no v_proj and V comes from K.

        A separate predicate from is_global() even though the two coincide in Gemma-4, because the
        thing the graph needs to know is "is there a v projection", and a model can plausibly split
        those two ways differently.
        """
        return not (self.v_from_k_on_global and self.is_global(layer))

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

        # EVERY geometry, not just the uniform one. Under per-layer geometry the global layers have
        # their own head_dim/q_dim/kv_dim, and checking only `self.*` checks the sliding layers and
        # leaves the others to whatever the placer happens to accept. Gemma-4-12B's global shapes
        # pass -- but they passed before this loop existed too, which is the reason to assert them.
        geoms = [("", self.head_dim, self.q_dim, self.kv_dim, self.n_kv_heads)]
        if not self.geometry_is_uniform():
            gl = self.global_head_dim
            gkv = self.global_n_kv_heads
            geoms.append((" [global]", gl, self.n_q_heads * gl, gkv * gl, gkv))
        for tag, hd, qd, kvd, kvh in geoms:
            for label, m in ((f"q_dim{tag}", qd), (f"kv_dim{tag}", kvd), ("d_model", self.d_model),
                             (f"head_dim{tag}", hd), ("ffn", self.ffn), ("vocab", self.vocab)):
                if m % cols:
                    raise ValueError(f"{self.name}: GEMV M={label}={m} not divisible by cols={cols}")
                if (m // cols) % tsi:
                    raise ValueError(f"{self.name}: GEMV {label}: (M//cols)={m//cols} not a multiple of "
                                     f"tile_size_input={tsi}")
            for label, k in (("d_model", self.d_model), (f"q_dim{tag}", qd),
                             (f"head_dim{tag}", hd), ("ffn", self.ffn)):
                if k % 64:
                    raise ValueError(f"{self.name}: GEMV K={label}={k} not a multiple of "
                                     f"kernel_vector_size=64")
            if hd % 32:
                raise ValueError(f"{self.name}: head_dim{tag}={hd} % 32 != 0 (Transpose n=32)")
            if self.n_q_heads % kvh:
                raise ValueError(f"{self.name}: n_q_heads={self.n_q_heads} not a multiple of "
                                 f"n_kv_heads{tag}={kvh}")
        # No n_q_heads % 16 rule here any more. It cited iron/operators/softmax/op.py, which had
        # rejected `rows % 16` with no stated derivation and has since dropped it: the real
        # requirements are rows >= num_aie_columns*num_channels and rows % num_aie_columns == 0,
        # both of which softmax_cols() satisfies by construction. The stale copy outlived its
        # source and was the ONLY thing failing gemma3-270m, whose 4 heads run at 4 columns.
        if self.softmax_cols(cols) * 1 > self.n_q_heads:
            raise ValueError(f"{self.name}: Softmax rows=n_q_heads={self.n_q_heads} cannot be "
                             f"split across {cols} columns")
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
    def _check_prefill_tiles_and_batch(self, batch: int, tile_m: int, tile_k: int, tile_n: int,
                                        cols: int, bfp16: bool) -> None:
        """The batch- and tile-level constraints that do not depend on any one projection's shape.

        `n_aie_rows=4` is hardcoded in `iron/operators/gemm/design.py::my_matmul`; `op.py`'s
        `GEMM.__post_init__` states it back as `min_M = tile_m * num_aie_rows` and asserts
        `M % min_M == 0` -- M here is the token batch (A is token-major: `C[batch,Nout] = A[batch,K]
        @ B[K,Nout]`, B stored `[Nout,K]` and read `b_col_maj`).

        The kernel's own tile rules are NARROWER than what `op.py` checks; `gemm_mac_dims` and
        `gemm_batch_tile_rejection` at the top of this file own the derivation, so the sweep and
        this raise cannot disagree about which candidate is legal.
        """
        rej = gemm_batch_tile_rejection(batch, tile_m, tile_k, tile_n, cols, bfp16=bfp16)
        if rej is not None:
            raise ValueError(f"{self.name}: prefill {rej.detail}")

    def _check_prefill_ops(self, ops, tile_m: int, tile_k: int, tile_n: int, cols: int,
                            bfp16: bool, overrides: dict, prio_accuracy: bool = False) -> None:
        for label, K, Nout in ops:
            eff_tile_n = overrides.get(label, tile_n)
            rej = (gemm_tile_rejection(tile_m, tile_k, eff_tile_n, cols, bfp16=bfp16)
                   or gemm_shape_rejection(K, Nout, tile_m, tile_k, eff_tile_n, cols,
                                           bfp16=bfp16, prio_accuracy=prio_accuracy))
            if rej is None:
                continue
            detail = rej.detail
            if rej.code == "N" and "would satisfy it" in detail:
                fix = largest_valid_tile_n(Nout, cols, bfp16)
                detail = detail.replace(f"; tile_n={fix} would satisfy it",
                                        f"; tile_n_overrides={{{label!r}: {fix}}} would satisfy it")
            raise ValueError(f"{self.name}: prefill {label} {detail}")

    def check_prefill_projections(self, batch: int, ops, tile_m: int = 64, tile_k: int = 64,
                                   tile_n: int = 64, cols: int = 8, bfp16: bool = True,
                                   tile_n_overrides: dict | None = None,
                                   prio_accuracy: bool = False) -> None:
        """The same checks for an EXPLICIT list of `(label, K, Nout)` GEMMs.

        `check_prefill`/`check_prefill_seq` below are the two spec-shaped callers. A generator that
        splits or fuses projections differently -- gen_llm_prefill.py projects q, k and v as three
        GEMMs rather than one, because at M>1 a token's v rows sit between its k rows and the next
        token's q rows, so no contiguous slice reaches the q or k head rows alone -- checks the
        shapes it ACTUALLY builds through here (K007), instead of a nearby list that happens to
        pass.
        """
        self._check_prefill_tiles_and_batch(batch, tile_m, tile_k, tile_n, cols, bfp16)
        self._check_prefill_ops(ops, tile_m, tile_k, tile_n, cols, bfp16, tile_n_overrides or {},
                                prio_accuracy=prio_accuracy)

    def check_prefill(self, batch: int, tile_m: int = 64, tile_k: int = 64, tile_n: int = 64,
                       cols: int = 8, bfp16: bool = True, tile_n_overrides: dict | None = None,
                       prio_accuracy: bool = False) -> None:
        """Every batched-prefill (GEMM) tiling constraint for the projections whose shape depends
        only on the spec, not on the runtime KV window: qkv, o, gate, up, down. `scores`/`ctx` need
        the KV window S and are checked by `check_prefill_seq` instead (mirrors `check`/`check_seq`
        above). The lm-head is NOT here -- see the module note at the top of this section.

        Pass `tile_n_overrides={"o": 16, ...}` for any op whose Nout does not divide
        `tile_n*cols`; the raised error names a tile_n that would work.
        """
        ops = (
            ("qkv", self.d_model, self.q_dim + 2 * self.kv_dim),
            ("o", self.q_dim, self.d_model),
            ("gate", self.d_model, self.ffn),
            ("up", self.d_model, self.ffn),
            ("down", self.ffn, self.d_model),
        )
        self.check_prefill_projections(batch, ops, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n,
                                        cols=cols, bfp16=bfp16,
                                        tile_n_overrides=tile_n_overrides,
                                        prio_accuracy=prio_accuracy)

    def check_prefill_seq(self, batch: int, S: int, tile_m: int = 64, tile_k: int = 64,
                           tile_n: int = 64, cols: int = 8, bfp16: bool = True,
                           tile_n_overrides: dict | None = None,
                           prio_accuracy: bool = False) -> None:
        """The two per-q-head prefill GEMMs that depend on the KV window S: `scores`
        (K=head_dim, Nout=S) and `ctx` (K=S, Nout=head_dim). Separate from `check_prefill` for the
        same reason `check_seq` is separate from `check`: S is a runtime choice, not a spec field.

        `ctx`'s Nout=head_dim is usually far narrower than the other ops' Nout (128 for Qwen3,
        256 for Gemma-3) and almost always needs a `tile_n_overrides={"ctx": ...}` entry -- this
        does NOT also run `check_prefill`'s batch-independent ops; call both when both apply.
        """
        ops = (
            ("scores", self.head_dim, S),
            ("ctx", S, self.head_dim),
        )
        self.check_prefill_projections(batch, ops, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n,
                                        cols=cols, bfp16=bfp16,
                                        tile_n_overrides=tile_n_overrides,
                                        prio_accuracy=prio_accuracy)

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
