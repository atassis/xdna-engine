# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel decode SwiGLU MLP: every core runs every stage on its own 1/N slice, unlike
fuse/mlp-block's spatial 5-core PIPELINE (measured +45% slower -- one core streamed all 12.58 MB
of gate+up weights while 27 of 32 cores sat idle). Fusion stays TEMPORAL (one aie.device, one
aiex.configure); parallelism is SPATIAL, across N cores, at every stage:

    x1  = cur + a                             every core, full D (replicated -- cheap, 2 KB)
    hf  = RMSNorm_weighted(x1, n_pf)          every core, full D (replicated -- avoids a reduction)
    g   = Wg[c*FF/N:(c+1)*FF/N] @ hf          core c's own FF/N output rows
    u   = Wu[c*FF/N:(c+1)*FF/N] @ hf          core c's own FF/N output rows
    gh  = silu(g) * u                         core c's own FF/N slice
    -- ALL-GATHER gh (the one unavoidable exchange -- down's K is the full FF) --
    d   = Wd[c*D/N:(c+1)*D/N] @ gh            core c's own D/N output rows
    nxt[c*D/N:(c+1)*D/N] = x1[same] + d       core c's own D/N output rows

N compute tiles means N cores each with the SAME 2-input/2-output DMA-channel budget the
single-core fuse/mlp-block attempt tripped over. Three consolidations make N cores fit it:

  MISC (1 in): cur, a, n_pf are D-shaped, needed once each. The all-gathered gh is FF-shaped, but
  FF is a whole multiple of D (ratio R = FF/D), so it comes back as R separate D-sized reads on
  the SAME channel and is reassembled by an explicit-offset copy kernel. depth=2 lets the core
  hold cur+a simultaneously (`.acquire(2)`) for the first add; every other use is one at a time.
  Broadcast to all N cores via N `.cons()` handles on one producer -- fan-out is in the
  stream-switch fabric, not the source's own DMA (the mechanism fuse/mlp-block's P1 already uses
  to feed both P2 and P3 from one output port).

  WEIGHT (1 in): Wg, Wu (row width D) and Wd (row width FF) are three different L3 buffers, but
  the shared ObjectFifo tile is sized so ONE flat shape serves all three: TSI_GU rows of D
  elements == TSI_D rows of FF elements (TSI_D = TSI_GU // R). Reused sequentially for Wg's tiles,
  then Wu's, then Wd's -- the same "same fifo, several fill() calls" idiom fuse/mlp-block's
  gate/up A-tile stream already uses.

  OUTPUT (1 of the 2 available, one spare): gh's own FF/N slice is emitted in R D/N-sized chunks
  (an offset-mul kernel reading straight out of the g/u buffers, no separate gh-sized scratch) onto
  the SAME small ObjectFifo the final residual reuses for nxt -- R+1 sequential produce/drain
  rounds share one channel.

The gather has no native all-to-all primitive: ObjectFifoLink is many-to-one XOR one-to-many,
never both (confirmed against attn_core's identical wall in this same codebase). It round-trips
through an internal DRAM scratch buffer instead: each core drains its own gh slice in R chunks
(N*R small, disjoint, offset-addressed drains -- no join, no memtile fan-in, since at n_aie_rows=1
each core already owns a whole column's slice), then every core re-reads the FULL scratch buffer
back over the MISC channel. Cost is trivial (2*FF elements moved, ~0.03% of the layer's traffic);
correctness rests only on a TaskGroup barrier separating the drains from the refill.

MEASURED (device-free, aiecc placement): n_aie_rows=1 (N=n_aie_cols<=8) places with plain flat
per-core ObjectFifos -- no explicit MemTile step needed, the automatic placer inserts whatever
staging one column needs (it column-major-fills 4 rows before moving to the next column, so N=8
lands on physical columns 0-1, not 0-7, and that is fine -- the design never assumes a "logical
core c" is "physical column c"). Both N=16 and N=32 fail there: aiecc's error is explicit --
"no ShimNOCTile ... free: all 8 ShimNOCTile(s) are at 16/16 input... channels used" -- the
DEVICE-WIDE ShimDMA budget (16, matching get_shim_dma_limit()) is spent one channel per DISTINCT
shim-facing ObjectFifo, not "2 per tile" as the compute-tile figure might suggest; misc(1) +
weight(N) already exceeds it at N=16.

n_aie_rows>1 fixes this with the SAME two combinators fuse/mlp-block's own report cites as the
proven multi-row pattern in this codebase (whole_array_silu_iron.py's A-split / C-join): per
GROUP of n_aie_rows cores sharing one shim source,
  - WEIGHT: one group-level ObjectFifo (n_aie_rows*WTILE_UNITS per fill) is `.split()` into
    n_aie_rows row sub-fifos (WTILE_ty each) at a MemTile. One fill per weight-tile ROUND gathers
    all n_aie_rows rows' data for that round with a strided TAP (rows are NOT adjacent in Wg/Wd's
    own row-major layout -- consecutive rows in one group are FF_PER_CORE, resp. D_PER_CORE,
    elements apart), correct because ObjectFifoLink's own offsets place them contiguously in the
    fetched tile in row order, matching the `.split()` offsets below.
  - OUTPUT: one group-level ObjectFifo is `.join()` from n_aie_rows row sub-fifos (DPC_ty each);
    cores write into the row sub-fifo exactly as at n_aie_rows=1. Draining gh needs the same
    strided TAP (destination rows are FF_PER_CORE apart) since the join's own buffer is
    contiguous by row; draining the final nxt round does not (D_PER_CORE apart on both sides).
This only reduces the number of DISTINCT shim-facing ObjectFifos from N to n_aie_cols for both
weight and output -- misc is already 1 regardless of n_aie_rows (see the MISC paragraph above).

FUSE_O (fuse_o=True, n_aie_rows=1 only -- see below): folds the attention output projection
`a = Wo @ cx` into this design too, deleting a whole standalone GEMV design/configure/run from the
decode runlist. `a` was the ONLY external input this design didn't already compute on-chip; now
`cx` (QD-wide) and `Wo` ([D, QD]) arrive instead, and every core computes its own D/N slice of `a`
from its own row-slice of Wo, exactly like the existing Wd/d_buf step. Two wrinkles this adds:

  cx reassembly: QD is a whole multiple of D (R_CX = QD/D), so cx arrives as R_CX D-sized misc
  broadcasts and is reassembled by the SAME explicit-offset-copy idiom gh already uses -- no new
  channel, just more rounds through the existing one.

  Wo's row-tile CANNOT share Wg/Wu/Wd's byte-identical WTILE_ty tile cleanly: the shared tile size
  is forced to lcm(D, FF, QD) = 6*D (FF=3D, QD=2D here), which makes TSI_O = 6*D/QD a multiple of
  3, and D_PER_CORE = D/N a power of two for every N this codebase places -- a multiple of 3 can
  never divide a power of two, so a uniform TSI_O-row tiling of D_PER_CORE always leaves a
  remainder, for ANY N. Reusing the channel anyway with a "short" final fill was rejected: nothing
  in this codebase does a partial-tile fill into a fixed-shape ObjectFifo object (see
  qkv_head_dp/design.py's identical refusal, "a D-wide tile would need a 128-of-1024 partial fill
  ... which nothing in this codebase does"), and a dedicated second weight channel for Wo is a
  non-starter on ITS OWN merits: misc(1)+weight(N)+weight_o(N) is 17 input channels at N=8, one
  over the same 16-channel device-wide budget that already caps this design at N=8 (see above) --
  confirmed independently by attn_core's fusion attempt, which hit exactly this wall trying to
  fold op_o in elsewhere ("all 8 ShimNOCTile(s) are at 9/16 input, 16/16 output channels used").

  The fix is a 1-row OVERLAP, not a partial fill: every core reads ceil(D_PER_CORE/TSI_O) FULL
  TSI_O-row tiles (a window of N_O_TILES*TSI_O rows, >= D_PER_CORE), starting at its own
  c*D_PER_CORE offset in Wo. For every core but the last this window simply reads a few of the
  NEXT core's real rows too (harmless -- Wo is read-only, and the extra rows are computed but
  never drained). Only the LAST core's window would run past Wo's true D rows, so Wo is padded
  with O_OVERLAP (< TSI_O) zero rows at the very end -- a single, tiny, explicit append, not an
  assumption about stale buffer contents. Every fill is a full, byte-identical WTILE_ty tile,
  identical in shape to the existing Wg/Wu/Wd fills; only the LAST core's window ever touches a
  padding row, and that row's own output (computed, never drained) is exactly zero. The per-core
  matvec output lands in a plain (n_aie_rows=1-scoped) O_WINDOW-sized scratch buffer, not
  ObjectFifo-backed, so the overlap/pad tail costs nothing beyond that buffer's own bytes; only
  its first D_PER_CORE elements -- this core's real slice -- are drained.

  `a`'s own all-gather reuses gh's exact mechanism: each core drains its D_PER_CORE-wide real
  slice onto the shared OUTPUT channel (now a NEW, first round ahead of gh's own R rounds), then
  every core re-reads the full D-wide result over MISC once the drains are barriered -- structured
  as its own TaskGroup pair (tg_a_drain/tg_a_refill) ahead of the existing gh pair, because this
  core now produces its FIRST output (the a-slice drain) before consuming cur/n_pf, not after (see
  op.py's Runtime docstring for why a stale two-group split would deadlock here).
"""

from ml_dtypes import bfloat16
import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern

from iron.operators._trace import maybe_enable_trace

# Shared weight-tile row counts (see module docstring, WEIGHT channel). Fixed, not searched: this
# design is gated at Qwen3-0.6B's D=1024/FF=3072 (R=3) shape only, and 6/2 is verified below to
# fit L1 at every N in {8, 16, 32} this file is built against.
TSI_GU = 6
TSI_D = 2


def _flat_tap(total, size, offset=0):
    """A contiguous [offset:offset+size) read/write into an L3 buffer of `total` elements.
    `total` is the FULL buffer's own declared size (TensorAccessPattern validates offset+extent
    against it), which is why a bare (size,) tensor_dims -- correct only at offset=0 -- silently
    rejects every sliced fill/drain this design needs once `total` differs from `size`."""
    return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])


def _group_tap(total, offset, n_rows, row_stride, run_hi, run_lo):
    """A group-of-`n_rows` gather/scatter: row r's `run_hi*run_lo`-element contiguous run sits
    `r*row_stride` elements apart in the L3 buffer, but CONTIGUOUS (row order) in the L2/L1 tile
    on the other end of the ObjectFifoLink -- exactly what `.split()`/`.join()`'s own `offsets=`
    (row r at r*run_hi*run_lo in the fetched/joined tile) assume. `run_hi*run_lo` splits a
    per-row run that exceeds the shim's 1023-element wrap cap into two dims (see _split_run) --
    at n_rows==1 this degenerates to _flat_tap's own [0,1,1,size] shape."""
    return TensorAccessPattern(
        (1, total), offset, [1, n_rows, run_hi, run_lo], [0, row_stride, run_lo, 1]
    )


def _split_run(total, lim=1023, gran=2):
    """Largest (hi, lo) with hi*lo == total, lo <= lim, lo a multiple of `gran` -- the shim BD
    4-dim wrap-size cap (mlir-aie's verifyStridesWraps), same constraint gemv/design.py's own
    split_run guards. Raises if no such split exists."""
    for lo in range(lim - (lim % gran), 0, -gran):
        if total % lo == 0:
            return total // lo, lo
    raise ValueError(f"{total} has no wrap-legal split (lim={lim}, gran={gran})")


def my_swiglu_mlp_dp(
    dev, D, FF, epsilon=1e-5, stack_size=0x800, func_prefix="", n_aie_cols=8, n_aie_rows=1,
    QD=None, fuse_o=False, trace_size=0, weight_dtype="bf16", group_size=0,
    weight_depth=2, tile_rows_gu=None,
):
    """`func_prefix` is required (not optional) by iron.common.sequence.FusedDispatch the moment
    this design is placed in an OperatorSequence -- see gemv/design.py's identical parameter for
    the same reason. N = n_aie_cols * n_aie_rows; n_aie_rows=1 is the plain-ObjectFifo topology,
    n_aie_rows>1 uses the MemTile split/join topology -- see the module docstring for both.

    `fuse_o=True` folds `a = Wo @ cx` into this design (see module docstring's FUSE_O section);
    it needs `QD` (the attention context width) and is currently n_aie_rows==1 only -- the
    overlap/pad arithmetic below is derived for the plain per-core-direct-fill topology and has
    not been re-derived for the MemTile split/join one.
    """
    # Local shadowing of the module defaults, so an arm can trade tile ROWS against fifo DEPTH at
    # constant L1: depth * TSI_GU * D * 2 bytes is what the budget below actually sees.
    TSI_GU = tile_rows_gu if tile_rows_gu else globals()["TSI_GU"]
    TSI_D = TSI_GU // (FF // D)
    assert TSI_D >= 1 and TSI_GU % (FF // D) == 0, (
        f"tile_rows_gu={TSI_GU} must be a multiple of R=FF/D={FF // D}")

    N = n_aie_cols * n_aie_rows
    assert FF % D == 0, f"this design assumes FF ({FF}) is a whole multiple of D ({D})"
    R = FF // D  # =3 at Qwen3-0.6B's shape; also N_GH_CHUNKS (misc) and N_GH_ROUNDS (output)
    assert D % N == 0 and FF % N == 0, f"D={D}, FF={FF} must both be divisible by N={N}"
    D_PER_CORE = D // N
    FF_PER_CORE = FF // N
    assert FF_PER_CORE % 32 == 0, (
        f"silu_tile_bf16 walks its buffer 32 lanes at a time with no tail handling; "
        f"FF/N ({FF_PER_CORE}) must be a multiple of 32"
    )
    # WEIGHT WIRE UNITS. bf16 weights are addressed in ELEMENTS; a group-quantized weight is a flat
    # byte row -- [n_groups x f32 scale][packed payload] for the symmetric dtypes, and
    # [n_groups x bf16 scale][n_groups x bf16 min][packed payload] for the affine ones ("int4a" /
    # "int8a") -- so every weight size, offset and stride
    # below is in whatever unit the wire format uses. Activations (hf, gh, nxt, cx) are ALWAYS bf16
    # and keep their element units; mixing the two is exactly the bytes-vs-elements seam that has
    # no owner, so the weight quantities are named WROW_* and nothing else changes.
    if weight_dtype == "bf16":
        WDT, WUNIT = bfloat16, 2
        WROW_D, WROW_FF = D, FF
        WROW_QD = QD
    else:
        from iron.operators.gemv.quant import row_stride_bytes
        assert weight_dtype in ("int4", "int8", "int4a", "int8a"), \
            f"unknown weight_dtype {weight_dtype!r}"
        assert group_size > 0, "weight_dtype != 'bf16' needs an explicit group_size > 0"
        assert n_aie_rows == 1, (
            "quantized weights are only derived for the plain (n_aie_rows=1) topology -- the "
            "MemTile split/join TAPs below still carry element strides"
        )
        for name, K in (("D", D), ("FF", FF)) + ((("QD", QD),) if fuse_o else ()):
            assert K % group_size == 0, f"{name}={K} must be a whole number of groups ({group_size})"
        WDT, WUNIT = np.int8, 1
        WROW_D = row_stride_bytes(D, group_size, weight_dtype)
        WROW_FF = row_stride_bytes(FF, group_size, weight_dtype)
        WROW_QD = row_stride_bytes(QD, group_size, weight_dtype) if fuse_o else None

    # The shared-tile invariant is what lets Wg/Wu (row width D) and Wd (row width FF) ride ONE
    # ObjectFifo. It survives quantization because row_stride_bytes is affine in K with the same
    # group size, so the R = FF/D ratio is preserved: at int4 g128, 6*544 == 2*1632 == 3264 B.
    assert TSI_GU * WROW_D == TSI_D * WROW_FF, (
        f"shared weight tile must be identical for gate/up and down: "
        f"{TSI_GU}*{WROW_D} != {TSI_D}*{WROW_FF} (weight_dtype={weight_dtype})"
    )
    assert FF_PER_CORE % TSI_GU == 0 and D_PER_CORE % TSI_D == 0, (
        f"N={N}: FF/N ({FF_PER_CORE}) must divide by TSI_GU ({TSI_GU}) and "
        f"D/N ({D_PER_CORE}) must divide by TSI_D ({TSI_D})"
    )
    N_GU_TILES = FF_PER_CORE // TSI_GU
    N_D_TILES = D_PER_CORE // TSI_D
    WTILE_UNITS = TSI_GU * WROW_D

    if fuse_o:
        assert n_aie_rows == 1, "fuse_o is only derived for the plain (n_aie_rows=1) topology"
        assert QD is not None, "fuse_o needs QD (the attention context width)"
        assert QD % D == 0, f"fuse_o assumes QD ({QD}) is a whole multiple of D ({D})"
        R_CX = QD // D
        assert WTILE_UNITS % WROW_QD == 0, (
            f"fuse_o needs the shared weight tile ({WTILE_UNITS} units) to be a whole number of "
            f"Wo rows ({WROW_QD} units each); it isn't, so Wo can't share this channel"
        )
        TSI_O = WTILE_UNITS // WROW_QD
        N_O_TILES = -(-D_PER_CORE // TSI_O)          # ceil division
        O_WINDOW = N_O_TILES * TSI_O                 # rows actually read per core (>= D_PER_CORE)
        O_OVERLAP = O_WINDOW - D_PER_CORE             # extra rows read past this core's own slice
        assert O_OVERLAP < TSI_O                      # ceil() guarantees this; sanity check
        WO_ROWS_PADDED = D + O_OVERLAP                # Wo's own arg spec size, in rows

    # The affine kernels keep one float per quant group on the STACK (mv_quant.cc's
    # `float bsum[n_groups]`, the per-group sums of B). It is the only stack term this design
    # controls, and stack_size is otherwise an opaque constant the budget below just adds -- so
    # size it here rather than let it be a hanging number. Worst case is the widest K, since
    # n_groups = K/group_size: at FF=3072 group_size=32 that is 96 floats = 384 B of the 2048 B
    # default. The 512 B floor left for everything else (two accums, the ones vector, the frame)
    # is a policy, not a measurement; aiecc validates the real requirement against stack_size per
    # core and fails the build if it is short, so this assert exists to fail EARLIER and to name
    # the term, not to be the only guard.
    if weight_dtype in ("int4a", "int8a"):
        bsum_bytes = 4 * (max(K for K in (D, FF) + ((QD,) if fuse_o else ())) // group_size)
        assert bsum_bytes + 512 <= stack_size, (
            f"affine bsum[] needs {bsum_bytes} B of the {stack_size} B core stack at "
            f"group_size={group_size}; raise stack_size or the group"
        )

    # L1 budget check (64 KB/core) -- see module docstring's channel accounting for what each
    # buffer is. Computed, not guessed: this is exactly the "hanging numbers are bugs" rule.
    L1_BYTES = 65536
    misc_bytes = 2 * (D * 2)  # depth=2
    weight_bytes = weight_depth * (WTILE_UNITS * WUNIT)
    out_bytes = 2 * (D_PER_CORE * 2)  # depth=2
    persistent_bytes = 2 * (D * 2) + (FF * 2) + 2 * (FF_PER_CORE * 2) + (D_PER_CORE * 2)
    # x1_buf + hf_buf         gh_buf      g_buf + u_buf          d_buf
    if fuse_o:
        persistent_bytes += (QD * 2) + (O_WINDOW * 2)
        # cx_buf                a_slice_buf
    total = misc_bytes + weight_bytes + out_bytes + persistent_bytes + stack_size
    assert total <= L1_BYTES, (
        f"N={N}: estimated L1 use {total} B exceeds {L1_BYTES} B "
        f"(misc={misc_bytes} weight={weight_bytes} out={out_bytes} "
        f"persistent={persistent_bytes} stack={stack_size})"
    )

    D_ty = np.ndarray[(D,), np.dtype[bfloat16]]
    FF_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    DPC_ty = np.ndarray[(D_PER_CORE,), np.dtype[bfloat16]]
    FFPC_ty = np.ndarray[(FF_PER_CORE,), np.dtype[bfloat16]]
    WTILE_ty = np.ndarray[(WTILE_UNITS,), np.dtype[WDT]]
    Wg_L3_ty = np.ndarray[(FF * WROW_D,), np.dtype[WDT]]
    Wd_L3_ty = np.ndarray[(D * WROW_FF,), np.dtype[WDT]]
    GH_SCRATCH_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    if fuse_o:
        QD_ty = np.ndarray[(QD,), np.dtype[bfloat16]]
        OWIN_ty = np.ndarray[(O_WINDOW,), np.dtype[bfloat16]]
        Wo_L3_ty = np.ndarray[(WO_ROWS_PADDED * WROW_QD,), np.dtype[WDT]]
        A_SCRATCH_ty = np.ndarray[(D,), np.dtype[bfloat16]]

    # ---- kernels (one archive per core -- every core plays every role) ----
    # The weight dtype is IN the archive name. Without it a bf16 build silently reuses a cached
    # int4 archive built earlier under the same name and dies at link with
    # "undefined symbol: <prefix>matvec_vectorized_bf16_bf16" -- the artifact-key collision this
    # tree already documents for the fused sequence name.
    _WTAG = "" if weight_dtype == "bf16" else f"_{weight_dtype}g{group_size}"
    CORE_ARCHIVE = f"{func_prefix}swiglu_mlp_dp_core{_WTAG}.a"
    # Two DIFFERENT bindings, not one reused: a Kernel() fixes ONE func.func signature for its
    # symbol, and the two call sites acquire differently-sized buffers (x1=cur+a is full-D; the
    # final residual is D/N-sized). The plain add costs nothing extra -- eltwise_add_bf16_vector
    # is already linked into every other design that touches add.cc.
    add_kernel = Kernel(
        f"{func_prefix}eltwise_add_bf16_vector", CORE_ARCHIVE, [D_ty, D_ty, D_ty, np.int32]
    )
    add_off_kernel = Kernel(
        f"{func_prefix}eltwise_add_offset_a_bf16_vector", CORE_ARCHIVE,
        [D_ty, DPC_ty, DPC_ty, np.int32, np.int32],
    )
    wnorm_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm_fixed", CORE_ARCHIVE, [D_ty, D_ty, D_ty, np.float32]
    )
    MV = f"matvec_vectorized_{weight_dtype}_bf16"
    mv_gu_kernel = Kernel(
        f"{func_prefix}{MV}", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, D_ty, FFPC_ty],
    )
    # Down's own matvec: different DIM_K, same extern "C" name as mv_gu_kernel -- symbol
    # uniqueness is device-wide (one aie.device, one symbol table), so op.py compiles this one
    # from a prefixed object (see fuse/mlp-block's identical mv.cc reuse for the same reason).
    mv_d_kernel = Kernel(
        f"{func_prefix}down_{MV}", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, FF_ty, DPC_ty],
    )
    silu_kernel = Kernel(f"{func_prefix}silu_tile_bf16", CORE_ARCHIVE, [np.int32, FFPC_ty])
    mul_off_kernel = Kernel(
        f"{func_prefix}eltwise_mul_offset_ab_bf16_vector", CORE_ARCHIVE,
        [FFPC_ty, FFPC_ty, DPC_ty, np.int32, np.int32],
    )
    copy_off_kernel = Kernel(
        f"{func_prefix}copy_offset_bf16_vector", CORE_ARCHIVE, [FF_ty, D_ty, np.int32, np.int32]
    )
    if fuse_o:
        # o's own matvec: DIM_K=QD, distinct from mv_gu (DIM_K=D) and mv_d (DIM_K=FF) -- same
        # symbol-uniqueness reasoning as mv_d_kernel above.
        mv_o_kernel = Kernel(
            f"{func_prefix}o_{MV}", CORE_ARCHIVE,
            [np.int32, np.int32, WTILE_ty, QD_ty, OWIN_ty],
        )
        # copy_offset_bf16_vector is (dst, src, size, dst_offset) over raw pointers -- no
        # compile-time size baked in -- but a func.func symbol is keyed by NAME only, and MLIR's
        # verifier refuses two declarations of the same symbol with different memref types
        # ("redefinition of symbol"), so each new call-site shape needs its own renamed object
        # (op.py's cx_copy_obj/oa_copy_obj), exactly like mv_d_kernel's "down_" prefix below.
        copy_off_cx_kernel = Kernel(
            f"{func_prefix}cx_copy_offset_bf16_vector", CORE_ARCHIVE,
            [QD_ty, D_ty, np.int32, np.int32],
        )
        copy_off_a_kernel = Kernel(
            f"{func_prefix}oa_copy_offset_bf16_vector", CORE_ARCHIVE,
            [DPC_ty, OWIN_ty, np.int32, np.int32],
        )

    # ---- ObjectFifos: misc(1, always) + weight(n_aie_cols groups) + output(n_aie_cols groups).
    # n_aie_rows==1: plain per-core ObjectFifos, direct L3<->L1 (no MemTile step in this file --
    # the automatic placer inserts whatever staging one column needs). n_aie_rows>1: one
    # group-level ObjectFifo per column, split (weight) / joined (output) into n_aie_rows row
    # sub-fifos at a MemTile -- see module docstring. Either way `weight_ofs[c]`/`out_ofs[c]`
    # (c = g*n_aie_rows + r) end up as the per-core handles core_fn acquires/releases from; it
    # does not know or care which path built them. fuse_o adds no new shim-facing ObjectFifo: Wo
    # rides the SAME weight_ofs/gweight_ps channel as Wg/Wu/Wd, and `a`'s all-gather rides the
    # SAME out_ofs/gout_cs channel gh's all-gather already uses (see module docstring). ----
    misc_of = ObjectFifo(D_ty, name="misc", depth=2)
    weight_ofs = [None] * N
    out_ofs = [None] * N
    if n_aie_rows == 1:
        for c in range(N):
            weight_ofs[c] = ObjectFifo(WTILE_ty, name=f"weight_{c}", depth=weight_depth)
            out_ofs[c] = ObjectFifo(DPC_ty, name=f"out_{c}", depth=2)
        group_weight_ofs = weight_ofs  # sequence() fills/drains these directly, one per "group"
        group_out_ofs = out_ofs
    else:
        RUN_HI, RUN_LO = _split_run(WTILE_UNITS)
        GROUP_WTILE_ty = np.ndarray[(n_aie_rows * WTILE_UNITS,), np.dtype[WDT]]
        GROUP_OTILE_ty = np.ndarray[(n_aie_rows * D_PER_CORE,), np.dtype[bfloat16]]
        group_weight_ofs = []
        group_out_ofs = []
        for g in range(n_aie_cols):
            gw = ObjectFifo(GROUP_WTILE_ty, name=f"weight_g{g}", depth=weight_depth)
            sub_w = gw.cons().split(
                [r * WTILE_UNITS for r in range(n_aie_rows)],
                obj_types=[WTILE_ty] * n_aie_rows,
                names=[f"weight_{g}_{r}" for r in range(n_aie_rows)],
                depths=[2] * n_aie_rows,
            )
            go = ObjectFifo(GROUP_OTILE_ty, name=f"out_g{g}", depth=2)
            sub_o = go.prod().join(
                [r * D_PER_CORE for r in range(n_aie_rows)],
                obj_types=[DPC_ty] * n_aie_rows,
                names=[f"out_{g}_{r}" for r in range(n_aie_rows)],
                depths=[2] * n_aie_rows,
            )
            for r in range(n_aie_rows):
                weight_ofs[g * n_aie_rows + r] = sub_w[r]
                out_ofs[g * n_aie_rows + r] = sub_o[r]
            group_weight_ofs.append(gw)
            group_out_ofs.append(go)

    def core_fn(misc_c, weight_c, out_p,
                x1_buf, hf_buf, gh_buf, g_buf, u_buf, d_buf,
                add_k, add_off_k, wnorm_k, mv_gu_k, mv_d_k, silu_k, mul_off_k, copy_off_k,
                core_id, *fo):
        if fuse_o:
            (cx_buf, a_slice_buf, mv_o_k, copy_off_cx_k, copy_off_a_k) = fo

            # step -1: reassemble cx (QD-wide) from R_CX D-sized misc broadcasts.
            for i in range(R_CX):
                chunk = misc_c.acquire(1)
                copy_off_cx_k(cx_buf, chunk, D, i * D)
                misc_c.release(1)

            # step 0: a_slice[0:O_WINDOW) = Wo[my window] @ cx -- a window of N_O_TILES full
            # TSI_O-row tiles, always >= D_PER_CORE rows (see module docstring's FUSE_O section).
            for j in range_(N_O_TILES):
                j32 = index.casts(T.i32(), j)
                row_off = j32 * TSI_O
                wt = weight_c.acquire(1)
                mv_o_k(TSI_O, row_off, wt, cx_buf, a_slice_buf)
                weight_c.release(1)

            # step 0b: drain only this core's real D_PER_CORE-wide prefix (discard the overlap
            # tail) onto the shared output channel -- the FIRST round through it now, ahead of
            # gh's own R rounds.
            ot = out_p.acquire(1)
            copy_off_a_k(ot, a_slice_buf, D_PER_CORE, 0)
            out_p.release(1)

            # step 0c: refill full `a` (barriered by the caller between 0b and here -- see
            # sequence()'s tg_a_drain/tg_a_refill split) and `cur`, adjacent in the misc queue by
            # construction (sequence() fills them as the last two items before this barrier and
            # the first item after it), so one acquire(2) still returns them as a pair exactly
            # like the non-fused-o arm below.
            pair = misc_c.acquire(2)
            add_k(pair[0], pair[1], x1_buf, D)
            misc_c.release(2)
        else:
            # step 1: x1 = cur + a, full D, replicated on every core.
            pair = misc_c.acquire(2)
            add_k(pair[0], pair[1], x1_buf, D)
            misc_c.release(2)

        # step 2: hf = weighted_rms_norm(x1, n_pf), full D, replicated.
        npf = misc_c.acquire(1)
        wnorm_k(x1_buf, npf, hf_buf, epsilon)
        misc_c.release(1)

        # step 3: g = Wg[my rows] @ hf, then u = Wu[my rows] @ hf -- same shared weight channel,
        # continued (Wg's N_GU_TILES tiles, then Wu's).
        for j in range_(N_GU_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_GU
            wt = weight_c.acquire(1)
            mv_gu_k(TSI_GU, row_off, wt, hf_buf, g_buf)
            weight_c.release(1)
        for j in range_(N_GU_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_GU
            wt = weight_c.acquire(1)
            mv_gu_k(TSI_GU, row_off, wt, hf_buf, u_buf)
            weight_c.release(1)

        # step 4: g = silu(g), in place over the whole FF/N slice.
        silu_k(FF_PER_CORE, g_buf)

        # step 5: emit gh = silu(g)*u in R chunks of D/N, straight onto the shared output fifo
        # (no separate gh-slice buffer -- the offset read comes out of g_buf/u_buf directly).
        for r in range(R):
            ot = out_p.acquire(1)
            mul_off_k(g_buf, u_buf, ot, D_PER_CORE, r * D_PER_CORE)
            out_p.release(1)

        # step 5b: reassemble the all-gathered gh from R D-sized misc reads (the Runtime's
        # sequence issues these only AFTER every core's R output drains land in gh_scratch --
        # see the TaskGroup barrier in sequence()).
        for i in range(R):
            chunk = misc_c.acquire(1)
            copy_off_k(gh_buf, chunk, D, i * D)
            misc_c.release(1)

        # step 6: d = Wd[my rows] @ gh -- same shared weight channel, continued (Wd's N_D_TILES).
        for j in range_(N_D_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_D
            wt = weight_c.acquire(1)
            mv_d_k(TSI_D, row_off, wt, gh_buf, d_buf)
            weight_c.release(1)

        # step 7: nxt[my rows] = x1[my rows] + d, onto the shared output fifo's (R+1)-th round.
        ot = out_p.acquire(1)
        add_off_k(x1_buf, d_buf, ot, D_PER_CORE, core_id * D_PER_CORE)
        out_p.release(1)

    workers = []
    for c in range(N):
        x1_buf = Buffer(D_ty, name=f"x1_{c}")
        hf_buf = Buffer(D_ty, name=f"hf_{c}")
        gh_buf = Buffer(FF_ty, name=f"gh_{c}")
        g_buf = Buffer(FFPC_ty, name=f"g_{c}")
        u_buf = Buffer(FFPC_ty, name=f"u_{c}")
        d_buf = Buffer(DPC_ty, name=f"d_{c}")
        core_args = [
            misc_of.cons(), weight_ofs[c].cons(), out_ofs[c].prod(),
            x1_buf, hf_buf, gh_buf, g_buf, u_buf, d_buf,
            add_kernel, add_off_kernel, wnorm_kernel, mv_gu_kernel, mv_d_kernel,
            silu_kernel, mul_off_kernel, copy_off_kernel,
            c,
        ]
        if fuse_o:
            cx_buf = Buffer(QD_ty, name=f"cx_{c}")
            a_slice_buf = Buffer(OWIN_ty, name=f"aslice_{c}")
            core_args += [cx_buf, a_slice_buf, mv_o_kernel, copy_off_cx_kernel, copy_off_a_kernel]
        workers.append(Worker(core_fn, core_args, stack_size=stack_size))

    def sequence(*args):
        if fuse_o:
            (cur, cx, npf, Wo, Wg, Wu, Wd, gh_scratch, a_scratch, nxt,
             misc_p, gweight_ps, gout_cs) = args
        else:
            (cur, a, npf, Wg, Wu, Wd, gh_scratch, nxt,
             misc_p, gweight_ps, gout_cs) = args
        # `wait=True` EVERYWHERE, not just on the drains: a plain TaskGroup.finish() with no
        # wait=True lowers to dma_free_task, which is compile-time BD-ID recycling ONLY -- no
        # hardware wait is emitted (AIEAssignRuntimeSequenceBDIDs.cpp; AIEDMATasksToNPU.cpp never
        # even sees dma_free_task). BD-ID pools are allocated PER SHIM TILE, shared across every
        # ObjectFifo mapped to that tile, so a later fill/drain on the SAME tile (misc, weight and
        # output are only 1-2 distinct shim tiles at N=8, since the placer fills 4 rows per column
        # before moving on) can get a recycled BD ID reprogrammed while the freed one's transfer
        # is still in flight -- a lock-count race, not a copy race, so it does not corrupt data,
        # it desyncs an ObjectFifo's acquire()/release() and hangs. Only wait=True lowers to a
        # real `dma_await_task`/NpuSyncOp barrier. MEASURED: without this, every N (including
        # N=8, which has no split/join to blame) hit a genuine device-side TDR
        # (aie2_tdr_detect, journalctl -k) and ERT_CMD_STATE_TIMEOUT, not just a host illusion.
        #
        # fuse_o inserts tg_a_drain/tg_a_refill AHEAD of this group (not merged into it): this
        # core now produces its FIRST output (the a-slice drain) before it has consumed cur/n_pf,
        # where the un-fused-o arm produces its first output (gh) only after consuming ALL of its
        # input. A TaskGroup boundary is a hard barrier (Runtime.finish_task_group awaits at group
        # close), so folding a's fills into tg1 unchanged while a's drain waits behind it would be
        # fine -- but folding the REFILL in with it would not: cur/n_pf are needed for step 1/2,
        # AFTER a is refilled, so their fill has to land in the group that FOLLOWS the a-slice
        # drain barrier, not the one that precedes it. Splitting fills vs drains across groups is
        # always safe; it is only unsafe to place a drain that depends on a not-yet-issued fill in
        # the SAME or an EARLIER group than that fill.
        tg1 = TaskGroup()
        if fuse_o:
            for i in range(R_CX):
                misc_p.fill(cx, _flat_tap(QD, D, i * D), wait=True, group=tg1)
            # cur is filled here (fills-only group, before the a barrier) but not CONSUMED until
            # after a is refilled -- see core_fn's step 0c. It stays adjacent to a's own refill in
            # the misc queue only because nothing else is filled into misc between here and
            # tg_a_refill below (n_pf is deliberately deferred to that same later group).
            misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg1)
            for g in range(n_aie_cols):
                gweight_ps[g].fill(
                    Wo, _flat_tap(WO_ROWS_PADDED * WROW_QD, O_WINDOW * WROW_QD, g * D_PER_CORE * WROW_QD),
                    wait=True, group=tg1,
                )
        else:
            misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg1)
            misc_p.fill(a, _flat_tap(D, D), wait=True, group=tg1)
            misc_p.fill(npf, _flat_tap(D, D), wait=True, group=tg1)
        if n_aie_rows == 1:
            # Wg/Wu belong in tg1 ONLY when nothing barriers the core between its Wo reads and its
            # Wg reads. With fuse_o the core PRODUCES its a-slice in between, and that drain is in
            # tg_a_drain -- so a Wg fill here can never complete: the core cannot reach step 3 to
            # consume it until a drain that tg1.finish() is itself blocking gets issued.
            # MEASURED as ERT_CMD_STATE_TIMEOUT with `Fatal error type: 0x0`. They are issued
            # after tg_a_refill instead; see the invariant note there.
            if not fuse_o:
                for g in range(n_aie_cols):
                    gweight_ps[g].fill(
                        Wg, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                        wait=True, group=tg1,
                    )
                for g in range(n_aie_cols):
                    gweight_ps[g].fill(
                        Wu, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                        wait=True, group=tg1,
                    )
            tg1.finish()
        else:
            tg1.finish()
            # Round-major, group-minor, with a finish() per round: each round issues exactly
            # n_aie_cols fills (one per shim tile), so no tile ever has more than 1 in flight.
            # Batching all rounds into one TaskGroup instead hit aiecc's real per-tile BD queue
            # depth (16) -- shim (0,0) alone would have queued N_GU_TILES*2 (Wg+Wu) unfreed
            # descriptors for group 0. Gathers all n_aie_rows rows' data for one round with a
            # strided TAP (see _group_tap / module docstring).
            for i in range(N_GU_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * FF_PER_CORE * D
                    gweight_ps[g].fill(
                        Wg,
                        _group_tap(FF * D, base + i * TSI_GU * D, n_aie_rows,
                                   FF_PER_CORE * D, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()
            for i in range(N_GU_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * FF_PER_CORE * D
                    gweight_ps[g].fill(
                        Wu,
                        _group_tap(FF * D, base + i * TSI_GU * D, n_aie_rows,
                                   FF_PER_CORE * D, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()

        if fuse_o:
            # tg_a_drain: every core's real D_PER_CORE-wide a-slice, the FIRST round through the
            # shared output channel (gh's own R rounds and the final residual follow it).
            tg_a_drain = TaskGroup()
            for g in range(n_aie_cols):
                gout_cs[g].drain(
                    a_scratch, _flat_tap(D, D_PER_CORE, g * D_PER_CORE),
                    wait=True, group=tg_a_drain,
                )
            tg_a_drain.finish()

            # tg_a_refill: full `a` back to every core (misc), plus n_pf (deferred here so it
            # stays AFTER cur in the misc queue -- core_fn's pair-acquire needs cur and this fill
            # adjacent, and n_pf is consumed only after that pair, so its position here is fine).
            tg_a_refill = TaskGroup()
            misc_p.fill(a_scratch, _flat_tap(D, D), wait=True, group=tg_a_refill)
            misc_p.fill(npf, _flat_tap(D, D), wait=True, group=tg_a_refill)
            tg_a_refill.finish()

            # Wg/Wu, moved here from tg1. THE INVARIANT, stated one-directionally in TIME rather
            # than by task kind: every task in group k must be reachable by the core using only
            # groups <= k. A group is unsafe both when it holds a drain waiting on a later fill
            # AND -- the case that hung this design -- when it holds a fill the core cannot reach
            # until a later group's drain is issued.
            tg_gu = TaskGroup()
            for g in range(n_aie_cols):
                gweight_ps[g].fill(
                    Wg, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                    wait=True, group=tg_gu,
                )
            for g in range(n_aie_cols):
                gweight_ps[g].fill(
                    Wu, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                    wait=True, group=tg_gu,
                )
            tg_gu.finish()

        # Barrier: gh_scratch must be fully written before any core reads it back. Every core's R
        # output-fifo drains for gh land at disjoint, contiguous offsets that together cover all
        # of gh_scratch exactly once, in the natural FF order Wd's rows expect.
        tg2 = TaskGroup()
        for g in range(n_aie_cols):
            for r in range(R):
                if n_aie_rows == 1:
                    tap = _flat_tap(FF, D_PER_CORE, g * FF_PER_CORE + r * D_PER_CORE)
                else:
                    tap = _group_tap(
                        FF, g * n_aie_rows * FF_PER_CORE + r * D_PER_CORE,
                        n_aie_rows, FF_PER_CORE, 1, D_PER_CORE,
                    )
                gout_cs[g].drain(gh_scratch, tap, wait=True, group=tg2)
        tg2.finish()

        # gh_scratch refill AND Wd share this group: both are exactly what the core's down-matvec
        # step needs next, and neither has an ordering hazard against anything still pending.
        tg3 = TaskGroup()
        for i in range(R):
            misc_p.fill(gh_scratch, _flat_tap(FF, D, i * D), wait=True, group=tg3)
        if n_aie_rows == 1:
            for g in range(n_aie_cols):
                gweight_ps[g].fill(
                    Wd, _flat_tap(D * WROW_FF, D_PER_CORE * WROW_FF, g * D_PER_CORE * WROW_FF),
                    wait=True, group=tg3,
                )
            tg3.finish()
        else:
            tg3.finish()
            for i in range(N_D_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * D_PER_CORE * FF
                    gweight_ps[g].fill(
                        Wd,
                        _group_tap(D * FF, base + i * TSI_D * FF, n_aie_rows,
                                   D_PER_CORE * FF, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()

        tg4 = TaskGroup()
        for g in range(n_aie_cols):
            # Final residual: joined-buffer row order and nxt's own indexing both step by
            # D_PER_CORE, so this drain -- unlike gh's -- is a plain contiguous run even at
            # n_aie_rows>1.
            gout_cs[g].drain(
                nxt, _flat_tap(D, n_aie_rows * D_PER_CORE, g * n_aie_rows * D_PER_CORE),
                wait=True, group=tg4,
            )
        tg4.finish()

    if fuse_o:
        rt_args = [
            D_ty, QD_ty, D_ty, Wo_L3_ty, Wg_L3_ty, Wg_L3_ty, Wd_L3_ty, GH_SCRATCH_ty, A_SCRATCH_ty,
            D_ty,
            misc_of.prod(),
            [of.prod() for of in group_weight_ofs], [of.cons() for of in group_out_ofs],
        ]
    else:
        rt_args = [
            D_ty, D_ty, D_ty, Wg_L3_ty, Wg_L3_ty, Wd_L3_ty, GH_SCRATCH_ty, D_ty,
            misc_of.prod(),
            [of.prod() for of in group_weight_ofs], [of.cons() for of in group_out_ofs],
        ]
    rt = Runtime(sequence, rt_args)

    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
