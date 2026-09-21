#
# mha_decode_s2 -- on-chip SINGLE-QUERY (M=1) decode-step self-attention for the S2 SLOW
# (text/semantic) AR transformer's incremental, KV-cached step: 32 Q heads, 8 KV heads, GQA
# repeat-interleave n_rep=4, head_dim=128. Reuses mha_decode.cc UNCHANGED (-DMHA_HD=128
# -DMHA_TKV=32, forced by L1 -- see that file's "L1 FOOTPRINT" comment), device-gated for one
# (Q head, KV head) pair at rel-L2 1.689e-02 by verify_mha_decode_hd128.py. This is the
# integration piece that kernel's GQA header names as unbuilt. The S2 FAST transformer is out
# of scope: its attention is a fresh M<=11 causal prefill with no cross-call cache
# (s2_model.cpp:1099-1255), a different kernel shape (docs/s2-ar-graph-map.md section 4).
#
# NOT mha_decode_iron.py parameterized: this driver's KV host buffer is sized at N_HEAD_KV=8,
# a different SHAPE than Whisper's [NHEADS,...], not a different parameter value. Also,
# rust/npu-asr/src/kernel_registry.rs:339-356 and rust/npu-dev/src/cmd/mha_decode.rs
# hardcode the exact stem "mha_decode_448" / filename final_mha_decode_${S}.xclbin, so
# Makefile.mha's tag format is an external contract, not free to repurpose for a second shape.
#
# ONE XCLBIN COVERS ONE KV HEAD, NOT ALL 32 Q HEADS. The first build unrolled all 32 heads x
# n_tiles into one dispatch; the device returned ERT_CMD_STATE_TIMEOUT against amdxdna's
# 2000 ms watchdog (timeout_in_sec=2, xdna-driver/src/driver/amdxdna/aie2_tdr.c:12). A raw
# per-dispatch COMPUTE-TIME overrun does NOT explain it: mha_decode_iron.py's shipped Whisper
# config (N_HEAD=12, S_MAX=448, TKV=64 -> n_tiles=7) issues 12*7=84 mha_tile calls in ONE
# dispatch and is device-gated, MORE calls than the 32*2=64 that timed out here, and per-call
# compute is structurally IDENTICAL across HD/TKV configs (TKV*HD=4096 held constant,
# mha_decode.cc's "L1 FOOTPRINT") -- so 64 slower-per-call-equivalent calls timing out while 84
# succeed rules out a simple calls/compute-time budget as the cause.
#
# The lowered IR points at a different, structural cause: the single-dispatch design issued 8
# separate `kv_h.fill()` calls -- ALL on the SAME `kv_in` objectFifo/shim channel, each its own
# `aiex.dma_start_task`, none awaited before the next is pushed (verified by generating both
# designs' MLIR and diffing the runtime_sequence: Whisper emits exactly ONE
# `dma_configure_task_for @kv_in` + `dma_start_task`; the single-dispatch S2 design emits EIGHT,
# back to back, before the one `dma_await_task` at the end). AIE2's per-channel DMA status
# register exposes a bounded hardware task queue for exactly this: `DMA_S2MM_Status_0`'s
# `Task_Queue_Size` (3 bits, bits 22:20 -- so queue depth is a single-digit hardware constant,
# not a software choice) and a `Task_Queue_Overflow` sticky bit for "attempt to write to full
# task queue" (mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json:26880,26902). Pushing 8
# tasks onto one channel with no intervening await is exactly the shape that register exists to
# flag, and the STREAM-A precedent this design cites never does it: relpos_rowtiled_stream_
# iron.py's per-head kpv fill (lines 490-494) issues the SAME stride-0-repeat single-BD tap
# per head, but each head's fill lands on its OWN column's shim -- that file's own comment says
# so explicitly, "24 tasks NEVER on one shim" (line 485) -- never stacked on one channel the way
# the single-dispatch S2 design stacked all 8 KV heads onto the one core it has. This is the
# best evidence available WITHOUT a device dispatch (which this fix is not permitted to run);
# it is not a confirmed root cause, only the structural difference that best fits the symptom
# and the one piece of this driver that has no precedent in the code it cites.
#
# HEADS_PER_DISPATCH=N_REP removes that difference: each dispatch now processes exactly one KV
# head's N_REP=4 Q heads with a SINGLE kv fill (repeat_count=N_REP-1, the same tap mechanism,
# just no longer looped 8 times per dispatch) -- one `dma_start_task` per channel per dispatch,
# matching both Whisper's shape and relpos's per-channel discipline. 3 shim DMA tasks per
# dispatch total (q fill, kv fill, ctx drain), also comfortably under the STATIC 16-BD-per-
# shim-tile allocation limit that file hit at 22 (scripts/count_tile_resources.py:15) -- a
# separate, build-time constraint from the runtime queue above, both satisfied by the same fix.
# The caller now issues N_HEAD_KV=8 dispatches per layer instead of 1 (BLOCK_COUNT stays the
# per-layer loop below, unchanged); KV DRAM stays sized at N_HEAD_KV, never a materialized
# 32-head copy.
#
# CALLS_GUARD below is UNRELATED to the fix above and should not be read as validating it: no
# probe_device_ms.py-style measurement exists for mha_tile (unlike conv-1d/conv_transpose,
# which stage_shapes.py sizes against a MEASURED ms/KiB rate), so per-dispatch compute time at
# very large s_max is genuinely UNSIZED. The guard is a soft, admittedly weak proxy (Whisper's
# proven-safe 84-call dispatch x TDR_MARGIN) kept only so an absurd s_max fails loudly at build
# time rather than silently; do not cite it as a measured or validated TDR budget.
#
# GQA index map. Per `scripts/s2_ar_ref.py::repeat_kv`'s docstring -- REPEAT-INTERLEAVE, i.e.
# `np.repeat(x, n_rep, axis=1)`, explicitly NOT `np.tile` -- output (Q) head `oh` reads KV head
# `oh // n_rep`, so a KV head's N_REP CONSECUTIVE Q heads are exactly this design's per-dispatch
# group. core_body stays IDENTICAL in shape to mha_decode_iron.py's ("for head: for tile:
# acquire/kern/release"), just HEADS_PER_DISPATCH iterations instead of N_HEAD; GQA never
# becomes core_body state (checked host-side by golden_hd128._selftest_index_equals_expand).
#
# S_MAX IS UNSIZED -- a PRODUCT decision, not derivable from source; build_design()/the CLI
# both require it, no default. Dispatch-count consequence (one mha_tile call per (head,
# kv-tile), per decode step -- the per-dispatch split above changes only how those calls are
# grouped into device round-trips, not the total math work):
#   calls        = N_HEAD * ceil(S_MAX / TKV) * BLOCK_COUNT     (same math work as before)
#   dispatches   = N_HEAD_KV * BLOCK_COUNT                       (was BLOCK_COUNT before)
# BLOCK_COUNT=36 (slow transformer layers) is NOT baked into this design: one compiled xclbin
# (fixed by S_MAX/TKV/HD alone) is dispatched N_HEAD_KV times per layer, 36 layers, against
# that layer's own Q/KV/ctx buffers; per-layer KV-cache growth AND the per-kv_head buffer
# offset are the calling driver's job, same division as mha_decode_iron.py's Whisper caller
# already uses for layers. See the task report for worked values.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
import sys
from pathlib import Path

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorAccessPattern

MHA_CC = Path(__file__).parent / "mha_decode.cc"

# Mirrors stage_shapes.TDR_MARGIN (designs/codec_block/stage_shapes.py:158), not
# imported: that module's own top-level `codec_paths.gguf()` call requires the S2 checkpoint on
# disk, a dependency this GGUF-free IRON driver has no other reason to acquire. If
# stage_shapes.TDR_MARGIN ever changes, change this too.
TDR_MARGIN = 0.7

# ARHParams (scripts/s2_ar_ref.py:319-348), cross-checked s2_model.cpp -- both transformers
# share these; see this file's header for why only the SLOW transformer's decode step is in
# scope here.
HD = 128        # head_dim: q_norm/k_norm shape s2_model.cpp:882-885 (slow, attention_qk_norm
                 # forces this over embedding_length/head_count=80); fast_head_dim
                 # s2_model.cpp:1121-1123. scripts/s2_ar_ref.py:353-355(head_dim property)/343.
N_HEAD = 32      # head_count: s2_ar_ref.py:325, s2_model.cpp:263 (attention.head_count).
N_HEAD_KV = 8    # head_count_kv: s2_ar_ref.py:326, s2_model.cpp:264 (attention.head_count_kv).
assert N_HEAD % N_HEAD_KV == 0, "GQA needs an integer repeat factor"
N_REP = N_HEAD // N_HEAD_KV  # 4. repeat-interleave: s2_model.cpp:57-68
                              # repeat_interleave_heads (output head oh -> kv head oh//N_REP,
                              # NOT a tile/np.tile map -- see that function's ne-order reshape),
                              # scripts/s2_ar_ref.py:571-579 repeat_kv (np.repeat, axis=1).
TKV = 32         # keys per K/V tile; FORCED at HD=128 by the 64 KB core L1 budget, not a free
                 # choice -- mha_decode.cc's "L1 FOOTPRINT" comment derives this exactly: at
                 # depth=2 the KV objectFifo's steady-state footprint is
                 # 2*(2*TKV*HD+2) bf16 bytes; TKV=64 gives 64.02 KB (does not fit a 64 KB
                 # tile alongside q/ctx/code/stack), TKV=32 gives 32.02 KB (fits, byte-for-
                 # byte the shipped HD=64/TKV=64 footprint, since TKV*HD=4096 is held
                 # constant). MUST match mha_decode.cc's -DMHA_TKV at kernel-build time.
BLOCK_COUNT = 36  # slow transformer layer count (documentation only; see header dispatch note).

HEADS_PER_DISPATCH = N_REP  # one KV head, one fill() call, per dispatch -- see module header
                             # for why (shim DMA task-queue overflow avoidance, not a time budget).

# NOT a validated TDR budget -- see module header. mha_tile has no measured ms/call rate, so
# per-dispatch compute time at very large s_max is UNSIZED; this is only a weak, explicitly
# unmeasured guard (Whisper's proven-safe 84-call dispatch x TDR_MARGIN) against that unknown,
# kept so an absurd s_max fails loudly at build time instead of silently.
WHISPER_SAFE_CALLS = 12 * 7  # mha_decode_iron.py: NHEADS=12, ceil(S_MAX=448/TKV=64)=7, shipped.
CALLS_GUARD = int(WHISPER_SAFE_CALLS * TDR_MARGIN)  # 58.


def ceildiv(a, b):
    return (a + b - 1) // b


def build_design(dev, s_max: int, trace_size: int = 0):
    """s_max: REQUIRED, no default -- see this file's header. Fixes n_tiles = ceil(s_max/TKV),
    this xclbin's unrolled tile count (RUNTIME S still applies within that budget: the real
    per-tile key count is read at runtime from each tile's header, exactly as in
    mha_decode.cc/mha_decode_iron.py -- s_max only bounds how long the KV cache can grow before
    this xclbin needs rebuilding at a larger s_max). Builds ONE kv_head's worth
    (HEADS_PER_DISPATCH Q heads) -- see module header for why (shim DMA task-queue overflow
    avoidance) -- and the unmeasured CALLS_GUARD this asserts against."""
    assert HD % 16 == 0
    from ml_dtypes import bfloat16 as _bf16

    n_tiles = ceildiv(s_max, TKV)
    calls = HEADS_PER_DISPATCH * n_tiles
    assert calls <= CALLS_GUARD, (
        f"s_max={s_max} -> n_tiles={n_tiles} -> {calls} mha_tile calls/dispatch exceeds the "
        f"unmeasured CALLS_GUARD={CALLS_GUARD} (see module header -- this is NOT a validated "
        f"TDR budget); max s_max at HEADS_PER_DISPATCH={HEADS_PER_DISPATCH} is "
        f"{(CALLS_GUARD // HEADS_PER_DISPATCH) * TKV}")
    KV_TILE = 2 * TKV * HD + 2  # K-tile | V-tile | 2-bf16 (int32) runtime-S header.
    kv_head_len = n_tiles * KV_TILE  # elements in the ONE kv_head this dispatch's `kv` arg names.

    q_tile_ty = np.ndarray[(HD,), np.dtype[_bf16]]
    kv_tile_ty = np.ndarray[(KV_TILE,), np.dtype[_bf16]]
    ctx_tile_ty = np.ndarray[(HD,), np.dtype[np.float32]]

    # Per-dispatch host buffers: this kv_head's HEADS_PER_DISPATCH=N_REP query heads and their
    # ctx, plus that kv_head's own KV region -- no N_HEAD_KV multiplicity here any more. The
    # caller slices the layer's full [N_HEAD_KV, kv_head_len] KV buffer (and [N_HEAD,HD] Q/ctx
    # buffers) into one kv_head's region per dispatch, the same "calling driver's job" division
    # BLOCK_COUNT already uses for per-layer buffers.
    q_group_ty = np.ndarray[(HEADS_PER_DISPATCH * HD,), np.dtype[_bf16]]
    ctx_group_ty = np.ndarray[(HEADS_PER_DISPATCH * HD,), np.dtype[np.float32]]
    kv_group_ty = np.ndarray[(kv_head_len,), np.dtype[_bf16]]

    mha_kernel = Kernel(
        "mha_tile",
        "mha_decode_s2.o",
        [q_tile_ty, kv_tile_ty, ctx_tile_ty, np.int32, np.int32],
    )

    of_q = ObjectFifo(q_tile_ty, name="q_in", depth=2)
    of_kv = ObjectFifo(kv_tile_ty, name="kv_in", depth=2)
    of_ctx = ObjectFifo(ctx_tile_ty, name="ctx_out", depth=2)

    def core_body(q_cons, kv_cons, ctx_prod, kern):
        # Same shape as mha_decode_iron.py's core_body (Python-unrolled head/tile loops --
        # tile_idx must be a compile-time constant per call, see mha_decode.cc), just
        # HEADS_PER_DISPATCH iterations instead of N_HEAD. GQA is INVISIBLE here: this whole
        # group shares the one kv_head the caller sliced into `kv` below.
        for _oh in range(HEADS_PER_DISPATCH):
            eq = q_cons.acquire(1)
            ec = ctx_prod.acquire(1)
            for t in range(n_tiles):
                ekv = kv_cons.acquire(1)
                kern(eq, ekv, ec, t, 0)
                kv_cons.release(1)
            q_cons.release(1)
            ctx_prod.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_q.cons(), of_kv.cons(), of_ctx.prod(), mha_kernel],
    )

    def sequence(q, kv, ctx, q_h, kv_h, ctx_h):
        q_h.fill(q)  # whole group buffer, natural Q-head order (no tap needed).
        # STREAM-A repeat (see module header): ONE tap now, not N_HEAD_KV of them -- `kv`
        # already names one kv_head's region, so dim0 (size=HEADS_PER_DISPATCH, stride=0)
        # replays it in place (shim BD repeat_count=HEADS_PER_DISPATCH-1), no per-kv_head
        # offset needed since that offset moved to the caller's buffer slicing above.
        kv_tap = TensorAccessPattern(
            [kv_head_len], 0,
            [HEADS_PER_DISPATCH, 1, n_tiles, KV_TILE], [0, 0, KV_TILE, 1])
        kv_h.fill(kv, tap=kv_tap)
        ctx_h.drain(ctx, wait=True)

    rt = Runtime(
        sequence,
        [q_group_ty, kv_group_ty, ctx_group_ty, of_q.prod(), of_kv.prod(), of_ctx.cons()],
    )
    return Program(dev, rt, workers=[worker]).resolve_program()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, dest="device")
    p.add_argument("-s", "--s_max", required=True, dest="s_max", type=int,
                   help="max decode steps (KV cache length) this xclbin supports -- a "
                        "PRODUCT decision, no default (see this file's header).")
    p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
    opts = p.parse_args(sys.argv[1:])

    dev = NPU2() if opts.device == "npu2" else NPU1()
    print(build_design(dev, int(opts.s_max), int(opts.trace_size)))
