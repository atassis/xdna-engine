#
# prefill_attn_s2 -- S2 SLOW (text/semantic) transformer's PROMPT PREFILL: M x M causal, GQA
# attention over a real text prompt (HD=128, 32 Q heads, 8 KV heads, n_rep=4), as opposed to
# prefill_attn_row's M<=16 FAST-decoder residual-codebook prefill (fast_context_length<=11) or
# mha_decode.cc's M=1 decode step. prefill-attn's resident-KV kernel (prefill_attn_row) does not
# generalize past a small M -- see
# "WHY NOT prefill_attn_row" below for the recomputed ceiling. This driver closes that gap by
# DISPATCHING the ALREADY-EXISTING flash/chunked kernel (prefill_attn.cc's `prefill_attn_chunk`,
# built for the S2 quantizer's RVQ post_module transformer) at its existing HD/NChunks macros --
# ZERO kernel source changes. See "BRICK VS PARAMETERIZE" below.
#
# WHY NOT prefill_attn_row (the M<=16 resident kernel) -- recomputed, not trusted. Its own header
# (prefill_attn.cc:153-165) states an L1 total of 512*M + 12*SPAD + 2048 bytes (SPAD =
# 16*ceil(M/16)), verified here by reproducing its own M=11 worked example exactly (7872 B). But
# that arithmetic is moot: `prefill_attn_row_impl`'s OWN static_assert (prefill_attn.cc:277-279)
# already caps Mq <= SPAD = 16 at COMPILE TIME (softmax_core<VL> takes exactly one 16-wide chunk;
# the file's own "prefill_attn_chunk" section header says explicitly not to call it at cols>16 on
# this Peano pin -- untested, and the kernel-internal-loop failure class this brick's own
# Revision-1 already paid for). So the code-enforced ceiling is M<=16, not ~125: the docs/tasks
# entry's "M ~ 125" describes the L1-only bound of a SPAD-scaled variant that was never built and
# is explicitly not recommended. Recomputing that hypothetical bound anyway (their own formula,
# solved for equality) gives THREE different numbers depending on what stack budget you charge
# against the SAME 64 KB (0x10000, AIE2 core tile) L1:
#   L1 objectFifo+scratch only, no stack charged   : M<=121 (512*121+12*128+2048 = 65536 exactly)
#   AIE dialect's own unconfigured default (0x400) : M<=119 (65536-1024=64512 avail; 512*119+
#                                                     12*128+2048 = 62464, the exact boundary)
#   THIS BRICK'S OWN stack_size=0xD00 (3328 B; verify_prefill_attn.py:104, raised from 0x400
#   because 0x400 overwrote the adjacent objectFifo buffers and returned NaN on every head)
#                                                   : M<=114 (65536-3328=62208 avail; 512*114+
#                                                     12*128+2048 = 61952 <= 62208, margin 256 B;
#                                                     M=115 overflows by 256 B)
# All three sit BELOW 125, and the number that matters for real device behavior (the brick's own
# configured 0xD00) is 114 -- 8% under the recorded ~125. None of this is reachable anyway: the
# static_assert bites at M=17, seven times tighter than even the worst of these three.
#
# BRICK VS PARAMETERIZE. prefill_attn.cc already carries a SECOND, flash/chunked kernel,
# `prefill_attn_chunk` (that file's own header, added for the S2 quantizer's RVQ post_module
# transformer: HD=64, no GQA, sliding window). Read against this task's needs it is ALREADY the
# right shape on every axis that matters:
#   * head_dim is a template parameter (`Hd`, PREFILL2_HD macro) with NO internal Hd-dependent
#     local array (only `alignas(64) float score[VL]`, VL=16 fixed, and scalar floats) -- so its
#     stack usage should not scale with Hd; unverified independently (inherited from the RVQ
#     config's own 1152 B measurement, not re-measured here, see STACK below).
#   * the causal mask IS the windowed mask's mechanism (additive DATA, prefill_attn.cc's own
#     "MASK IS DATA" doctrine) -- a growing causal prefix needs no new kernel branch, only a
#     different HOST-computed mask (build_causal_mask_chunked below vs codec_quantizer_ref's
#     _causal_window_mask).
#   * GQA needs no kernel change either (prefill_attn.cc's own "GQA -- INDEX, NEVER MATERIALIZE":
#     one call sees one query row against one KV head's chunk, whichever head that is -- entirely
#     a host-buffer-layout concern, same doctrine mha_decode.cc and prefill_attn_row already use).
# So this is a PARAMETERIZATION (new -D value: PREFILL2_HD=128, a NEW PREFILL2_NCHUNKS sized to
# the prompt cap below), not a new kernel or a new extern "C" symbol -- prefill_attn.cc is
# UNCHANGED by this driver. mha_decode.cc (settled, excluded from this task) is untouched and
# irrelevant here: M=1 decode is the wrong shape for an M x M prefill regardless.
#
# TWO LEVELS OF TILING (the design brief's explicit ask: query rows AND keys, both tiled).
#   * QUERY ROWS: the `range_(m_max)` WORKER loop (bricklib's own proven single-call-site device
#     dispatch loop, `_build_streamed`'s shape) -- ONE call site in the emitted program regardless
#     of m_max, which is why this axis can be M_MAX-sized without repeating prefill_attn_row's
#     Revision-1 program-memory overflow (352 Python-unrolled call sites; see
#     kernel-internal-loops-miscompile-put-volume-in-the-worker.md).
#   * KEYS: Python `for c in range(n_chunks)`, NESTED INSIDE the row loop -- n_chunks call sites
#     TOTAL in the emitted program, independent of m_max (mha_decode_s2_iron.py's own
#     justification for Python-unrolling `tile_idx`, at a comparably small count). Every row
#     processes the SAME n_chunks (fixed by the compiled xclbin, mirroring mha_decode.cc's
#     "RUNTIME S ... one xclbin serves every cache length <= S_MAX" convention): a row shorter
#     than the full causal range simply has its excess chunks masked to -1e9 by
#     build_causal_mask_chunked, contributing exp(-1e9)=0 to the softmax -- correct, but not
#     compute- or DMA-optimal (every row pays for n_chunks calls even if only 1 is real). Named,
#     not fixed here: see WASTE below.
#   THIS EXACT NESTING (`range_()` outer, Python-unrolled inner) IS UNVERIFIED END TO END --
#   inherited, not new: verify_prefill_attn.py's `_build_flash_design` (the RVQ config this
#   driver's shape is modeled on) already flags it: "Unverified: whether IRON accepts a plain-
#   Python-unrolled inner loop nested inside a range_() body at all." Neither that config nor
#   this one has ever reached a device. If IRON rejects the nesting, this needs restructuring
#   (e.g. n_chunks separate `range_()` regions), not a bigger n_chunks bound.
#
# KV: ONE PHYSICAL COPY PER DISPATCH, REPLAYED m_max TIMES IN HARDWARE, NOT DUPLICATED ON HOST.
# Because this is a growing causal PREFIX (not codec_quantizer_ref's sliding WINDOW), every row
# that needs chunk c needs the SAME physical K/V bytes for chunk c -- unlike pack_head_chunks'
# RVQ case, where each row's window genuinely differs. So `kv`'s host buffer holds n_chunks
# chunks (this head's WHOLE resident K/V, chunked), not m_max*n_chunks, and the shim DMA REPEATS
# it across the row axis via a stride-0 TensorAccessPattern dimension -- copied directly from
# mha_decode_s2_iron.py's own kv_tap (`[HEADS_PER_DISPATCH,1,n_tiles,KV_TILE],[0,0,KV_TILE,1]`,
# there repeating a KV head across its N_REP=4 sibling Q heads; here repeating one head's KV
# across the m_max row axis). That mechanism is DEVICE-CONFIRMED
# (stacking-shim-dma-tasks-on-one-channel-hangs-the-dispatch.md: "1:1 determinism run2run_l2 =
# 0.000e+00 ... PASS" for the intervened, repeat-tap S2 mha_decode driver) -- reused here, not
# reinvented. This still issues exactly ONE `dma_start_task` per channel per dispatch (qm.fill,
# kv.fill, state.drain -- 3 total, matching mha_decode_s2_iron.py's own dispatch shape), so the
# stacking hazard that mechanism exists to avoid does not apply regardless.
#   REPEAT-COUNT HARDWARE CEILING, CHECKED: a stride-0 TAP dimension lowers to
#   `aiex.npu.push_queue`'s `repeat_count` operand (mlir-aie/lib/Dialect/AIEX/Transforms/
#   AIEDmaToNpu.cpp: "We allow users to encode the repeat_count as a dimension 3 stride of 0 ...
#   repeat the BD using the repeat_count in NpuPushQueueOp"), verified in-range at
#   `AIEXDialect.cpp`'s `NpuPushQueueOp::verify()`: "Repeat count exceeds the [0:255] range." That
#   is a DIFFERENT, wider field than the per-tile-DMA-BD `Iteration_Wrap` register (6 bits,
#   `aie_registers_aie2.json`'s DMA_BD0_4, range [1:64]) a first pass over this mistook it for --
#   Iteration_Wrap is the genuine-stride (non-zero) iteration axis, not this repeat-tap path. So
#   the real ceiling on this driver's row-repeat axis is m_max <~ 255 (not 64); M_MAX=64 below
#   sits at 25% of it, not at the boundary this driver almost mis-derived.
#
# WASTE, NAMED NOT FIXED (prototype-first per the owner's 2026-09-02 steer on this task's parent
# epic -- perf work is explicitly out of scope until a working prototype exists):
#   * Every row pays for the FULL n_chunks sweep regardless of its real causal length (see TWO
#     LEVELS above) -- row 0 does the same n_chunks calls as row m_max-1.
#   * KV read bandwidth is still O(m_max * n_chunks) (the shim re-reads the same DRAM bytes
#     m_max times, per REPEAT-COUNT above) even though STORAGE is only O(n_chunks) -- the repeat
#     tap removes the host-side duplication, not the DMA read traffic.
#   * GQA head-sharing (mha_decode_s2_iron.py's HEADS_PER_DISPATCH=N_REP, one KV fill serving 4
#     sibling Q heads) is NOT built here: this driver dispatches ONE Q head at a time (32
#     dispatches/layer), because grouping heads would ALSO need an N_REP Python loop wrapping (or
#     wrapped by) the already-unverified `range_(m_max)` nesting above, a SECOND untested nesting
#     shape this driver declines to introduce on top of the first.
# Both are real, quantified in the accompanying report's worked M_MAX=64 example (~2 MB of KV DMA
# per Q-head dispatch, ~68.9 MB/layer, ~2.48 GB across 36 layers for a 64-token prompt) -- a
# genuine lever, deferred, not hidden.
#
# STACK: reuses verify_prefill_attn.py's own `_build_flash_design` stack_size=0xD00 (that file's
# comment: "measured deepest path 1152 B (0x480)" -- measured for PREFILL2_HD=64, NOT
# independently re-measured here for HD=128). Structurally this kernel has no Hd-sized local
# array (see BRICK VS PARAMETERIZE), so stack usage is not expected to scale with Hd, but that is
# an argument, not a measurement -- state it as such, do not claim it as sized.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

BRICK_CC = Path(__file__).parent / "prefill_attn.cc"

sys.path.insert(0, str(Path(__file__).parent.parent / "_verify"))
import bricklib  # noqa: E402  -- the shared brick build rail this file's __main__ already cites.

# ARHParams (scripts/s2_ar_ref.py:319-343), cross-checked against golden.py's own constants and
# the mha_decode_s2 driver's shape-confirmation worklog entry (2026-09-02 pass 7: "head_dim is
# 128 from the q_norm shape, NOT embedding_length/head_count=80" -- true for BOTH transformers).
HD = 128       # head_dim: hp.head_dim property (slow), fast_head_dim (fast) -- both 128 in
               # practice (attention_qk_norm forces it), same value mha_decode_s2_iron.py uses.
N_HEAD = 32    # head_count.
N_HEAD_KV = 8  # head_count_kv.
assert N_HEAD % N_HEAD_KV == 0
N_REP = N_HEAD // N_HEAD_KV  # 4, repeat-interleave (s2_model.cpp:57-68, s2_ar_ref.py:571-579).

TKV = 16  # keys per chunk -- MUST equal prefill_attn.cc's fixed VL (its `score[VL]` local and
          # every chunk-width loop in prefill_attn_chunk_impl are hardcoded to 16, not a template
          # parameter). Widening this needs a kernel change (an internal multi-vector loop over
          # Tkv/VL blocks) that this driver deliberately does not make -- see BRICK VS
          # PARAMETERIZE above and kernel-internal-loops-miscompile-put-volume-in-the-worker.md.

# M_MAX IS UNSIZED -- a PRODUCT decision (max text-prompt length this xclbin supports), not
# derivable from source, exactly like mha_decode_s2_iron.py's S_MAX. No default: an unset M_MAX
# must fail loud. The CLI/build_design below both require it.


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_design(dev, m_max: int, trace_size: int = 0):
    """One Q head's worth of prompt prefill: m_max query rows (range_ device loop) x n_chunks =
    ceil(m_max/TKV) key chunks (Python-unrolled, n_chunks call sites total -- see module header
    TWO LEVELS OF TILING). Dispatched once per Q head by the caller (32x/layer); GQA head
    selection is a host-buffer-layout concern (golden.pack_head_chunks_causal), never plumbed
    into this design -- the kernel and this design are both head-agnostic by construction, same
    division of labor as mha_decode.cc / mha_decode_s2_iron.py."""
    n_chunks = ceildiv(m_max, TKV)
    kv_chunk = 2 * TKV * HD          # elements: K-chunk | V-chunk, this head's data.
    qm_bf16 = HD + TKV               # q_row | mask_chunk, packed (bf16).
    state_f32 = HD + 2               # running V-acc | running max | running sum (f32).

    # The -I is NOT optional and NOT cosmetic: this box's instance is the lean kind that only
    # symlinks src/third_party/aie_api, so Peano never sees the headers and every brick .cc dies
    # on "'aie_api/aie.hpp' file not found" (bricklib._aie_api_include's own docstring). Verified
    # on the committed pin: the helper returns a non-empty flag here.
    compile_flags = bricklib._aie_api_include() + [
        f"-DPREFILL2_HD={HD}", f"-DPREFILL2_NCHUNKS={n_chunks}"]

    import ml_dtypes
    _bf16 = ml_dtypes.bfloat16

    qm_ty = np.ndarray[(qm_bf16,), np.dtype[_bf16]]
    kv_ty = np.ndarray[(kv_chunk,), np.dtype[_bf16]]
    state_ty = np.ndarray[(state_f32,), np.dtype[np.float32]]
    qm_full_ty = np.ndarray[(m_max * n_chunks * qm_bf16,), np.dtype[_bf16]]
    kv_full_ty = np.ndarray[(n_chunks * kv_chunk,), np.dtype[_bf16]]  # ONE copy -- see KV header.
    state_full_ty = np.ndarray[(m_max * state_f32,), np.dtype[np.float32]]

    def design(qm_in: In, kv_in: In, state_out: Out):
        kern = ExternalFunction(
            "prefill_attn_chunk", source_file=str(BRICK_CC),
            arg_types=[qm_ty, kv_ty, state_ty, np.int32],
            compile_flags=compile_flags,
        )
        of_qm = ObjectFifo(qm_ty, name="qm_in", depth=2)
        of_kv = ObjectFifo(kv_ty, name="kv_in", depth=2)
        of_state = ObjectFifo(state_ty, name="state_out", depth=2)

        def core(qm_cons, kv_cons, state_prod, kern):
            for _ in range_(m_max):           # genuine device loop, ONE call-site body.
                es = state_prod.acquire(1)     # resident for this row's whole chunk sweep.
                for c in range(n_chunks):      # Python-unrolled, n_chunks literal call sites.
                    eqm = qm_cons.acquire(1)
                    ekv = kv_cons.acquire(1)
                    kern(eqm, ekv, es, c)
                    kv_cons.release(1)
                    qm_cons.release(1)
                state_prod.release(1)

        worker = Worker(core, fn_args=[of_qm.cons(), of_kv.cons(), of_state.prod(), kern],
                        stack_size=0xD00)  # see module header STACK.

        def sequence(qm, kv, st, qm_h, kv_h, st_h):
            qm_h.fill(qm)  # m_max*n_chunks distinct tiles -- genuinely differs per row.
            # KV REPEAT-TAP (see module header KV): dim3 (size=m_max, stride=0) replays the
            # SAME n_chunks*kv_chunk region for every row -- one dma_start_task, repeat_count in
            # NpuPushQueueOp, checked <= 255 at m_max=64 (see module header REPEAT-COUNT).
            kv_tap = TensorAccessPattern(
                [n_chunks * kv_chunk], 0,
                [m_max, 1, n_chunks, kv_chunk], [0, 0, kv_chunk, 1])
            kv_h.fill(kv, tap=kv_tap)
            st_h.drain(st, wait=True)

        rt = Runtime(
            sequence,
            [qm_full_ty, kv_full_ty, state_full_ty, of_qm.prod(), of_kv.prod(), of_state.cons()],
        )
        return Program(dev, rt, workers=[worker]).resolve_program()

    # use_cache=True keys on __qualname__, so the name must carry everything that changes the
    # PROGRAM. A shape-only name is not enough here: prefill_attn.cc pulls in
    # `#include "../softmax/softmax.cc"` (line 183), and an edit to an INCLUDED file moves no
    # directory mtime, so ExternalFunction's own digest misses it and the stale .o is served
    # (bricklib._include_closure_digest). Same key construction as this brick's existing
    # verify_prefill_attn.py:191-194.
    base = bricklib._design_key(
        "prefill_attn_chunk", compile_flags,
        bricklib._include_closure_digest(BRICK_CC, compile_flags))
    design.__name__ = design.__qualname__ = f"{base}_s2_hd{HD}_nchunks{n_chunks}_m{m_max}"
    return iron.jit(design, use_cache=True)


if __name__ == "__main__":
    # DEVICE-FREE ONLY: `.compile()` runs MLIR generation + aiecc (--get-xclbin/--get-npu-insts
    # under the hood) and writes the two artifacts below; it never touches /dev/accel or dispatches
    # anything (aie.utils.callabledesign.CallableDesign.compile's own docstring: "Eagerly compile
    # this design"). Mirrors mha_decode_s2_iron.py's naming (final_<tag>.xclbin/insts_<tag>.txt)
    # without needing that file's Makefile+mlir-aie-sandbox-copy mechanism -- bricks build via
    # iron.jit directly (see aie_kernels/_test/bricklib.py, compile_check.sh).
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, dest="device")
    p.add_argument("-m", "--m_max", required=True, dest="m_max", type=int,
                   help="max prompt length this xclbin supports -- a PRODUCT decision, no "
                        "default (see this file's header).")
    p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
    p.add_argument("-o", "--outdir", required=False, dest="outdir", default="build")
    opts = p.parse_args(sys.argv[1:])

    dev = NPU2() if opts.device == "npu2" else NPU1()
    design = build_design(dev, int(opts.m_max), int(opts.trace_size))

    outdir = Path(opts.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"prefill_attn_s2_{opts.m_max}"
    xclbin_path = outdir / f"final_{tag}.xclbin"
    inst_path = outdir / f"insts_{tag}.txt"
    got_xclbin, got_inst = design.compile(xclbin_path=xclbin_path, inst_path=inst_path)
    print(f"xclbin: {got_xclbin}")
    print(f"insts:  {got_inst}")
