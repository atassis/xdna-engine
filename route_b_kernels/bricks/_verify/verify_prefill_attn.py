#!/usr/bin/env python3
"""Device rel-L2 verify for the prefill-attn brick: causal, GQA, no-relative-position attention.
Gate 3e-2. Run under the device lock.

// cachebust 2026-07-31d -- unlike bricklib's own hash, this comment does not gate the build
// (bricklib._design_name hashes prefill_attn.cc + its includes + compile flags, not this file's
// text) -- it is a human-visible signal only. Edit prefill_attn.cc to force a rebuild.

REVISION 3: NOW USES bricklib.verify_streamed DIRECTLY (read prefill_attn.cc's header "ONE ROW PER
CALL, REVISION 2" and "MASK IS DATA, NOT A SCALAR" first). Revision 1 hand-rolled a design with a
per-call `row_idx` scalar and a Python-unrolled outer loop (352 call sites for the 32-head x
11-row case) because bricklib's generic builders cannot carry an extra scalar argument -- that
compiled but FAILED TO LOAD ON DEVICE (`_XAie_LoadProgMemSection(): Overflow of program memory`):
352 emitted call sites do not fit an AIE core's small PROGRAM memory budget (separate from the 64
KB DATA memory this brick's L1 footprint was already sized against). Revision 3 removes the
scalar entirely (the causal mask moved from a `row_idx` branch to an ADDITIVE MASK ROW, streamed
as data alongside q_row -- see prefill_attn.cc's header) specifically so this script can go back
to the proven, GENERIC bricklib rail every other green brick in this catalog uses (snake, gather-
rows, softmax, gelu-erf): `bricklib.verify_streamed`'s WORKER loop (`for _ in range_(n_tiles)`
inside `_build_streamed`) is a SINGLE call site in the emitted program regardless of `n_tiles`,
because it is a genuine device dispatch loop, not source duplication -- the opposite failure mode
from Revision 1's Python unroll, and not implicated by the earlier (different) kernel-body-loop
NaN defect either (softmax_core's OWN internal loop, and this file's per-key loops, are unaffected
-- see prefill_attn.cc's header for the full defect/fix history).

ONE DESIGN PER HEAD, NOT PER CALL. GQA means different Q heads need DIFFERENT resident K/V (see
prefill_attn.cc's header GQA section) and a bricklib resident operand is fixed for one whole
design/build -- so this script calls `bricklib.verify_streamed` once per head (HEADS below), each
a fully independent build+device-run. This is a host-side compile-time cost (32 builds for the
main case), NOT a device program-memory cost: every one of those 32 builds emits the SAME single-
call-site shape, so none of them risk Revision 1's overflow. If a real run finds 32 builds too
slow or otherwise impractical, cut HEADS down to a smaller list (e.g. `[5]`) -- gating one head
green is a better outcome than not gating at all; the per-head loop below makes that a one-line
change.

SCOPE NOTE on the head sweep (mirrors mha_decode.cc's own scope boundary, see its header "GQA"
section): prefill_attn.cc is HEAD-AGNOSTIC BY CONSTRUCTION -- one call only ever sees ONE query
row of ONE (Q head, KV head) pair, and NEVER internally expands KV to n_head. Building one design
per head, each with that head's own physical KV data resident, is per-head DMA/build plumbing for
a test harness, not the kernel-internal "materialize an [n_head,...] expanded copy" the doctrine
forbids; a real S2 driver integrating this brick would instead hold the 8 physical KV heads
resident and reuse the same on-chip region for the 4 Q-head calls that share it (the same "host-
buffer-layout change belongs in the driver, not this file" scope boundary mha_decode.cc's own
header states).
"""
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bricklib  # noqa: E402

BRICK_DIR = (HERE.parent / "prefill-attn").resolve()
BRICK_CC = BRICK_DIR / "prefill_attn.cc"
sys.path.insert(0, str(BRICK_DIR))
import golden  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402

GATE = 3e-2
_bf16 = ml_dtypes.bfloat16
FLASH_GATE = 3.0e-2


def _verify_head(oh: int, q_full, k_full, v_full, m_tokens: int, mask: np.ndarray = None,
                  name_suffix: str = ""):
    """One bricklib.verify_streamed build+run for head `oh`: streamed [q_row|mask_row] tiles (one
    per query row), resident KV (this head's h_kv, depth=1), streamed ctx_row out. `mask`
    defaults to the real causal mask (golden.build_causal_mask); passing an explicit mask here is
    how the mask-as-data negative test below exercises a deliberately BROKEN mask on real device
    output. Returns bricklib's own result dict (has 'ok', 'rel_l2', 'run2run', 'got', ...)."""
    in_tiles, kv_resident, h_kv = golden.pack_head_rows(q_full, k_full, v_full, oh, mask=mask)
    exp = golden.reference_direct(q_full, k_full, v_full, oh) if mask is None else None
    return bricklib.verify_streamed(
        name=f"prefill_attn_row_h{oh}_m{m_tokens}{name_suffix}",
        shim=BRICK_CC,
        symbol="prefill_attn_row",  # bound directly: prefill_attn.cc's own extern "C" symbol is
                                     # already pure-buffers (qm_row, kv, ctx_row), no wrapper shim
                                     # needed -- same as gather_rows_f32's own verify script.
        in_tiles=in_tiles,
        out_tile_numel=golden.HD,
        resident=kv_resident,
        unpack=lambda d: d,  # device shape (m_tokens, HD) already matches golden's shape exactly
        golden=exp if exp is not None else np.zeros((m_tokens, golden.HD), np.float32),
        gate=GATE,
        in_dt=_bf16, out_dt=np.float32, resident_dt=_bf16,
        # depth=1: KV is resident (acquired once, held for this head's whole row stream), never
        # double-buffered -- bricklib's own documented reasoning (see gather_rows' verify script,
        # which measured depth=2 fail to build / depth=1 build+gate-green at a comparable shape).
        resident_depth=1,
        compile_flags=[f"-DPREFILL_HD={golden.HD}", f"-DPREFILL_M={m_tokens}"],
        # This kernel's frame exceeds the 0x400 dialect default, so at the default it overruns the
        # adjacent objectFIFO buffers and every head returns NaN -- the long-standing red on this
        # brick, and NOT a numerics defect (see the ~10 arithmetic hypotheses refuted against it).
        # Measured head 0: 0x400 nan/1408 non-finite, 0x800 rel-L2 8.290e-08 and 0/1408. 0xD00 is
        # 0x800 plus margin, matches amd/IRON's own constant for its gemm/mha operators, and costs
        # nothing here -- the design still builds at 0x4000, so L1 is not tight at this shape.
        stack_size=0xD00,
    )


#===============================================================================================#
# FLASH/CHUNKED CONFIGURATION -- prefill_attn_chunk, the S2 quantizer post_module transformer's
# capacity extension (RVQ_HEAD_DIM=64, up to RVQ_WINDOW_SIZE=128 visible keys; see
# prefill_attn.cc's "prefill_attn_chunk" section header for the design and the route-choice
# rationale). NOT bricklib's verify_streamed/verify_rowwise, for the same class of reason
# verify_mha_decode_hd128.py hand-rolls its own design: prefill_attn_chunk's ABI (2 streamed
# inputs, ONE RESIDENT-PER-ROW output DOUBLING as online-softmax state, plus a per-call literal
# chunk_idx) does not fit bricklib's generic `kern(in_tile,[resident],out_tile)` no-scalar
# contract. Bespoke @iron.jit design below, same pieces bricklib's own builders use.
#
# TWO LEVELS OF ITERATION, DELIBERATELY DIFFERENT MECHANISMS -- this is the one thing to get right
# here, matching the exact "PROGRAM MEMORY" hazard prefill_attn.cc's own header records
# (Revision 1b: 352 Python-unrolled call sites overflowed the core's program memory and failed to
# load, a genuine device error, not a numerics bug). t_tokens ROWS is the axis that scales with
# the real problem (up to the T=66 captured clip, or more for a longer one) -- that axis uses
# `range_(t_tokens)`, bricklib's own proven single-call-site device dispatch loop (see
# `_build_streamed`'s own core function above for the identical shape), NOT Python's `range`.
# n_chunks CHUNKS is a small, HARDWARE-CAPACITY-bounded axis (<=8 for the RVQ_WINDOW_SIZE=128
# cap) that genuinely needs a per-call COMPILE-TIME LITERAL `chunk_idx` (the reset/finalize
# signal) -- exactly mha_decode_iron.py's own justification for Python-unrolling its `tile_idx`
# loop, at a comparably small n_tiles=4 in its own device-green test. So: Python `for c in
# range(n_chunks)` NESTED INSIDE the `range_(t_tokens)` body -- n_chunks call sites TOTAL in the
# emitted program (independent of t_tokens), never t_tokens*n_chunks. Unverified: whether IRON
# accepts a plain-Python-unrolled inner loop nested inside a `range_()` body at all (no exact
# precedent for that nesting was found in this codebase's other verify scripts); if it does not,
# this design needs restructuring, not a bigger n_chunks bound.
#===============================================================================================#

def _build_flash_design(t_tokens: int, n_chunks: int, hd: int):
    """One resident-per-row state buffer (RESIDENT ACROSS one row's n_chunks calls, released and
    re-acquired every row -- see prefill_attn_chunk's header NO STATIC L1 STATE), streamed over
    t_tokens*n_chunks (qm_row, kv_chunk) tile pairs. `sequence` fills the FULL padded qm/kv arrays
    (t_tokens*n_chunks tiles each, built by golden.pack_head_chunks) and drains t_tokens state
    tiles of [hd+2] f32 each (the extra 2 are the row's final running max/sum -- unused by the
    caller, see prefill_attn_chunk's header; only ctx[:, :hd] is read back)."""
    compile_flags = bricklib._aie_api_include() + [f"-DPREFILL2_HD={hd}", f"-DPREFILL2_NCHUNKS={n_chunks}"]

    qm_bf16 = hd + golden.VL
    kv_bf16 = 2 * golden.VL * hd
    state_f32 = hd + 2

    qm_ty = np.ndarray[(qm_bf16,), np.dtype[_bf16]]
    kv_ty = np.ndarray[(kv_bf16,), np.dtype[_bf16]]
    state_ty = np.ndarray[(state_f32,), np.dtype[np.float32]]
    qm_full_ty = np.ndarray[(t_tokens * n_chunks * qm_bf16,), np.dtype[_bf16]]
    kv_full_ty = np.ndarray[(t_tokens * n_chunks * kv_bf16,), np.dtype[_bf16]]
    state_full_ty = np.ndarray[(t_tokens * state_f32,), np.dtype[np.float32]]

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
            for _ in range_(t_tokens):  # genuine device loop, ONE call-site body (see header)
                es = state_prod.acquire(1)  # resident for this row's whole chunk sweep
                for c in range(n_chunks):   # Python-unrolled, n_chunks literal call sites TOTAL
                    eqm = qm_cons.acquire(1)
                    ekv = kv_cons.acquire(1)
                    kern(eqm, ekv, es, c)
                    kv_cons.release(1)
                    qm_cons.release(1)
                state_prod.release(1)

        worker = Worker(core, fn_args=[of_qm.cons(), of_kv.cons(), of_state.prod(), kern],
                        stack_size=0xD00)  # measured deepest path 1152 B (0x480); see report

        def sequence(qm, kv, st, qm_h, kv_h, st_h):
            qm_h.fill(qm)
            kv_h.fill(kv)
            st_h.drain(st, wait=True)

        rt = Runtime(
            sequence,
            [qm_full_ty, kv_full_ty, state_full_ty, of_qm.prod(), of_kv.prod(), of_state.cons()],
        )
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    base = bricklib._design_key(
        "prefill_attn_chunk", compile_flags,
        bricklib._include_closure_digest(BRICK_CC, compile_flags))
    design.__name__ = design.__qualname__ = f"{base}_hd{hd}_nchunks{n_chunks}_t{t_tokens}"
    return iron.jit(design, use_cache=False)


def _verify_flash_head(cq, q_full, k_full, v_full, head: int, t_tokens: int, n_chunks: int,
                        hd: int, mask: np.ndarray = None):
    """Build+run prefill_attn_chunk for one head, all t_tokens rows, gate rel-L2 against
    `golden.reference_rvq_head`. Returns a bricklib-shaped result dict (has 'ok', 'rel_l2',
    'run2run', 'got', ...)."""
    qm_tiles, kv_tiles = golden.pack_head_chunks(cq, q_full, k_full, v_full, head, t_tokens,
                                                 n_chunks, mask=mask)
    if mask is None:
        mask = cq._causal_window_mask(t_tokens, cq.RVQ_WINDOW_SIZE)
    exp = golden.reference_rvq_head(cq, q_full, k_full, v_full, head, mask)  # [t_tokens, hd] f32
    design = _build_flash_design(t_tokens, n_chunks, hd)

    def run_once():
        qm_t = iron.tensor(np.ascontiguousarray(qm_tiles.reshape(-1)), dtype=_bf16, device="npu")
        kv_t = iron.tensor(np.ascontiguousarray(kv_tiles.reshape(-1)), dtype=_bf16, device="npu")
        st_t = iron.zeros((t_tokens * (hd + 2),), dtype=np.float32, device="npu")
        design(qm_t, kv_t, st_t)
        return st_t.numpy().reshape(t_tokens, hd + 2)[:, :hd].copy()  # drop the 2 scratch scalars

    dev1 = run_once()
    dev2 = run_once()
    got = np.asarray(dev1, np.float64)
    e = np.asarray(exp, np.float64)
    den = np.linalg.norm(e)
    rel_l2 = float(np.linalg.norm(got - e) / den) if den else float(np.linalg.norm(got))
    determ = float(np.linalg.norm(dev1.astype(np.float64) - dev2.astype(np.float64)))
    nz = float(np.abs(dev1).sum())
    ok = (nz > 0.0) and (rel_l2 <= FLASH_GATE)
    status = "PASS" if ok else ("FAIL-ZERO" if nz == 0.0 else "FAIL")
    print(f"  [flash] head={head} T={t_tokens} n_chunks={n_chunks} HD={hd}: "
          f"rel_l2={rel_l2:.3e} run2run={determ:.3e} -> {status}")
    return dict(ok=ok, rel_l2=rel_l2, run2run=determ, nonzero=nz, status=status, got=got)


def main():
    # ---- device-free self-checks (also exercised standalone by `python3 golden.py`) ----
    golden._selftest_index_equals_expand()
    golden._selftest_causal_mask_has_teeth()
    golden._selftest_gqa_mapping_has_teeth()
    golden._selftest_mask_data_has_teeth()
    golden._selftest_m1_degenerate()

    # 32 builds (one per head) -- cut this list down (e.g. `[5]`) if a real run finds that
    # impractical; see this file's header "ONE DESIGN PER HEAD, NOT PER CALL".
    HEADS = list(range(golden.N_HEAD))

    ar_ref, q_full, k_full, v_full = golden.build_qkv(seed=0, m_tokens=golden.M)

    overall_ok = True
    per_head = {}
    for oh in HEADS:
        res = _verify_head(oh, q_full, k_full, v_full, golden.M)
        per_head[oh] = res
        overall_ok &= res["ok"]
        print(f"  head {oh:2d}: rel_l2={res['rel_l2']:.3e} run2run={res['run2run']:.3e} "
              f"-> {res['status']}")

    worst_head = max(per_head, key=lambda h: per_head[h]["rel_l2"])
    print(f"[prefill_attn_row M={golden.M} HD={golden.HD}] {len(HEADS)} heads, one "
          f"bricklib.verify_streamed build per head")
    print(f"  worst head: {worst_head} rel_l2={per_head[worst_head]['rel_l2']:.3e} gate={GATE:.1e}")
    print(f"  -> {'PASS' if overall_ok else 'FAIL'}")
    assert overall_ok, f"prefill_attn main gate failed for head(s): " \
        f"{[h for h in per_head if not per_head[h]['ok']]}"

    # ---- NEGATIVE TEST 1: causal mask has teeth, checked against MEASURED device output.
    # Position i cannot see i+1 (or anything after it): row 0 of ANY head can only ever attend to
    # key 0, so device ctx[head,0,:] must equal that head's V[0] almost exactly, and must be FAR
    # from the bidirectional (unmasked) reference for row 0. Reuses the already-collected device
    # result for `probe_head` -- no extra device call needed. ----
    probe_head = 5
    assert probe_head in per_head, "probe_head must be in HEADS for this test to reuse its result"
    h_kv_probe = probe_head // golden.N_REP
    dev_row0 = np.asarray(per_head[probe_head]["got"])[0]  # (HD,) float64
    v0 = v_full[0, h_kv_probe, :].astype(np.float64)
    exp_bidir_row0 = golden.reference_bidirectional(q_full, k_full, v_full, probe_head)[0].astype(np.float64)
    d_causal = golden.rel_l2(dev_row0, v0)
    d_bidir = golden.rel_l2(dev_row0, exp_bidir_row0)
    print(f"[causal-mask negative test] head={probe_head} row 0 (can only see key 0)")
    print(f"  device row0 vs V[0]                (want small, <= gate) rel_l2={d_causal:.3e}")
    print(f"  device row0 vs bidirectional row0  (want LARGE, > 0.1)   rel_l2={d_bidir:.3e}")
    causal_ok = (d_causal <= GATE) and (d_bidir > 0.1)
    overall_ok &= causal_ok
    assert d_causal <= GATE, f"device row 0 does not match V[0] -- causal mask is broken (or " \
        f"leaking future keys): rel_l2={d_causal:.3e}"
    assert d_bidir > 0.1, f"device row 0 is suspiciously close to the UNMASKED reference " \
        f"(rel_l2={d_bidir:.3e}) -- this test would not catch a non-causal kernel"
    print(f"  -> {'PASS' if causal_ok else 'FAIL'}")

    # ---- NEGATIVE TEST 2: mask-as-data has teeth, checked with a DELIBERATELY BROKEN mask fed to
    # a real device run (not just reused data): the causal part zeroed (padding beyond m_tokens
    # still correctly masked -- see golden._selftest_mask_data_has_teeth for why THAT split, not a
    # literal all-zero mask, is the meaningful negative control). If the kernel genuinely consumes
    # the mask as data, this run's device output must land near the bidirectional (unmasked)
    # reference and far from the correctly-masked device output already collected for this head.
    # This is the strongest available evidence that the mask is not being ignored, misaligned, or
    # optimized away. ----
    broken_mask = np.zeros((golden.M, golden.SPAD), dtype=np.float32)
    broken_mask[:, golden.M:] = -1.0e9
    exp_bidir_full = golden.reference_bidirectional(q_full, k_full, v_full, probe_head)
    res_broken = _verify_head(probe_head, q_full, k_full, v_full, golden.M, mask=broken_mask,
                               name_suffix="_brokenmask")
    # res_broken was gated against a placeholder golden (zeros) inside _verify_head since its
    # purpose is a DIFFERENT comparison; recompute rel_l2 here against the right references.
    dev_broken = np.asarray(res_broken["got"])
    d_broken_vs_bidir = golden.rel_l2(dev_broken, exp_bidir_full)
    d_broken_vs_correct = golden.rel_l2(dev_broken, per_head[probe_head]["got"])
    print(f"[mask-as-data negative test] head={probe_head}, causal part of mask zeroed on device")
    print(f"  device (broken mask) vs bidirectional reference   (want <= gate) "
          f"rel_l2={d_broken_vs_bidir:.3e}")
    print(f"  device (broken mask) vs device (correct mask)     (want LARGE)   "
          f"rel_l2={d_broken_vs_correct:.3e}")
    mask_ok = (d_broken_vs_bidir <= GATE) and (d_broken_vs_correct > 0.1)
    overall_ok &= mask_ok
    assert d_broken_vs_bidir <= GATE, f"device with a broken (causal-zeroed) mask does not " \
        f"match bidirectional attention -- the kernel may not be consuming the mask as data at " \
        f"all: rel_l2={d_broken_vs_bidir:.3e}"
    assert d_broken_vs_correct > 0.1, f"device output barely changed when the mask was broken " \
        f"(rel_l2={d_broken_vs_correct:.3e}) -- this test would not catch a mask-plumbing bug"
    print(f"  -> {'PASS' if mask_ok else 'FAIL'}")

    # ---- NEGATIVE TEST 3: GQA mapping, checked against MEASURED device output, across ALL
    # gated heads (not just head 0 -- a broken kernel/driver could get head 0 right, since h_kv=0
    # is identical whether you compute oh//n_rep or oh%n_head_kv there, and still be wrong for
    # every other head). Already exercised by the per-head gate above (each head's golden used the
    # CORRECT oh//n_rep mapping); this adds an explicit divergence check against the WRONG (tile)
    # convention for a head where the two conventions disagree, using the actual device numbers
    # already collected. ----
    wrong_head = 8
    assert wrong_head in per_head, "wrong_head must be in HEADS for this test to reuse its result"
    exp_wrong = golden.reference_direct_wrong_tile(q_full, k_full, v_full, wrong_head).astype(np.float64)
    dev_wrong_head = np.asarray(per_head[wrong_head]["got"])
    d_correct = per_head[wrong_head]["rel_l2"]
    d_wrong = golden.rel_l2(dev_wrong_head, exp_wrong)
    print(f"[GQA negative test] head={wrong_head} (correct h_kv={wrong_head // golden.N_REP}, "
          f"wrong/tile h_kv={wrong_head % golden.N_HEAD_KV})")
    print(f"  device vs correct (repeat-interleave) reference  (want <= gate) rel_l2={d_correct:.3e}")
    print(f"  device vs wrong (np.tile) reference               (want LARGE)  rel_l2={d_wrong:.3e}")
    gqa_ok = (d_correct <= GATE) and (d_wrong > 0.1)
    overall_ok &= gqa_ok
    assert d_correct <= GATE, f"device disagrees with the correct GQA index map: {d_correct:.3e}"
    assert d_wrong > 0.1, f"device is suspiciously close to the WRONG (tile) GQA convention " \
        f"(rel_l2={d_wrong:.3e}) -- this test would not catch a backwards driver"
    print(f"  -> {'PASS' if gqa_ok else 'FAIL'}")

    # ---- DEGENERATE CASE: M=1 (PREFILL_M is compile-time -> a separate build). A handful of
    # heads, spanning the first and last KV group, is enough -- M=1's only new code path vs M=11
    # is the SPAD=16 padding boundary (15 of 16 softmax slots masked), which does not depend on
    # WHICH head is used. ----
    m1_heads = [0, 17, 31]
    ar_ref1, q1, k1, v1 = golden.build_qkv(seed=1, m_tokens=1)
    m1_ok = True
    for oh in m1_heads:
        res1 = _verify_head(oh, q1, k1, v1, m_tokens=1)
        m1_ok &= res1["ok"]
        print(f"[prefill_attn_row M=1 degenerate] head={oh}: rel_l2={res1['rel_l2']:.3e} "
              f"run2run={res1['run2run']:.3e} -> {res1['status']}")
    overall_ok &= m1_ok
    assert m1_ok, "prefill_attn M=1 gate failed for one or more heads"

    # ---- FLASH/CHUNKED CONFIGURATION: prefill_attn_chunk, the S2 quantizer post_module
    # transformer's capacity extension (RVQ_HEAD_DIM=64, no GQA, sliding-window causal mask).
    # ADDITIONAL cases, ADDED alongside the HD=128/M<=11 gate above, which this section does not
    # touch. Two configs: the real captured clip (T=66, 5 chunks) and the window cap (T=128, 8
    # chunks, ceil(RVQ_WINDOW_SIZE/16)) -- see prefill_attn.cc's "prefill_attn_chunk" section and
    # this file's own FLASH/CHUNKED CONFIGURATION header above for the design. UNVERIFIED end to
    # end: authored device-free, never run (see the accompanying report). ----
    golden._selftest_rvq_chunked_matches_windowed()  # device-free: proves the merge algebra alone

    cq = golden._load_codec_quantizer_ref()
    flash_ok = True
    for t_tokens, n_chunks, seed, label in [
        (66, 5, 3, "T=66 (real captured clip)"),
        (128, 8, 4, "T=128 (RVQ_WINDOW_SIZE cap)"),
    ]:
        _, qf, kf, vf = golden.build_qkv_rvq(seed=seed, t_tokens=t_tokens, n_head=4)
        res = _verify_flash_head(cq, qf, kf, vf, head=1, t_tokens=t_tokens, n_chunks=n_chunks,
                                 hd=cq.RVQ_HEAD_DIM)
        flash_ok &= res["ok"]
        print(f"[prefill_attn_chunk {label}] -> {res['status']}")
    overall_ok &= flash_ok
    assert flash_ok, "prefill_attn_chunk (flash/windowed) gate failed for one or more configs"

    assert overall_ok, "prefill_attn: one or more device checks failed (see above)"
    print("PASS")


if __name__ == "__main__":
    main()
