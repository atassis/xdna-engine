#!/usr/bin/env python3
"""Device gate for mha_decode_s2_iron.py (designs/mha_decode/) -- the S2 slow-
transformer decode-step driver: 32 Q heads, 8 KV heads, GQA n_rep=4, HD=128. Run under the
device lock: ./run.sh verify_mha_decode_s2.py

GATE DISCIPLINE (error-metrics-are-notes-not-gates, this project's own convention): 1:1
run-to-run determinism is the BLOCKING check (dispatch twice, must land bit-identical).
rel-L2 vs the numpy golden is printed as a NOTE, not gated -- unlike
verify_mha_decode_hd128.py (which gates on rel-L2 for its single-head case), matching what
this driver's own task brief asked for explicitly.

GOLDEN IS s2_ar_ref's OWN attention, not a self-authored fixture: golden_hd128.py already
wraps scripts/s2_ar_ref.py::repeat_kv/causal_attention (its own docstring: "index (oh ->
oh//n_rep) vs repeat_kv-expand-then-slice ... asserts identical" for all 32 heads, host-only,
run automatically below). This script extends that single-head harness to the FULL 32-head,
8-KV-head case this repo's real driver dispatches -- now as N_HEAD_KV=8 SEPARATE device
dispatches per run (mha_decode_s2_iron.py's per-kv_head split, see that file's header for why:
a single dispatch stacking 8 same-channel kv fills timed out), each exercising ONE
stride-0-repeat shim BD rather than one hand-fed (Q head, KV head) pair.

NOT bricklib's verify_streamed/verify_rowwise (same reasons verify_mha_decode_hd128.py gives:
mha_tile's ABI needs an explicit unrolled tile_idx and resident, not streamed, q/ctx). NOT a
call into mha_decode_s2_iron.build_design() either: that function returns Program(...).
resolve_program() text for the Makefile/aiecc build path (CLI dev/s_max/trace_size args, no
In/Out runtime-tensor params), not an iron.jit-callable design -- same reason
verify_mha_decode_hd128.py hand-rolls its own design rather than reusing mha_decode_iron.py's
mha_decode(). _build_design() below mirrors mha_decode_s2_iron.py's core_body/sequence
logic; keep the two in sync if either changes.
"""
import sys
from pathlib import Path

import ml_dtypes
import os

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bricklib  # noqa: E402

MHA_DIR = (HERE.parent.parent / "mha_decode").resolve()
MHA_CC = MHA_DIR / "mha_decode.cc"
sys.path.insert(0, str(MHA_DIR))
import golden_hd128 as golden  # noqa: E402
import mha_decode_s2_iron as s2  # noqa: E402 (constants + ceildiv only, no aie.iron re-entry)

import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

_bf16 = ml_dtypes.bfloat16
HD, N_HEAD, N_HEAD_KV, N_REP, TKV = s2.HD, s2.N_HEAD, s2.N_HEAD_KV, s2.N_REP, s2.TKV
KV_TILE = 2 * TKV * HD + 2


def build_full_case(seed: int = 0, s_tokens: int = 100):
    """One device-test case, all 32 heads. s_max is the smallest TKV multiple >= s_tokens
    (a TEST value, not a recommendation -- S_MAX is a product decision, see this driver's
    module docstring)."""
    ar_ref, q_full, k_full, v_full = golden.build_qkv(seed, s_tokens)
    s_max = TKV * s2.ceildiv(s_tokens, TKV)
    n_tiles = s2.ceildiv(s_max, TKV)

    exp_direct = np.stack(
        [golden.reference_direct(q_full, k_full, v_full, oh) for oh in range(N_HEAD)])
    exp_full = np.stack(
        [golden.reference_full(ar_ref, q_full, k_full, v_full, oh) for oh in range(N_HEAD)])

    q_bf = q_full[-1].astype(_bf16).reshape(-1)  # [N_HEAD*HD], Q-head order
    kv_bf = np.concatenate([
        golden.pack_kv_tiles(k_full[:, kvh, :], v_full[:, kvh, :], TKV).reshape(-1)
        for kvh in range(N_HEAD_KV)])  # kv-head order, N_HEAD_KV*n_tiles*KV_TILE elements
    assert kv_bf.shape[0] == N_HEAD_KV * n_tiles * KV_TILE

    return dict(exp_direct=exp_direct, exp_full=exp_full, q_bf=q_bf, kv_bf=kv_bf,
                n_tiles=n_tiles, s_max=s_max, s_tokens=s_tokens)


def _build_design(n_tiles: int):
    """Mirrors mha_decode_s2_iron.build_design()'s current shape -- ONE kv_head's
    HEADS_PER_DISPATCH=N_REP Q heads per dispatch, single kv fill (see that file's header for
    why: avoids stacking multiple same-channel dma_start_tasks) -- wrapped for iron.jit's
    In/Out calling convention (see module docstring for why this isn't a shared helper)."""
    kv_head_len = n_tiles * KV_TILE
    calls = s2.HEADS_PER_DISPATCH * n_tiles
    assert calls <= s2.CALLS_GUARD, (
        f"n_tiles={n_tiles} -> {calls} calls/dispatch exceeds s2.CALLS_GUARD={s2.CALLS_GUARD} "
        f"(an unmeasured guard, not a validated TDR budget -- see mha_decode_s2_iron.py header)")
    compile_flags = bricklib._aie_api_include() + [f"-DMHA_HD={HD}", f"-DMHA_TKV={TKV}"]

    q_tile_ty = np.ndarray[(HD,), np.dtype[_bf16]]
    kv_tile_ty = np.ndarray[(KV_TILE,), np.dtype[_bf16]]
    ctx_tile_ty = np.ndarray[(HD,), np.dtype[np.float32]]
    q_group_ty = np.ndarray[(s2.HEADS_PER_DISPATCH * HD,), np.dtype[_bf16]]
    kv_group_ty = np.ndarray[(kv_head_len,), np.dtype[_bf16]]
    ctx_group_ty = np.ndarray[(s2.HEADS_PER_DISPATCH * HD,), np.dtype[np.float32]]

    def design(q_in: In, kv_in: In, ctx_out: Out):
        kern = ExternalFunction(
            "mha_tile", source_file=str(MHA_CC),
            arg_types=[q_tile_ty, kv_tile_ty, ctx_tile_ty, np.int32, np.int32],
            compile_flags=compile_flags,
        )
        of_q = ObjectFifo(q_tile_ty, name="q_in", depth=2)
        of_kv = ObjectFifo(kv_tile_ty, name="kv_in", depth=2)
        of_ctx = ObjectFifo(ctx_tile_ty, name="ctx_out", depth=2)

        def core(q_cons, kv_cons, ctx_prod, kern):
            # Same shape as mha_decode_iron.py's core_body; GQA is invisible here (see
            # mha_decode_s2_iron.py's header) -- this whole group shares the one kv_head
            # sliced into `kv` by main()'s per-kv_head dispatch loop below.
            for _oh in range(s2.HEADS_PER_DISPATCH):
                eq = q_cons.acquire(1)
                ec = ctx_prod.acquire(1)
                for t in range(n_tiles):
                    ekv = kv_cons.acquire(1)
                    kern(eq, ekv, ec, t, 0)
                    kv_cons.release(1)
                q_cons.release(1)
                ctx_prod.release(1)

        worker = Worker(core, fn_args=[of_q.cons(), of_kv.cons(), of_ctx.prod(), kern])

        def sequence(q, kv, ctx, q_h, kv_h, ctx_h):
            q_h.fill(q)
            # STREAM-A repeat tap (designs/relpos_mha/relpos_rowtiled_stream_iron.py
            # header): ONE tap now (not N_HEAD_KV of them -- `kv` already names one kv_head's
            # region). dim0 (size=HEADS_PER_DISPATCH, stride=0) -> shim BD
            # repeat_count=HEADS_PER_DISPATCH-1, re-reading this kv_head's n_tiles*KV_TILE
            # elements from DDR HEADS_PER_DISPATCH times.
            kv_tap = TensorAccessPattern(
                [kv_head_len], 0,
                [s2.HEADS_PER_DISPATCH, 1, n_tiles, KV_TILE], [0, 0, KV_TILE, 1])
            kv_h.fill(kv, tap=kv_tap)
            ctx_h.drain(ctx, wait=True)

        rt = Runtime(
            sequence,
            [q_group_ty, kv_group_ty, ctx_group_ty, of_q.prod(), of_kv.prod(), of_ctx.cons()],
        )
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    base = bricklib._design_key(
        "mha_tile", compile_flags,
        bricklib._include_closure_digest(MHA_CC, compile_flags))
    design.__name__ = design.__qualname__ = (
        f"{base}_s2_hd{HD}_tkv{TKV}_ntiles{n_tiles}_grp{s2.HEADS_PER_DISPATCH}")
    return iron.jit(design, use_cache=False)


def rel_l2(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(a - b))


def main():
    # Host-only self-checks (golden_hd128.py): the GQA index map and the flash-tiled kernel
    # algorithm, both against scripts/s2_ar_ref.py directly.
    golden._selftest_index_equals_expand()
    golden._selftest_flash_matches_direct()

    # S_TOKENS sweepable so the accuracy note can be read against tile count: flash-attention
    # rescales once per KV tile, so bf16 accumulation error should GROW with n_tiles while a
    # structural defect would not. Default is the original 100 (n_tiles=4).
    s_tokens = int(os.environ.get("MHA_S2_S_TOKENS", "100"))
    case = build_full_case(seed=0, s_tokens=s_tokens)
    design = _build_design(case["n_tiles"])  # ONE xclbin, dispatched N_HEAD_KV times below.
    kv_head_len = case["n_tiles"] * KV_TILE
    hpd = s2.HEADS_PER_DISPATCH

    def run_once():
        # N_HEAD_KV separate dispatches of the SAME design, one per kv_head -- mirrors
        # mha_decode_s2_iron.py's per-layer calling convention (see that file's header): the
        # caller slices the full Q/KV/ctx buffers into one kv_head's region per dispatch.
        ctx_all = np.zeros((N_HEAD, HD), dtype=np.float32)
        for kvh in range(N_HEAD_KV):
            q_slice = case["q_bf"][kvh * hpd * HD:(kvh + 1) * hpd * HD]
            kv_slice = case["kv_bf"][kvh * kv_head_len:(kvh + 1) * kv_head_len]
            q_t = iron.tensor(np.ascontiguousarray(q_slice), dtype=_bf16, device="npu")
            kv_t = iron.tensor(np.ascontiguousarray(kv_slice), dtype=_bf16, device="npu")
            ctx_t = iron.zeros((hpd * HD,), dtype=np.float32, device="npu")
            design(q_t, kv_t, ctx_t)
            ctx_all[kvh * hpd:(kvh + 1) * hpd] = ctx_t.numpy().reshape(hpd, HD)
        return ctx_all.copy()

    dev1 = run_once()
    dev2 = run_once()  # run-twice self-check -- guards the CLFLUSH host-only-BO read race,
                        # now across all N_HEAD_KV dispatches so every rotating slot is visited.

    determ = float(np.linalg.norm(dev1.astype(np.float64) - dev2.astype(np.float64)))
    nz = float(np.abs(dev1).sum())
    det_ok = (determ == 0.0) and (nz > 0.0)  # BLOCKING gate

    got = np.asarray(dev1, np.float64)
    exp_direct = np.asarray(case["exp_direct"], np.float64)
    exp_full = np.asarray(case["exp_full"], np.float64)
    r_direct = rel_l2(got, exp_direct)
    r_full = rel_l2(got, exp_full)
    per_head = [rel_l2(got[oh], exp_direct[oh]) for oh in range(N_HEAD)]

    print(f"[mha_decode_s2] N_HEAD={N_HEAD} N_HEAD_KV={N_HEAD_KV} N_REP={N_REP} HD={HD} "
          f"TKV={TKV} s_tokens={case['s_tokens']} s_max={case['s_max']} "
          f"n_tiles={case['n_tiles']}")
    print(f"  1:1 determinism (BLOCKING gate)              run2run_l2={determ:.3e} "
          f"nonzero={nz:.3e} -> {'PASS' if det_ok else 'FAIL'}")
    print(f"  rel_l2 vs s2_ar_ref index-map reference       (NOTE) = {r_direct:.3e}")
    print(f"  rel_l2 vs s2_ar_ref repeat_kv expand-then-slice(NOTE) = {r_full:.3e}")
    print(f"  per-head rel_l2 range                         (NOTE) = "
          f"[{min(per_head):.3e}, {max(per_head):.3e}]")

    assert det_ok, f"mha_decode_s2 determinism gate FAILED: run2run_l2={determ:.3e} nonzero={nz:.3e}"
    print("PASS (determinism gate)")


if __name__ == "__main__":
    main()
