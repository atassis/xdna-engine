#!/usr/bin/env python3
"""Device rel-L2 verify for mha_decode's S2 fast-decoder instance: HD=128, 8->32 GQA (n_rep=4).
Gate 3e-2. Run under the device lock.

SCOPE: gates the SINGLE-QUERY step (M=1 decode), one query row against a given K/V -- per the
task brief, `scripts/s2_ar_ref.py::causal_attention` recomputes the whole sequence every call (see
its docstring), so "this decode step" is the LAST row of that recompute, which is exactly what
`golden_hd128.reference_full`/`reference_direct` return. The GQA "index, don't materialize" claim
is checked device-free by `golden_hd128._selftest_index_equals_expand` (run automatically below,
also runnable standalone: `python3 route_b_kernels/mha_decode/golden_hd128.py`) -- this script
only needs to feed the kernel ONE Q head against its (unexpanded) KV head's data, because the
device kernel is head-agnostic (see mha_decode.cc's GQA header comment) and that equivalence is
what makes doing so correct.

NOT bricklib's verify_streamed/verify_rowwise: those assume a pure `kern(in_tile, [resident],
out_tile)` call with NO extra scalar args and a genuinely STREAMED output (one output tile per
input tile). mha_decode's `mha_tile(q_h, kv_tile, ctx_h, tile_idx, s_in_tile)` needs an explicit,
per-call UNROLLED tile_idx (the online-softmax reset/finalize signal) and its q/ctx are RESIDENT
(acquired once, held across the whole tile sweep, released once) rather than streamed -- the same
shape mha_decode_iron.py itself uses, just with NHEADS=1 here. So this script hand-rolls one
@iron.jit design (same iron.jit/Program/Worker/Runtime/ObjectFifo pieces bricklib's own builders
use) instead of forcing an unrelated ABI through the generic ones.

RoPE is applied on the HOST (golden_hd128.py, via scripts/s2_ar_ref.py::rope_interleaved) --
mha_decode.cc does not do RoPE (that is a separate op in the real pipeline: "RoPE(q,k) ->
GQA-repeat(k,v) -> causal attn"), and this script does not depend on the concurrently-authored
rope-interleaved brick.
"""
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bricklib  # noqa: E402  (reuse _aie_api_include / _design_name; see its docstrings)

MHA_DIR = (HERE.parent.parent / "mha_decode").resolve()
MHA_CC = MHA_DIR / "mha_decode.cc"
sys.path.insert(0, str(MHA_DIR))
import golden_hd128 as golden  # noqa: E402

import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction

GATE = 3e-2
_bf16 = ml_dtypes.bfloat16


def _build_design(n_tiles: int, hd: int, tkv: int):
    """One resident query + one resident ctx output, streamed over `n_tiles` K/V tiles, with an
    explicit UNROLLED tile_idx per call -- mirrors mha_decode_iron.py's own core_body (NHEADS
    collapsed to 1: this script tests exactly one (Q head, KV head) pair per the task's "single
    query row against a given K/V" scope)."""
    kv_tile_bf16 = 2 * tkv * hd + 2
    compile_flags = bricklib._aie_api_include() + [f"-DMHA_HD={hd}", f"-DMHA_TKV={tkv}"]

    q_ty = np.ndarray[(hd,), np.dtype[_bf16]]
    kv_tile_ty = np.ndarray[(kv_tile_bf16,), np.dtype[_bf16]]
    ctx_ty = np.ndarray[(hd,), np.dtype[np.float32]]
    kv_full_ty = np.ndarray[(n_tiles * kv_tile_bf16,), np.dtype[_bf16]]

    def design(q_in: In, kv_in: In, ctx_out: Out):
        kern = ExternalFunction(
            "mha_tile", source_file=str(MHA_CC),
            arg_types=[q_ty, kv_tile_ty, ctx_ty, np.int32, np.int32],
            compile_flags=compile_flags,
        )
        of_q = ObjectFifo(q_ty, name="q_in", depth=2)
        of_kv = ObjectFifo(kv_tile_ty, name="kv_in", depth=2)
        of_ctx = ObjectFifo(ctx_ty, name="ctx_out", depth=2)

        def core(q_cons, kv_cons, ctx_prod, kern):
            eq = q_cons.acquire(1)
            ec = ctx_prod.acquire(1)
            # UNROLLED at Python/MLIR-generation time (not a device-side loop): mha_tile needs the
            # literal tile_idx per call (online-softmax reset/finalize signal), exactly like
            # mha_decode_iron.py's own core_body.
            for t in range(n_tiles):
                ekv = kv_cons.acquire(1)
                kern(eq, ekv, ec, t, 0)
                kv_cons.release(1)
            q_cons.release(1)
            ctx_prod.release(1)

        worker = Worker(core, fn_args=[of_q.cons(), of_kv.cons(), of_ctx.prod(), kern])

        def sequence(q, kv, ctx, q_h, kv_h, ctx_h):
            q_h.fill(q)
            kv_h.fill(kv)
            ctx_h.drain(ctx, wait=True)

        rt = Runtime(
            sequence,
            [q_ty, kv_full_ty, ctx_ty, of_q.prod(), of_kv.prod(), of_ctx.cons()],
        )
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    # Cache-bust on every real input (kernel source + compile flags, via bricklib's own hashing)
    # PLUS n_tiles, which _design_name cannot see (it only hashes shim text + flags, and n_tiles
    # is a Python closure baked into the unrolled MLIR, not part of either) -- see bricklib's
    # `_design_name` docstring for why a stale name silently reruns an old build.
    base = bricklib._design_name("mha_tile", MHA_CC, compile_flags)
    design.__name__ = design.__qualname__ = f"{base}_hd{hd}_tkv{tkv}_ntiles{n_tiles}"
    return iron.jit(design, use_cache=False)


def main():
    golden._selftest_index_equals_expand()
    golden._selftest_flash_matches_direct()

    case = golden.build_case(seed=0, s_tokens=100, out_head=5)
    design = _build_design(case["n_tiles"], golden.HD, golden.TKV)

    def run_once():
        q_t = iron.tensor(np.ascontiguousarray(case["q_bf"]), dtype=_bf16, device="npu")
        kv_t = iron.tensor(np.ascontiguousarray(case["kv_bf"].reshape(-1)), dtype=_bf16,
                           device="npu")
        ctx_t = iron.zeros((golden.HD,), dtype=np.float32, device="npu")
        design(q_t, kv_t, ctx_t)
        return ctx_t.numpy().reshape(-1).copy()

    dev1 = run_once()
    dev2 = run_once()  # run-twice self-check, guards the CLFLUSH read race (bricklib convention)

    got = np.asarray(dev1, np.float64)
    exp_direct = np.asarray(case["exp_direct"], np.float64)
    exp_full = np.asarray(case["exp_full"], np.float64)
    den = np.linalg.norm(exp_direct)
    rel_l2 = float(np.linalg.norm(got - exp_direct) / den) if den else float(np.linalg.norm(got))
    rel_l2_vs_full = float(np.linalg.norm(got - exp_full) / den) if den else rel_l2
    determ = float(np.linalg.norm(dev1.astype(np.float64) - dev2.astype(np.float64)))
    nz = float(np.abs(dev1).sum())
    ok = (nz > 0.0) and (rel_l2 <= GATE)
    status = "PASS" if ok else ("FAIL-ZERO" if nz == 0.0 else "FAIL")

    print(f"[mha_decode_hd128] out_head={case['out_head']} h_kv={case['h_kv']} "
          f"S={case['s_tokens']} n_tiles={case['n_tiles']} TKV={golden.TKV} HD={golden.HD}")
    print(f"  device vs index-map reference (unexpanded KV)      rel_l2={rel_l2:.3e} "
          f"gate={GATE:.1e}")
    print(f"  device vs repeat_kv expand-then-slice reference    rel_l2={rel_l2_vs_full:.3e}")
    print(f"  run2run determinism                                {determ:.3e}")
    print(f"  nonzero                                             {nz:.3e}")
    print(f"  -> {status}")
    assert ok, f"mha_decode HD=128 GQA gate failed: {status} rel_l2={rel_l2:.3e}"
    print("PASS")


if __name__ == "__main__":
    main()
