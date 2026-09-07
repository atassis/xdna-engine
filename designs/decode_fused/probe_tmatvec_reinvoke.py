#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does TMatVec stop being a function when invoked TWICE in ONE runtime sequence?

The race recorded in [[tmatvec-races-only-when-invoked-more-than-once-per-sequence]] was measured
on the FULL fused decode (28 designs, 336 dumped buffers). The isolation that ran alongside it was
ONE invocation per dispatch, 30 dispatches. Nobody ran the middle rung -- the same design invoked
twice in one runtime sequence, with nothing else in the graph -- so "TMatVec is not a function" and
"something in the decode graph around it" were never separated.

That matters because the recorded CAUSE does not survive its own controls. Read off the generated
MLIR of both designs:

    op                    A objectfifo             depth  A fill BD    objects/BD  self-determinism
    TMatVec (TMV arm)     memref<8192xbf16>          2    len=262144       32       22-51 / 336
    GEMV op_ctx (kv arm)  memref<4x2048xbf16>        2    len=524288       64        0 / 336
    GEMV op_q  (kv arm)   memref<4x1024xbf16>        2    len=262144       64        0 / 336

op_q's A fill is byte-for-byte the same BD length into a SMALLER object, so it cycles the ring
twice as many times, shim->core with no MemTile, and it is clean on every layer of the deterministic
arm. "One BD into a ring with no per-object lock" is the construct the op it replaces already uses,
harder.

Two outcomes, and they point at different work:
  reproduces     -> the defect is TMatVec's, and this is a 3-design reproducer to bisect against
                    instead of a 28-design one.
  does not       -> the defect is in the decode graph AROUND op_ctx, and routing A through a
                    MemTile would not have touched it.

Arms (env):
  PROBE_OP=tmv|gemv     gemv is the harness control at op_ctx's own shape; it must come out clean.
  PROBE_SEP=1|0         1 (default) puts a filler design between the two invocations, so each gets
                        its OWN aiex.ConfigureOp -- which is what the decode does, where ~20 other
                        designs separate one layer's op_ctx from the next. 0 puts them back to back,
                        and fuse_mlir then shares ONE configure point between them.
  PROBE_INV=2           invocations of the operator under test.
  PROBE_RUNS=3          dispatches; every run is compared against run 0 buffer by buffer.

The assay is SELF-determinism, not the golden: identical inputs into every invocation, so C_i must
be bit-identical to C_0 within a run AND across runs. The golden is reported as a note.
"""
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports
from iron.common import AIEContext  # noqa: E402
from iron.common.sequence import OperatorSequence  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402
from iron.operators.gemv.op import GEMV  # noqa: E402
from iron.operators.tmatvec.op import TMatVec  # noqa: E402

BF16 = ml_dtypes.bfloat16

# op_ctx's own shape in the qwen3-0.6b decode: head_dim 128, max_seq 2048, 16 query heads over
# 8 kv heads. Not a reduced shape -- the race is reported here and nowhere smaller.
M, K = 128, 2048
HQ, HKV = 16, 8
BGRP = HQ // HKV
COLS = 8
RPC = int(os.environ.get("TMV_RPC", "64"))

OP = os.environ.get("PROBE_OP", "tmv")
SEP = os.environ.get("PROBE_SEP", "1") == "1"
INV = int(os.environ.get("PROBE_INV", "2"))
RUNS = int(os.environ.get("PROBE_RUNS", "3"))
# The decode builds with this; placement is not obviously irrelevant to a race, so match it.
FLAGS = os.environ.get("PROBE_PLACER_FLAGS", "--cores-per-col 1").split()

FILL = 4096  # filler ElementwiseMul width; its only job is to break the configure run


def build():
    ctx = AIEContext()
    if OP == "tmv":
        op = TMatVec(M=M, K=K, num_aie_columns=HKV, num_batches=HQ, batch_group=BGRP,
                     rows_per_chunk=RPC, context=ctx)
        n_mat = HKV
    elif OP == "gemv":
        # op_ctx as it is TODAY: same M/K/batches, the contraction the transpose feeds.
        op = GEMV(M=M, K=K, num_aie_columns=COLS, tile_size_input=4, tile_size_output=16,
                  num_batches=HQ, context=ctx)
        n_mat = HQ
    else:
        raise SystemExit(f"PROBE_OP must be tmv|gemv, got {OP!r}")

    filler = ElementwiseMul(size=FILL, tile_size=FILL // COLS, num_aie_columns=COLS, context=ctx)

    outs = [f"C{i}" for i in range(INV)]
    runlist = []
    for i in range(INV):
        # THE SAME op OBJECT every time. unique_operators() de-dupes by id(), so this is one
        # design with INV aiex.RunOps -- exactly how the decode reuses one op_ctx across layers.
        # A second TMatVec(...) with equal parameters would be a second DESIGN and test nothing.
        runlist.append((op, "A", "W", outs[i]))
        if SEP and i != INV - 1:
            runlist.append((filler, "fin", "fones", "fout"))

    a_elems = n_mat * K * M
    sizes = {"A": a_elems * 2, "W": HQ * K * 2,
             "fin": FILL * 2, "fones": FILL * 2, "fout": FILL * 2}
    sizes.update({o: HQ * M * 2 for o in outs})

    name = f"tmv_reinvoke_{OP}_inv{INV}_{'sep' if SEP else 'btb'}_oarena"
    # HARNESS CONTRACT, and the first arm of this probe got it wrong: SequenceFullELFCallable
    # syncs ONLY the input arena host->device and the output arena device->host. The SCRATCH
    # arena is never synced either way, so a buffer the host reads back must live in the
    # OUTPUT arena or it is read out of stale host cache lines. Measured with C in scratch:
    # the GEMV control -- an operator with no known defect -- "raced" at 32/96/192 elements,
    # i.e. exactly 1/3/6 x 64-byte lines, the fingerprint probe_fusion_roundtrip.py records.
    # A and W stay in scratch because that is where the decode keeps vc and sw, and they are
    # written once before the first dispatch and never read back.
    seq = OperatorSequence(name, runlist,
                           input_args=["fin", "fones"], output_args=["fout"] + outs,
                           buffer_sizes=sizes, context=ctx, extra_flags=FLAGS)
    seq.compile()
    return seq, outs, n_mat


def main():
    print(f"[probe] op={OP} invocations={INV} separate_configure={SEP} runs={RUNS} "
          f"shape M={M} K={K} batches={HQ} bgrp={BGRP} rpc={RPC}", flush=True)
    seq, outs, n_mat = build()
    if os.environ.get("PROBE_BUILD_ONLY") == "1":
        print("[probe] build only -- ELF is built, no device run requested")
        return 0
    c = seq.get_callable()

    rng = np.random.default_rng(20260907)
    A = np.asarray(rng.standard_normal((n_mat, K, M), dtype=np.float32), BF16)
    W = np.asarray(rng.standard_normal((HQ, K), dtype=np.float32), BF16)
    np.copyto(c.get_buffer("A").data, A.reshape(-1))
    np.copyto(c.get_buffer("W").data, W.reshape(-1))
    np.copyto(c.get_buffer("fin").data, np.asarray(np.ones(FILL, np.float32), BF16))
    np.copyto(c.get_buffer("fones").data, np.asarray(np.ones(FILL, np.float32), BF16))

    # Nothing syncs scratch, so make the one host->device transfer of A and W explicit
    # rather than relying on allocation-time residency.
    c.scratch_buffer.device = "cpu"
    c.scratch_buffer.to("npu")

    if OP == "tmv":
        gold = np.stack([(W[b].astype(np.float32) @ A[b // BGRP].astype(np.float32))
                         for b in range(HQ)])
    else:
        gold = None

    runs = []
    for r in range(RUNS):
        # Deliberately NOT pre-zeroed: this callable does not flush the output arena before
        # the run, so a pre-fill through .data leaves dirty host lines over the region the
        # DMA is about to write and the readback shadows the device.
        c()
        runs.append([np.array(c.get_buffer(o).data, copy=True).reshape(HQ, M) for o in outs])
        print(f"  run {r}: dispatched", flush=True)

    def nbits(x, y):
        return int((np.asarray(x, BF16) != np.asarray(y, BF16)).sum())

    print(f"\n[within a run] every invocation gets identical A and W, so C_i must equal C_0")
    within = 0
    for r, cs in enumerate(runs):
        for i in range(1, INV):
            d = nbits(cs[i], cs[0])
            within += d
            print(f"  run {r}: C{i} vs C0  {d:6d} / {HQ * M} elements differ"
                  f"{'' if d == 0 else '   <-- NOT A FUNCTION'}")

    print(f"\n[across runs] identical inputs, so run r must equal run 0")
    across = 0
    for r in range(1, RUNS):
        for i, o in enumerate(outs):
            d = nbits(runs[r][i], runs[0][i])
            across += d
            print(f"  run {r} vs run 0: {o}  {d:6d} / {HQ * M} elements differ"
                  f"{'' if d == 0 else '   <-- NONDETERMINISTIC'}")

    if gold is not None:
        g = np.asarray(gold, np.float64)
        for i, o in enumerate(outs):
            v = np.asarray(runs[0][i], np.float64)
            rel = float(np.linalg.norm(v - g) / np.linalg.norm(g))
            print(f"\n[note] {o} rel-L2 vs f32 golden: {rel:.4e}  (a note, not the gate)")

    # A buffer nobody wrote compares identical to itself. Gate on the result being REAL
    # before gating on it being stable: an all-zero readback reported CLEAN once already,
    # when a stale cached ELF was still writing C into the scratch arena.
    nz = [int((np.asarray(runs[0][i], BF16).astype(np.float32) != 0).sum())
          for i in range(INV)]
    live = all(n > HQ * M // 2 for n in nz)
    print(f"\n[sanity] nonzero elements in run 0: "
          + ", ".join(f"C{i} {n}/{HQ * M}" for i, n in enumerate(nz)))
    if not live:
        print("\n[probe] VERDICT: VOID -- the readback is (mostly) zero, so nothing was "
              "measured. Suspect a stale cached ELF or the wrong arena; do NOT read the "
              "determinism numbers above.")
        return 2
    if gold is not None:
        g = np.asarray(gold, np.float64)
        rel0 = float(np.linalg.norm(np.asarray(runs[0][0], np.float64) - g)
                     / np.linalg.norm(g))
        if rel0 > 0.05:
            print(f"\n[probe] VERDICT: VOID -- rel-L2 {rel0:.4e} against the golden is not "
                  "this operator computing its function at all.")
            return 2
    ok = within == 0 and across == 0
    print(f"\n[probe] VERDICT: {'CLEAN' if ok else 'RACES'} -- "
          f"{within} within-run and {across} across-run differing elements")
    if OP == "tmv" and ok:
        print("  TMatVec IS a function when invoked twice in isolation. The fused-decode "
              "nondeterminism is therefore NOT this operator on its own; it comes from the "
              "graph around it, and MemTile routing of A would not have addressed it.")
    elif OP == "tmv":
        print("  Reproduced in isolation. This is now a 3-design reproducer for bisection "
              "(drop the acc Buffer, vary fifo depths, vary rows_per_chunk).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
