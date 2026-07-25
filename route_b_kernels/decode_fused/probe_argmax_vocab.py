# SPDX-License-Identifier: Apache-2.0
"""Device gate for the per-column partial Argmax at GEMMA's vocab size (N=262144).

WHY. In the Gemma-3-270M decode graph the host reads back the FULL logits vector every
token -- 262144 bf16 = 512 KB -- purely so numpy can argmax it. That is the largest
host<->device edge in the graph. `Argmax` (argmax_op.py + argmax_slice.cc) already exists
and collapses it to cols x 8 bytes, but it has only ever been wired at the ASR proj_out
size (VOCAB_PAD=52224, slice 6528). Gemma needs slice 32768 -- 5x longer.

Two things are worth checking before trusting it there, and neither is obvious:
  * argmax_slice.cc is a SCALAR bf16->float scan. Scalar float loops are a known-bad family
    on aie2p (the rope-lut key loop was miscompiled that way, and isolated scalar-float
    probes hung the core outright earlier today). A 5x longer scalar loop is exactly where
    that would show.
  * the gate is INDEX-exactness, not rel-L2. A near-miss index is a wrong token, so the
    only acceptable result is an exact match against the host argmax -- including the
    tie-break rule (argmax_slice.cc uses strict `>`, so first-max wins, matching numpy).

This runs the op standalone -- no decode graph, no weights -- so it is independent of the
open port/re-pin decision on gen_gemma_decode.py.

Run under the NPU lock:  python probe_argmax_vocab.py
"""
import os
import sys

import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 -- MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator
from argmax_op import Argmax

BF16 = ml_dtypes.bfloat16
N = 262144        # Gemma-3-270M / Gemma-4-E2B tied-embedding vocab
COLS = 8
SLICE = N // COLS  # 32768 bf16 = 64 KB -- the WHOLE L1 of a core, so it must be tiled
CHUNK = 4096       # 8 KB tiles; 8 per column -> 64 partials total (512 bytes back, vs 512 KB)
NCHUNKS = SLICE // CHUNK


def host_reduce(raw_bf16, cols=COLS, slice_n=SLICE, chunk=CHUNK, nchunks=None):
    """(cols x nchunks) partials of [val:f32 | idx:i32] -> global argmax index.

    Partial p of column i covers absolute range [i*slice_n + p*chunk, +chunk), and the kernel
    reports an index LOCAL to that tile -- so the global index is i*slice_n + p*chunk + local.
    Scanning in (col, chunk) order with a strict `>` keeps the first-max-wins tie-break that
    both the kernel and numpy use."""
    nchunks = nchunks if nchunks is not None else slice_n // chunk
    b = np.asarray(raw_bf16, BF16).tobytes()
    best_v, best_i = -np.inf, -1
    parts = []
    for i in range(cols):
        for p in range(nchunks):
            o = (i * nchunks + p) * 8
            v = np.frombuffer(b[o:o + 4], np.float32)[0]
            li = int(np.frombuffer(b[o + 4:o + 8], np.int32)[0])
            g = i * slice_n + p * chunk + li
            parts.append((v, g))
            if v > best_v:
                best_v, best_i = v, g
    return best_i, best_v, parts


def make_case(name, rng):
    """Return (logits_bf16, description). Cases target where the max LIVES, since the
    per-column split means column placement is the interesting axis."""
    x = rng.standard_normal(N).astype(np.float32) * 0.5
    if name == "random":
        pass
    elif name == "max_in_first_col":
        x[17] = 50.0
    elif name == "max_in_last_col":
        x[N - 3] = 50.0
    elif name == "max_at_slice_boundary":     # last element of column 0
        x[SLICE - 1] = 50.0
    elif name == "duplicate_max":             # exact tie -> lowest index must win
        x[100] = 50.0
        x[N - 100] = 50.0
    return np.asarray(x, BF16), name


def main():
    ctx = AIEContext()
    op = Argmax(N=N, cols=COLS, chunk=CHUNK, context=ctx)
    fused = FusedMLIROperator(
        "argmax_vocab_probe",
        [(op, "logits", "amax")],
        input_args=["logits"],
        output_args=["amax"],
        buffer_sizes={"logits": N * 2, "amax": COLS * NCHUNKS * 8},
        context=ctx,
    )
    fused.compile()
    print(f"[argmax-probe] compiled N={N} cols={COLS} slice={SLICE} chunk={CHUNK} nchunks={NCHUNKS} -> {COLS*NCHUNKS} partials ({COLS*NCHUNKS*8} B back)", flush=True)

    c = fused.get_callable()
    lg = c.get_buffer("logits")
    am = c.get_buffer("amax")

    rng = np.random.default_rng(7)
    cases = ["random", "max_in_first_col", "max_in_last_col",
             "max_at_slice_boundary", "duplicate_max"]
    npass = 0
    for nm in cases:
        x, _ = make_case(nm, rng)
        ref = int(np.argmax(x.astype(np.float32)))       # numpy: first-max wins ties
        # Run TWICE and require the two reads to agree. The XRT shim in use is unfenced and
        # has a known CLFLUSH read race that returns stale / cold-zero output; a fenced shim
        # build is the real fix. The brick verify rail guards it the same way (run twice,
        # report run-to-run delta).
        # Running once here produced exactly that signature: 0/64 partials on most cases and
        # 24/64 on one, varying run to run -- which reads like a tiling bug but is the race.
        def run_once():
            np.copyto(lg.data, x.reshape(-1))
            am.data[:] = 0
            c.input_buffer.device = "cpu"
            c.output_buffer.device = "npu"
            c()
            return np.array(am.data, copy=True)

        # Retry until two consecutive reads agree AND are not the all-zero cold read. A single
        # pair is not enough here: the race yields all-zeros often enough that two zero reads
        # can agree with each other and look "stable" while being pure artifact.
        prev, r2, stable, tries = run_once(), None, False, 1
        for tries in range(2, 13):
            r2 = run_once()
            if np.array_equal(np.asarray(prev, BF16), np.asarray(r2, BF16)) and np.abs(np.asarray(r2, BF16).astype(np.float32)).sum() > 0:
                stable = True
                break
            prev = r2
        got, got_v, per_col = host_reduce(r2 if r2 is not None else prev)
        ok = got == ref
        npass += ok
        print(f"  [{nm:22s}] stable={str(stable):5s}({tries}) dev_idx={got:7d} ref_idx={ref:7d} "
              f"dev_val={float(got_v):+.4f} ref_val={float(x.astype(np.float32)[ref]):+.4f} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            # WHICH partials came back tells us the shape of the bug directly:
            #   only COLS nonzero  -> the nchunks partials per column collapse onto one slot
            #                         (output tap stride wrong), last chunk wins
            #   all nonzero, wrong -> the input tiling is off (each partial describes the
            #                         wrong slice of data)
            #   fewer than COLS    -> the stream is not being consumed at all
            nz = [(k, v, g) for k, (v, g) in enumerate(per_col) if not (v == 0.0 and g % CHUNK == 0)]
            print(f"      partials nonzero: {len(nz)}/{len(per_col)} "
                  f"(cols={COLS} nchunks={NCHUNKS})", flush=True)
            print(f"      col0 partials (chunk, val, global_idx): "
                  f"{[(k, round(float(v), 3), g) for k, v, g in nz if k < NCHUNKS]}", flush=True)
            print(f"      top partials: {sorted(per_col, reverse=True)[:4]}", flush=True)
            print(f"      ref lives in column {ref // SLICE} chunk {(ref % SLICE) // CHUNK} "
                  f"local {ref % CHUNK}", flush=True)

    print(f"\n[argmax-probe] {npass}/{len(cases)} index-exact", flush=True)
    if npass == len(cases):
        print("  VERDICT: Argmax is index-exact at the Gemma vocab size. The 512 KB/token "
              "logits readback can collapse to 64 bytes -- append (op_argmax, 'logits', 'amax') "
              "to the decode runlist and reduce cols-way on the host.")
    else:
        print("  VERDICT: NOT index-exact at this size -- do NOT wire it into the decode graph.")
    return 0 if npass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
