# SPDX-License-Identifier: Apache-2.0
"""Does the FusedMLIROperator path round-trip AT ALL?

Write a known pattern -> identity kernel copies in to out -> read it back. No arithmetic, no
reduction, no packed output, no tiling cleverness. If this fails, the fusion path is broken
outright and nothing computed on it can be trusted -- and there is no point debugging any
real kernel that runs there.

WHY THIS EXISTS. probe_argmax_vocab.py returns all-zero reads through this path. That
survived (a) a residency/sync fix to FusedFullELFCallable (parent= on sub-views + forced
output sync, mirroring OperatorSequence) and (b) a bisect down to the STOCK un-chunked
ASR-size shape, which fails identically. Meanwhile the same argmax kernel is index-exact on
the brick verify rail. So the fault is the path, not the kernel and not the tiling.

Doubles as the permanent regression guard for that whole class: three separate
harness/runtime bugs in one day presented as kernel bugs, and a round-trip assert catches
every one of them at the point of failure instead of five bisect stages later.

Run:
    python probe_fusion_roundtrip.py            # default N=4096, tile=1024
    PROBE_N=8192 PROBE_TILE=8192 python ...     # single-tile variant
"""
import os
import sys

import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 -- MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.common.sequence import OperatorSequence
from identity_op import Identity

BF16 = ml_dtypes.bfloat16
N = int(os.environ.get("PROBE_N", 4096))
TILE = int(os.environ.get("PROBE_TILE", 1024))
# PROBE_PATH=fused    -> FusedMLIROperator (emits ELFs for the Rust engine; its Python
#                        callable path appears to be exercised by nobody)
# PROBE_PATH=sequence -> OperatorSequence, the path gen_gemma_decode drives to token parity
#                        and the one that received the residency fix in 2970f8a
PATH = os.environ.get("PROBE_PATH", "fused")


def main():
    ctx = AIEContext()
    op = Identity(N=N, tile=TILE, context=ctx)
    kwargs = dict(input_args=["src"], output_args=["dst"],
                  buffer_sizes={"src": N * 2, "dst": N * 2}, context=ctx)
    if PATH == "sequence":
        fused = OperatorSequence("roundtrip_seq", [(op, "src", "dst")],
                                 dispatch="fused", **kwargs)
    else:
        fused = FusedMLIROperator("fusion_roundtrip_probe", [(op, "src", "dst")], **kwargs)
    fused.compile()
    print(f"[roundtrip] path={PATH} N={N} tile={TILE} ntiles={N // TILE}", flush=True)

    c = fused.get_callable()
    src, dst = c.get_buffer("src"), c.get_buffer("dst")

    # A ramp with no zeros anywhere: then "returned zeros" is unambiguously "never written",
    # and any partial delivery shows up as a prefix/suffix boundary rather than as noise.
    x = ((np.arange(N) % 4096) + 1).astype(np.float32)
    x = np.asarray(x, BF16)

    npass = 0
    for trial in range(3):
        np.copyto(src.data, x.reshape(-1))
        dst.data[:] = 0
        c()
        got = np.array(dst.data, copy=True)

        exact = int((np.asarray(got, BF16) == x).sum())
        nz = int((np.asarray(got, BF16).astype(np.float32) != 0).sum())
        ok = exact == N
        npass += ok
        print(f"  trial {trial}: exact {exact}/{N}  nonzero {nz}/{N} -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            g = np.asarray(got, BF16).astype(np.float32)
            e = x.astype(np.float32)
            bad = np.flatnonzero(g != e)
            print(f"      first bad index {bad[0]}, last bad {bad[-1]}, "
                  f"{len(bad)} wrong", flush=True)
            print(f"      exp[:6]={e[:6]}  got[:6]={g[:6]}", flush=True)
            print(f"      exp[-6:]={e[-6:]}  got[-6:]={g[-6:]}", flush=True)

    print(f"\n[roundtrip] {npass}/3 trials exact", flush=True)
    if npass == 3:
        v = ("the fusion path DOES round-trip. The all-zero argmax reads are therefore NOT a "
             "generic path failure -- look at that op's arg-spec/layout binding specifically.")
    elif npass == 0:
        v = ("the fusion path does NOT round-trip even for a plain copy. Nothing computed on "
             "this path can be trusted; fix the path before debugging any kernel on it.")
    else:
        v = (f"the fusion path round-trips INTERMITTENTLY ({npass}/3) -- a race, not a wiring "
             "error. Suspect the unfenced-shim CLFLUSH read race.")
    print(f"  VERDICT: {v}", flush=True)
    return 0 if npass == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
