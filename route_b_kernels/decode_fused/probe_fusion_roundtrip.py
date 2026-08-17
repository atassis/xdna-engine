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

ROOT CAUSE (2026-08-17, and it is NOT what this file recorded for three weeks). The
cache-line-quantised staleness was real -- correct-element counts of 32/64/128 bf16 = 1/2/4
x 64-byte lines, varying run to run -- but it was not an XRT invalidation bug, and no fenced
shim was needed. The dirty lines were OURS. `dst.data[:] = 0` below writes the output arena
through the raw `.data` array, which `NpuTensor.data` documents as the one write that is not
reconciled: nothing is recorded, so nothing is flushed, and the host keeps dirty zero lines
over the region the DMA is about to write. The device-to-host sync afterwards does not
discard them, so they shadow the device's output and this probe read back its own pre-fill.
Which lines survived was eviction luck, which is the run-to-run variation that read as a race.

MEASURED with a sentinel pre-fill (probe_fusion_output_prefill.py, three arms over one ELF):
pre-fill 0 -> 1024/1024 read 0; pre-fill 7168 -> 1024/1024 read 7168 and none read 0, so the
device had written the whole output; pre-fill 7168 then flushed before dispatch -> 1024/1024
exact. Fixed in `FusedFullELFCallable._sync_inputs` (and `_FusedArenaCallable`'s copy in
sequence.py) by flushing the output arena to the device alongside the input, which retires
the dirty lines. This probe now passes 3/3 at N=1024 and 4096, tile 1024 and 4096, for the
hand-written Identity and for IRON's built-in ElementwiseMul.

Kept as the regression guard for that class: a caller that pre-fills an output arena through
`.data` and reads plausible-looking stale bytes is silent, and this is where it shows.

Doubles as the permanent regression guard for that whole class: four separate
harness/runtime bugs in one day presented as kernel bugs, and a round-trip assert catches
them at the point of failure instead of five bisect stages later.

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
from iron.operators.elementwise_mul.op import ElementwiseMul
from identity_op import Identity

BF16 = ml_dtypes.bfloat16
N = int(os.environ.get("PROBE_N", 4096))
TILE = int(os.environ.get("PROBE_TILE", 1024))
# PROBE_PATH=fused    -> FusedMLIROperator (emits ELFs for the Rust engine; its Python
#                        callable path appears to be exercised by nobody)
# PROBE_PATH=sequence -> OperatorSequence, the path gen_gemma_decode drives to token parity
#                        and the one that received the residency fix in 2970f8a
PATH = os.environ.get("PROBE_PATH", "fused")
# PROBE_OP=identity -> the hand-written Identity op in this directory
# PROBE_OP=memcopy  -> IRON's BUILT-IN MemCopy. This is the discriminator: gen_gemma_decode
#                     drives built-in operators to 8/8 token parity on this same runtime, so
#                     if a built-in round-trips and the hand-written one does not, the fault
#                     is in how a locally-defined MLIROperator declares itself, not the path.
OP = os.environ.get("PROBE_OP", "identity")


def main():
    ctx = AIEContext()
    if OP == "emul":
        op = ElementwiseMul(size=N, tile_size=N // 8, num_aie_columns=8, context=ctx)
        runlist = [(op, "src", "ones", "dst")]
        kwargs = dict(input_args=["src", "ones"], output_args=["dst"],
                      buffer_sizes={"src": N * 2, "ones": N * 2, "dst": N * 2}, context=ctx)
    else:
        op = Identity(N=N, tile=TILE, context=ctx)
        runlist = [(op, "src", "dst")]
        kwargs = dict(input_args=["src"], output_args=["dst"],
                      buffer_sizes={"src": N * 2, "dst": N * 2}, context=ctx)
    if PATH == "sequence":
        fused = OperatorSequence("roundtrip_seq", runlist, dispatch="fused", **kwargs)
    else:
        fused = FusedMLIROperator("fusion_roundtrip_probe", runlist, **kwargs)
    fused.compile()
    print(f"[roundtrip] op={OP} path={PATH} N={N} tile={TILE} ntiles={N // TILE}", flush=True)

    c = fused.get_callable()
    src, dst = c.get_buffer("src"), c.get_buffer("dst")
    ones = c.get_buffer("ones") if OP == "emul" else None

    # A ramp with no zeros anywhere: then "returned zeros" is unambiguously "never written",
    # and any partial delivery shows up as a prefix/suffix boundary rather than as noise.
    x = ((np.arange(N) % 4096) + 1).astype(np.float32)
    x = np.asarray(x, BF16)

    npass = 0
    for trial in range(3):
        np.copyto(src.data, x.reshape(-1))
        if ones is not None:
            np.copyto(ones.data, np.ones(N, BF16))
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
