# SPDX-License-Identifier: Apache-2.0
"""Which side loses the fused round-trip: the device's write, or the host's stale lines?

`probe_fusion_roundtrip` delivers a cache-line-quantised PREFIX of the identity copy and
reads zeros for the rest, varying run to run. Zero is ambiguous there, because the harness
itself writes zeros into the output arena host-side (`dst.data[:] = 0`) through the raw
`.data` array, which records nothing in the coherence map. So a "missing" element is either
a byte the device never wrote or the harness's own zero read back.

Three arms over the same built ELF disambiguate it:

  data-zero      the current harness: pre-zero through `.data`, nothing recorded. Baseline.
  data-sentinel  pre-fill with a sentinel instead of zero. A wrong element reading SENTINEL
                 is the HOST's own pre-fill surviving the dispatch -- the device's bytes were
                 overwritten from the host side. A wrong element reading 0 means the device
                 genuinely wrote nothing there.
  prefill-flush  pre-fill with the sentinel and then flush the output arena to the device
                 before dispatch, so no dirty host line for that region outlives the DMA.
                 If this arm is exact, the defect is the un-flushed pre-fill, not the ELF,
                 not the argument binding and not the device write.

Run:
    python probe_fusion_output_prefill.py           # N=1024, tile=1024, 3 trials per arm
    PROBE_N=4096 PROBE_TRIALS=5 python ...
"""
import os
import sys

import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 -- MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator
from identity_op import Identity

BF16 = ml_dtypes.bfloat16
N = int(os.environ.get("PROBE_N", 1024))
TILE = int(os.environ.get("PROBE_TILE", 1024))
TRIALS = int(os.environ.get("PROBE_TRIALS", 3))
# Exactly representable in bf16 and nowhere in the ramp, so a sentinel read is unambiguous.
SENTINEL = 7168.0
ARMS = ("data-zero", "data-sentinel", "prefill-flush")


def run_arm(arm, c, src, dst, x):
    """One arm, TRIALS times. Returns (n_exact_trials, per-trial breakdown)."""
    rows = []
    for _ in range(TRIALS):
        np.copyto(src.data, x.reshape(-1))
        if arm == "data-zero":
            dst.data[:] = 0
        else:
            dst.data[:] = np.asarray(SENTINEL, BF16)
        if arm == "prefill-flush":
            # Record the pre-fill and push it, so the host holds no dirty line over the
            # region the DMA is about to write.
            c.output_buffer.device = "cpu"
            c.output_buffer.to("npu")
        c()
        got = np.asarray(np.array(dst.data, copy=True), BF16).astype(np.float32)
        e = x.astype(np.float32)
        rows.append((
            int((got == e).sum()),
            int((got == 0).sum()),
            int((got == SENTINEL).sum()),
        ))
    return rows


def main():
    ctx = AIEContext()
    op = Identity(N=N, tile=TILE, context=ctx)
    fused = FusedMLIROperator(
        "fusion_roundtrip_probe",
        [(op, "src", "dst")],
        input_args=["src"],
        output_args=["dst"],
        buffer_sizes={"src": N * 2, "dst": N * 2},
    )
    fused.compile()
    print(f"[prefill] N={N} tile={TILE} ntiles={N // TILE} trials={TRIALS} "
          f"sentinel={SENTINEL}", flush=True)

    c = fused.get_callable()
    src, dst = c.get_buffer("src"), c.get_buffer("dst")
    x = np.asarray(((np.arange(N) % 4096) + 1).astype(np.float32), BF16)

    results = {}
    for arm in ARMS:
        rows = run_arm(arm, c, src, dst, x)
        results[arm] = rows
        print(f"  {arm:14s}", flush=True)
        for i, (exact, zeros, sent) in enumerate(rows):
            print(f"    trial {i}: exact {exact}/{N}  zero {zeros}/{N}  "
                  f"sentinel {sent}/{N} -> {'PASS' if exact == N else 'FAIL'}", flush=True)

    zero_arm = results["data-zero"]
    sent_arm = results["data-sentinel"]
    flush_arm = results["prefill-flush"]

    print("\n[prefill] VERDICT", flush=True)
    sent_lost = sum(s for _, _, s in sent_arm)
    zero_lost = sum(z for _, z, _ in sent_arm)
    if sent_lost > 0 and zero_lost == 0:
        print("  the missing bytes are the HOST's own pre-fill, not an unwritten device "
              "region: with a sentinel pre-fill every wrong element reads the SENTINEL and "
              "none reads 0. The device wrote the whole output; the host's un-recorded "
              "pre-fill lines were written back over it.", flush=True)
    elif zero_lost > 0 and sent_lost == 0:
        print("  the missing bytes were never written by the device: a sentinel pre-fill "
              "still reads 0 there, so the DMA did not cover the region. Look at the "
              "argument binding / arena layout, not the sync.", flush=True)
    else:
        print(f"  MIXED: sentinel {sent_lost}, zero {zero_lost} over {TRIALS} trials -- "
              "both mechanisms present, or the arm did not reproduce.", flush=True)

    if all(e == N for e, _, _ in flush_arm):
        print(f"  and flushing the pre-fill before dispatch is EXACT {TRIALS}/{TRIALS}: the "
              "fix is to record/flush the output arena's host pre-fill, in the callable or "
              "in the harness.", flush=True)
    else:
        print("  flushing the pre-fill did NOT fix it -- the mechanism is not (only) the "
              "un-recorded pre-fill.", flush=True)

    return 0 if all(e == N for e, _, _ in flush_arm) else 1


if __name__ == "__main__":
    sys.exit(main())
