#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""WHERE does the fused decode first diverge between two identical runs?

The earlier assay reported a
COUNT -- 22 of 24 dumped buffers at 2 layers, 22-51 of 336 at full depth -- and localised the
defect from it to `op_ctx`. A count cannot do that. The decode is sequential, so one differing
value contaminates every buffer after it, and 22/24 says divergence starts around the THIRD buffer
of layer 0, which is upstream of the context step entirely. This reports the ORDER instead: the
first (step, runlist index, buffer) whose two passes disagree.

That script was never committed, so its number could not be re-run or audited. This one is the
instrument.

WHY A NEGATIVE HERE IS WEAKER THAN A POSITIVE, and the asymmetry is not symmetric hedging.
Every decode intermediate lives in the SCRATCH arena, which SequenceFullELFCallable syncs in
NEITHER direction. A host read of scratch can therefore return a clean-but-stale cache line
holding the PREVIOUS pass's value, which MASKS a real difference. It cannot invent one: a false
difference needs the host to have WRITTEN the buffer, leaving dirty lines that shadow the DMA --
which is exactly what made the GEMV control "race" at 64-byte granularity in
probe_tmatvec_reinvoke.py. So this probe never pre-fills a dumped buffer, and it explicitly
invalidates scratch before every snapshot and flushes it after every host write.

Arms:
  TMV_CTX=1  the arm that races
  TMV_CTX=0  the kv arm, the 0/336 control -- run it and expect NO divergence

  PROBE_LAYERS=2   the ~50 s reproducer depth
  PROBE_STEPS=3    decode positions per pass
  PROBE_NO_SYNC=1  DROP both scratch syncs, reproducing a harness that does not do them.
                   This is the discriminator: the KV caches are host-zeroed between passes,
                   and without the flush those dirty zero lines sit over a region the device
                   is about to DMA into. An eviction at an arbitrary moment clobbers
                   device-written KV -- real corruption, host-caused, timing-dependent. If
                   the race appears only here, the defect is in the measuring instrument.
"""
import argparse
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph, load_weight_buffer, isolate_build_dir  # noqa: E402
from verify_llm_decode import rope_row  # noqa: E402

BF16 = ml_dtypes.bfloat16
NO_SYNC = False


def dumped_buffers(runlist):
    """Every op's OUTPUT buffer, in runlist order, de-duplicated on first appearance.

    The output is the last name in each entry. Slice notation is reduced to its base buffer:
    get_buffer() resolves slices, but two slices of one buffer are one region to compare.
    """
    seen, order = set(), []
    for entry in runlist:
        name = entry[-1]
        base = name[: name.index("[")] if "[" in name else name
        if base not in seen:
            seen.add(base)
            order.append(base)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt-token", type=int, default=785)
    a = ap.parse_args()
    isolate_build_dir("probe")

    layers = int(os.environ.get("PROBE_LAYERS", "2"))
    steps = int(os.environ.get("PROBE_STEPS", "3"))
    tmv = os.environ.get("TMV_CTX") == "1"
    global NO_SYNC
    NO_SYNC = os.environ.get("PROBE_NO_SYNC") == "1"
    print(f"[probe] arm={'TMV_CTX' if tmv else 'kv'} layers={layers} steps={steps} "
          f"scratch_sync={'OFF' if NO_SYNC else 'on'}", flush=True)

    sp, fused, weights, md = build_graph(a.spec, a.weights, layers)
    HD, VOCAB = sp.head_dim, sp.vocab
    c = fused.get_callable()
    params = c.params
    names = dumped_buffers(fused.runlist)
    print(f"[probe] {len(names)} distinct output buffers in runlist order", flush=True)

    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    scale = np.sqrt(sp.d_model) if sp.embed_scale == "sqrt_d_model" else 1.0
    xin, rope_buf = c.get_buffer("x"), c.get_buffer("rope_global")

    def load_all():
        """Reset every device-visible buffer the host owns, then FLUSH scratch.

        The KV caches are host-zeroed here. Without the flush those zero lines stay dirty in the
        host's cache over a region the device is about to DMA into, and an eviction at an arbitrary
        moment clobbers device-written KV -- a real corruption, timing-dependent, and one that
        would read as exactly the nondeterminism this probe is trying to locate.
        """
        for name, arr in weights.items():
            load_weight_buffer(c.get_buffer(name), arr)
        if not NO_SYNC:
            c.scratch_buffer.device = "cpu"
            c.scratch_buffer.to("npu")

    def snapshot():
        # Invalidate scratch first: without it a read can return a clean-but-stale line holding
        # the previous pass's value, which would MASK a difference.
        if not NO_SYNC:
            c.scratch_buffer.device = "npu"
            c.scratch_buffer.to("cpu")
        return {n: np.array(c.get_buffer(n).data, copy=True) for n in names}

    def one_pass(tag):
        load_all()
        snaps, toks = [], []
        tok = a.prompt_token
        for pos in range(steps):
            with xin.overwrite() as _buf:
                _buf[:] = np.asarray(embed[tok] * scale, BF16).reshape(-1)
            with rope_buf.overwrite() as _buf:
                _buf[:] = rope_row(pos, HD, sp.rope_theta_global).reshape(-1)
            params.write("kv_off", int(pos * HD))
            params.write("sm_mask", int(pos + 1))
            params.sync()
            c()
            snaps.append(snapshot())
            tok = int(np.argmax(np.asarray(c.get_buffer("logits").data[:VOCAB], np.float32)))
            toks.append(tok)
        print(f"  pass {tag}: tokens {toks}", flush=True)
        return snaps, toks

    A, tokA = one_pass("A")
    B, tokB = one_pass("B")

    print(f"\n[first divergence] runlist order, step-major")
    first = None
    ndiff = 0
    for pos in range(steps):
        for idx, n in enumerate(names):
            d = int((np.asarray(A[pos][n], BF16) != np.asarray(B[pos][n], BF16)).sum())
            if d:
                ndiff += 1
                if first is None:
                    first = (pos, idx, n, d, A[pos][n].size)
    if first is None:
        print(f"  NONE -- {len(names) * steps} buffer snapshots identical across both passes")
        print(f"\n[probe] VERDICT: DETERMINISTIC on this arm at {layers} layers, {steps} steps.")
        return 0
    pos, idx, n, d, sz = first
    print(f"  step {pos}, runlist index {idx}/{len(names)}, buffer '{n}': "
          f"{d}/{sz} elements differ")
    print(f"  {ndiff} of {len(names) * steps} snapshots differ in total")
    print(f"\n  buffers BEFORE it that are stable at step {pos}: {names[:idx]}")
    print(f"\n[probe] VERDICT: diverges first at '{n}'. Everything after it in runlist order is "
          f"downstream contamination, not independent evidence.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
