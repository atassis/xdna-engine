#!/usr/bin/env python3
"""Does the restream price track how DISSIMILAR the two streams are, or is it flat per change?

Two sound measurements of "the cost of changing the instruction stream inside one hw_context"
disagree by ~750x: [[bd-restream-inside-one-context-is-free]] measured < 3 us for a pair differing
only in BD size/stride, and [[2026-08-22-a-stream-change-costs-what-a-transition-costs]] measured
+2.258 ms in the encoder for fc1's panel-drain stream against an identity stream. Two variables moved
at once -- the streams' SIMILARITY, and isolated-vs-in-encoder. This holds the second fixed (everything
here is isolated) and sweeps the first.

Construction, unchanged from `switch-cost-per-transition-controlled`: ONE multiset of dispatches, half
from each stream, timed in two orders. ALTERNATING pays total-1 stream changes, GROUPED pays 1. Both
arms run identical work, so the difference divided by the change count is the per-change price. Arms
are interleaved per rep and the leading arm alternates, so box drift cancels rather than loading onto
whichever arm ran second.

The SELF pair is the control the original construction lacked: the same instruction file loaded into
two separate BOs. Its content dissimilarity is zero, so any cost it shows is the price of pointing the
dispatch at a different instruction BO -- which would mean the price is not about dissimilarity at all.

TIMING ONLY: nothing reads C, and the streams' embedded rtp_write picks the epilogue mode, so running
a foreign stream on a given xclbin computes the foreign stream's mode. That is intended -- all three
designs link the same core objects and differ only in constants.

ACTIVITY AXES (--interleave-contexts / --interleave-bos). Dissimilarity is flat at zero and PASSIVE
context residency is flat at zero, so the two surviving suspects for the encoder's +2.3 ms are BO
churn and instruction-BO eviction. Both predict the price turns on with ACTIVITY, which is the axis
passive ballast held fixed. A ballast dispatch is therefore issued after every measured dispatch,
round-robin, identically in both arms -- so the ballast's own cost cancels in ALT-GRP and what
survives is whether ALT's working set, one stream larger than GRP's, crosses a capacity threshold
GRP's does not. The two axes separate the resource: --interleave-contexts spreads the ballast over
its own hw_contexts, --interleave-bos keeps it on the measured pair's context as extra live
instruction BOs with content pinned identical.

Usage (single-tenant, under scripts/npu_lock.sh, services quiesced):
  restream_similarity_ab.py --root <repo> --xclbin <stem> --pair <name>=<specA>,<specB> [...]
  where spec = <instsStem>:<K>:<N>
"""
import argparse
import json
import statistics
import time

import numpy as np
import pyxrt

WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
M = 512
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--xclbin", required=True, help="xclbin stem, without final_ / .xclbin")
ap.add_argument("--pair", action="append", required=True,
                help="name=spec,spec where spec = stem:K:N[@xclbinStem] (@ gives the stream its own "
                     "hw_context, making the pair a real xclbin transition -- the positive control)")
ap.add_argument("--total", type=int, default=256, help="dispatches per arm (half from each stream)")
ap.add_argument("--reps", type=int, default=25)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--extra-contexts", type=int, default=0,
                help="hold N additional live hw_contexts open, dispatching into none of them. The "
                     "encoder runs 9; the isolated probe runs 1, and that is the only surviving "
                     "difference between their restream prices.")
ap.add_argument("--interleave-contexts", type=int, default=0,
                help="dispatch a ballast stream on each of N OWN hw_contexts, round-robin between "
                     "the measured dispatches. Activity across contexts, not mere residency.")
ap.add_argument("--interleave-bos", type=int, default=0,
                help="same round-robin ballast, but all N share the measured pair's context as "
                     "distinct live instruction BOs of identical content -- many streams in flight "
                     "with dissimilarity pinned at zero.")
ap.add_argument("--ballast", default=None,
                help="spec dispatched as ballast (default: the first pair's first spec)")
ap.add_argument("--out", default=None, help="write JSON here")
a = ap.parse_args()

if a.interleave_contexts and a.interleave_bos:
    ap.error("--interleave-contexts and --interleave-bos confound the resource; sweep one at a time")

d = pyxrt.device(0)
_xb = {}
_ctx = {}
_hw = []
_ballast = []


def xclbin_of(stem):
    """Registered once per stem: re-registering the same xclbin is not idempotent on XRT."""
    if stem not in _xb:
        xb = pyxrt.xclbin(f"{a.root}/{WA}/final_{stem}.xclbin")
        d.register_xclbin(xb)
        _xb[stem] = xb
    return _xb[stem]


def kernel_on_new_context(stem):
    """A kernel on its own fresh hw_context. The context is held live for the process."""
    xb = xclbin_of(stem)
    hw = pyxrt.hw_context(d, xb.get_uuid())
    _hw.append(hw)
    return pyxrt.kernel(hw, xb.get_kernels()[0].get_name())


def context(stem):
    """The one memoised hw_context per stem, so a stem is only ever loaded once."""
    if stem not in _ctx:
        _ctx[stem] = kernel_on_new_context(stem)
    return _ctx[stem]


def load(spec, kernel=None):
    """One dispatchable stream: its instruction BO plus operand BOs sized from its own K and N."""
    spec, _, xstem = spec.partition("@")
    xstem = xstem or a.xclbin
    k = kernel if kernel is not None else context(xstem)
    stem, kres, n = spec.split(":")
    kres, n = int(kres), int(n)
    instr = np.fromfile(f"{a.root}/{WA}/{stem}.txt", dtype=np.uint32)
    bo_i = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(TO)
    bo_a = pyxrt.bo(d, M * kres * 2, pyxrt.bo.host_only, k.group_id(3))
    bo_b = pyxrt.bo(d, kres * n * 2, pyxrt.bo.host_only, k.group_id(4))
    bo_c = pyxrt.bo(d, M * n * 2, pyxrt.bo.host_only, k.group_id(5))
    bo_t = pyxrt.bo(d, 1, pyxrt.bo.host_only, k.group_id(6))
    bo_r = pyxrt.bo(d, 4, pyxrt.bo.host_only, k.group_id(7))
    for bo in (bo_a, bo_b, bo_r):
        bo.write(bytearray(bo.size()), 0)
        bo.sync(TO)
    return dict(stem=stem, kres=kres, n=n, words=int(instr.size), instr=instr, xclbin=xstem, k=k,
                args=(3, bo_i, instr.size, bo_a, bo_b, bo_c, bo_t, bo_r))


def word_diff(s0, s1):
    x, y = s0["instr"], s1["instr"]
    n = min(x.size, y.size)
    return int((x[:n] != y[:n]).sum()) + abs(x.size - y.size)


def timed(streams, order, ballast):
    """Time one arm. The ballast index tracks POSITION, so both arms see an identical ballast
    sequence and its cost cancels in the ALT-GRP difference."""
    t0 = time.perf_counter()
    for j, i in enumerate(order):
        s = streams[i]
        s["k"](*s["args"]).wait()
        if ballast:
            b = ballast[j % len(ballast)]
            b["k"](*b["args"]).wait()
    return (time.perf_counter() - t0) * 1e6


half = a.total // 2
ALT = [i % 2 for i in range(2 * half)]
GRP = [0] * half + [1] * half
CH_ALT = sum(1 for x, y in zip(ALT, ALT[1:]) if x != y)
CH_GRP = sum(1 for x, y in zip(GRP, GRP[1:]) if x != y)
DCH = CH_ALT - CH_GRP

print(f"xclbin        final_{a.xclbin}")
print(f"dispatches/arm {len(ALT)}   changes ALT {CH_ALT} vs GRP {CH_GRP}   (delta {DCH})")
print(f"reps {a.reps}\n")

if a.extra_contexts:
    bx = xclbin_of(a.xclbin)
    for _ in range(a.extra_contexts):
        _ballast.append(pyxrt.hw_context(d, bx.get_uuid()))
    print(f"holding {len(_ballast)} extra live hw_context(s), never dispatched into\n")

BALLAST_SPEC = a.ballast or a.pair[0].split("=", 1)[1].split(",")[0]
n_ilv = a.interleave_contexts or a.interleave_bos
interleaved = []
for _ in range(n_ilv):
    k = kernel_on_new_context(a.xclbin) if a.interleave_contexts else context(a.xclbin)
    interleaved.append(load(BALLAST_SPEC, kernel=k))
if interleaved:
    axis = "own hw_context" if a.interleave_contexts else "the measured context"
    print(f"interleaving {len(interleaved)} ballast dispatch(es) of {BALLAST_SPEC.split(':')[0]}"
          f" on {axis}, round-robin between measured dispatches\n")

results = []
for spec in a.pair:
    name, rhs = spec.split("=", 1)
    sa, sb = rhs.split(",")
    S = [load(sa), load(sb)]
    wd = word_diff(S[0], S[1])
    print(f"### {name}")
    for s in S:
        print(f"    {s['stem']}  K={s['kres']} N={s['n']}  {s['words']} words  ctx={s['xclbin'][:28]}")
    print(f"    differing words: {wd} ({100 * wd / S[0]['words']:.1f}%)")

    for s in S + interleaved:
        for _ in range(a.warmup):
            s["k"](*s["args"]).wait()

    deltas, alts, grps = [], [], []
    for r in range(a.reps):
        # Alternate which arm leads so any within-rep warming loads on both arms equally.
        if r % 2 == 0:
            ta = timed(S, ALT, interleaved)
            tg = timed(S, GRP, interleaved)
        else:
            tg = timed(S, GRP, interleaved)
            ta = timed(S, ALT, interleaved)
        alts.append(ta)
        grps.append(tg)
        deltas.append((ta - tg) / DCH)

    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    se = sd / len(deltas) ** 0.5
    ci = 1.96 * se
    pos = sum(1 for x in deltas if x > 0)
    print(f"    ALT  {statistics.mean(alts) / 1e3:9.2f} ms/arm     GRP {statistics.mean(grps) / 1e3:9.2f} ms/arm")
    print(f"    => restream cost {mean:8.2f} us/change   95% CI [{mean - ci:.2f}, {mean + ci:.2f}]"
          f"   {pos}/{len(deltas)} reps positive\n")
    results.append(dict(pair=name, streams=[s["stem"] for s in S], xclbins=[s["xclbin"] for s in S],
                        kres=[s["kres"] for s in S],
                        n=[s["n"] for s in S], word_diff=wd, words=S[0]["words"],
                        mean_us=mean, ci95_us=ci, sd_us=sd, reps=a.reps, pos=pos,
                        alt_ms=statistics.mean(alts) / 1e3, grp_ms=statistics.mean(grps) / 1e3,
                        deltas_us=deltas))

print(f"{'pair':28s} {'diff words':>10s} {'us/change':>12s} {'95% CI':>22s}")
for r in results:
    lo, hi = r["mean_us"] - r["ci95_us"], r["mean_us"] + r["ci95_us"]
    print(f"{r['pair']:28s} {r['word_diff']:10d} {r['mean_us']:12.2f}   [{lo:8.2f}, {hi:8.2f}]")

if a.out:
    with open(a.out, "w") as f:
        json.dump(dict(xclbin=a.xclbin, total=a.total, reps=a.reps, dch=DCH,
                       extra_contexts=a.extra_contexts,
                       interleave_contexts=a.interleave_contexts, interleave_bos=a.interleave_bos,
                       ballast=BALLAST_SPEC if interleaved else None,
                       results=results), f, indent=1)
    print(f"\nwrote {a.out}")
