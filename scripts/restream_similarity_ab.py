#!/usr/bin/env python3
"""Does the restream price track how DISSIMILAR the two streams are, or is it flat per change?

Written when two measurements of "the cost of changing the instruction stream inside one
hw_context" appeared to disagree by ~750x: an isolated probe measured < 3 us for a pair differing only
in BD size/stride, while the encoder appeared to charge +2.258 ms for fc1's panel-drain stream against
an identity stream. Two variables moved at once -- the streams' SIMILARITY, and isolated-vs-in-encoder
-- so this holds the second fixed (everything here is isolated) and sweeps the first.

THE DISAGREEMENT WAS AN ARTIFACT (resolved 2026-08-22). The encoder was not measuring a restream: two
byte-identical copies of one xclbin loaded from two directories became two hw_contexts, and the
dispatch log names a context by the xclbin file stem, so a full program transition was booked as a
same-xclbin restream. A same-context restream is free in the encoder too. This sweep's own answer
stands and is the one that generalises: the price does not track dissimilarity, it tracks the
hw_context boundary. `npu-xrt` now warns when one stem lands in two contexts.

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

FOOTPRINT AXIS (--footprint-bos). Dissimilarity, passive context residency, cross-context ACTIVITY
and live instruction-BO count are all refuted flat at zero, and operand IDENTITY is refuted by the
self pair (load() allocates fresh operands per stream, so `self` already alternated two distinct
operand sets). What no isolated instrument has ever scaled is HOW MUCH IS LIVE: this probe holds a
dozen small zero-filled BOs, while the encoder pins 1.25 GB across one resident multi-MB weight BO
per op and reads a different one every dispatch. Both surviving mechanisms -- eviction of the next
dispatch's working set, and address-translation state where each BO is a distinct DMA mapping --
scale with footprint and mapping count, not with identity. So --footprint-bos N adds N resident,
NON-ZERO, multi-MB ballast streams on the measured context, each actually dispatched against,
round-robin; sweeping N carries the isolated working set up to the encoder's.

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
ap.add_argument("--footprint-bos", type=int, default=0,
                help="N resident NON-ZERO ballast streams on the measured context, dispatched "
                     "round-robin -- carries live footprint and DMA-mapping count toward the "
                     "encoder's 1.25 GB / ~200 resident weight BOs")
ap.add_argument("--footprint-spec", default=None,
                help="spec for --footprint-bos ballast (default: the second stream of the first "
                     "pair, i.e. the larger operand set)")
ap.add_argument("--ballast", default=None,
                help="spec dispatched as ballast (default: the first pair's first spec)")
ap.add_argument("--out", default=None, help="write JSON here")
a = ap.parse_args()

_axes = [n for n, v in (("--interleave-contexts", a.interleave_contexts),
                        ("--interleave-bos", a.interleave_bos),
                        ("--footprint-bos", a.footprint_bos)) if v]
if len(_axes) > 1:
    ap.error(f"{' and '.join(_axes)} confound the resource; sweep one axis at a time")

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


def load(spec, kernel=None, nonzero=False):
    """One dispatchable stream: its instruction BO plus operand BOs sized from its own K and N.

    `nonzero` fills the operands with finite bf16 instead of zeros. Nothing reads C, so the content
    is numerically irrelevant -- it exists so a footprint ballast cannot be served by any zero-page
    or all-zero-DMA shortcut, which would make the pinned bytes nominal rather than real."""
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
        if nonzero and bo is not bo_r:
            # 0x3000..0x3fff is positive normal bf16 in ~[1.2e-4, 2.0): non-zero, finite, no
            # denormal/NaN slow path in the core. Deterministic, so a point is reproducible.
            rng = np.random.default_rng(bo.size())
            bo.write(rng.integers(0x3000, 0x4000, bo.size() // 2, dtype=np.uint16).tobytes(), 0)
        else:
            bo.write(bytearray(bo.size()), 0)
        bo.sync(TO)
    return dict(stem=stem, kres=kres, n=n, words=int(instr.size), instr=instr, xclbin=xstem, k=k,
                bytes=sum(b.size() for b in (bo_i, bo_a, bo_b, bo_c, bo_t, bo_r)), bos=6,
                args=(3, bo_i, instr.size, bo_a, bo_b, bo_c, bo_t, bo_r))


def word_diff(s0, s1):
    x, y = s0["instr"], s1["instr"]
    n = min(x.size, y.size)
    return int((x[:n] != y[:n]).sum()) + abs(x.size - y.size)


def timed(streams, order, ballast):
    """Time one arm, splitting MEASURED dispatches from ballast ones.

    The ballast index tracks POSITION, so both arms see an identical ballast sequence and its cost
    cancels in the ALT-GRP difference. The split exists because the differential is blind to a
    UNIFORM per-dispatch effect that hits both arms equally -- exactly what an eviction driven by
    total footprint would be. The measured-only absolute is what makes a point commensurable with
    the encoder's per-predecessor-class ms/dispatch."""
    tm = 0.0
    t0 = time.perf_counter()
    for j, i in enumerate(order):
        s = streams[i]
        tc = time.perf_counter()
        s["k"](*s["args"]).wait()
        tm += time.perf_counter() - tc
        if ballast:
            b = ballast[j % len(ballast)]
            b["k"](*b["args"]).wait()
    return (time.perf_counter() - t0) * 1e6, tm * 1e6


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

# The footprint axis defaults to the LARGER of the first pair's two streams, so each ballast pins
# the most bytes available without inventing a spec the encoder has no analogue for.
if a.footprint_bos:
    BALLAST_SPEC = a.footprint_spec or a.pair[0].split("=", 1)[1].split(",")[1]
else:
    BALLAST_SPEC = a.ballast or a.pair[0].split("=", 1)[1].split(",")[0]
n_ilv = a.interleave_contexts or a.interleave_bos or a.footprint_bos
interleaved = []
for _ in range(n_ilv):
    k = kernel_on_new_context(a.xclbin) if a.interleave_contexts else context(a.xclbin)
    interleaved.append(load(BALLAST_SPEC, kernel=k, nonzero=bool(a.footprint_bos)))
if interleaved:
    axis = ("own hw_context" if a.interleave_contexts else
            "the measured context, NON-ZERO" if a.footprint_bos else "the measured context")
    print(f"interleaving {len(interleaved)} ballast dispatch(es) of {BALLAST_SPEC.split(':')[0]}"
          f" on {axis}, round-robin between measured dispatches")
BALLAST_BYTES = sum(s["bytes"] for s in interleaved)
BALLAST_BOS = sum(s["bos"] for s in interleaved)
print(f"ballast pins {BALLAST_BYTES / 2**20:.0f} MiB over {BALLAST_BOS} BOs"
      f"   (encoder reference: 1.25 GB resident, one weight BO per op)\n")

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
    mdeltas, malts, mgrps = [], [], []
    for r in range(a.reps):
        # Alternate which arm leads so any within-rep warming loads on both arms equally.
        if r % 2 == 0:
            ta, ma = timed(S, ALT, interleaved)
            tg, mg = timed(S, GRP, interleaved)
        else:
            tg, mg = timed(S, GRP, interleaved)
            ta, ma = timed(S, ALT, interleaved)
        alts.append(ta)
        grps.append(tg)
        deltas.append((ta - tg) / DCH)
        malts.append(ma)
        mgrps.append(mg)
        mdeltas.append((ma - mg) / DCH)

    def stats(xs):
        m = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return m, sd, 1.96 * sd / len(xs) ** 0.5, sum(1 for x in xs if x > 0)

    mean, sd, ci, pos = stats(deltas)
    mmean, msd, mci, mpos = stats(mdeltas)
    # Absolute per MEASURED dispatch, ballast excluded: the number the encoder's per-class
    # ms/dispatch can be read against directly (same-context restream cell = 2.57-2.62 ms).
    alt_pd = statistics.mean(malts) / len(ALT)
    grp_pd = statistics.mean(mgrps) / len(GRP)
    print(f"    ALT  {statistics.mean(alts) / 1e3:9.2f} ms/arm     GRP {statistics.mean(grps) / 1e3:9.2f} ms/arm")
    print(f"    => restream cost {mean:8.2f} us/change   95% CI [{mean - ci:.2f}, {mean + ci:.2f}]"
          f"   {pos}/{len(deltas)} reps positive")
    print(f"    measured-only    {mmean:8.2f} us/change   95% CI [{mmean - mci:.2f}, {mmean + mci:.2f}]"
          f"   {mpos}/{len(mdeltas)} reps positive")
    print(f"    abs/dispatch (measured only)  ALT {alt_pd:8.1f} us   GRP {grp_pd:8.1f} us\n")
    results.append(dict(pair=name, streams=[s["stem"] for s in S], xclbins=[s["xclbin"] for s in S],
                        kres=[s["kres"] for s in S],
                        n=[s["n"] for s in S], word_diff=wd, words=S[0]["words"],
                        mean_us=mean, ci95_us=ci, sd_us=sd, reps=a.reps, pos=pos,
                        alt_ms=statistics.mean(alts) / 1e3, grp_ms=statistics.mean(grps) / 1e3,
                        meas_mean_us=mmean, meas_ci95_us=mci, meas_pos=mpos,
                        alt_us_per_disp=alt_pd, grp_us_per_disp=grp_pd,
                        ballast_bytes=BALLAST_BYTES, ballast_bos=BALLAST_BOS,
                        pinned_bytes=BALLAST_BYTES + sum(s["bytes"] for s in S),
                        deltas_us=deltas, meas_deltas_us=mdeltas))

print(f"{'pair':28s} {'pinned MiB':>10s} {'us/change':>12s} {'95% CI':>22s} {'abs us/disp ALT':>16s}")
for r in results:
    lo, hi = r["mean_us"] - r["ci95_us"], r["mean_us"] + r["ci95_us"]
    print(f"{r['pair']:28s} {r['pinned_bytes'] / 2**20:10.0f} {r['mean_us']:12.2f}"
          f"   [{lo:8.2f}, {hi:8.2f}] {r['alt_us_per_disp']:16.1f}")

if a.out:
    with open(a.out, "w") as f:
        json.dump(dict(xclbin=a.xclbin, total=a.total, reps=a.reps, dch=DCH,
                       extra_contexts=a.extra_contexts,
                       interleave_contexts=a.interleave_contexts, interleave_bos=a.interleave_bos,
                       footprint_bos=a.footprint_bos,
                       ballast=BALLAST_SPEC if interleaved else None,
                       results=results), f, indent=1)
    print(f"\nwrote {a.out}")
