#!/usr/bin/env python3
# What the decode runlist's ORDER costs, and which reorders are legal.
#
# `iron/common/compilation/sequence.py` emits one `aiex.ConfigureOp` per runlist entry, skipped only
# when the same design repeats CONSECUTIVELY -- so configures = the number of runs in the design-name
# sequence, and ADJACENCY, not design count, is what sets it. Merging every mergeable design saves
# nothing if no two are ever neighbours.
#
# Device-free and COMPILE-free: OperatorSequence.compile is replaced by its buffer-layout half, so
# `unique_designs()` (which owns the operator -> design collapse, including share_designs) is the
# real one while aiecc never runs. The design name reproduces iron/common/sequence.py::to_comp.
#
# Legality comes from AIERuntimeArgSpec.direction ('in'/'out'/'inout') plus the `buf[start:end]`
# slice notation, so a dependency is a byte-range overlap on the shared arena, not a name match.
#
#   python3 scripts/runlist_adjacency_census.py --spec gemma4-12b --weights <dir> --layers 48
#   python3 scripts/runlist_adjacency_census.py ... --json out.json
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "designs", "decode_fused"))


def _no_compile(gen):
    """Make OperatorSequence lay out buffers and stop, so build_graph runs without aiecc."""
    base = gen.OperatorSequence

    class Seq(base):
        def compile(self):
            self.subbuffer_layout, self.buffer_sizes, self.slice_info = (
                self.calculate_buffer_layout())

    gen.OperatorSequence = Seq
    return base


def design_sequence(seq):
    """(design names in runlist order, name -> operator) exactly as fuse_mlir will see them."""
    designs, design_of = seq.unique_designs()
    names = [f"op{i}_{op.__class__.__name__}" for i, op in enumerate(designs)]
    return ([names[design_of[id(op)]] for op, *_ in seq.runlist],
            dict(zip(names, designs)))


def runs(names):
    """[(design, length)] -- one entry per configure the emitter will issue."""
    out = []
    for n in names:
        if out and out[-1][0] == n:
            out[-1][1] += 1
        else:
            out.append([n, 1])
    return [tuple(r) for r in out]


def rw_sets(seq):
    """[(reads, writes)] per runlist entry, as {(base, start, end)} byte ranges on the arena.

    An unsliced buffer is recorded as (name, None, None), which overlaps every slice of itself --
    the conservative reading, and the correct one: an operator handed the whole buffer may touch
    any of it.
    """
    def rng(buf):
        if "[" in buf and buf.endswith("]"):
            base = buf[:buf.index("[")]
            lo, hi = buf[buf.index("[") + 1:-1].split(":")
            return (base, int(lo), int(hi))
        return (buf, None, None)

    ent = []
    for op, *bufs in seq.runlist:
        r, w = set(), set()
        for spec, buf in zip(op.get_arg_spec(), bufs):
            if spec.direction in ("in", "inout"):
                r.add(rng(buf))
            if spec.direction in ("out", "inout"):
                w.add(rng(buf))
        ent.append((r, w))
    return ent


def overlaps(a, b):
    """Do two (base, start, end) ranges touch the same bytes?"""
    if a[0] != b[0]:
        return False
    if a[1] is None or b[1] is None:
        return True
    return a[1] < b[2] and b[1] < a[2]


def conflict(i, j, ent):
    """True if entries i and j cannot be swapped: RAW, WAR or WAW on any shared byte range."""
    ri, wi = ent[i]
    rj, wj = ent[j]
    for x in wi:
        if any(overlaps(x, y) for y in rj | wj):
            return True
    for x in wj:
        if any(overlaps(x, y) for y in ri):
            return True
    return False


def bubble_gain(names, ent):
    """For every adjacent-in-time pair of same-design runs, whether they can be made adjacent.

    A run of design D at positions [a..b] and the NEXT run of D at [c..d] merge into one configure
    if every entry strictly between them commutes with one side or the other. Reported per design:
    how many of its run boundaries are removable this way, which is exactly the configures saved.
    """
    pos = collections.defaultdict(list)
    idx, rs = 0, runs(names)
    for name, n in rs:
        pos[name].append((idx, idx + n - 1))
        idx += n

    res = {}
    for name, spans in pos.items():
        movable = blocked = 0
        for (a, b), (c, d) in zip(spans, spans[1:]):
            between = range(b + 1, c)
            # Hoist the later run BACK over the gap, or push the earlier run FORWARD over it.
            back = all(not conflict(k, m, ent) for m in range(c, d + 1) for k in between)
            fwd = all(not conflict(k, m, ent) for m in range(a, b + 1) for k in between)
            movable += bool(back or fwd)
            blocked += not (back or fwd)
        res[name] = dict(runs=len(spans), boundaries=len(spans) - 1,
                         removable=movable, blocked=blocked)
    return res


def dep_edges(ent):
    """i -> j for every i < j that cannot cross: the DAG whose topological orders are the legal runlists."""
    succ = [[] for _ in ent]
    pred = [0] * len(ent)
    for j in range(len(ent)):
        for i in range(j):
            if conflict(i, j, ent):
                succ[i].append(j)
                pred[j] += 1
    return succ, pred


def greedy_order(names, ent):
    """Configures of a legal order that always stays on the current design when it can.

    List scheduling over the dependency DAG, preferring a ready entry of the design just emitted.
    An UPPER bound on what any reorder achieves -- paired with `forced_configures` below, which is
    a lower bound, it brackets the whole reorder search space without enumerating it.
    """
    succ, pred = dep_edges(ent)
    ready = {i for i, p in enumerate(pred) if p == 0}
    out, last = [], None
    while ready:
        same = [i for i in ready if names[i] == last]
        pick = min(same) if same else min(ready)
        ready.discard(pick)
        out.append(pick)
        last = names[pick]
        for j in succ[pick]:
            pred[j] -= 1
            if pred[j] == 0:
                ready.add(j)
    return len(runs([names[i] for i in out])), out


def forced_configures(names, ent):
    """1 + the most design changes on any dependency path -- a LOWER bound on configures.

    Every edge of the DAG fixes the relative order of its endpoints in EVERY legal runlist, and two
    ordered entries of different designs force at least one configure between them. Interleaving
    other entries can only add configures, never remove one, so the longest such path is a floor
    that no reorder can go under.
    """
    succ, _ = dep_edges(ent)
    # Edges only ever point forward, so descending index is already a reverse topological order.
    best = [0] * len(ent)
    for i in range(len(ent) - 1, -1, -1):
        for j in succ[i]:
            best[i] = max(best[i], best[j] + (names[i] != names[j]))
    return max(best) + 1 if best else 0


def from_trace(path):
    """Replay a --trace file as (names, entries) per segment, so analysis needs no rebuild."""
    tr = json.load(open(path))
    segs = collections.OrderedDict()
    for e in tr:
        segs.setdefault(e["seg"], []).append(e)
    out = []
    for si, rows in segs.items():
        names = [r["design"] for r in rows]
        ent = [({tuple(x) for x in r["reads"]}, {tuple(x) for x in r["writes"]}) for r in rows]
        out.append((si, names, ent))
    return out


def main(o):
    trace, report = [], {}
    if o.from_trace:
        segs_in = from_trace(o.from_trace)
        report = dict(source=o.from_trace, segments=len(segs_in))
    else:
        import gen_llm_decode as gen  # noqa: E402  -- after sys.path is set
        _no_compile(gen)
        sp, fused, weights, md = gen.build_graph(o.spec, o.weights, o.layers, o.max_seq)
        report = dict(spec=sp.name, layers=md["NL"], max_seq=md["S"],
                      segments=len(md["segments"]), lm_head_separate=md.get("head") is not None)
        segs_in = []
        for si, s in enumerate(md["segments"]):
            names, _ = design_sequence(s["seq"])
            ent = rw_sets(s["seq"])
            segs_in.append((si, names, ent))
            for n, (r, w), (_op, *bufs) in zip(names, ent, s["seq"].runlist):
                trace.append(dict(seg=si, design=n, bufs=list(bufs),
                                  reads=sorted(map(list, r)), writes=sorted(map(list, w))))

    tot_disp = tot_cfg = 0
    per_seg = []
    for si, names, ent in segs_in:
        rs = runs(names)
        tot_disp += len(names)
        tot_cfg += len(rs)
        hist = collections.Counter(names)
        adj = collections.Counter((a[0], b[0]) for a, b in zip(rs, rs[1:]))
        gain = bubble_gain(names, ent)
        row = dict(segment=si, dispatches=len(names), configures=len(rs), designs=len(hist),
                   per_design={n: dict(dispatches=hist[n], **gain[n]) for n in sorted(hist)},
                   boundaries={f"{a}>{b}": c for (a, b), c in adj.most_common()})
        if o.bound or o.check:
            best, _order = greedy_order(names, ent)
            row.update(greedy_configures=best, forced_configures=forced_configures(names, ent))
        per_seg.append(row)
    report.update(dispatches=tot_disp, configures=tot_cfg, segment_detail=per_seg)

    if trace and o.trace:
        # The whole graph as data, so every later reorder question is answered offline instead of
        # by re-running build_graph (minutes, and ~3 GB of packed weights, per question).
        with open(o.trace, "w") as f:
            json.dump(trace, f)
        print(f"[census] wrote {len(trace)}-entry trace to {o.trace}")

    print(f"[census] dispatches={tot_disp} configures={tot_cfg}"
          + "".join(f" seg{r['segment']}: greedy={r['greedy_configures']} "
                    f"floor={r['forced_configures']}" for r in per_seg if o.bound or o.check))
    if o.json:
        with open(o.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[census] wrote {o.json}")
    elif not (o.bound or o.check):
        print(json.dumps(report, indent=2))

    if o.check:
        slack = [(r["segment"], r["configures"], r["greedy_configures"])
                 for r in per_seg if r["greedy_configures"] < r["configures"]]
        for si, have, want in slack:
            print(f"[census] segment {si}: {have} configures, but a legal reorder reaches "
                  f"{want} -- {have - want} are paid for ORDER, not for dependencies.",
                  file=sys.stderr)
        return 1 if slack else 0
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--weights")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--from-trace", help="analyse a previously written --trace instead of rebuilding")
    ap.add_argument("--bound", action="store_true",
                    help="bracket every legal reorder: greedy upper bound + dependency-path floor")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any configure in the emitted order is paid for order alone")
    ap.add_argument("--json")
    ap.add_argument("--trace", help="write the whole runlist (design + read/write ranges) as JSON")
    sys.exit(main(ap.parse_args()))
