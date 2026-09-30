#!/usr/bin/env python3
"""Merge one build's aiecc --profile-trace files, kcc.jsonl kernel-compile log, and an
optional generator span/cProfile dump into one Perfetto-loadable trace.json plus a
tree.txt (inclusive ms / % of parent / % of total / count, aggregating repeated same-name
siblings -- the thing that shows "same pass on N identical devices" instead of N lines).

aiecc's `ts` (ProfileTrace.h) is already absolute wall-clock epoch microseconds, so no
per-file offset math is needed to line two aiecc invocations up on one axis; `otherData
.processStartEpochUs` is carried through as a display aid (when a build's trace predates
that field, this falls back to the earliest event's own ts).

    python3 scripts/buildtrace_merge.py <dir> -o trace.json --tree tree.txt
"""
import argparse
import glob
import json
import os
import statistics
import sys

PID_AIECC_BASE = 0
PID_KCC = 900
PID_GEN = 901


def load_aiecc_traces(d):
    """One (source_label, events) pair per *.trace.json in `d`, pid-namespaced apart."""
    out = []
    for i, path in enumerate(sorted(glob.glob(os.path.join(d, "*.trace.json")))):
        with open(path) as f:
            doc = json.load(f)
        events = doc.get("traceEvents", [])
        start = doc.get("otherData", {}).get("processStartEpochUs")
        if start is None and events:
            start = min(e["ts"] for e in events)
        pid = PID_AIECC_BASE + i
        for e in events:
            e["pid"] = pid
        label = os.path.basename(path)
        out.append((label, pid, start, events))
    return out


def load_kcc(d):
    """kcc.jsonl -> one X event per compile, epoch-seconds start/end -> us. Overlapping
    compiles (parallel -j builds) get lanes by greedy earliest-free-lane assignment, so
    concurrent compiles land on separate tid rows instead of stacking on tid 0."""
    path = os.path.join(d, "kcc.jsonl")
    if not os.path.exists(path):
        return []
    lane_free_at = []
    events = []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["start"])
    for r in rows:
        ts = r["start"] * 1e6
        dur = (r["end"] - r["start"]) * 1e6
        lane = next((i for i, free in enumerate(lane_free_at) if free <= ts), None)
        if lane is None:
            lane = len(lane_free_at)
            lane_free_at.append(0)
        lane_free_at[lane] = ts + dur
        name = os.path.basename(r["source"])
        events.append({"name": name, "ph": "X", "pid": PID_KCC, "tid": lane,
                       "ts": ts, "dur": dur,
                       "args": {"rc": r.get("rc"), "compiler": r.get("compiler")}})
    return events


def load_generator(d):
    """generator.json: {"start": epoch_s, "end": epoch_s[, "name"]} -> one X event. If
    generator.prof (cProfile -o output) also exists, its call tree is folded into tree.txt
    only (see prof_call_tree) -- pstats has no wall-clock timestamps to place spans by."""
    path = os.path.join(d, "generator.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        g = json.load(f)
    ts = g["start"] * 1e6
    dur = (g["end"] - g["start"]) * 1e6
    return [{"name": g.get("name", "generator"), "ph": "X", "pid": PID_GEN, "tid": 0,
             "ts": ts, "dur": dur, "args": {}}]


def prof_call_tree(path, depth=5):
    """generator.prof (cProfile) -> nested dict by cumulative time, depth-limited. Node:
    {name, ms (cumulative), count, children}. Best-effort: pstats' caller graph is not a
    single tree (a function can have multiple callers), so each function is attached under
    its highest-cumulative-time caller only."""
    import pstats
    stats = pstats.Stats(path)
    funcs = stats.stats  # func -> (cc, nc, tt, ct, callers)

    def label(func):
        fn, ln, name = func
        return f"{os.path.basename(fn)}:{ln}({name})"

    roots = [f for f, v in funcs.items() if not v[4]]
    if not roots:
        roots = sorted(funcs, key=lambda f: funcs[f][3], reverse=True)[:1]

    def build(func, d_left, seen):
        cc, nc, tt, ct, callers = funcs[func]
        node = {"name": label(func), "ms": ct * 1000.0, "count": nc, "children": []}
        if d_left <= 0 or func in seen:
            return node
        callees = [f for f, v in funcs.items() if func in v[4]]
        callees.sort(key=lambda f: funcs[f][3], reverse=True)
        for c in callees:
            node["children"].append(build(c, d_left - 1, seen | {func}))
        return node

    return [build(r, depth, frozenset()) for r in sorted(
        roots, key=lambda f: funcs[f][3], reverse=True)]


def build_forest(events):
    """One level: edge/task spans (no args.pass -- aiecc scheduler tasks, incl. kcc/generator
    rows) are flat siblings, never nested into one another -- two edges/items legitimately
    overlap under -j>1 parallelism, which is concurrency, not containment. A pass span (has
    args.pass) is attached to the tightest-fitting edge whose [ts, ts+dur) contains it (its
    own tid differs from that edge's by design -- see ProfileTrace.h -- so containment has to
    be time-based, not a per-tid stack); an edge with no attributable passes is a leaf."""
    edges = [e for e in events if "pass" not in e.get("args", {})]
    passes = [e for e in events if "pass" in e.get("args", {})]
    roots = [{"event": e, "children": []} for e in edges]
    for p in passes:
        candidates = [n for n in roots
                     if n["event"]["ts"] <= p["ts"]
                     and p["ts"] + p["dur"] <= n["event"]["ts"] + n["event"]["dur"]]
        if not candidates:
            continue
        best = min(candidates, key=lambda n: n["event"]["dur"])
        best["children"].append({"event": p, "children": []})
    roots.sort(key=lambda n: n["event"]["ts"])
    return roots


def group_and_render(nodes, out, indent, parent_ms, total_ms):
    """Render `nodes` (build_forest's children at one level), collapsing siblings that
    share a name into one aggregated line (count + min/median/max), per node's own inclusive
    ms as the sum over its instances. Single-count nodes recurse; aggregated ones do not
    (their instances' sub-trees may differ in name/structure across N copies)."""
    groups = {}
    order = []
    for n in nodes:
        name = n["event"]["name"]
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(n)
    for name in order:
        instances = groups[name]
        durs = [n["event"]["dur"] / 1000.0 for n in instances]
        total_dur = sum(durs)
        pct_parent = 100.0 * total_dur / parent_ms if parent_ms else 0.0
        pct_total = 100.0 * total_dur / total_ms if total_ms else 0.0
        if len(instances) == 1:
            out.append(f"{indent}{name}  {total_dur:.2f}ms  {pct_parent:.1f}%parent "
                       f"{pct_total:.1f}%total")
            group_and_render(instances[0]["children"], out, indent + "  ",
                             total_dur, total_ms)
        else:
            mn, mx = min(durs), max(durs)
            med = statistics.median(durs)
            out.append(f"{indent}{name}  x{len(instances)}  sum={total_dur:.2f}ms "
                       f"min={mn:.2f} median={med:.2f} max={mx:.2f}  "
                       f"{pct_parent:.1f}%parent {pct_total:.1f}%total")


def render_prof_tree(nodes, out, indent, parent_ms, total_ms):
    for n in nodes:
        pct_parent = 100.0 * n["ms"] / parent_ms if parent_ms else 0.0
        pct_total = 100.0 * n["ms"] / total_ms if total_ms else 0.0
        out.append(f"{indent}{n['name']}  {n['ms']:.2f}ms  x{n['count']}  "
                   f"{pct_parent:.1f}%parent {pct_total:.1f}%total")
        render_prof_tree(n["children"], out, indent + "  ", n["ms"], total_ms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--prof-depth", type=int, default=5)
    args = ap.parse_args()

    aiecc = load_aiecc_traces(args.dir)
    kcc_events = load_kcc(args.dir)
    gen_events = load_generator(args.dir)
    prof_path = os.path.join(args.dir, "generator.prof")
    prof_tree = prof_call_tree(prof_path, args.prof_depth) if os.path.exists(prof_path) else None

    all_events = list(kcc_events) + list(gen_events)
    for _, _, _, events in aiecc:
        all_events.extend(events)

    proc_names = {PID_KCC: "kernel-compiles", PID_GEN: "generator"}
    for label, pid, _, _ in aiecc:
        proc_names[pid] = f"aiecc:{label}"
    meta = [{"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": n}}
           for pid, n in proc_names.items()]

    with open(args.out, "w") as f:
        json.dump({"traceEvents": all_events + meta, "displayTimeUnit": "ms"}, f)

    starts = [e["ts"] for e in all_events]
    ends = [e["ts"] + e["dur"] for e in all_events]
    total_ms = ((max(ends) - min(starts)) / 1000.0) if all_events else 0.0
    if prof_tree:
        total_ms = max(total_ms, sum(n["ms"] for n in prof_tree))

    lines = [f"build  {total_ms:.2f}ms  100.0%total"]
    if gen_events:
        g = gen_events[0]
        gms = g["dur"] / 1000.0
        lines.append(f"  generator  {gms:.2f}ms  {100.0*gms/total_ms:.1f}%total")
        if prof_tree:
            render_prof_tree(prof_tree, lines, "    ", gms, total_ms)
    if kcc_events:
        kms = sum(e["dur"] for e in kcc_events) / 1000.0
        lines.append(f"  kernel-compiles  {kms:.2f}ms (sum)  "
                     f"{100.0*kms/total_ms:.1f}%total  count={len(kcc_events)}")
        group_and_render(build_forest(kcc_events), lines, "    ", kms, total_ms)
    for label, pid, _, events in aiecc:
        if not events:
            continue
        ams = (max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)) / 1000.0
        lines.append(f"  aiecc:{label}  {ams:.2f}ms  {100.0*ams/total_ms:.1f}%total")
        group_and_render(build_forest(events), lines, "    ", ams, total_ms)

    with open(args.tree, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
