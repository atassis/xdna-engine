#!/usr/bin/env python3
"""DDR bytes a fused dispatch reads more than once, resolved to arena addresses.

    python3 scripts/ddr_reread_census.py <fused.mlir> [meta.json]

decode_ddr_bytes.py counts every shim BD and flags only a stride-0 OUTER repeat. A range delivered
by several separate BDs, or by several runs of one design, is the same DDR read paid again and has
no repeat field to flag. This walks the top-level runtime sequence(s) in order, maps each BD
through its run's memref.view to an absolute offset, and counts a read byte as re-read when it was
already read with no write to it since. Totals must equal decode_ddr_bytes.py's; that is the check.

A prefill GEMM's column taps deliver ~1e5 disjoint runs per BD. Checking each one individually
against the growing "already read" set costs a Python list-splice per run -- O(size of the set) --
so a BD with N runs costs O(N * set_size). Per BD instead of per run: collapse the BD's own runs
into a merged, non-overlapping [lo,hi) union once (position-independent, cached across its repeat
invocations), and intersect that union against the set in one batched sweep. Re-reads AMONG a BD's
own duplicate/overlapping runs (e.g. a stride-0 broadcast tap) don't disappear in the union -- they
are exactly `raw bytes folded into a segment - that segment's own span` and are counted separately,
so collapsing changes representation, not the count.
"""
import bisect, collections, json, re, sys

ELEM = {"bf16": 2, "f32": 4, "i8": 1, "i32": 4}
DEV = re.compile(r"aie\.device\(\w+\)\s*@(\w+)\s*\{")
# The producer operand carries an optional `dimensionsToStream [...]` clause (an L2->L3 output
# stream's shim-side reshape) before the comma. Accepting only a bare `%\w+,` silently misread
# every such objectfifo as direction "unknown" and dropped its bytes -- 208 of 1251 objectfifos
# in a ring prefill dispatch, all on the GEMM C (output) path.
FIFO = re.compile(r"aie\.objectfifo @(\w+)\((%\w+)(?:\s+dimensionsToStream\s*\[[^\]]*\])?,\s*\{([^}]*)\}")
SEQ_ARGS = re.compile(r"(%arg\d+):\s*memref<((?:\d+x)+)(bf16|f32|i8|i32)>")
TASK = re.compile(r"aiex\.dma_configure_task_for @(\w+)")
BD = re.compile(r"aie\.dma_bd\((%arg\d+)\s*:\s*memref<[^>]*>\s*offset\s*=\s*(\d+)\s+len\s*=\s*(\d+)"
                r"\s+sizes\s*=\s*\[([^\]]*)\]\s+strides\s*=\s*\[([^\]]*)\]")


def body_of(src, start):
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start:j]
    raise ValueError("unbalanced braces")


def bd_raw_runs(off, ln, sizes, strides, esz):
    """Element-offset BD -> raw (byte_start, byte_len) runs, generation order, unmerged."""
    if len(sizes) != 4:
        return [(off * esz, ln * esz)]
    runs = []
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            for k in range(sizes[2]):
                base = off + i * strides[0] + j * strides[1] + k * strides[2]
                if strides[3] == 1:
                    runs.append((base * esz, sizes[3] * esz))
                else:
                    runs.extend(((base + l * strides[3]) * esz, esz) for l in range(sizes[3]))
    return runs


def bd_union(off, ln, sizes, strides, esz):
    """Raw runs -> sorted merged (start, end, raw_bytes_folded_in) segments.

    raw_bytes_folded_in - (end - start) is bytes re-read AMONG this BD's own runs (e.g. a
    stride-0 broadcast tap reading the same span N times): real re-reads, not a merge artifact.
    """
    runs = sorted(bd_raw_runs(off, ln, sizes, strides, esz))
    merged = []
    for s, n in runs:
        e = s + n
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
            merged[-1][2] += n
        else:
            merged.append([s, e, n])
    return merged


class Intervals:
    """Disjoint half-open byte intervals, queried/updated a whole BD's union at a time.

    `ranges` in every method is a sorted, pairwise-disjoint list of (start, end). Each call does
    ONE bisect span lookup and ONE list-splice over that span, instead of one pair per range --
    the splice is what is O(len(self.s)) per call, so paying it once per BD instead of once per
    run is the whole fix.
    """

    def __init__(self):
        self.s, self.e = [], []

    def _span(self, a, b):
        i = bisect.bisect_right(self.e, a)
        j = bisect.bisect_left(self.s, b)
        return i, j

    def overlap(self, ranges, on_overlap=None):
        """Total bytes of `ranges` already covered. on_overlap(lo, hi) fires per matched piece."""
        if not ranges:
            return 0
        i, j = self._span(ranges[0][0], ranges[-1][1])
        total = 0
        p, q, nr = i, 0, len(ranges)
        while p < j and q < nr:
            es, ee = self.s[p], self.e[p]
            a, b = ranges[q]
            lo, hi = (es if es > a else a), (ee if ee < b else b)
            if lo < hi:
                total += hi - lo
                if on_overlap is not None:
                    on_overlap(lo, hi)
            if ee < b:
                p += 1
            elif b < ee:
                q += 1
            else:
                p += 1
                q += 1
        return total

    def add(self, ranges):
        if not ranges:
            return
        i, j = self._span(ranges[0][0] - 1, ranges[-1][1] + 1)
        merged_in, p, q, nr = [], i, 0, len(ranges)
        while p < j and q < nr:
            if self.s[p] <= ranges[q][0]:
                merged_in.append((self.s[p], self.e[p])); p += 1
            else:
                merged_in.append(ranges[q]); q += 1
        merged_in.extend((self.s[k], self.e[k]) for k in range(p, j))
        merged_in.extend(ranges[q:])
        out_s, out_e = [], []
        for a, b in merged_in:
            if out_s and a <= out_e[-1]:
                if b > out_e[-1]:
                    out_e[-1] = b
            else:
                out_s.append(a); out_e.append(b)
        self.s[i:j], self.e[i:j] = out_s, out_e

    def remove(self, ranges):
        if not ranges:
            return
        i, j = self._span(ranges[0][0], ranges[-1][1])
        keep_s, keep_e = [], []
        ri, nr = 0, len(ranges)
        for p in range(i, j):
            cur, ee = self.s[p], self.e[p]
            while ri < nr and ranges[ri][1] <= cur:
                ri += 1
            rj = ri
            while rj < nr and ranges[rj][0] < ee:
                a, b = ranges[rj]
                if cur < a:
                    keep_s.append(cur); keep_e.append(a)
                cur = max(cur, b)
                if b <= ee:
                    rj += 1
                else:
                    break
            if cur < ee:
                keep_s.append(cur); keep_e.append(ee)
            ri = rj
        self.s[i:j], self.e[i:j] = keep_s, keep_e


src = open(sys.argv[1]).read()
layout = []
if len(sys.argv) > 2:
    lay = json.load(open(sys.argv[2]))["layout"]
    layout = sorted((v["offset"], v["offset"] + v["len"], k) for k, v in lay.items() if v["type"] == "scratch")
starts = [a for a, _, _ in layout]


def owner(addr):
    i = bisect.bisect_right(starts, addr) - 1
    if i >= 0 and addr < layout[i][1]:
        return re.sub(r"^L\d+_", "L*_", layout[i][2])
    return "-"


# Per design: its BDs in textual order as (direction, arg index, union, raw_total, internal_dup).
# union/raw_total/internal_dup are position-independent (computed once here, not per invocation);
# only the +base shift at invocation time depends on which run this is.
designs = {}
for m in DEV.finditer(src):
    body = body_of(src, m.end() - 1)
    fifo_dir = {}
    for f in FIFO.finditer(body):
        prod, cons = f.group(2), f.group(3)
        fifo_dir[f.group(1)] = ("read" if prod.startswith("%logical_shim") else
                                "write" if "%logical_shim" in cons else "internal")
    seq = body[body.find("aie.runtime_sequence("):]
    head = seq.split("\n", 1)[0]
    esz = {a: ELEM[t] for a, _, t in SEQ_ARGS.findall(head)}
    bds, task = [], None
    for line in seq.splitlines():
        t = TASK.search(line)
        if t:
            task = t.group(1)
            continue
        b = BD.search(line)
        if b:
            arg = b.group(1)
            sizes = [int(x) for x in b.group(4).split(",")]
            strides = [int(x) for x in b.group(5).split(",")]
            union = bd_union(int(b.group(2)), int(b.group(3)), sizes, strides, esz[arg])
            raw_total = sum(n for _, _, n in union)
            internal_dup = raw_total - sum(e - s for s, e, _ in union)
            bds.append((fifo_dir.get(task, "unknown"), int(arg[4:]), union, raw_total, internal_dup))
    designs[m.group(1)] = bds

top = src[src.rindex("aie.device(npu2) {"):]
spaces = collections.defaultdict(Intervals)
stat = collections.defaultdict(lambda: [0, 0, 0, 0])      # design -> runs, read, write, re-read
by_owner = collections.Counter()
consts, views, cur = {}, {}, None
for line in top.splitlines():
    s = line.strip()
    c = re.match(r"aiex\.configure @(\w+)", s)
    if c:
        cur = c.group(1)
        continue
    k = re.match(r"(%\w+) = arith\.constant (\d+) : index", s)
    if k:
        consts[k.group(1)] = int(k.group(2))
        continue
    v = re.match(r"(%\w+) = memref\.view (%arg\d+)\[(%\w+)\]", s)
    if v:
        views[v.group(1)] = (v.group(2), consts[v.group(3)])
        continue
    r = re.match(r"aiex\.run @\w+\(([^)]*)\)", s)
    if r and cur:
        ops = [views.get(o.strip(), (o.strip(), 0)) for o in r.group(1).split(",")]
        st = stat[cur]
        st[0] += 1
        for want in ("read", "write"):
            for direction, argi, union, raw_total, internal_dup in designs[cur]:
                if direction != want:
                    continue
                space, base = ops[argi]
                shifted = [(s0 + base, e0 + base) for s0, e0, _ in union]
                if want == "read":
                    def cb(lo, hi, cur=cur, space=space):
                        by_owner[(cur, owner(lo) if space == "%arg2" else space)] += hi - lo
                    ext = spaces[space].overlap(shifted, cb)
                    if internal_dup:
                        for (s0, e0, n0), (s1, _) in zip(union, shifted):
                            seg_dup = n0 - (e0 - s0)
                            if seg_dup:
                                by_owner[(cur, owner(s1) if space == "%arg2" else space)] += seg_dup
                    st[1] += raw_total
                    st[3] += internal_dup + ext
                    spaces[space].add(shifted)
                else:
                    st[2] += raw_total
                    spaces[space].remove(shifted)

tot = [sum(x[i] for x in stat.values()) for i in range(4)]
print(f"{'design':26} {'runs':>5} {'read MB':>10} {'write MB':>9} {'re-read MB':>11}")
for d, (n, rd, wr, rr) in sorted(stat.items(), key=lambda kv: -kv[1][3]):
    print(f"{d:26} {n:5} {rd/1e6:10.2f} {wr/1e6:9.2f} {rr/1e6:11.2f}")
print(f"\nread {tot[1]/1e6:.2f} MB + write {tot[2]/1e6:.2f} MB = {(tot[1]+tot[2])/1e6:.2f} MB "
      f"(decode_ddr_bytes.py's total)")
print(f"re-read {tot[3]/1e6:.2f} MB = {100*tot[3]/(tot[1]+tot[2]):.2f}% of all DDR bytes, "
      f"{100*tot[3]/tot[1]:.2f}% of reads")
print("\nlargest re-read sources (design, buffer):")
for (d, o), n in by_owner.most_common(8):
    print(f"  {d:26} {o:24} {n/1e6:10.2f} MB")
