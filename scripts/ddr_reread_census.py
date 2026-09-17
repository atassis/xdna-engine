#!/usr/bin/env python3
"""DDR bytes a fused dispatch reads more than once, resolved to arena addresses.

    python3 scripts/ddr_reread_census.py <fused.mlir> [meta.json]

decode_ddr_bytes.py counts every shim BD and flags only a stride-0 OUTER repeat. A range delivered
by several separate BDs, or by several runs of one design, is the same DDR read paid again and has
no repeat field to flag. This walks the top-level runtime sequence(s) in order, maps each BD
through its run's memref.view to an absolute offset, and counts a read byte as re-read when it was
already read with no write to it since. Totals must equal decode_ddr_bytes.py's; that is the check.
Decode runs in about a second. A prefill GEMM's column taps expand to ~10^5 ranges per BD, which
this pure-Python walk does not finish in minutes.
"""
import bisect, collections, json, re, sys

ELEM = {"bf16": 2, "f32": 4, "i8": 1, "i32": 4}
DEV = re.compile(r"aie\.device\(\w+\)\s*@(\w+)\s*\{")
FIFO = re.compile(r"aie\.objectfifo @(\w+)\((%\w+),\s*\{([^}]*)\}")
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


def bd_ranges(off, ln, sizes, strides, esz):
    """Element-offset BD -> list of (byte_start, byte_len) runs it delivers."""
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
    runs.sort()
    merged = []
    for s, n in runs:
        if merged and merged[-1][0] + merged[-1][1] == s:
            merged[-1] = (merged[-1][0], merged[-1][1] + n)
        else:
            merged.append((s, n))
    return merged


class Intervals:
    """Disjoint half-open byte intervals: overlap, add, remove."""

    def __init__(self):
        self.s, self.e = [], []

    def _span(self, a, b):
        i = bisect.bisect_right(self.e, a)
        j = bisect.bisect_left(self.s, b)
        return i, j

    def overlap(self, a, b):
        i, j = self._span(a, b)
        return sum(min(b, self.e[k]) - max(a, self.s[k]) for k in range(i, j))

    def add(self, a, b):
        i, j = self._span(a - 1, b + 1)
        if i < j:
            a, b = min(a, self.s[i]), max(b, self.e[j - 1])
        self.s[i:j], self.e[i:j] = [a], [b]

    def remove(self, a, b):
        i, j = self._span(a, b)
        keep_s, keep_e = [], []
        for k in range(i, j):
            if self.s[k] < a:
                keep_s.append(self.s[k]); keep_e.append(a)
            if self.e[k] > b:
                keep_s.append(b); keep_e.append(self.e[k])
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


# Per design: its BDs in textual order as (direction, arg index, [(byte_start, byte_len)]).
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
            bds.append((fifo_dir.get(task, "unknown"), int(arg[4:]),
                        bd_ranges(int(b.group(2)), int(b.group(3)), sizes, strides, esz[arg])))
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
            for direction, argi, ranges in designs[cur]:
                if direction != want:
                    continue
                space, base = ops[argi]
                for off, n in ranges:
                    a, b = base + off, base + off + n
                    if want == "read":
                        again = spaces[space].overlap(a, b)
                        st[1] += n
                        st[3] += again
                        if again:
                            by_owner[(cur, owner(a) if space == "%arg2" else space)] += again
                        spaces[space].add(a, b)
                    else:
                        st[2] += n
                        spaces[space].remove(a, b)

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
