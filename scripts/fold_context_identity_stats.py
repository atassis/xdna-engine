#!/usr/bin/env python3
"""Summarise the three fold-context-identity arms: contexts, wall clock, restream population.

The load-bearing column is hw_contexts. It is deterministic per arm, so one rep settles whether an
arm's fold merged two hw_contexts or merely renamed a transition; the reps are there for the wall
clock, which is paired per rep across arms so box drift cancels.
"""
import math
import re
import statistics
import sys
from pathlib import Path

CTX = re.compile(r"hw_contexts:\s*(\d+)/")
ENC = re.compile(r"mean encode ([0-9.]+)s/clip")
ROW = re.compile(r"^\s{4}(\S+#\d+@0x[0-9a-f]+)\s+(\d+)\s+(.*)$")


def cells(rest):
    toks = rest.split("(")[0].split()
    out, i = [], 0
    while i < len(toks) and len(out) < 3:
        if toks[i] == "-":
            out.append(None); i += 1
        elif toks[i].startswith("x") and i + 1 < len(toks):
            out.append((int(toks[i][1:]), float(toks[i + 1]))); i += 2
        else:
            i += 1
    return out + [None] * (3 - len(out))


def parse(p):
    txt = p.read_text()
    rows = {}
    for line in txt.splitlines():
        m = ROW.match(line)
        if m:
            rows[m.group(1).split("@")[0]] = cells(m.group(3))
    return {
        "ctx": int(CTX.search(txt).group(1)) if CTX.search(txt) else None,
        "enc": float(ENC.search(txt).group(1)) if ENC.search(txt) else None,
        "rows": rows,
    }


def ci(xs):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    h = 2.2 * sd / math.sqrt(len(xs))
    return (m - h, m + h)


d = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/fold_ctx_identity")
arms = {}
for arm in ("base", "fold", "foldc"):
    fs = sorted(d.glob(f"{arm}_rep*.txt"), key=lambda p: int(re.search(r"rep(\d+)", p.name).group(1)))
    arms[arm] = [parse(f) for f in fs]

print("arm     reps  hw_contexts   mean encode s/clip        95% CI")
for arm, rs in arms.items():
    if not rs:
        continue
    ctxs = sorted({r["ctx"] for r in rs})
    encs = [r["enc"] for r in rs if r["enc"] is not None]
    lo, hi = ci(encs)
    print(f"{arm:<7} {len(rs):>4}  {str(ctxs):<12}  {statistics.mean(encs):>10.4f}"
          f"          [{lo:.4f}, {hi:.4f}]")

# Paired wall clock against base, rep by rep -- the arms interleave inside a rep, so pairing on the
# rep is what cancels box drift rather than merely averaging it.
base = arms.get("base") or []
for arm in ("fold", "foldc"):
    rs = arms.get(arm) or []
    n = min(len(base), len(rs))
    if n < 2:
        continue
    ds = [rs[i]["enc"] - base[i]["enc"] for i in range(n)]
    lo, hi = ci(ds)
    pos = sum(1 for x in ds if x < 0)
    print(f"\npaired {arm} - base: {statistics.mean(ds)*1e3:+.1f} ms/clip  "
          f"CI [{lo*1e3:+.1f}, {hi*1e3:+.1f}]  faster in {pos}/{n} reps")

print("\nper-predecessor rows (rep 1 of each arm; mean ms)")
for arm, rs in arms.items():
    if not rs:
        continue
    print(f"  --- {arm} ---")
    for name, c in rs[0]["rows"].items():
        f = lambda x: "      -     " if x is None else f"x{x[0]:<4} {x[1]:>7.3f}"
        print(f"    {name:<62} same {f(c[0])}  restream {f(c[1])}  xclbin {f(c[2])}")
