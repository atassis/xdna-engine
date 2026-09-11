"""Paired arm-vs-arm comparison from saved per-position NLL.

The sweep reports each arm against the bf16 control. Comparing two ARMS by differencing their
two control-relative percentages is not a test -- it throws away the pairing that makes the
comparison powerful. These arms share a corpus and share positions, so the right statistic is a
paired t on the per-position NLL difference between the two arms directly.
"""
import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/xdna/qlab")
import sys, glob, os, numpy as np


def paired(a, b):
    d = a - b
    n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    return dict(n=n, ratio=float(np.exp(mu)), pct=float((np.exp(mu) - 1) * 100),
                ci=[float((np.exp(lo) - 1) * 100), float((np.exp(hi) - 1) * 100)],
                t=float(mu / se) if se > 0 else float("nan"),
                frac_worse=float((d > 0).mean()))


def load(tag, arm):
    return np.load(f"{QLAB}/runs/{tag}--{arm}.nll.npy")


if __name__ == "__main__":
    tag = sys.argv[1]
    pairs = [tuple(p.split(":")) for p in sys.argv[2:]]
    print(f"{'A vs B (A worse by)':46s} {'pct':>7s} {'95% CI':>16s} {'t':>7s} {'worse@':>7s}")
    for A, B in pairs:
        try:
            r = paired(load(tag, A), load(tag, B))
        except FileNotFoundError as e:
            print(f"{A} vs {B}: missing {os.path.basename(e.filename)}"); continue
        print(f"{A + ' vs ' + B:46s} {r['pct']:7.2f} [{r['ci'][0]:6.2f},{r['ci'][1]:6.2f}] "
              f"{r['t']:7.2f} {r['frac_worse']:7.3f}")
