#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Byte-identity gate between two `verify_llm_decode.py --dump-logits` runs.

    python3 scripts/compare_logits_dumps.py control.npz refactor.npz [--self-test]

For a graph refactor that claims unchanged arithmetic, only bit identity is a gate: argmax parity
passes a reduction-order change that moves every logit. Exit 0 identical, 1 different, 2 not
comparable (different spec, depth, window or token history).
"""
import sys

import numpy as np


def compare(a, b):
    for key in ("spec", "layers", "max_seq"):
        if a[key].item() != b[key].item():
            return 2, f"not comparable: {key} {a[key].item()} vs {b[key].item()}"
    ta, tb = a["tokens_fed"], b["tokens_fed"]
    n = min(len(ta), len(tb))
    if n == 0 or not np.array_equal(ta[:n], tb[:n]):
        first = int(np.argmax(ta[:n] != tb[:n])) if n else 0
        return 2, f"not comparable: token history differs at position {first}"
    la, lb = a["logits"][:n], b["logits"][:n]
    # Compare the bit patterns: -0.0 == 0.0 and NaN != NaN under float equality.
    diff = la.view(np.uint32) != lb.view(np.uint32)
    if not diff.any():
        return 0, f"IDENTICAL: {n} positions x {la.shape[1]} logits, bit for bit"
    rows = np.flatnonzero(diff.any(axis=1))
    p = int(rows[0])
    absd = np.abs(la.astype(np.float64) - lb.astype(np.float64))
    argmax_same = int((la.argmax(axis=1) == lb.argmax(axis=1)).sum())
    return 1, (f"DIFFERENT: first at position {p}, {int(diff[p].sum())} of {la.shape[1]} logits there; "
               f"{len(rows)} of {n} positions differ; max |diff| {absd.max():.6g}; "
               f"argmax agrees at {argmax_same}/{n} positions")


def self_test():
    rng = np.random.default_rng(0)
    base = dict(spec="t", layers=6, max_seq=6912, head="graph",
                tokens_fed=np.arange(4), logits=rng.standard_normal((4, 64)).astype(np.float32))
    def run(**over):
        d = {k: np.asarray(v) for k, v in {**base, **over}.items()}
        return compare(base_np, d)[0]
    base_np = {k: np.asarray(v) for k, v in base.items()}
    one = base["logits"].copy(); one[2, 7] = np.nextafter(one[2, 7], np.float32(np.inf))
    zero = base["logits"].copy(); zero[1, 3] = 0.0
    negz = zero.copy(); negz[1, 3] = -0.0
    checks = [
        ("identical", run(), 0),
        ("one ulp at one logit", run(logits=one), 1),
        ("different token history", run(tokens_fed=np.array([0, 1, 9, 3])), 2),
        ("different depth", run(layers=12), 2),
        ("-0.0 vs 0.0", compare({**base_np, "logits": zero}, {**base_np, "logits": negz})[0], 1),
    ]
    bad = [(name, got, want) for name, got, want in checks if got != want]
    for name, got, want in checks:
        print(f"  {'ok  ' if got == want else 'FAIL'} {name}: exit {got} (want {want})")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    code, msg = compare(np.load(sys.argv[1]), np.load(sys.argv[2]))
    print(msg)
    raise SystemExit(code)
