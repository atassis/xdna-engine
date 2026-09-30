#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""openjev's NLI prompt, decision hypotheses and long-premise windowing, defined once.

`rust/npu-models/src/nli.rs` mirrors this file; `--golden` writes the cases its tests read, so a
drift between the two is a test failure rather than a quiet accuracy loss.

  python scripts/nli_common.py --golden tests/refs/nli/common_golden.json
"""
import argparse
import json

DECIDE_HYPOTHESIS = 'The answer to "{instr}" is {label}: {crit}'
RERANK_HYPOTHESIS = "The correct answer is: {text}"


def py_strip(s):
    return s.strip()


def prompt(template, premise, hypothesis):
    """`template.format(...)` with openjev's `_encode` strips; one substitution per field."""
    return template.format(premise=py_strip(premise), hypothesis=py_strip(hypothesis))


def decide_hypotheses(q):
    """(label, hypothesis) per option, in the order the NPU answers them (noul: yes first)."""
    instr = py_strip(q["instructions"])
    crit = q.get("criteria")
    if q["type"] in ("noul", "boolean"):
        crit = crit or {}
        opts = [("yes", crit.get("true", "Yes")), ("no", crit.get("false", "No"))]  # the adapter's .get, not `or`
    elif q["type"] == "choice":
        opts = [(k, v or k) for k, v in crit.items()]
    else:
        opts = [(str(i), lvl) for i, lvl in enumerate(crit)]
    return [(k, DECIDE_HYPOTHESIS.format(instr=instr, label=k, crit=c)) for k, c in opts]


def char_windows(premise, w):
    """openjev's windowing (24000 chars, 2000 overlap) at width `w`, overlap w // 12."""
    if len(premise) <= w:
        return [premise]
    ov = w // 12
    step = w - ov
    return [premise[s:s + w] for s in range(0, max(len(premise) - ov, 1), step)]


def fit_windows(template, premise, hyps, n_tokens, s):
    """The widest windows at which every (window, hypothesis) prompt fits `s` tokens."""
    w = max(len(premise), 1)
    while True:
        wins = char_windows(premise, w)
        if all(n_tokens(prompt(template, x, h)) <= s for x in wins for h in hyps):
            return wins
        if w <= 64:
            raise ValueError(f"a hypothesis alone does not fit {s} tokens")
        w = w * 9 // 10


def normalise_entailment(p_ent):
    """[window][option] P(entailment) -> per option max over windows, normalised over options."""
    best = [max(col) for col in zip(*p_ent)]
    z = max(sum(best), 1e-9)
    return [x / z for x in best]


def write_golden(path):
    T = "Premise: {premise}\nHypothesis: {hypothesis}"
    toks = lambda s: len(s) // 4 + 1  # the toy tokenizer both test suites use
    long_p = "Zürich — ünïcödé " * 40
    qs = [{"type": "noul", "instructions": " Is it allowed? ", "criteria": {"true": "All hold."}},
          {"type": "noul", "instructions": "Q", "criteria": None},
          {"type": "choice", "instructions": "Route?", "criteria": {"billing": "", "tech": "A fault."}},
          {"type": "score", "instructions": "Rate", "criteria": ["bad", "good", "great"]}]
    g = {
        "template": T,
        "prompt": [{"premise": p, "hypothesis": h, "out": prompt(T, p, h)}
                   for p, h in [("  a {hypothesis} b \x1c", " h "), ("x", "y"), ("\t\n", " ")]],
        "hypotheses": [{"q": q, "out": decide_hypotheses(q)} for q in qs],
        "windows": [{"premise": long_p, "w": w, "out": char_windows(long_p, w)} for w in (24, 100, 5000)],
        "fit": [{"premise": long_p, "hyps": ["short", "a much longer hypothesis text"], "s": s,
                 "out": fit_windows(T, long_p, ["short", "a much longer hypothesis text"], toks, s)}
                for s in (64, 300, 10_000)],
        "normalise": [{"p": p, "out": normalise_entailment(p)}
                      for p in ([[0.2, 0.6], [0.4, 0.1]], [[0.0, 0.0]], [[0.5, 0.25, 0.25]])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    write_golden(ap.parse_args().golden)
