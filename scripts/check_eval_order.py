#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Flag statements passing >=2 side-effecting builder calls as sibling arguments.

C++ leaves argument evaluation order unspecified, so each call's arith.constant lands in an order
that depends on the compiler that built aiecc. Usage: check_eval_order.py <dir>...; exit 1 on hits.
"""
import re
import sys
from pathlib import Path

CALLS = ("createConstantI32(", "createConstantI64(")
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\])*"')


def _blank(m):
    return re.sub(r"[^\n]", " ", m.group(0))


def _check(text, start, end, hits):
    stmt = text[start:end]
    if sum(stmt.count(c) for c in CALLS) >= 2:
        first = min(p for p in (stmt.find(c) for c in CALLS) if p >= 0)
        hits.append((text.count("\n", 0, start + first) + 1, stmt.strip()[:80]))


def find_hits(text):
    # Each `{` opens a fresh lexical scope (function/lambda/if/for body) with
    # its own paren-depth counter, so a call that takes a multi-statement
    # lambda argument (`module.walk([&](...) { ...; ...; })`) does not
    # swallow the lambda body into one statement: the body's own `;`/`{`/`}`
    # split normally, and the outer call resumes its depth on the match `}`.
    text = _STRING.sub(_blank, _COMMENT.sub(_blank, text))
    hits, stack, start = [], [0], 0
    for i, ch in enumerate(text):
        if ch == "(":
            stack[-1] += 1
        elif ch == ")":
            stack[-1] -= 1
        elif ch == "{":
            if stack[-1] == 0:
                _check(text, start, i, hits)
                start = i + 1
            stack.append(0)
        elif ch == "}":
            if stack[-1] == 0:
                _check(text, start, i, hits)
                start = i + 1
            if len(stack) > 1:
                stack.pop()
        elif ch == ";" and stack[-1] == 0:
            _check(text, start, i, hits)
            start = i + 1
    return hits


def main(dirs):
    total = 0
    for d in dirs:
        for f in sorted(Path(d).rglob("*.cpp")):
            for line, snippet in find_hits(f.read_text(errors="replace")):
                print(f"{f}:{line}: sibling builder calls: {snippet}")
                total += 1
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
