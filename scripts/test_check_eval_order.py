# SPDX-License-Identifier: Apache-2.0
"""check_eval_order.py flags >=2 side-effecting builder calls in one statement's argument list."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ceo", HERE / "check_eval_order.py")
ceo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ceo)

BAD = """
void f() {
  NpuMaskWrite32Op::create(rewriter, loc, createConstantI32(rewriter, loc, a),
                           createConstantI32(rewriter, loc, b));
}
"""
GOOD = """
void f() {
  Value x = createConstantI32(rewriter, loc, a);
  Value y = createConstantI32(rewriter, loc, b);
  NpuMaskWrite32Op::create(rewriter, loc, x, y);
  for (int i = 0; i < n; ++i) { use(createConstantI32(rewriter, loc, i)); }
  // createConstantI32(a), createConstantI32(b) in a comment
}
"""


def test_flags_sibling_calls():
    hits = ceo.find_hits(BAD)
    assert len(hits) == 1 and hits[0][0] == 3


def test_ignores_named_locals_loops_and_comments():
    assert ceo.find_hits(GOOD) == []
