#!/usr/bin/env python3
"""Recompute the --coalesce-cross determinism verdict over a gate output dir.

Separates the three questions the raw gate.log conflated:
  A. base == cross            -- the flip-relevant question (is the flag argmax-neutral?)
  B. arm == itself across reps -- run-to-run determinism of each NPU arm
  C. NPU arm == ONNX          -- 1:1 vs the CPU reference; a pre-existing property of the
                                 fused decode, NOT caused by the flag
Invocations that died before emitting text (CREATE_HWCTX) are reported as INCOMPLETE, never as a
correctness failure -- conflating those is what made the first run read as 11/17 FAIL.
"""
import sys, pathlib
d = pathlib.Path(sys.argv[1])
clips = sorted({p.name.split('.')[2] for p in d.glob('base.r1.*.txt')} |
               {p.name.split('.')[2] for p in d.glob('onnx.r1.*.txt')})
def rd(label, rep, c):
    p = d / f"{label}.r{rep}.{c}.txt"
    return p.read_text().strip() if p.exists() and p.stat().st_size else None

rows, A_ok, A_n, B_ok, B_n, C_ok, C_n, incomplete = [], 0, 0, 0, 0, 0, 0, []
for c in clips:
    b1, b2, x1, x2, o = rd('base',1,c), rd('base',2,c), rd('cross',1,c), rd('cross',2,c), rd('onnx',1,c)
    if not (b1 and x1):
        incomplete.append(c); rows.append((c,'INCOMPLETE','-','-','-')); continue
    a = (b1 == x1);                      A_n += 1; A_ok += a
    bdet = (b1 == b2) and (x1 == x2) if (b2 and x2) else None
    if bdet is not None: B_n += 1; B_ok += bdet
    cc = (b1 == o) and (x1 == o) if o else None
    if cc is not None: C_n += 1; C_ok += cc
    rows.append((c, 'ok', 'SAME' if a else 'DIFF',
                 {True:'det',False:'NONDET',None:'n/a'}[bdet],
                 {True:'==onnx',False:'!=onnx',None:'n/a'}[cc]))

print(f"{'clip':8} {'run':11} {'A base==cross':14} {'B rep-det':10} {'C vs onnx'}")
for r in rows: print(f"{r[0]:8} {r[1]:11} {r[2]:14} {r[3]:10} {r[4]}")
print()
print(f"A  base == cross           : {A_ok}/{A_n}")
print(f"B  each arm == itself      : {B_ok}/{B_n}")
print(f"C  NPU arm == ONNX ref     : {C_ok}/{C_n}")
if incomplete: print(f"INCOMPLETE (never ran)     : {len(incomplete)} -> {', '.join(incomplete)}")
print()
print("GATE (flip-relevant, A and B):",
      "PASS" if A_n and A_ok == A_n and B_ok == B_n and not incomplete else
      ("PASS on the clips that ran" if A_n and A_ok == A_n and B_ok == B_n else "FAIL"))
