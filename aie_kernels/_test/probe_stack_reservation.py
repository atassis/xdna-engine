#!/usr/bin/env python3
"""Does raising the core's stack reservation turn the half-fill arm green?

The mechanism on record:
the generated linker script places the stack immediately below the objectFIFO buffers with zero
clearance, so a frame larger than the reservation overwrites them. Straight-line vector code was
measured at `paddxm [sp], #0x700` = 1792 B against a 0x400 = 1024 B reservation; the same
arithmetic written as a loop emits no frame at all and passes.

That was an inference from adjacency. This tests it directly, and the reservation is NOT a
hardcoded literal -- AIETargetLdScript.cpp emits `core.getStackSize()`, whose dialect default is
0x400 (AIEOps.td) and which `Worker(stack_size=)` sets per design.

A single larger-stack point would only show "something changed". The sweep is the sharper claim:
if the frame is 0x700 and `main` adds 0x40, the deepest path needs 0x740, so fill must step from
0.5 to 1.0 BETWEEN 0x700 and 0x740 and nowhere else. A transition at that boundary is the
mechanism; a transition somewhere else, or none, refutes it.

  loop  arm : zero frame, must pass at EVERY reservation (control -- if it ever fails, the
              reservation is not the only variable and the flat readings mean nothing)
  flat  arm : 1792 B frame, must fail below 0x740 and pass at/above it

Run:  ./run.sh probe_stack_reservation.py
"""
import time

import numpy as np

import bricklib

GEN = bricklib.GEN
ROWS, COLS, GATE, D = 32, 16, 3e-2, 32
CB = int(time.time() * 1000) % 10**9

# Reservations bracketing the predicted 0x740 need. 0x400 is the dialect default (the shipped
# failing condition); 0x700 is the kernel frame alone, still short by main's own frame.
STACKS = [0x400, 0x600, 0x700, 0x740, 0x800, 0xD00]

rng = np.random.default_rng(7)
x = rng.uniform(-1.0, 1.0, (ROWS, COLS)).astype(np.float32)
CS = [0.11, -0.23, 0.37, -0.41, 0.53, -0.61, 0.71, -0.79]
co = [CS[i % len(CS)] for i in range(D)]
ref = np.full_like(x.astype(np.float64), co[0])
for c in co[1:]:
    ref = ref * x.astype(np.float64) + c

flat_lines = [f"  ::aie::vector<float,16> p = ::aie::broadcast<float,16>({co[0]:.9e}f);"]
for c in co[1:]:
    flat_lines += ["  p = ::aie::mul(p, v);",
                   f"  p = ::aie::add(p, ::aie::broadcast<float,16>({c:.9e}f));"]

arr = ", ".join(f"{c:.9e}f" for c in co)
ARMS = {
    "flat": "\n".join(flat_lines),
    "loop": (f"  static const float K[{D}] = {{{arr}}};\n"
             "  ::aie::vector<float,16> p = ::aie::broadcast<float,16>(K[0]);\n"
             "  #pragma clang loop unroll(disable)\n"
             f"  for (int i = 1; i < {D}; i++) {{\n"
             "    p = ::aie::mul(p, v);\n"
             "    p = ::aie::add(p, ::aie::broadcast<float,16>(K[i]));\n  }"),
}

for tag, body in ARMS.items():
    src = ("#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
           f"// cachebust {CB}\n"
           f'extern "C" void sr_{tag}(float *inp, float *out) {{\n  event0();\n'
           "  ::aie::vector<float,16> v = ::aie::load_v<16>(inp);\n"
           + body + "\n  ::aie::store_v(out, p);\n  event1();\n}\n")
    (GEN / f"sr_{tag}.cc").write_text(src)

print(f"{'arm':6s} {'stack':>7} {'rel-L2':>12} {'fill':>8} {'run2run':>10}  verdict")
fills = {}
for tag in ARMS:
    cc = GEN / f"sr_{tag}.cc"
    for st in STACKS:
        try:
            res = bricklib.verify_rowwise(
                name=f"sr_{tag}_{st:x}", brick_cc=cc, shim_body="", symbol=f"sr_{tag}",
                m=ROWS, in_cols=COLS, out_cols=COLS, x=x, expected=ref, gate=GATE,
                stack_size=st)
            got = np.asarray(res["got"], np.float64)
            fill = float((got != 0).mean())
            fills[(tag, st)] = fill
            print(f"{tag:6s} {st:>#7x} {res['rel_l2']:12.3e} {fill:8.4f} {res['run2run']:10.2e}  "
                  f"{'PASS' if res['ok'] else 'FAIL'}")
        except Exception as ex:
            print(f"{tag:6s} {st:>#7x} ERROR {type(ex).__name__}: {str(ex)[:110]}")

print()
ctrl = [s for s in STACKS if fills.get(("loop", s), 0.0) < 0.999]
if ctrl:
    print(f"CONTROL BROKEN: the zero-frame loop arm did not fill at {[hex(s) for s in ctrl]}.")
    print("         The reservation is not the only variable; the flat column is uninterpretable.")
else:
    red = [s for s in STACKS if fills.get(("flat", s), 1.0) < 0.999]
    green = [s for s in STACKS if fills.get(("flat", s), 0.0) >= 0.999]
    if red and green and max(red) < min(green):
        print(f"CONFIRMED: flat fills only once the reservation reaches {min(green):#x} "
              f"(largest failing {max(red):#x}).")
        if max(red) < 0x740 <= min(green):
            print("         The step brackets 0x740 = kernel frame 0x700 + main 0x40, which is the"
                  "\n         predicted deepest path. Mechanism proven to the byte.")
        else:
            print("         The step is REAL but not at the predicted 0x740 -- the reservation is"
                  "\n         the variable, the frame arithmetic is not yet accounted for.")
    elif not red:
        print("NO REPRO: flat filled at every reservation including the 0x400 default. The failing"
              "\n         condition is not reproduced here -- fix that before reading the sweep.")
    else:
        print(f"REFUTED: raising the reservation did not turn flat green "
              f"(failing at {[hex(s) for s in red]}).")
