#!/usr/bin/env python3
"""BISECT: what structural difference between probe_inline_helper.py's GREEN standalone
and gelu_erf.cc's RED real file actually flips the result?

Prior work (all recorded in gelu_erf.cc's header) established the polynomial itself is not the
defect: primitives exonerated (probe_abs_min.py), long Horner chains survive when fully inlined
into the bound symbol (probe_horner_degree.py, degree 9 green at 1.273e-07), the `static inline`
template-helper spelling is exonerated head-to-head against inline (probe_inline_helper.py, both
1.515e-07), the runtime-vs-compile-time loop trip count is exonerated (bit-identical device number
either way), and chunk count is exonerated (red at ONE chunk too, probe_gelu_chunk.py).

What NONE of those isolate: probe_inline_helper.py's green "helper" case is a ONE-hop call from
the bound extern "C" symbol to the static-inline template helper, with event0()/event1() and the
loop living in that SAME exported function. gelu_erf.cc is a TWO-hop call: the bound/exported
`gelu_erf_f32` is a bare forwarder with no loop and no event0()/event1(); a second, namespaced,
NOT-static/NOT-inline template function `gelu_erf_core<N>` owns the loop AND event0()/event1();
only THAT calls the static-inline leaf helper. `gelu_erf_core` also takes `const .. *restrict`
parameters that `gelu_erf_f32` (the caller) does not declare on its own signature.

This is a LADDER: start from the known-good one-hop spelling and add exactly one of those
structural features per rung, ending at the real file. Same input, same float64-exact polynomial
reference (not the looser GELU-fit gate) at every rung, so a flip is unambiguous. The polynomial
coefficients are read from golden.py, never retyped, and never change across rungs -- if you find
yourself editing a coefficient here, stop, that is the failure mode four prior rounds fell into.

DEVICE RESULT (round 1, coordinator): r0/r1/r2a/r2b/r3 all PASS (~1e-7); the ORIGINAL r4 (runtime
`n` AND wrapper-bound, added in the same rung) FAILED at 7.097e-01 with the recorded symptom
(large positive x -> exact 0.0); r5 (the real file) reproduced r4 EXACTLY, confirming the ladder
reaches the real defect. But r4 moved two variables at once, so it named no culprit -- and
route_b_kernels/bricks/snake/snake.cc is a counterexample that rules out EITHER alone: snake_f32
is a 2-arg wrapper calling a 4-arg entry that threads a runtime int32_t through `chunks = t / N`,
i.e. it has BOTH of r4's changes, and snake is device-green (5.143e-06). So the trigger is either
an interaction between wrapper + runtime-n, or a third thing r4 also changed without naming it.

The other live lead: snake_core does NOT feed sin_v's return value straight into store_v the way
sin_core/gelu_erf_core do. It computes `ax` itself, calls `sin_v<N>(ax)` for just the transcendental
part, then does its OWN `mul`/`mul`/`add` on the result before storing -- sin_core (RED, same
symptom family) and the original gelu_erf_core both do `store_v(out, helper<N>(load_v(...)))`, one
call whose return IS the whole answer. That is a real pattern across three bricks, unexplained.

Round 2 (this ladder) splits the bundled r4 into r4a (wrapper only, compile-time count) and r4b
(runtime count only, bound directly) and adds r4c: r4's exact skeleton (runtime n, wrapper-bound)
but with the loop body doing its own abs/min/add/mul around a narrower helper call for just the
Horner polynomial -- snake's call shape, not sin's -- bit-identical arithmetic, no coefficient
touched. r0-r3 and r5 are kept as-is so the ladder still anchors both ends.

DEVICE RESULT (round 2, coordinator): r3 PASS, r4a (wrapper only) PASS, r4b (runtime-n only) FAIL
7.097e-01, r4c (snake call shape) FAIL 7.097e-01, r5 FAIL 7.097e-01 (identical to r4b/r4c). Two
hypotheses killed, one confirmed:
  * wrapper indirection: EXONERATED (r4a green).
  * call-shape / leaf-helper-indirection: REFUTED (r4c matches snake_core's shape -- core does its
    own abs/min/add/mul around a NARROWER helper call rather than feeding a whole-result helper
    straight into store_v -- and is still red). This was the coordinator's leading cross-brick
    hypothesis (it also covered sin.cc and the new prefill-attn brick, which returned NaN on
    device the same day); it does not explain gelu-erf, so whatever the eventual answer is, it
    must explain sin/prefill-attn SEPARATELY, not via this mechanism.
  * runtime `chunks = n / N` ALONE is sufficient to flip r3 (green) to red, at n=16 / 1 chunk.

THE CONTRADICTION, RESOLVED AS A SECOND FINDING. This round's r4b (runtime n, 1 chunk, direct
bind) is RED. But the coordinator's own earlier compile-time-vs-runtime A/B on the real file, run
at COLS=64 (4 chunks), found compile-time and runtime BIT-IDENTICAL (both red, 9.538126245315652).
Both device results stand; they were run at different chunk counts. So compile-time trip count
fixes the defect at 1 chunk (r3 green) and does NOT fix it at 4 chunks (coordinator's earlier
test, red either way) -- meaning there are (at least) TWO independent problems, and "compile-time
CHUNKS above 1" has never actually been tested in isolation by anyone this session: every green
compile-time probe so far (probe_horner_degree.py, probe_inline_helper.py, r0-r3, r4a) ran exactly
ONE chunk; every multi-chunk run has had a runtime count, except the coordinator's single 4-chunk
compile-time test, which was red.

Round 3 (this ladder) adds a compile-time CHUNKS sweep (r6a=1, r6b=2, r6c=4, r6d=8; direct bind,
no wrapper, no runtime division, everything else identical to r3 -- the untested cell) and r7
(r4b's runtime form but with the division replaced by a DIRECTLY-passed `int32_t chunks`
parameter, no `n/N` division in the kernel at all, same width/chunk-count as r4b for a direct
compare) to separate "runtime loop bound" from "runtime integer division" as the trigger. r0-r5
are kept as-is so the ladder still anchors both ends.

Run under the device lock (from route_b_kernels/bricks/_verify/): ./run.sh probe_gelu_bisect.py
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

GEN = bricklib.GEN
HERE = Path(__file__).parent
BRICK = (HERE.parent / "gelu-erf").resolve()
GELU_ERF_CC = BRICK / "gelu_erf.cc"
GATE = 3e-2
ROWS, COLS = 32, 16

# --- coefficients, read from golden.py so they cannot drift (same technique as
# probe_inline_helper.py) -- NEVER retyped, NEVER changed across rungs. ---
spec = importlib.util.spec_from_file_location("gelu_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)
CS = [float(c) for c in golden._D_COEFFS]
CS_FMT = [f"{c:.9e}f" for c in CS]

# --- shared input + exact float64 reference (the kernel's OWN formula, not the looser GELU-fit
# gate) -- same for every rung AT A GIVEN WIDTH, so a flip is a codegen fact, not a data artifact.
# Most rungs share width COLS=16; the r6 compile-time CHUNKS sweep needs wider inputs (CHUNKS*16
# cols), so the reference computation is factored out and reused at each width via xref_for(). ---
def exact_ref(xarr):
    axc = np.minimum(np.abs(xarr.astype(np.float64)), 5.0)
    p = np.full_like(axc, CS[0])
    for c in CS[1:]:
        p = p * axc + c
    return 0.5 * (xarr.astype(np.float64) + np.abs(xarr.astype(np.float64))) - p


def make_xref(rows, cols, seed):
    rng_local = np.random.default_rng(seed)
    xx = np.concatenate([
        rng_local.uniform(-12.0, -4.0, rows * cols // 3),
        rng_local.uniform(-3.0, 3.0, rows * cols // 3),
        rng_local.uniform(4.0, 12.0, rows * cols - 2 * (rows * cols // 3)),
    ]).astype(np.float32)
    rng_local.shuffle(xx)
    xx = xx.reshape(rows, cols)
    return xx, exact_ref(xx)


rng = np.random.default_rng(23)
x = np.concatenate([
    rng.uniform(-12.0, -4.0, ROWS * COLS // 3),
    rng.uniform(-3.0, 3.0, ROWS * COLS // 3),
    rng.uniform(4.0, 12.0, ROWS * COLS - 2 * (ROWS * COLS // 3)),
]).astype(np.float32)
rng.shuffle(x)
x = x.reshape(ROWS, COLS)
REF = exact_ref(x)

# cache keyed by column width, so every rung sharing a width (all of r0-r5/r4a-c/r7 at COLS=16)
# reuses the SAME array already validated across two device rounds -- only wider r6 rungs get a
# freshly generated (but still deterministic) array.
XREF_CACHE = {COLS: (x, REF)}


def xref_for(cols):
    if cols not in XREF_CACHE:
        XREF_CACHE[cols] = make_xref(ROWS, cols, seed=20000 + cols)
    return XREF_CACHE[cols]


def cachebust():
    return "// cachebust " + str(int(time.time() * 1000) % 10**9)


# --- the leaf helper: IDENTICAL text in every rung. This is the polynomial; it never changes. ---
HELPER_LINES = [
    "template <int N>",
    "static inline ::aie::vector<float, N> gelu_v(::aie::vector<float, N> v) {",
    "  ::aie::vector<float, N> ax = ::aie::abs(v);",
    "  ::aie::vector<float, N> axc = ::aie::min(ax, ::aie::broadcast<float, N>(5.0f));",
    "  ::aie::vector<float, N> s = ::aie::add(v, ax);",
    "  s = ::aie::mul(s, ::aie::broadcast<float, N>(0.5f));",
    "  ::aie::vector<float, N> p = ::aie::broadcast<float, N>(" + CS_FMT[0] + ");",
]
for _c in CS_FMT[1:]:
    HELPER_LINES.append("  p = ::aie::mul(p, axc);")
    HELPER_LINES.append("  p = ::aie::add(p, ::aie::broadcast<float, N>(" + _c + "));")
HELPER_LINES.append("  return ::aie::sub(s, p);")
HELPER_LINES.append("}")
HELPER = "\n".join(HELPER_LINES)

PREAMBLE = "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"


def src_r0():
    """R0 ANCHOR: probe_inline_helper.py's green 'helper' spelling, reproduced verbatim in
    shape. Entry directly hosts the loop + event0()/event1(); ONE hop to the static-inline
    template helper; no namespace; N=16 compile-time; 1 chunk. Bound directly."""
    return "\n".join([
        PREAMBLE + cachebust(),
        HELPER,
        'extern "C" void gelu_r0(float *inp, float *out) {',
        "  event0();",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < 1; i++) {",
        "    ::aie::store_v(out + i*16, gelu_v<16>(::aie::load_v<16>(inp + i*16)));",
        "  }",
        "  event1();",
        "}",
    ])


def src_r1():
    """R1: + wrap the helper in `namespace route_b_bricks {}` (matches gelu_erf.cc). Entry
    still hosts the loop + event0()/event1() directly and still calls the helper ONE hop
    deep -- only the namespace changed."""
    return "\n".join([
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        HELPER,
        "}  // namespace route_b_bricks",
        'extern "C" void gelu_r1(float *inp, float *out) {',
        "  event0();",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < 1; i++) {",
        "    ::aie::store_v(out + i*16, route_b_bricks::gelu_v<16>(::aie::load_v<16>(inp + i*16)));",
        "  }",
        "  event1();",
        "}",
    ])


def src_r2a():
    """R2a: + delegate the loop to a SECOND namespaced function `gelu_loop_core<N>` (NOT
    static/inline -- matches gelu_erf_core's declaration), one hop deeper than R1.
    event0()/event1() STAY in the exported entry (not moved) -- isolates the extra hop
    from event placement."""
    return "\n".join([
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        HELPER,
        "",
        "template <int N>",
        "void gelu_loop_core(float *input, float *output) {",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < 1; i++) {",
        "    ::aie::store_v(output + i*N, gelu_v<N>(::aie::load_v<N>(input + i*N)));",
        "  }",
        "}",
        "}  // namespace route_b_bricks",
        'extern "C" void gelu_r2a(float *inp, float *out) {',
        "  event0();",
        "  route_b_bricks::gelu_loop_core<16>(inp, out);",
        "  event1();",
        "}",
    ])


def src_r2b():
    """R2b: + move event0()/event1() INTO the core function too (matches gelu_erf_core
    exactly). The exported entry is now a bare one-line forwarder -- same shape as
    gelu_erf_f32 minus the `n` parameter and the restrict/const qualifiers."""
    return "\n".join([
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        HELPER,
        "",
        "template <int N>",
        "void gelu_core(float *input, float *output) {",
        "  event0();",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < 1; i++) {",
        "    ::aie::store_v(output + i*N, gelu_v<N>(::aie::load_v<N>(input + i*N)));",
        "  }",
        "  event1();",
        "}",
        "}  // namespace route_b_bricks",
        'extern "C" void gelu_r2b(float *inp, float *out) {',
        "  route_b_bricks::gelu_core<16>(inp, out);",
        "}",
    ])


def src_r3():
    """R3: + add `const`+`restrict` to the CORE function's parameters only (matches
    gelu_erf_core's exact signature `const float *restrict input, float *restrict output`).
    The exported entry keeps plain unqualified pointers (matches gelu_erf_f32's exact
    signature) -- this is the qualifier MISMATCH across the call boundary, present verbatim
    in gelu_erf.cc."""
    return "\n".join([
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        HELPER,
        "",
        "template <int N>",
        "void gelu_core_r3(const float *restrict input, float *restrict output) {",
        "  event0();",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < 1; i++) {",
        "    ::aie::store_v(output + i*N, gelu_v<N>(::aie::load_v<N>(input + i*N)));",
        "  }",
        "  event1();",
        "}",
        "}  // namespace route_b_bricks",
        'extern "C" void gelu_r3(float *inp, float *out) {',
        "  route_b_bricks::gelu_core_r3<16>(inp, out);",
        "}",
    ])


def src_r4a_core():
    """R4a: WRAPPER ONLY. Core+entry are r3-shaped -- COMPILE-TIME count (1 hardcoded
    chunk), restrict/const on the core, entry a bare 2-arg forwarder -- i.e. nothing about
    the core's trip count changes from r3. The only difference from r3 is at the BINDING
    site: the test does not bind to this file's entry directly, it binds to a separate
    wrapper (defined in the shim, exactly like r4/r5's wrapper) that calls the entry.
    Isolates: does merely adding a harness-level wrapper hop matter, holding the trip count
    compile-time and the arity unchanged (2-arg wrapper -> 2-arg entry)?"""
    return "\n".join([
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        HELPER,
        "",
        "template <int N>",
        "void gelu_core_r4a(const float *restrict input, float *restrict output) {",
        "  event0();",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < 1; i++) {",
        "    ::aie::store_v(output + i*N, gelu_v<N>(::aie::load_v<N>(input + i*N)));",
        "  }",
        "  event1();",
        "}",
        "}  // namespace route_b_bricks",
        'extern "C" void gelu_f32_r4a(float *input, float *output) {',
        "  route_b_bricks::gelu_core_r4a<16>(input, output);",
        "}",
    ])


def src_r4b_core():
    """R4b: RUNTIME N ONLY. The core now takes a genuine int32_t `n` and computes
    `chunks = n / N` by integer division at runtime (matching gelu_erf_core's real
    computation) -- but the exported entry stays a bare 2-arg forwarder, BOUND DIRECTLY (no
    wrapper hop added), and supplies `n` via a local literal at its own call site. The core
    is a separate, not-static/not-inline function, so it cannot see that the caller's `n`
    happens to be a compile-time literal -- `chunks = n / N` is still a genuine runtime
    division from the core's point of view. Isolates: does the runtime division inside the
    (already-separate, already-restrict-qualified) core matter alone, holding the binding
    path direct?"""
    return "\n".join([
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        HELPER,
        "",
        "template <int N>",
        "void gelu_core_r4b(const float *restrict input, float *restrict output, int32_t n) {",
        "  event0();",
        "  const int chunks = n / N;",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < chunks; i++) {",
        "    ::aie::store_v(output + i*N, gelu_v<N>(::aie::load_v<N>(input + i*N)));",
        "  }",
        "  event1();",
        "}",
        "}  // namespace route_b_bricks",
        'extern "C" void gelu_r4b(float *inp, float *out) {',
        "  const int32_t n = 16;",
        "  route_b_bricks::gelu_core_r4b<16>(inp, out, n);",
        "}",
    ])


def src_r4c_core():
    """R4c: SNAKE-SHAPED POSITIVE CONTROL. Same skeleton as the original bundled r4
    (runtime `n`, `chunks = n/N`, restrict-qualified core, wrapper-bound entry -- both of
    r4's changes are present) -- but the loop body no longer feeds a whole-result-returning
    helper call straight into store_v. Exactly like snake_core computes `ax` itself, calls
    `sin_v<N>(ax)` for just the transcendental part, then does its OWN mul/mul/add before
    storing: this core computes `ax`/`axc`/`s` itself and calls a NARROWER leaf helper
    `D_v<N>` for just the 5-term Horner polynomial, then does its OWN final `sub` before
    storing. Bit-identical arithmetic to gelu_erf_v/HELPER above -- same coefficients, same
    literals, just relocated between two functions instead of living in one. This tests the
    call/compute SHAPE, not the maths."""
    lines = [
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        "",
        "template <int N>",
        "static inline ::aie::vector<float, N> D_v(::aie::vector<float, N> axc) {",
        "  ::aie::vector<float, N> p = ::aie::broadcast<float, N>(" + CS_FMT[0] + ");",
    ]
    for _c in CS_FMT[1:]:
        lines.append("  p = ::aie::mul(p, axc);")
        lines.append("  p = ::aie::add(p, ::aie::broadcast<float, N>(" + _c + "));")
    lines.append("  return p;")
    lines.append("}")
    lines += [
        "",
        "template <int N>",
        "void gelu_core_r4c(const float *restrict input, float *restrict output, int32_t n) {",
        "  event0();",
        "  const int chunks = n / N;",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < chunks; i++) {",
        "    ::aie::vector<float, N> x = ::aie::load_v<N>(input + i*N);",
        "    ::aie::vector<float, N> ax = ::aie::abs(x);",
        "    ::aie::vector<float, N> axc = ::aie::min(ax, ::aie::broadcast<float, N>(5.0f));",
        "    ::aie::vector<float, N> s = ::aie::add(x, ax);",
        "    s = ::aie::mul(s, ::aie::broadcast<float, N>(0.5f));",
        "    ::aie::vector<float, N> p = D_v<N>(axc);",
        "    ::aie::store_v(output + i*N, ::aie::sub(s, p));",
        "  }",
        "  event1();",
        "}",
        "}  // namespace route_b_bricks",
        'extern "C" void gelu_f32_r4c(float *input, float *output, int32_t n) {',
        "  route_b_bricks::gelu_core_r4c<16>(input, output, n);",
        "}",
    ]
    return "\n".join(lines)


def src_r6_core(tag, chunks_ct):
    """Compile-time CHUNKS sweep, one generator shared by r6a-d. Same core shape as r3
    (namespaced, NOT static/inline, restrict-qualified, owns loop+event0/1) but the loop
    trip count is a SECOND compile-time template parameter `CHUNKS`, not r3's hardcoded
    literal `1`. Bound DIRECTLY, no wrapper, no runtime division anywhere -- isolates chunk
    count alone, compile-time, against r3's already-proven-green single-chunk case."""
    fn = f"gelu_core_{tag}"
    entry = f"gelu_{tag}"
    return "\n".join([
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        HELPER,
        "",
        "template <int N, int CHUNKS>",
        f"void {fn}(const float *restrict input, float *restrict output) {{",
        "  event0();",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < CHUNKS; i++) {",
        "    ::aie::store_v(output + i*N, gelu_v<N>(::aie::load_v<N>(input + i*N)));",
        "  }",
        "  event1();",
        "}",
        "}  // namespace route_b_bricks",
        f'extern "C" void {entry}(float *inp, float *out) {{',
        f"  route_b_bricks::{fn}<16, {chunks_ct}>(inp, out);",
        "}",
    ])


def src_r6a():
    return src_r6_core("r6a", 1)


def src_r6b():
    return src_r6_core("r6b", 2)


def src_r6c():
    return src_r6_core("r6c", 4)


def src_r6d():
    return src_r6_core("r6d", 8)


def src_r7_core():
    """R7: runtime trip count WITHOUT integer division. r4b threaded a runtime `n` through
    `chunks = n / N`; this instead takes `int32_t chunks` directly as the loop bound -- same
    runtime-ness (the core cannot see across the call boundary that the caller's value
    happens to be a compile-time literal), but there is no division anywhere in the kernel.
    Isolates: is it the runtime LOOP BOUND that matters, or specifically the runtime integer
    DIVISION used to derive it in r4b/gelu_erf_core? Same width/chunk-count as r4b (16 cols,
    1 chunk) for a direct comparison; bound DIRECTLY, no wrapper, matching r4b's binding."""
    return "\n".join([
        PREAMBLE + cachebust(),
        "namespace route_b_bricks {",
        HELPER,
        "",
        "template <int N>",
        "void gelu_core_r7(const float *restrict input, float *restrict output, int32_t chunks) {",
        "  event0();",
        "  #pragma clang loop unroll(disable)",
        "  for (int i = 0; i < chunks; i++) {",
        "    ::aie::store_v(output + i*N, gelu_v<N>(::aie::load_v<N>(input + i*N)));",
        "  }",
        "  event1();",
        "}",
        "}  // namespace route_b_bricks",
        'extern "C" void gelu_r7(float *inp, float *out) {',
        "  const int32_t chunks = 1;",
        "  route_b_bricks::gelu_core_r7<16>(inp, out, chunks);",
        "}",
    ])


# --- the ladder, in order. Each entry either provides its own self-contained source (bound
# directly, shim_body="") or a (brick_cc, shim_body) pair that mirrors exactly how
# verify_gelu_erf.py / probe_gelu_chunk.py bind the real gelu_erf_f32. `cols` (default COLS=16)
# lets a rung request a wider input -- only the r6 compile-time CHUNKS sweep needs this. ---
RUNGS = [
    dict(tag="r0_helper_baseline", fn=src_r0, symbol="gelu_r0", shim_body="",
         desc="ANCHOR (= probe_inline_helper.py's green 'helper' case): 1 hop, no namespace, "
              "event0/1 + loop in the exported entry itself."),
    dict(tag="r1_namespace", fn=src_r1, symbol="gelu_r1", shim_body="",
         desc="+ wrap the helper in `namespace route_b_bricks {}`. Still 1 hop, still "
              "event0/1 + loop in the exported entry."),
    dict(tag="r2a_extra_hop_event_stays", fn=src_r2a, symbol="gelu_r2a", shim_body="",
         desc="+ delegate the loop to a 2nd namespaced, NOT-static/inline function "
              "(gelu_erf_core's shape). event0/1 STAY in the exported entry."),
    dict(tag="r2b_event_moves_to_core", fn=src_r2b, symbol="gelu_r2b", shim_body="",
         desc="+ move event0()/event1() INTO the core function (matches gelu_erf_core "
              "exactly). Entry is now a bare 1-line forwarder."),
    dict(tag="r3_restrict_mismatch", fn=src_r3, symbol="gelu_r3", shim_body="",
         desc="+ core gets `const T *restrict` params; entry keeps plain unqualified "
              "pointers -- the exact qualifier mismatch gelu_erf.cc has. "
              "[round 1 device: PASS 1.497e-07]"),
    dict(tag="r4a_wrapper_only", fn=src_r4a_core, symbol="gelu_r4a_w",
         shim_body='extern "C" void gelu_r4a_w(float *x, float *out) '
                    '{ gelu_f32_r4a(x, out); }\n',
         desc="vs r3: ONLY change is binding through an extra 2-arg wrapper hop "
              "(shim calls the entry rather than being the entry). Count stays "
              "compile-time, 1 chunk, arity unchanged (2->2)."),
    dict(tag="r4b_runtime_n_only", fn=src_r4b_core, symbol="gelu_r4b", shim_body="",
         desc="vs r3: ONLY change is the core taking a genuine int32_t n and computing "
              "chunks=n/N at runtime; still bound DIRECTLY, no wrapper hop added."),
    dict(tag="r4c_snake_shaped_control", fn=src_r4c_core, symbol="gelu_r4c_w",
         shim_body='extern "C" void gelu_r4c_w(float *x, float *out) '
                    '{ gelu_f32_r4c(x, out, 16); }\n',
         desc="r4's exact skeleton (runtime n, wrapper-bound, restrict core) but the loop "
              "body computes ax/axc/s itself and calls a NARROWER helper (just the Horner "
              "poly) then does its OWN final sub -- snake_core's call shape, not sin_core's. "
              "Compare directly against r5: only the call/compute shape differs."),
    dict(tag="r5_real_file", fn=None, symbol="gelu_r5_w",
         shim_body='extern "C" void gelu_r5_w(float *x, float *out) '
                    '{ gelu_erf_f32(x, out, 16); }\n',
         desc="the ACTUAL gelu_erf.cc, unchanged, called exactly like the original bundled "
              "r4 (n=16, 1 chunk). [round 2 device: FAIL 7.097e-01, identical to r4b/r4c]"),
    dict(tag="r6a_ct_chunks1", fn=src_r6a, symbol="gelu_r6a", shim_body="", cols=16,
         desc="compile-time CHUNKS sweep 1/4: CHUNKS=1 via a 2nd template param (r3 used a "
              "bare literal `1`). SANITY ANCHOR -- should reproduce r3, not a new datum."),
    dict(tag="r6b_ct_chunks2", fn=src_r6b, symbol="gelu_r6b", shim_body="", cols=32,
         desc="compile-time CHUNKS sweep 2/4: CHUNKS=2, cols=32. Direct bind, no wrapper, "
              "no runtime division -- only CHUNKS changed vs r6a."),
    dict(tag="r6c_ct_chunks4", fn=src_r6c, symbol="gelu_r6c", shim_body="", cols=64,
         desc="compile-time CHUNKS sweep 3/4: CHUNKS=4, cols=64 -- the coordinator's earlier "
              "4-chunk compile-time A/B on the real file was RED at this chunk count."),
    dict(tag="r6d_ct_chunks8", fn=src_r6d, symbol="gelu_r6d", shim_body="", cols=128,
         desc="compile-time CHUNKS sweep 4/4: CHUNKS=8, cols=128."),
    dict(tag="r7_runtime_chunks_no_div", fn=src_r7_core, symbol="gelu_r7", shim_body="",
         cols=16,
         desc="vs r4b: runtime trip count passed DIRECTLY as `int32_t chunks` (no `n/N` "
              "division in the kernel). Same width/chunk-count as r4b (1 chunk) -- isolates "
              "the division itself from 'runtime loop bound' in general."),
]

print(f"{'rung':28s} {'cols':>5} {'rel-L2':>12} {'max-abs':>12}  verdict   change vs previous")
print("-" * 118)
results = []
for r in RUNGS:
    cols = r.get("cols", COLS)
    xw, refw = xref_for(cols)
    if r["fn"] is not None:
        cc = GEN / f"{r['tag']}.cc"
        cc.write_text(r["fn"]())
        brick_cc = cc
    else:
        brick_cc = GELU_ERF_CC
    res = bricklib.verify_rowwise(
        name=r["tag"], brick_cc=brick_cc, shim_body=r["shim_body"], symbol=r["symbol"],
        m=ROWS, in_cols=cols, out_cols=cols, x=xw, expected=refw, gate=GATE)
    got = np.asarray(res["got"], np.float64)
    ma = float(np.max(np.abs(got - refw)))
    # positive-tail diagnostic: same symptom check as verify_gelu_erf.py / gelu_erf.cc header
    # ("large POSITIVE x returns exactly 0.0"). Confirms whether the SAME symptom reproduces
    # at the flip point, or something else (NaN, garbage) does.
    pos_mask = xw >= 8.0
    nzero_pos = int((got[pos_mask] == 0.0).sum()) if np.any(pos_mask) else 0
    results.append(dict(tag=r["tag"], desc=r["desc"], ok=res["ok"], rel_l2=res["rel_l2"],
                        max_abs=ma, nzero_pos=nzero_pos, pos_total=int(pos_mask.sum())))
    print(f"{r['tag']:28s} {cols:5d} {res['rel_l2']:12.3e} {ma:12.3e}  "
          f"{'PASS' if res['ok'] else 'FAIL':7s}  {r['desc']}")
    if not res["ok"]:
        print(f"{'':28s} positive-tail (x>=8) exact-zeros: {nzero_pos}/{int(pos_mask.sum())}")
        w = np.argsort(np.abs(got - refw).ravel())[::-1][:4]
        for fi in w:
            wr, wc = divmod(int(fi), cols)
            print(f"{'':28s} x={xw[wr,wc]:+9.4f}  golden={refw[wr,wc]:+11.5f}  "
                  f"device={got[wr,wc]:+13.5e}")

print()
print("=" * 110)
by_tag = {r["tag"]: r for r in results}


def tag_verdict(r):
    return "PASS" if r["ok"] else "FAIL"


first_fail = next((i for i, r in enumerate(results) if not r["ok"]), None)
if first_fail is None:
    print("VERDICT: every rung GREEN, including r5_real_file. This contradicts round 1's "
          "device history (gelu_erf.cc / r5 red at 7.097e-01 / 9.538e+00) -- re-run "
          "verify_gelu_erf.py itself before trusting this; something about the harness or "
          "inputs changed.")
elif first_fail == 0:
    print("VERDICT: r0 (the known-good anchor) is ALREADY red in this run. The bisect ladder "
          "did not reproduce the reference green result -- do not trust the rest of this "
          "table; something about this run (not the code) is off (stale cache? wrong "
          "toolchain instance?). Re-run before concluding anything.")
else:
    prev = results[first_fail - 1]
    culprit = results[first_fail]
    print(f"FIRST FLIP: '{prev['tag']}' PASSED (rel-L2 {prev['rel_l2']:.3e}); "
          f"'{culprit['tag']}' FAILED (rel-L2 {culprit['rel_l2']:.3e}).")
    print(f"  single change there: {culprit['desc']}")
    if culprit["nzero_pos"] > 0:
        print(f"  symptom match: {culprit['nzero_pos']}/{culprit['pos_total']} exact-zero "
              f"outputs on the positive tail (x>=8) -- same symptom as gelu_erf.cc "
              f"(x=+11.9882 -> +0.00000e+00).")

print()
print("-" * 110)
print("SUB-HYPOTHESIS BREAKDOWN (r4a = wrapper alone, r4b = runtime-n alone, "
      "r4c = snake's call shape):")
r3r, r4ar, r4br, r4cr, r5r = (by_tag.get(t) for t in
    ("r3_restrict_mismatch", "r4a_wrapper_only", "r4b_runtime_n_only",
     "r4c_snake_shaped_control", "r5_real_file"))
if None in (r3r, r4ar, r4br, r4cr, r5r):
    print("  (one or more tagged rungs missing from results -- skipping breakdown)")
else:
    print(f"  r3  (anchor)          : {tag_verdict(r3r)}  rel-L2 {r3r['rel_l2']:.3e}")
    print(f"  r4a (wrapper only)    : {tag_verdict(r4ar)}  rel-L2 {r4ar['rel_l2']:.3e}")
    print(f"  r4b (runtime-n only)  : {tag_verdict(r4br)}  rel-L2 {r4br['rel_l2']:.3e}")
    print(f"  r4c (snake call shape): {tag_verdict(r4cr)}  rel-L2 {r4cr['rel_l2']:.3e}")
    print(f"  r5  (real file)       : {tag_verdict(r5r)}  rel-L2 {r5r['rel_l2']:.3e}")
    print()
    if r3r["ok"]:
        if not r4ar["ok"]:
            print("  -> wrapper indirection ALONE (compile-time count) is sufficient to flip "
                  "r3 to red. But snake_f32 is bound via an equivalent 2-arg wrapper and is "
                  "green, so wrapper alone cannot be universally sufficient -- something "
                  "about THIS wrapper/entry pairing differs from snake's, or this is real but "
                  "conditional on something else r4a still shares with the red cases.")
        else:
            print("  -> wrapper indirection ALONE (compile-time count) does NOT flip it. "
                  "Wrapper-vs-direct-bind is exonerated as a sole cause.")
        if not r4br["ok"]:
            print("  -> the runtime chunks=n/N division ALONE (direct bind, no wrapper) is "
                  "sufficient to flip r3 to red. This would directly contradict the "
                  "session's earlier compile-time-vs-runtime A/B on the real file (bit-"
                  "identical either way) -- re-check that earlier test's binding path before "
                  "trusting either result over the other.")
        else:
            print("  -> the runtime chunks=n/N division ALONE (direct bind) does NOT flip it. "
                  "Runtime-vs-compile-time trip count is exonerated as a sole cause, "
                  "consistent with the session's earlier compile-time A/B on the real file.")
        if r4ar["ok"] and r4br["ok"] and not r5r["ok"]:
            print("  -> NEITHER alone flips it, but the real file (both together, plus "
                  "whatever else) is still red: this is an INTERACTION between wrapper and "
                  "runtime-n, or r4a/r4b still miss a third factor r4/r5 both carry.")
    if r4cr["ok"] and not r5r["ok"]:
        print("  -> CALL-SHAPE CONFIRMED: r4c (core does its own abs/min/add/mul around a "
              "narrower helper call, snake's shape) is GREEN while r5 (helper call returns "
              "the whole result straight into store_v, sin's shape) is RED, with everything "
              "else -- runtime n, wrapper, restrict, namespace -- held equal. This is the "
              "differentiator, and it explains sin.cc (same shape as r5) vs snake.cc (same "
              "shape as r4c) in the same stroke.")
    elif not r4cr["ok"] and not r5r["ok"]:
        print("  -> r4c is ALSO red despite matching snake's call shape: the call-shape "
              "hypothesis does NOT explain this brick either. The remaining difference from "
              "snake_core must be something not yet isolated here -- candidates: snake_core "
              "has NO `#pragma clang loop unroll(disable)` (gelu_core/r4c does); sin_v's "
              "argument-reduction work (the round-trip add/sub magic-number fold) ahead of "
              "its own Horner chain, which D_v/gelu_v do not have; or the specific constants/"
              "degree. Do not touch gelu_erf.cc on this evidence -- report the narrowed set "
              "and design a further rung.")
    elif r4cr["ok"] and r5r["ok"]:
        print("  -> r4c and r5 are BOTH green in this run -- that contradicts round 1's "
              "device history for r5. Re-run before trusting this table.")

print()
print("-" * 110)
print("COMPILE-TIME CHUNKS SWEEP (r6a-d, the untested cell) + division-vs-bound isolation (r7):")
r6ar, r6br, r6cr, r6dr, r7r = (by_tag.get(t) for t in
    ("r6a_ct_chunks1", "r6b_ct_chunks2", "r6c_ct_chunks4", "r6d_ct_chunks8",
     "r7_runtime_chunks_no_div"))
if None in (r6ar, r6br, r6cr, r6dr, r7r):
    print("  (one or more r6/r7 rungs missing from results -- skipping breakdown)")
else:
    print(f"  r6a (CHUNKS=1, ct)         : {tag_verdict(r6ar)}  rel-L2 {r6ar['rel_l2']:.3e}")
    print(f"  r6b (CHUNKS=2, ct)         : {tag_verdict(r6br)}  rel-L2 {r6br['rel_l2']:.3e}")
    print(f"  r6c (CHUNKS=4, ct)         : {tag_verdict(r6cr)}  rel-L2 {r6cr['rel_l2']:.3e}")
    print(f"  r6d (CHUNKS=8, ct)         : {tag_verdict(r6dr)}  rel-L2 {r6dr['rel_l2']:.3e}")
    print(f"  r7  (runtime chunks,no div): {tag_verdict(r7r)}  rel-L2 {r7r['rel_l2']:.3e}")
    print()
    if not r6ar["ok"]:
        print("  -> r6a (CHUNKS=1 via a 2nd template param) is ALREADY red, and r3 (same "
              "shape/width, CHUNKS as a bare literal `1`) was green -- that is a flip caused "
              "purely by HOW the compile-time bound is spelled (template param vs literal). "
              "Do not trust r6b/c/d's readings as a clean chunk-count sweep until this is "
              "re-examined; the sweep's premise (r6a anchors r3) did not hold.")
    else:
        multi = [("r6b", r6br), ("r6c", r6cr), ("r6d", r6dr)]
        multi_fail = [t for t, rr in multi if not rr["ok"]]
        if not multi_fail:
            print("  -> ALL of r6b/r6c/r6d GREEN. The ONLY defect in this whole ladder is the "
                  "runtime division (r4b/r5's failure mode) -- chunk count itself, compile-time, "
                  "is not independently guilty up to CHUNKS=8. The coordinator's earlier 4-chunk "
                  "compile-time A/B on the real gelu_erf.cc (red, bit-identical to runtime) "
                  "needs re-examination: something about THAT specific test -- not chunk count "
                  "-- produced a red result. A compile-time trip count is therefore a live "
                  "candidate fix for gelu_erf.cc, at least through CHUNKS=8.")
        else:
            print(f"  -> {multi_fail} went RED while r6a (CHUNKS=1) stayed GREEN: a SECOND, "
                  "chunk-count-dependent defect exists on top of the runtime-division one, "
                  "confirming the coordinator's 4-chunk compile-time A/B. 'One chunk per call' "
                  "is the only currently-known-good configuration for this kernel shape; a "
                  "compile-time trip count fixes the division defect but does NOT make "
                  "multi-chunk calls safe on its own.")
    if r7r["ok"] and not r4br["ok"]:
        print("  -> r7 (runtime chunks passed directly, no division) is GREEN while r4b "
              "(runtime chunks via n/N division) is RED, same width/chunk-count: the DIVISION "
              "itself is the defect, not 'a runtime loop bound' in general. A miscompiled or "
              "mis-lowered int32_t division is now the leading candidate -- worth an isolated "
              "single-op repro and an upstream llvm-aie/Peano report, independent of gelu-erf.")
    elif not r7r["ok"] and not r4br["ok"]:
        print("  -> r7 is ALSO red: the defect is not specific to the division op itself -- a "
              "runtime trip count, however it is produced, is sufficient on its own at this "
              "chunk count. The division hypothesis is narrowed out; the trigger is 'the core "
              "receives ANY runtime-valued loop bound', not specifically how it is computed.")
    elif r7r["ok"] and r4br["ok"]:
        print("  -> r7 and r4b are BOTH green in this run -- that contradicts round 2's device "
              "history for r4b (red at 7.097e-01). Re-run before trusting this table.")
    else:
        print("  -> r7 is red and r4b is green in this run -- that contradicts round 2's device "
              "history for r4b (red at 7.097e-01). Re-run before trusting this table.")
