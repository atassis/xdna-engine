//===- gelu_erf.cc ---------------------------------------------*- C++ -*-===//
//
// GENERIC vectorized GELU brick (route_b_kernels/bricks, group: activation), matching
// ggml_vec_gelu_erf_f32 / scripts/codec_quantizer_ref.py::gelu_erf's target curve:
//
//   y = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// Consumer: convnext_block (~line 292) applies this elementwise to the [4096, T] pwconv1 output,
// between pwconv1 and pwconv2 -- a plain UNARY activation over a large tensor.
//
// NOT route_b_kernels/bricks/geglu/geglu.cc, on two independent counts: geglu.cc is (1) the TANH
// approximation, not erf -- a different curve, and (2) GATED (GeGLU: gelu(gate)*up, two operands).
// This brick is the plain ungated GELU: one input row, one output row.
//
// ==========================================================================================
// THREE ROUNDS OF DEVICE FAILURE -- READ BEFORE CHANGING THE DEGREE OR THE VARIABLE ORDER BELOW.
// ==========================================================================================
// v1 computed erf via Abramowitz & Stegun 7.1.26 (aie::inv reciprocal + aie::exp2<bfloat16> SFU).
// Host predicted rel-L2 1.171e-05; device measured 0.70 (gate 3e-2). "pos tail (x>=8) max
// |device-x|" = 12.0 at x=12 (device output exactly 0.0).
//
// v2 added an x-clamp before the exp2 argument plus an unconditional final invariant clamp
// (0<=y<=x for x>=0, exact for the true GELU). Device measured 0.7043 -- ESSENTIALLY UNCHANGED. A
// device probe (oneshot vs rowwise, same kernel) showed BOTH rails red with row0 identical, ruling
// out a rail/codegen defect and proving the arithmetic itself wrong broadly (16/16 rows), not just
// at extreme |x|. x=12 STILL read exactly 0.0 -- the v2 final clamp mapped a wrong negative raw
// value to exactly 0, a "plausible" GELU value at some x, hiding the defect for a full round.
// LESSON: a "can only reduce error" safety clamp is only safe for DIAGNOSIS if you can see through
// it. v3 (and this version) carry NO final clamp for that reason.
//
// v3 dropped erf/exp2/aie::inv entirely for a direct polynomial fit of D(ax) := ax - GELU_ref(ax)
// (GELU(-x)=GELU(x)-x lets `y = 0.5*(x+ax) - D(ax)` reproduce GELU_ref(x) for all real x from a
// function of ax=|x| alone), degree 9, over ax in [0,5]. Every LANDMARK in verify_gelu_erf.py came
// back correct on device, including the predicted small constant tail offset -- so the FORMULA was
// right. But a device probe (coordinator's probe_gelu_rail.py) isolating abs/min/mul/add and the
// literal term `0.5*(x+abs(x))` found ALL FOUR bit-exact on device, while a non-landmark point
// (x=11.9882) read exactly 0.0 and later chunks in a multi-chunk run blew up to -1.85e5. That
// combination -- primitives individually exonerated, landmarks correct, non-landmark values wrong,
// "zeros interleaved" -- is textbook spill/alias codegen, described almost verbatim in sin.cc's own
// header (an earlier sin.cc version with ~17 live named intermediates came back with results
// ALIASED across variables: p3 returned r2's value, zeros interleaved; sin itself is STILL red
// today for this reason, llvm-aie #1155 territory). v3's chain held x, ax, axc, p, s live across a
// TEN-coefficient Horner loop, each `aie::broadcast` materialising another vector -- longer than
// sin_v's six-term chain, which already spills on this hardware.
//
// v4 (current): same formula as v3, TWO mechanical changes only, no new maths.
//   1. Degree cut 9 -> 5 (6 coefficients instead of 10): the gate is 3e-2 and v3's own host numbers
//      showed degree 9 accurate to ~1e-5, about 3000x more precision than needed. Refit at
//      successively lower degree until the host fit cleared ~1e-3 (rel-L2) with room under the 3e-2
//      gate: degree 4 measured rel-L2 2.389e-03 / max-abs 3.064e-02 (too close to the gate itself
//      -- not "comfortable"); degree 5 measured rel-L2 5.422e-04 / max-abs 6.701e-03 (~55x / ~4.5x
//      margin) and was kept. Each dropped term removes one live broadcast and one mul/add pair.
//   2. Reordered so `s = 0.5*(x+ax)` is computed BEFORE the Horner loop, not after: x and ax are
//      then dead for the rest of the function, so only `axc`, `p`, and `s` are live across the
//      polynomial evaluation -- down from v3's five (x, ax, axc, p, s all live simultaneously).
// Both changes are purely mechanical (shorten the live chain); the arithmetic, the D(ax) definition,
// and the fit domain [0,5] are unchanged from v3. If THIS still shows non-landmark-only failures,
// the spill hypothesis itself needs revisiting (e.g. 2-accumulator Estrin split, or the per-call
// width), not another degree cut -- degree 5 already has a large accuracy margin to spend.
//
// APPROXIMATION ROUTE: D(ax) := ax - GELU_ref(ax), fit as a degree-5 polynomial in ax over
// ax in [0, 5] (least squares, numpy.polyfit against the A&S-based golden). D(0)=0 by construction
// of GELU_ref, and D(ax)->0 as ax->inf, so the domain [0,5] already captures the whole shape; ax is
// clamped into that domain (aie::min; ax is already >=0, so no lower clamp is needed) before the
// polynomial is evaluated, so it NEVER extrapolates -- for x outside [-5,5] the output is (x or 0)
// plus a FIXED, small (~6.7e-3 host), non-growing offset D(5), never an unbounded blow-up.
// Coefficients (highest degree first), fit against GELU_ref, host-verified in float64:
//   x^5    0.002835879616732109      x^4   -0.043260768673460806
//   x^3    0.24472919119017997       x^2   -0.6094418780222849
//   x^1    0.5663938552943448        x^0   -0.004514140008753372
//
// HOST MODEL RELIABILITY -- THREE STRIKES NOW, READ THIS. v1 and v2's host simulations predicted
// numbers orders of magnitude better than the device measured (1.171e-05 vs 0.70, then vs 0.7043)
// for a formula that was WRONG. v3's host simulation was numerically fine (2.232e-05 rel-L2) for a
// formula that was RIGHT but a codegen chain that was too long -- so a clean host number is
// necessary but nowhere near sufficient; it says nothing about spill/alias codegen, which is a
// property of the compiled chain length, not the maths. v4's host numbers below are reported ONLY
// as fit quality (how well does a degree-5 poly approximate GELU_ref), not as evidence this will
// pass -- that requires a device run. Host (float32, this file's exact reordered formula; see
// golden.py): rel-L2 5.422e-04, max-abs 6.701e-03 vs GELU_ref over ax in [0,5] and over the full
// tested range |x|<=30 plus N(0,3) noise.
//
// HARDWARE PRIMITIVES USED, aie2p-verified via compile_check.sh: aie::abs(vector<float,N>) (32-bit
// generic aie2 path, no aie2p override), aie::min(vector<float,N>, vector<float,N>) (same
// op_min/op_max family relu2.cc's ReLU clamp uses, confirmed to return a vector directly, not an
// accum -- unlike aie::mul/aie::add, which can return an accum<accfloat,N> and therefore must
// always be assigned into a named vector before being passed as an argument to another aie::
// call). All four of abs/min/mul/add (plus the literal term 0.5*(x+abs(x))) were independently
// device-confirmed bit-exact by the coordinator's probe_abs_min.py -- they are NOT under suspicion;
// the defect that motivated v4 was chain length, not any primitive. NOT used, deliberately:
// aie::exp2, aie::tanh, aie::inv, aie::lut<4>/parallel_lookup (open ab/cd-layout question, as in
// sin.cc).
//
// PER-CALL WIDTH: 64 elements/call (mirrors sin.cc/snake.cc's proven-safe width). REGISTER
// PRESSURE: `x` and `ax` are dead once `s` is computed, so only THREE named f32 vectors (`axc`,
// `p`, `s`) are live across the 5-term Horner loop; `p` is the one accumulator reassigned in place.
// Unrolling DISABLED (`#pragma clang loop unroll(disable)`) -- sin.cc needs this even at six terms
// and this chain, while shortened, is still longer than sin_v's.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// Vector-level GELU: y = 0.5*(x+ax) - D(clamp(ax,0,5)), D a degree-5 Horner fit (see header).
// `s = 0.5*(x+ax)` is computed FIRST so `x`/`ax` are dead before the polynomial loop -- only
// `axc`, `p`, `s` are live across it. Exposed (like sin_v<N>) so composites can reuse it.
template <int N>
static inline ::aie::vector<float, N> gelu_erf_v(::aie::vector<float, N> x) {
  ::aie::vector<float, N> ax = ::aie::abs(x);
  // ax is already >= 0, so a single aie::min (no aie::max) clamps it to the fit domain [0, 5].
  ::aie::vector<float, N> axc = ::aie::min(ax, ::aie::broadcast<float, N>(5.0f));

  // s = 0.5*(x+ax), computed BEFORE the polynomial loop: `x` and `ax` are dead after this, so
  // they do not add to the live set across the Horner chain below (register-pressure fix, v3->v4;
  // see header -- v3 kept x/ax/axc/p/s all live simultaneously across a longer, 9-degree chain).
  ::aie::vector<float, N> s = ::aie::add(x, ax);
  s = ::aie::mul(s, ::aie::broadcast<float, N>(0.5f));

  // D(ax) Horner, degree 5 (6 coefficients; cut from v3's degree 9, see header). ONE accumulator
  // `p` reassigned in place. Every op assigns into a NAMED vector<float,N> before feeding the
  // next (aie::mul of two vectors can return an accum<accfloat,N>, not a vector; only an
  // assignment triggers the implicit narrowing -- measured via compile_check.sh).
  ::aie::vector<float, N> p = ::aie::broadcast<float, N>(0.002835879616732109f);
  p = ::aie::mul(p, axc);
  p = ::aie::add(p, ::aie::broadcast<float, N>(-0.043260768673460806f));
  p = ::aie::mul(p, axc);
  p = ::aie::add(p, ::aie::broadcast<float, N>(0.24472919119017997f));
  p = ::aie::mul(p, axc);
  p = ::aie::add(p, ::aie::broadcast<float, N>(-0.6094418780222849f));
  p = ::aie::mul(p, axc);
  p = ::aie::add(p, ::aie::broadcast<float, N>(0.5663938552943448f));
  p = ::aie::mul(p, axc);
  p = ::aie::add(p, ::aie::broadcast<float, N>(-0.004514140008753372f)); // p == D(axc)

  // y = s - D(axc). NO final clamp (see header: a v2 safety clamp masked the exact diagnostic
  // that broke that round open, so this version keeps the raw value visible).
  return ::aie::sub(s, p);
}

// WHAT HAS BEEN EXCLUDED AS THE CAUSE OF THIS BRICK'S DEVICE FAILURE, each measured 2026-07-31.
// Recorded so nobody re-runs them, and because four rewrites of the MATHS all chased the wrong
// thing: the maths is not what is wrong.
//   * the primitives -- abs / min / min(abs(x),5) / 0.5*(x+abs(x)) are each rel-L2 EXACTLY 0
//     in isolation (_verify/probe_abs_min.py);
//   * register pressure -- a degree-9 Horner (10 live vectors) with a COMPILE-TIME bound is green
//     at 1.273e-07 (_verify/probe_horner_degree.py), so chain length is not the trigger;
//   * the `static inline` helper spelling -- identical maths inline in the bound symbol vs behind
//     a `template<int N> static inline` helper are BIT-IDENTICAL at 1.515e-07
//     (_verify/probe_inline_helper.py). The task file's "macro-expand the body" constraint did not
//     reproduce here.
//   * the RUNTIME trip count -- the parent task root-caused exactly this for the sibling `sin`
//     brick ("sin_core's RUNTIME loop trip count corrupts after exactly 4 iterations"), so the
//     count was made a compile-time template parameter here. The device result was BIT-IDENTICAL
//     (9.538126245315652 both ways, since -DGELU_N=64 yields the same 4 chunks the runtime
//     division did). Refuted, and the ABI change was reverted as unjustified.
//   * chunks-per-call -- red at ONE chunk per call too (n=16, rel-L2 6.948e-01,
//     _verify/probe_gelu_chunk.py). Not a multi-iteration effect.
//
// THE SHARP REMAINING CLUE, and where the next session should start. In
// _verify/probe_inline_helper.py, a STANDALONE .cc containing this brick's exact degree-5
// coefficients (read from golden.py so they cannot drift), the same clamp, the same
// `0.5*(x+|x|) - D(min(|x|,5))` form, at ONE chunk per call, is GREEN at 1.515e-07 -- in BOTH the
// inline and the `template<int N> static inline` helper spelling, bit-identically. This file, with
// the same maths, is RED. So the defect is a difference between THIS TRANSLATION UNIT and that
// standalone one, not the algorithm, not the primitives, not the chain length, not the loop.
// BISECT gelu_erf.cc against that probe rather than touching the polynomial again.
//
// Symptom to bisect toward: large POSITIVE x returns exactly 0.0 (x=+11.9882 -> +0.00000e+00)
// while x=+1 is right to six digits. Note 50% of outputs being exactly 0 is NOT a clue -- the test
// pool is ~50% negative, where GELU legitimately underflows to 0. That coincidence cost time.
// ONE N-wide chunk per call, NO loop and NO runtime bound. Both are required, measured 2026-07-31.
// Volume comes from the harness's tile loop (bricklib streams n_tiles of N elements), so this costs
// nothing -- the iteration just moves from inside the kernel to the objectFIFO driver, which is
// where the resident-stream contract wants it anyway.
//
// The grid that forces this (probe_gelu_degree_chunks.py, this exact body, compile-time bounds):
//
//   Horner degree   live vecs    1 chunk        2 chunks
//         1             2        4.796e-09      4.493e-09   ok
//         2             3        3.458e-08      7.077e-01   FAIL
//         3             4        4.467e-08      8.651e-01   FAIL
//         5             6        1.151e-07      1.007e+01   FAIL
//         9            10        1.275e-07      6.048e+03   FAIL
//
// Every degree is correct at one chunk; only degree 1 survives two, and the error grows with both
// degree and chunk count. A runtime loop bound fails too, at ANY chunk count, and NOT because of
// the integer division -- passing the trip count in directly fails identically.
template <int N>
void gelu_erf_core(const float *restrict input, float *restrict output) {
  event0();
  ::aie::store_v(output, gelu_erf_v<N>(::aie::load_v<N>(input)));
  event1();
}

} // namespace route_b_bricks

extern "C" {

// GELU over EXACTLY 16 f32 elements -- one vector, no element count, no loop.
//
// The caller supplies volume by invoking this once per streamed tile; bricklib's rowwise/streamed
// rails already do exactly that. Taking an `n` and looping internally is what made every earlier
// version of this brick fail, at any n, for reasons that have nothing to do with the maths -- see
// gelu_erf_core's header for the measured grid and the six hypotheses excluded on device.
void gelu_erf_f32(float *input, float *output) {
  route_b_bricks::gelu_erf_core<16>(input, output);
}

} // extern "C"
