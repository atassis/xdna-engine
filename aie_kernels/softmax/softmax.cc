//===- softmax.cc ----------------------------------------------*- C++ -*-===//
//
// GENERIC AIE2P brick: softmax over one row (op-TYPE `softmax{axis}`, no existing instance in this
// catalog before this file). A reduce-then-normalize row op, the same family as norm{LN|RMS}
// (layernorm.cc / rmsnorm.cc), but THREE passes instead of two: max, sum-of-exp, normalize.
//
// Reference (scripts/codec_quantizer_ref.py::_softmax_over_keys, ~line 357, the assigned oracle):
//   s = scores.astype(f64); s = s - s.max(axis=1, keepdims=True); e = exp(s); return e/e.sum(...)
// `_softmax_over_keys` reduces over axis=1 of a [n_head, T_k, T_q] tensor (keys); its consumer
// rvq_transformer (~line 366) adds an additive mask (0 / -1e9 from _causal_window_mask, ~line 323)
// BEFORE calling it, so -1e9-masked entries are the normal input, not an edge case.
// Cross-checked (golden.py __main__) against scripts/s2_ar_ref.py::causal_attention (~line 582),
// an INDEPENDENT second formulation (mask = -inf, reduces over axis=-1 of a [n_head,T_q,T_k]
// tensor). Both are literally the same algorithm (max-subtract, exp, normalize) modulo which array
// axis is "the reduction axis" and the mask sentinel (-1e9 vs -inf) -- golden.py transcribes both
// verbatim and confirms they agree to float64 precision on identical row data.
//
// CONTRACT: one call computes softmax over ONE ROW of `cols` contiguous values (the reduction
// axis). Axis-agnostic like every other brick here: the caller lays out memory so the group being
// normalized is contiguous (e.g. one (head, query) column's T_k keys, already transposed on host
// if the source tensor's natural layout doesn't already make keys contiguous -- that transpose is
// out of scope for this brick, same division of labor as layernorm.cc's D-axis contract). The
// max-subtract is IMPLEMENTED HERE, on device, not assumed away by the caller.
//
// ---------------------------------------------------------------------------------------------
// EXP ON DEVICE. There is no hardware TRANSCENDENTAL instruction we trust for this: aie_api DOES
// expose `aie::exp2<TR,N>(vector<float,N>)` and it DOES instantiate on aie2p for TR=bfloat16 (SFU
// `::exp2` on an accfloat accumulator -- verified via compile_check.sh on a throwaway probe, see
// worklog). But: (1) TR=float is gated `requires(arch::is(AIE_MLv2))` in aie.hpp -- aie2p compiles
// as XDNA2 (__AIE_ARCH__==21), not AIE_MLv2 (==22), so f32-precision hardware exp2 is NOT available
// on this target at all, only a bf16-output path; (2) this SFU family is the same class of hardware
// (fixed-function transcendental unit, sin/cos/tanh/exp2 all share the ElementaryOp dispatch) that
// sin.cc's header already documents rejecting for `aie::sin` -- which ALSO compiles and instantiates
// for float on aie2p (aie.hpp:7304) yet sin.cc writes its own Taylor polynomial instead of calling
// it, because the sibling `aie::lut`-based approach in this same repo was measured returning
// permuted entries on real hardware (probe_linear_approx_abcd.py) despite compiling cleanly --
// i.e. compiling is not evidence of runtime correctness for this hardware family here, and there
// is no device access in this task to independently re-verify aie::exp2's behavior at the extreme
// negative arguments (-1e9-scale, post max-subtract) softmax actually needs. Given sin.cc's own
// precedent chose the polynomial route for exactly this class of risk, softmax does the same:
// write `exp2_v<N>`, host-verify it exhaustively (see below and the worklog), and never touch
// `aie::exp2`/`aie::sin`/`aie::lut`.
//
// No exp2/expf vector helper existed anywhere in this tree before this file (grepped
// route_b_kernels/ for exp2/expf/exp2f_vec/expf_vec: no hits) -- `exp2_v<N>` here is the first
// instance of the upstream-brick-exp2f-vec primitive, defined inline (this brick owns no other
// file to host it in). A natural follow-up is factoring it into its own bricks/exp2/ file, exactly
// as sin.cc's sin_v was later composed by snake.cc via `#include "../sin/sin.cc"`.
//
// exp2_v<N> implements 2^y = 2^n * 2^f, n=round(y) (integer), f=y-n in [-0.5, 0.5):
//   * ARGUMENT CLAMP: y is first clamped to >= -120 before anything else. Domain is y<=0 always
//     (softmax always calls this with y = (x-row_max)*log2(e), and x<=row_max), so the only
//     direction that needs guarding is deep-negative -- exactly the "masked entry" case
//     (x-row_max ~ -1e9, y ~ -1.4e9). Reconstructing 2^n via exponent-field bit manipulation
//     (below) is only well-defined for n in the normal float range; feeding it the raw
//     billion-scale n would EITHER overflow the round-trick's valid range (see next bullet) or
//     wrap the exponent field into a bogus large FINITE value -- exactly the "underflow must be
//     explicit, not an exponent-field wrap into NaN/Inf" hazard this task calls out. Clamping to
//     -120 forces every masked entry to evaluate the SAME safe point (2^-120 ~= 7.52e-37), which
//     is correct: any row that reaches softmax always has its own max element contributing
//     exp2(0)=1 to the sum, so a competitor at 2^-120 relative to a sum >= 1 is many orders below
//     float32 epsilon (1.19e-7) -- indistinguishable from the true 0 the math wants. Host-checked
//     with x as low as -1e9 and with -inf: clamp -> 2^-120 exactly, never NaN/Inf (see worklog).
//
//   * ROUND -- REVISED after a DEVICE FAILURE. The first version rounded via the classic float
//     magic-constant identity, n = (y + 1.5*2^23) - 1.5*2^23 (relying on the add forcing f32
//     rounding to lose y's sub-ULP bits, then the subtract recovering only the rounded integer).
//     That version passed compile_check.sh and every host-side numpy check (host float32 DOES
//     round this way), but FAILED catastrophically on the real device gate: rel_l2=2.075e+30,
//     negative row sums, and even the trivial uniform-row case (all-equal input, max-subtract
//     gives all zeros, expected output exactly 1/cols) returned 0.0 instead. run2run=0.00e+00
//     (fully deterministic, not a race). The main session independently reproduced the same root
//     cause on sin.cc: changing its magic constant from 8388608.0f to 12582912.0f (a change that
//     moves the host float32 result from wrong to correct, see below) left the ON-DEVICE sin
//     output BIT-IDENTICAL, while an unrelated mutation (scale sin's output by 3) DID change the
//     device result -- proving edits reach the device and this specific identity just does not
//     round on this hardware/toolchain (most likely an extended-precision intermediate for
//     chained vector add/sub, or a reassociating compile-time fold of `(y+C)-C` -> `y`; either
//     way `f = y - n` silently evaluates to ~0 for EVERY row, not just uniform, so pass 2/3 both
//     degenerate to `pow2f == 1.0` and the wrong integer part alone drives the result).
//     FIX: do not rely on any add/sub identity for rounding at all. `aie::to_fixed<int32>`
//     (Float2Fix) is a dedicated hardware convert instruction, not an arithmetic identity, and it
//     is ALREADY proven correct on this exact device by cast-quant-bf16-int8.cc's
//     quantize_bf16_to_int8_row (that brick's own header records replacing a DIFFERENT broken
//     bit-reinterpret path with `to_fixed` and measuring the fix on device). That file also
//     establishes the precondition this needs: the VECTOR `to_fixed` overload's rounding is
//     governed by the software rounding-mode register (unlike the SCALAR `to_fixed(float,...)`
//     overload, whose own doc block says global rounding settings have no effect on it -- two
//     different hardware paths), so `aie::set_rounding(aie::rounding_mode::conv_even)` is called
//     once at the top of softmax_core before any to_fixed call, exactly as that brick does before
//     its own to_fixed<int8_t>. `n_int = to_fixed<int32>(y)` then gives round(y) directly, in ONE
//     hardware instruction, with no magic constant and nothing for the compiler to fold away.
//   * n AS A FLOAT, EXACTLY, WITHOUT to_float. The polynomial below still needs `f = y - n` as an
//     ordinary float subtraction, which needs n back in float form -- and plain `aie::to_float`
//     (Fix2Float) on a vector<int32,N> is the direction sin.cc's own header already documents as
//     not instantiating on aie2p. Reconstructed instead via an EXACT bit trick that needs no
//     value-conversion primitive and, critically, no lossy rounding step for correctness (so it
//     cannot suffer the same failure as the magic-constant trick above): shift n_int into a known
//     non-negative range with a plain INTEGER add (exact by definition, two's complement handles
//     the negative n_int case correctly), integer-add that into the literal bit pattern of 2^23
//     (0x4B000000, exact), reinterpret the sum as a float via `cast_to<float>` (a bitcast, zero
//     computation) -- this is EXACTLY 2^23 + (n_int+256) because the low 23 bits of 0x4B000000 are
//     zero and n_int+256 never exceeds them, then subtract the exact float constant 8388608+256 to
//     recover n_int. That last subtraction's true result is exactly representable in f32 (n_int is
//     a small integer), so unlike the broken trick above it needs NO rounding at all to be
//     correct -- an extended-precision accumulator computes an exact result exactly too, it only
//     breaks tricks that depend on discarding precision. Host-verified bit-exact (0 error) for
//     every n_int in [-120, 0] (see worklog); the full exp2_v pipeline built on this reconstruction
//     re-measures the SAME 2.378e-07 max relative error as the original (broken-on-device) version
//     did on host, confirming the fix changes nothing about the math, only how it reaches silicon.
//   * RECONSTRUCT 2^n: unchanged from the original design and NOT implicated in the device failure
//     (integer add + `aie::upshift` + `cast_to<float>` only, no rounding-dependent step anywhere
//     in this part). n_int is already the exact integer from `to_fixed<int32>` above; bias by the
//     IEEE-754 exponent bias (127), left-shift into the exponent field with `aie::upshift(.., 23)`,
//     then `cast_to<float>` reinterprets the resulting bit pattern as a float -- mantissa 0,
//     exponent = n_int+127, i.e. exactly 2^n_int.
//   * 2^f: Taylor series of 2^f = e^(f*ln2) about f=0, Horner in f (one accumulator reassigned in
//     place, constants materialised at point of use -- same register-pressure discipline as
//     sin.cc's sin_v). Coefficients unchanged from the original design (fitting is independent of
//     how n/f are derived). Nominal domain is f in [-0.5, 0.5) since n_int=round(y), but the same
//     degree-6 (7-term) poly was host-verified accurate to 2.8e-5 max relative error over the much
//     wider f in [-1, 1] too (see worklog) -- deliberate slack in case `to_fixed`'s rounding turns
//     out not to be exactly round-half-to-even in practice; still far inside the 3e-2 device gate
//     either way.
//
// ROW WIDTH PER CALL: this brick's own verify (bricks/_verify/verify_softmax.py) uses cols=64,
// matching sin.cc's OWN measured ceiling ("exact at 64 floats per call and wrong above that").
// exp2_v<N> is heavier than sin_v (round-trick + int32 detour + 7-term Horner vs sin_v's 6-term
// odd-only Horner), so if 64 was sin_v's proven ceiling, 64 is not a number to exceed here without
// new evidence -- there is no device access in this task to measure a higher one. Also notable:
// scripts/codec_quantizer_ref.py's own module docstring (kernel-map note #3) states this brick's
// actual first consumer runs at "T<=65 keys" -- i.e. the conservative 64-wide call this file is
// verified at is close to a real deployment width, not an arbitrarily small test size. (A separate
// instruction for this task cited "c=1024" for this segment; that number could not be reconciled
// against the T<=65 figure documented in-repo and is flagged here rather than silently picked.)
// If a caller needs a wider row, the brick stays mathematically correct (cols is a runtime param,
// any multiple of N) but is UNVALIDATED above 64 pending a device run -- watch for sin.cc's own
// failure signature (plausible-looking but ALIASED results, not a crash) if you push past it.
//
// NO SCRATCH BUFFER. softmax structurally needs exp(x-max) twice (once to sum, once to normalize),
// but rather than cache it in a new on-chip array (the alignas(64)/L1-scratch hazard this task
// flags), this kernel just RECOMPUTES exp2_v a second time from `input`, re-read a third time
// total -- the exact multi-pass-over-`input` idiom ln_2pass.cc/layernorm.cc/rmsnorm.cc already use
// (2 reads there, 3 here) instead of caching intermediates. Per this project's own optimization
// doctrine compute is next to free here (32 idle cores) while a wrongly-aliased or misaligned new
// local buffer is a proven, expensive failure mode -- so paying one extra pass over an already-
// resident `cols`-length row to avoid introducing any new buffer is the safer trade, not a missed
// optimization.
//
// NO STATIC L1 STATE (none used). No local scratch array (none used, see above, so alignas(64) does
// not apply here). Every #pragma clang loop unroll(disable) mirrors sin.cc: this kernel's per-chunk
// body is heavier than sin_v's, so the same spill risk applies, if not more.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// Generic vector-level 2^y, y <= 0 expected (softmax always calls this post max-subtract; see
// the clamp note above for what happens if a caller ever violates that). Composable like sin_v:
// exposed at namespace scope so a future kernel needing 2^x (not just softmax) can reuse it
// without a scratch buffer or second pass, same intent as sin.cc's sin_v.
template <int N>
static inline ::aie::vector<float, N> exp2_v(::aie::vector<float, N> y) {
  // clamp: see header. -120 keeps 2^n in the safe NORMAL float exponent range with wide margin
  // (smallest normal exponent is -126) and makes every masked/very-negative input converge to the
  // exact same negligible constant (2^-120) instead of risking an exponent-field wrap.
  y = ::aie::max(y, ::aie::broadcast<float, N>(-120.0f));

  // n_int = round(y) via a dedicated hardware convert instruction, NOT the add/sub magic-constant
  // identity (see header ROUND for why that identity does not round on this device -- it passed
  // every host check and still failed the device gate). Rounding mode must be set explicitly
  // (softmax_core does this once, before any pass calls exp2_v) -- the vector to_fixed overload
  // honors aie::set_rounding, unlike the scalar overload.
  ::aie::vector<int32, N> ni = ::aie::to_fixed<int32>(y);

  // n as a float, EXACTLY, via an integer-only construction (see header): shift ni into a known
  // non-negative range with a plain integer add (exact for any sign of ni), integer-add that into
  // the literal bit pattern of 2^23 (0x4B000000; the low 23 bits are zero so this cannot carry
  // into the exponent field as long as ni+256 < 2^23, true for our clamped ni in [-120,0]),
  // reinterpret as float (exact bitcast) -- giving exactly 2^23 + (ni+256) -- then subtract the
  // exact float constant 8388864.0f (=2^23+256) to recover ni. That last step's true result is
  // exactly representable (ni is a small integer), so it needs no rounding to be correct, unlike
  // the broken trick this replaces.
  ::aie::vector<int32, N> ni_shift = ::aie::add(ni, ::aie::broadcast<int32, N>(256));
  ::aie::vector<int32, N> bits = ::aie::add(ni_shift, ::aie::broadcast<int32, N>(0x4B000000));
  ::aie::vector<float, N> big = bits.template cast_to<float>();
  ::aie::vector<float, N> n = ::aie::sub(big, ::aie::broadcast<float, N>(8388864.0f));
  ::aie::vector<float, N> f = ::aie::sub(y, n);

  // 2^n: bias ni by the IEEE-754 exponent bias (127), shift into the exponent field, reinterpret
  // as float. Integer add + shift + bitcast only -- not implicated in the device failure above.
  ::aie::vector<int32, N> biased = ::aie::add(ni, ::aie::broadcast<int32, N>(127));
  ::aie::vector<int32, N> shifted = ::aie::upshift(biased, 23u);
  ::aie::vector<float, N> pow2n = shifted.template cast_to<float>();

  // 2^f, f in [-0.5, 0.5): degree-6 Taylor of e^(f*ln2), Horner, one accumulator reassigned in
  // place (register-pressure discipline per sin.cc). Host-measured max rel error 2.4e-7 (see
  // header / worklog) -- far inside the 3e-2 device gate.
  ::aie::vector<float, N> p = ::aie::mul(f, ::aie::broadcast<float, N>(1.5403530393e-04f));
  p = ::aie::add(p, ::aie::broadcast<float, N>(1.3333558146e-03f));
  p = ::aie::mul(p, f);
  p = ::aie::add(p, ::aie::broadcast<float, N>(9.6181291076e-03f));
  p = ::aie::mul(p, f);
  p = ::aie::add(p, ::aie::broadcast<float, N>(5.5504108665e-02f));
  p = ::aie::mul(p, f);
  p = ::aie::add(p, ::aie::broadcast<float, N>(2.4022650696e-01f));
  p = ::aie::mul(p, f);
  p = ::aie::add(p, ::aie::broadcast<float, N>(6.9314718056e-01f));
  p = ::aie::mul(p, f);
  p = ::aie::add(p, ::aie::broadcast<float, N>(1.0f));

  return ::aie::mul(p, pow2n);
}

// Generic softmax{axis} core. One call = one row of `cols` contiguous values (the reduction
// axis). f32 in / f32 out. `cols` must be a multiple of N. Three passes over `input`, no scratch
// buffer (see header "NO SCRATCH BUFFER").
template <int N>
void softmax_core(const float *restrict input, float *restrict output, int32_t cols) {
  event0();
  // Governs exp2_v's to_fixed<int32> rounding (round-half-to-even). Set ONCE per call, matching
  // cast_quant_bf16_int8.cc's own use of to_fixed -- see exp2_v's ROUND comment for why this is
  // required (the vector to_fixed overload honors this register; the scalar overload does not).
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const int chunks = cols / N;

  // pass 1: row_max = max(x). Running vector max, reassigned in place; one reduce at the end.
  ::aie::vector<float, N> max_v = ::aie::broadcast<float, N>(-3.389e38f); // ~ -FLT_MAX
  #pragma clang loop unroll(disable)
  for (int i = 0; i < chunks; i++) {
    max_v = ::aie::max(max_v, ::aie::load_v<N>(input + i * N));
  }
  const float row_max = ::aie::reduce_max(max_v);
  ::aie::vector<float, N> max_bv = ::aie::broadcast<float, N>(row_max);

  // pass 2: row_sum = Sigma(exp2((x-row_max)*log2e)). log2e materialised fresh each pass (not
  // hoisted across passes) per the "materialise constants at point of use" discipline.
  ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
  #pragma clang loop unroll(disable)
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), max_bv);
    ::aie::vector<float, N> y = ::aie::mul(d, ::aie::broadcast<float, N>(1.4426950408889634f));
    sum_v = ::aie::add(sum_v, exp2_v<N>(y));
  }
  const float row_sum = ::aie::reduce_add(sum_v);
  // row_sum >= 1 always: the row's own max element contributes exp2(0)=1 to the sum, so a plain
  // scalar reciprocal is safe (no near-zero-denominator case) -- same scalar-divide idiom
  // rmsnorm.cc already uses for `/float(cols)`.
  const float inv_sum = 1.0f / row_sum;
  ::aie::vector<float, N> inv_bv = ::aie::broadcast<float, N>(inv_sum);

  // pass 3: out = exp2((x-row_max)*log2e) * inv_sum. Recomputes exp2_v rather than caching pass
  // 2's values (see header "NO SCRATCH BUFFER").
  #pragma clang loop unroll(disable)
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), max_bv);
    ::aie::vector<float, N> y = ::aie::mul(d, ::aie::broadcast<float, N>(1.4426950408889634f));
    ::aie::vector<float, N> e = exp2_v<N>(y);
    ::aie::vector<float, N> o = ::aie::mul(e, inv_bv);
    ::aie::store_v(output + i * N, o);
  }
  event1();
}

} // namespace route_b_bricks

extern "C" {

// softmax over ONE ROW of `cols` contiguous f32 values (the reduction axis; caller-transposed
// into place, see header CONTRACT). cols must be a multiple of 16.
void softmax_f32(float *input, float *output, int32_t cols) {
  route_b_bricks::softmax_core<16>(input, output, cols);
}

} // extern "C"
