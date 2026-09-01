//===- conv_1d.cc ----------------------------------------------*- C++ -*-===//
//
// CAUSAL dilated 1-D convolution brick (route_b_kernels/bricks, group: conv).
//
//   y[co, t] = bias[co] + sum_ci sum_j w[ci, co, j] * x[ci, t - (k-1-j)*dilation]
//
// with x[.., <0] reading as zero. Reference semantics: s2.cpp/src/s2_codec.cpp:224-235. Every
// residual unit in the DAC codec decoder uses two of these -- one dilated (1, 3, 9) and one 1x1
// (s2_codec.cpp:381-393) -- so this brick plus snake is the whole residual unit.
//
// s2.cpp expresses it as left-pad-then-valid-conv; at stride 1 that is exactly (k-1)*dilation zeros
// on the left, so out_len == in_len. Evaluated here in the direct form instead: a kernel that
// materialised the padded copy would spend L1 on zeros, and the bound check is one comparison.
// golden.py asserts the two forms agree, and asserts causality directly, because a sign error in
// the shift is invisible to a shape check.
//
// ONE OUTPUT CHANNEL PER CALL. The residual-unit weights are [c_out, c_in, k] = 96*96*7 f32 =
// 258 KB and no core tile holds that, so the caller streams one output channel's slice at a time
// past a resident activation -- the same split the fused upsample stage uses, and for the same
// reason. The caller owns that decomposition; this kernel just does one channel.
//
// conv_1d_causal_core (below) is SCALAR and stays the correctness reference the vectorised form is
// gated against. This brick's gather is contiguous in t, so a vector form (conv_1d_causal_core_vec,
// below) is available directly. conv-transpose-1d's naive scatter form is NOT -- its stride-offset
// write is genuinely unaligned -- but conv_transpose_1d.cc's own conv_transpose_channel_core_vec
// gets there anyway, by reformulating the scatter into `stride` contiguous per-phase gathers
// instead of aligning it. See its own comment for how it stays alignment-safe.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// w_row: [c_in * k] weights for ONE output channel, laid out [ci*k + j]. With ggml_conv_1d's
//        [c_out, c_in, k] layout that slice is w[co] and is already contiguous -- no gather.
//        Note conv_transpose_1d uses the TRANSPOSED layout; see conv-1d/golden.py.
// x:     [c_in, t] activation.
// out:   [t] for that output channel.
static inline void conv_1d_causal_core(const float *restrict x, const float *restrict w_row,
                                       float bias, float *restrict out, int32_t c_in, int32_t k,
                                       int32_t t, int32_t dilation) {
  event0();
  for (int32_t p = 0; p < t; p++) {
    out[p] = bias;
  }
  for (int32_t ci = 0; ci < c_in; ci++) {
    const float *xr = x + (int32_t)(ci * t);
    const float *wr = w_row + (int32_t)(ci * k);
    for (int32_t j = 0; j < k; j++) {
      const float wv = wr[j];
      const int32_t shift = (k - 1 - j) * dilation;
      // Everything before `shift` reads padding, which is zero, so it contributes nothing --
      // start at the first output that has a real input rather than testing inside the loop.
      for (int32_t p = shift; p < t; p++) {
        out[p] += wv * xr[p - shift];
      }
    }
  }
  event1();
}

// VECTORISED form of conv_1d_causal_core. Same signature, same math, ADDITIVE -- the scalar core
// above is untouched and stays the reference this is gated against (see _verify/verify_conv_1d.py).
//
// Vectorises over t (output position p), N lanes at a time. The gather is contiguous in t (dilation
// only changes the scalar `shift`, never the read stride -- see file header), but `shift` is almost
// never a multiple of N, so a naive aligned-output/misaligned-input load would issue an unaligned
// load_v. On this toolchain that SNAPS to the aligned base and silently returns the wrong data
// instead of erroring (the same failure dwconv1d_shift's header records for its own FIR window,
// route_b_kernels/dwconv1d/dwconv1d.cc) -- so every load_v here is on an N-aligned address, and the
// misaligned window is built in-register with shuffle_down_fill, exactly that kernel's technique.
//
// dwconv1d_shift affords a padded scratch buffer because T/K/P are compile-time there; this core
// keeps c_in/k/t/dilation runtime, matching conv_1d_causal_core, so there is no compile-time bound
// to size a buffer from -- and it turns out not to need one. Per output chunk [oc, oc+N) the read
// window is x[oc-shift .. oc-shift+N), and since 0 <= oc <= t-N and shift >= 0, that window starts
// at most N-1 elements before x's own base and ends at most at x's last element. One shared,
// always-zero length-N vector stands in for that left slack; nothing is ever read outside
// x[0, c_in*t).
//
// Three cases per (ci, j, oc), decided by comparing oc/oc+N against shift:
//   oc + N <= shift : chunk is entirely left of this tap's causal window -> contributes 0, skip.
//   oc     >= shift : chunk is entirely inside it -> read_base = oc-shift >= 0: one aligned load if
//                      read_base is itself N-aligned, else an aligned pair + shuffle_down_fill.
//   otherwise       : the chunk straddles `shift` -> read_base in [-(N-1), -1]; splice the shared
//                      zero vector against x's own first N-wide chunk.
// The aligned-pair case never reads past x's t elements: read_base <= t - N always. A second vector
// is only read when read_base isn't already N-aligned, and since t % N == 0, the one read_base that
// IS N-aligned at the top of the range is read_base == t - N itself -- so whenever a second vector
// is needed, read_base < t - N, which places its aligned floor at or before t - 2N, keeping the pair
// inside [0, t). PRECONDITION: t must be a multiple of N (the same contract every vectorised f32
// brick in this tree carries -- softmax/layernorm/rmsnorm/relu2 all require cols % N == 0).
template <int N>
static inline void conv_1d_causal_core_vec(const float *restrict x, const float *restrict w_row,
                                            float bias, float *restrict out, int32_t c_in,
                                            int32_t k, int32_t t, int32_t dilation) {
  event0();
  const ::aie::vector<float, N> zero_v = ::aie::zeros<float, N>();
  const ::aie::vector<float, N> bias_bv = ::aie::broadcast<float, N>(bias);

  for (int32_t oc = 0; oc < t; oc += N) {
    ::aie::accum<accfloat, N> acc;
    acc.from_vector(bias_bv);
    for (int32_t ci = 0; ci < c_in; ci++) {
      const float *xr = x + (int32_t)(ci * t);
      const float *wr = w_row + (int32_t)(ci * k);
      for (int32_t j = 0; j < k; j++) {
        const int32_t shift = (k - 1 - j) * dilation;
        if (oc + N <= shift) continue; // entirely left of the causal window: no contribution
        const ::aie::vector<float, N> wbv = ::aie::broadcast<float, N>(wr[j]);
        const int32_t read_base = oc - shift;
        if (read_base < 0) {
          const int32_t r = read_base + N; // in [1, N-1]
          const ::aie::vector<float, N> b = ::aie::load_v<N>(xr);
          acc = ::aie::mac(acc, ::aie::shuffle_down_fill(zero_v, b, (unsigned)r), wbv);
        } else {
          const int32_t ra = (read_base / N) * N; // floor to N (read_base >= 0)
          const int32_t r = read_base - ra;
          if (r == 0) {
            acc = ::aie::mac(acc, ::aie::load_v<N>(xr + ra), wbv);
          } else {
            const ::aie::vector<float, N> a0 = ::aie::load_v<N>(xr + ra);
            const ::aie::vector<float, N> a1 = ::aie::load_v<N>(xr + ra + N);
            acc = ::aie::mac(acc, ::aie::shuffle_down_fill(a0, a1, (unsigned)r), wbv);
          }
        }
      }
    }
    ::aie::store_v(out + oc, acc.template to_vector<float>());
  }
  event1();
}

// Applies ONE tap's contribution (channel ci, dilation tap j) to acc_lane. Identical 3-case
// alignment logic to conv_1d_causal_core_vec's inner body (see that function's own comment for why
// each case is alignment-safe) -- factored out so the multi-accumulator form below can call it once
// per lane, per tap, without duplicating the case split.
template <int N>
static inline void conv1d_vec_acc_tap(::aie::accum<accfloat, N> &acc_lane, const float *restrict x,
                                       const float *restrict w_row, int32_t ci, int32_t k,
                                       int32_t t, int32_t j, int32_t oc, int32_t dilation,
                                       const ::aie::vector<float, N> &zero_v) {
  const int32_t shift = (k - 1 - j) * dilation;
  if (oc + N <= shift) return; // entirely left of the causal window: no contribution
  const float *xr = x + (int32_t)(ci * t);
  const float *wr = w_row + (int32_t)(ci * k);
  const ::aie::vector<float, N> wbv = ::aie::broadcast<float, N>(wr[j]);
  const int32_t read_base = oc - shift;
  if (read_base < 0) {
    const int32_t r = read_base + N; // in [1, N-1]
    const ::aie::vector<float, N> b = ::aie::load_v<N>(xr);
    acc_lane = ::aie::mac(acc_lane, ::aie::shuffle_down_fill(zero_v, b, (unsigned)r), wbv);
  } else {
    const int32_t ra = (read_base / N) * N; // floor to N (read_base >= 0)
    const int32_t r = read_base - ra;
    if (r == 0) {
      acc_lane = ::aie::mac(acc_lane, ::aie::load_v<N>(xr + ra), wbv);
    } else {
      const ::aie::vector<float, N> a0 = ::aie::load_v<N>(xr + ra);
      const ::aie::vector<float, N> a1 = ::aie::load_v<N>(xr + ra + N);
      acc_lane = ::aie::mac(acc_lane, ::aie::shuffle_down_fill(a0, a1, (unsigned)r), wbv);
    }
  }
}

// Applies tap j to lanes [Lane, NACC) of acc, channel ci0+Lane .. ci0+NACC-1, via COMPILE-TIME
// template recursion -- not a runtime `for` loop. This is deliberate: a plain `for (int lane = 0;
// lane < NACC; lane++)` loop here measured as NOT unrolled by Peano at -O2 (verified by compiling
// both forms to aie2p asm -- the runtime-loop version keeps a 4-deep nested loop with the lane trip
// count still a branch at runtime, vmul.f/vadd.f/vmsc.f counts identical across NACC=4/6/8, and only
// 5 distinct accfloat registers touched regardless of NACC), so the NACC independent accumulators
// stayed in ONE small runtime-indexed array instead of NACC separate registers -- exactly what this
// brick exists to avoid. Recursing on a template int, by contrast, generates NACC literal call
// sites with a compile-time-constant lane/array-index at each one, which is what lets the compiler
// promote `acc` to NACC independent registers and lets its local scheduler interleave their
// independent op-chains within this one straight-line block. See conv_1d_causal_core_vec_acc's own
// comment for why that interleaving is the point.
template <int N, int Lane, int NACC>
struct conv1d_vec_acc_unroll {
  static inline void run(::aie::accum<accfloat, N> (&acc)[NACC], const float *restrict x,
                         const float *restrict w_row, int32_t ci0, int32_t k, int32_t t, int32_t j,
                         int32_t oc, int32_t dilation, const ::aie::vector<float, N> &zero_v) {
    conv1d_vec_acc_tap<N>(acc[Lane], x, w_row, ci0 + Lane, k, t, j, oc, dilation, zero_v);
    conv1d_vec_acc_unroll<N, Lane + 1, NACC>::run(acc, x, w_row, ci0, k, t, j, oc, dilation, zero_v);
  }
};
template <int N, int NACC>
struct conv1d_vec_acc_unroll<N, NACC, NACC> {
  static inline void run(::aie::accum<accfloat, N> (&)[NACC], const float *restrict,
                         const float *restrict, int32_t, int32_t, int32_t, int32_t, int32_t,
                         int32_t, const ::aie::vector<float, N> &) {}
};

// MULTI-ACCUMULATOR form of conv_1d_causal_core_vec. Same signature plus NACC, same math, ADDITIVE
// -- the scalar core and the single-accumulator vector core above are both untouched and stay
// references this is gated against.
//
// WHY: conv_1d_causal_core_vec carries exactly ONE `aie::accum<accfloat, N> acc` through the whole
// (ci, j) tap loop (c_in*k taps), `acc = aie::mac(acc, ..., wbv)` each time -- a single serial
// recurrence. Confirmed by compiling that function to aie2p asm (clang -S): its innermost loop
// carries ZERO postpipeliner remarks (neither Passed nor Missed, in the -fsave-optimization-record
// YAML) -- the loop is not software-pipelined at all, so nothing overlaps between taps. And aie2p
// has no native f32 vector MAC: __AIE_API_FP32_EMULATION__'s mac_elem_32_accuracy_safe splits each
// f32 operand into a 3-term bf16 decomposition and does 9 cross products + 8 reduction adds + the
// splitting vmsc's per tap (visible in the compiled asm as vconv.bf16.fp32 / vmsc.f / vmul.f /
// vadd.f, ~28 real vector ops for what the C++ source spells as one aie::mac). Per
// AIE2PGenSchedule.td (llvm-aie, aie2p/AIE2PGenSchedule.td), the itinerary classes those lower to
// -- II_VMUL_f_vmul_bf_vmul_bf_core_X_X and II_VMSC_f_vmac_bf_vmul_bf_core_X_X, the plain
// (non-complex) float mul/mac classes -- both report `dst` operand latency 6 cycles. With the loop
// unpipelined, EVERY tap pays that ~28-op, multi-times-6-cycle critical path with no overlap to the
// next tap, which is what the device sweep's 0.185 MAC/cyc (~86 cycles/tap) is paying for.
//
// FIX: split the c_in axis into NACC independent lanes, each with its OWN accumulator, so the
// recurrence chain length drops from c_in*k taps to (c_in/NACC)*k taps per lane, and -- because all
// NACC lanes' work for a given (ci0, j) sits in ONE straight-line block (conv1d_vec_acc_unroll,
// above), unlike separate loop bodies -- the ordinary (non-modulo) instruction scheduler can
// interleave the NACC independent mac chains and hide each one's latency behind the others, with no
// software-pipelining pragma and no declared minimum trip count required (c_in and k stay fully
// runtime, same contract as both cores above). NACC is chosen by the caller; the ~6-cycle dst
// latency measured above is the concrete number NACC needs to cover to run resource- rather than
// latency-bound.
//
// c_in is not required to be a multiple of NACC: the main loop covers the largest multiple of NACC
// channels, and the remainder (< NACC channels) is folded into lane 0 with conv1d_vec_acc_tap
// directly, so this is correct for ANY c_in, including c_in < NACC (which degenerates to that
// single-accumulator path entirely).
template <int N, int NACC>
static inline void conv_1d_causal_core_vec_acc(const float *restrict x, const float *restrict w_row,
                                                float bias, float *restrict out, int32_t c_in,
                                                int32_t k, int32_t t, int32_t dilation) {
  static_assert(NACC >= 1, "need at least one accumulator");
  // c_in is runtime, so `c_in / NACC` cannot be constant-folded -- for a non-power-of-two NACC
  // Peano lowers it to a call to the softdiv library (__modsi3, confirmed by compiling NACC=6:
  // that symbol is undefined in this bare-metal core and would only surface as a LINK failure,
  // invisible to compile_check.sh's -c-only compile). Power-of-two NACC strength-reduces to a
  // shift/mask instead, so require it here rather than let a future caller hit that at link time.
  static_assert((NACC & (NACC - 1)) == 0, "NACC must be a power of two (c_in/NACC must strength-reduce, not call __modsi3)");
  event0();
  const ::aie::vector<float, N> zero_v = ::aie::zeros<float, N>();
  const ::aie::vector<float, N> bias_bv = ::aie::broadcast<float, N>(bias);
  const int32_t c_in_main = (c_in / NACC) * NACC; // largest multiple of NACC not exceeding c_in

  for (int32_t oc = 0; oc < t; oc += N) {
    ::aie::accum<accfloat, N> acc[NACC];
    acc[0].from_vector(bias_bv);
    for (int lane = 1; lane < NACC; lane++) acc[lane] = ::aie::zeros<accfloat, N>();

    for (int32_t ci0 = 0; ci0 < c_in_main; ci0 += NACC) {
      for (int32_t j = 0; j < k; j++) {
        conv1d_vec_acc_unroll<N, 0, NACC>::run(acc, x, w_row, ci0, k, t, j, oc, dilation, zero_v);
      }
    }

    // Remainder channels (c_in % NACC of them, always < NACC): same per-tap math, folded into
    // lane 0 -- summed into the total below regardless.
    for (int32_t ci = c_in_main; ci < c_in; ci++) {
      for (int32_t j = 0; j < k; j++) {
        conv1d_vec_acc_tap<N>(acc[0], x, w_row, ci, k, t, j, oc, dilation, zero_v);
      }
    }

    // Combine the NACC partials. Reassociation (NACC independent sums instead of one c_in*k-term
    // chain) is safe here: f32, and a host simulation against the float64 golden (real S2-Pro
    // weights, dilations 1/3/9, k=1 and k=7, NACC swept 1/4/6/8 even though only powers of two are
    // reachable through this template -- the numerics do not depend on the power-of-two guard above)
    // shows it does not move rel-L2 in a harmful direction: splitting one long serial f32 sum into
    // NACC shorter ones consistently REDUCED rel-L2 (fewer roundings per chain), from ~1e-7 (NACC=1)
    // to ~1-2e-7 (NACC=4/8), both far under the 3e-2 gate.
    ::aie::accum<accfloat, N> total = acc[0];
    for (int lane = 1; lane < NACC; lane++) total = ::aie::add(total, acc[lane]);
    ::aie::store_v(out + oc, total.template to_vector<float>());
  }
  event1();
}

} // namespace route_b_bricks
