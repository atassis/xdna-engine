//===- gatedeltanet.cc ------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// SPIKE brick: gated delta-rule linear-attention recurrence (op-TYPE
// `recurrence{delta,gated}`), studied against dwconv1d.cc's brick shape
// (resident [tile,D] stream, generic op over compile-time dims, extern "C"
// entry points) -- see route_b_kernels/dwconv1d/dwconv1d.cc for the sibling
// COMPUTE-FIR brick this one borrows its layout conventions from.
//
// WHY THIS IS A SPIKE, NOT A LANDED BRICK: causal conv1d (dwconv1d) is a
// SLIDING-WINDOW recurrence -- output t only needs a fixed K-wide halo of
// INPUT, so it vectorizes across time (aie::sliding_mul / shuffle_down_fill,
// L=16/32 time-steps per issue). The gated delta rule below is a STATE
// recurrence -- output t needs the FULL accumulated state S_{t-1}, which
// itself depends on every step 0..t-1. There is no time-parallel brick for
// this on AIE2P; the only vectorizable axis is the D_V (value) lane dimension
// *within* one step. This file is the "what does that look like" probe, not
// a claim that it is fast. See the findings note (adjacent) for the read.
//
// MATH (gated delta rule; one head, one resident state tile):
//   State S_t in R^{D_K x D_V} (row i = key-dim i, D_V-wide per-lane row).
//   Per step t, given q_t,k_t in R^{D_K}, v_t in R^{D_V}, and two ALREADY-
//   ACTIVATED scalar gates alpha_t in (0,1] (decay) and beta_t in (0,1)
//   (write strength) -- gate activation (sigmoid/softplus on raw logits) is
//   its OWN elementwise op-type, not fused in here, same separation-of-
//   concerns dwconv1d.cc's SILU note argues for (fused epilogues on a
//   per-row-loop kernel miscompiled on this toolchain; keep gates as a
//   separate producer):
//
//     pred_t   = S_{t-1}^T k_t                      (R^{D_V}; a K-reduction)
//     err_t    = v_t - pred_t                        (the delta / correction)
//     S_t[i,:] = alpha_t * S_{t-1}[i,:] + beta_t * k_t[i] * err_t   (i=0..D_K-1)
//     o_t      = S_t^T q_t                            (R^{D_V}; same reduction)
//
// This is the single-scalar-gate member of the Gated DeltaNet family (Yang et
// al., "Gated Delta Networks: Improving Mamba2 with Delta Rule"): alpha_t is a
// per-step (per-head) decay, beta_t the delta-rule write strength; S^T k / S^T
// q are the two "read" contractions, the outer-product term k_t (x) err_t is
// the "write". Precision: k/v/q arrive bf16 (resident-stream convention);
// state S and the two gates are carried f32 (accfloat-equivalent) because
// they are integrated over T steps -- bf16 state would compound rounding
// error every step (see findings note: this IS the open question the spike
// was asked to probe).
//
// BRICK SHAPE: the only data-parallel axis on chip is D_V (one aie::vector
// lane per value-channel); D_K is an unrolled row loop (each row read/write
// is one D_V-wide vector op), same shape as dwconv1d_tmajor_f32o's per-row
// mac loop over taps. T is a sequential (non-parallel, non-unrollable-for-
// correctness) outer loop -- this is the honest recurrent part.
//
// Tile ABI (matches the numpy golden in golden.py):
//   k     [T, DK] bf16   -- keys (assumed already L2-normalized upstream)
//   v     [T, DV] bf16   -- values
//   q     [T, DK] bf16   -- queries
//   gates [T, 2]  float  -- (alpha_t, beta_t) per step, ALREADY activated
//   s_in  [DK, DV] float -- initial state (0 for a fresh sequence, else the
//                           s_out tile carried from the previous dispatch --
//                           this in/out pair IS how state threads across
//                           resident-stream dispatches, same role KV-cache
//                           plays for attention bricks)
//   o     [T, DV] bf16   -- outputs (one per step)
//   s_out [DK, DV] float -- final state, to hand to the next dispatch
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef GDN_DK
#define GDN_DK 32 // key/query dim (must be a multiple of 1; rows are scalar-broadcast)
#endif
#ifndef GDN_DV
#define GDN_DV 32 // value dim (the vectorized lane axis; multiple of the bf16/f32 vector width)
#endif
#ifndef GDN_T
#define GDN_T 64 // steps per resident-stream tile dispatch
#endif

// out_v[0..DV-1] = S^T x, i.e. sum_i x[i] * S[i, 0..DV-1].
// S is row-major [DK, DV] f32; x is a length-DK f32 vector already unpacked
// into scalars (caller passes it as a `vector<float,DK>` so bf16 k/q are
// upcast once per step, not per row).
template <unsigned DK, unsigned DV>
static inline aie::vector<float, DV> gdn_state_read(const float *__restrict S,
                                                     const aie::vector<float, DK> &x) {
  aie::accum<accfloat, DV> acc;
  acc.from_vector(aie::broadcast<float, DV>(0.0f));
  for (unsigned i = 0; i < DK; ++i) {
    aie::vector<float, DV> row = aie::load_v<DV>(&S[i * DV]);
    acc = aie::mac(acc, row, aie::broadcast<float, DV>(x[i]));
  }
  return acc.template to_vector<float>();
}

// In-place gated delta-rule state update:
//   S[i,:] = alpha * S[i,:] + bk[i] * err[:]     for i=0..DK-1
// bk is beta*k, precomputed by the caller in the VECTOR domain: the aie2p
// scalar-f32 path miscompiles (see rope_lut.cc's header), and the `beta*k[i]`
// this loop used to do was a scalar float mul.
template <unsigned DK, unsigned DV>
static inline void gdn_state_write(float *__restrict S,
                                    const aie::vector<float, DK> &bk,
                                    const aie::vector<float, DV> &err,
                                    const aie::vector<float, DV> &alpha_v) {
  for (unsigned i = 0; i < DK; ++i) {
    aie::vector<float, DV> row = aie::load_v<DV>(&S[i * DV]);
    aie::accum<accfloat, DV> acc = aie::mul(row, alpha_v);  // alpha * S[i,:]
    acc = aie::mac(acc, err, aie::broadcast<float, DV>(bk[i]));
    aie::store_v(&S[i * DV], acc.template to_vector<float>());
  }
}

// Full T-step gated delta-rule recurrence over one resident state tile.
// This is the honest sequential core: step t reads S_{t-1} (gdn_state_read
// twice: once for the delta prediction, once for the query readout after the
// write), so there is no cross-t vector width to exploit -- D_V is the only
// lane axis, D_K/T are scalar-unrolled loops. NOT marked AIE_LOOP_UNROLL_FULL
// on the T loop: unrolling a state-carried loop buys nothing (each iteration
// depends on the last) and would just bloat program memory.
template <unsigned DK, unsigned DV, unsigned T>
static inline void gatedeltanet_core(const bfloat16 *__restrict k,
                                     const bfloat16 *__restrict v,
                                     const bfloat16 *__restrict q,
                                     const float *__restrict gates, // [T,2] = (alpha,beta)
                                     const float *__restrict s_in,  // [DK,DV]
                                     bfloat16 *__restrict o,        // [T,DV]
                                     float *__restrict s_out) {     // [DK,DV]
  event0();
  ::aie::set_rounding(::aie::rounding_mode::conv_even);

  // Working state lives in s_out for the duration of the call (avoids a
  // second DK*DV-sized on-chip buffer); seed it from s_in.
  for (unsigned i = 0; i < DK * DV; i += DV) {
    aie::vector<float, DV> row = aie::load_v<DV>(&s_in[i]);
    aie::store_v(&s_out[i], row);
  }

  for (unsigned t = 0; t < T; ++t) {
    // Unpack this step's bf16 k/q rows to f32 once (gdn_state_{read,write}
    // both need per-lane scalar broadcasts of k; q only needs the read).
    aie::vector<bfloat16, DK> kb = aie::load_v<DK>(&k[t * DK]);
    aie::vector<bfloat16, DK> qb = aie::load_v<DK>(&q[t * DK]);
    aie::accum<accfloat, DK> kacc, qacc;
    kacc.from_vector(kb);
    qacc.from_vector(qb);
    aie::vector<float, DK> kf = kacc.template to_vector<float>();
    aie::vector<float, DK> qf = qacc.template to_vector<float>();

    aie::vector<bfloat16, DV> vb = aie::load_v<DV>(&v[t * DV]);
    aie::accum<accfloat, DV> vacc;
    vacc.from_vector(vb);
    aie::vector<float, DV> vf = vacc.template to_vector<float>();

    // Both gates go straight from their scalar load into the vector domain;
    // no scalar float arithmetic happens on this path.
    const aie::vector<float, DV> alpha_v =
        aie::broadcast<float, DV>(gates[t * 2 + 0]);
    const aie::vector<float, DK> bk =
        aie::mul(kf, aie::broadcast<float, DK>(gates[t * 2 + 1]))
            .template to_vector<float>();

    // 1) predict what the (pre-update) state already stores for this key.
    aie::vector<float, DV> pred = gdn_state_read<DK, DV>(s_out, kf);
    // 2) delta / correction term.
    aie::vector<float, DV> err = aie::sub(vf, pred);
    // 3) decay + delta-rule write (in place on s_out).
    gdn_state_write<DK, DV>(s_out, bk, err, alpha_v);
    // 4) read the just-updated state back out with the query.
    aie::vector<float, DV> ov = gdn_state_read<DK, DV>(s_out, qf);

    aie::accum<accfloat, DV> oacc;
    oacc.from_vector(ov);
    aie::store_v(&o[t * DV], oacc.template to_vector<bfloat16>());
  }

  event1();
}

extern "C" {

// Generic entry point: DK/DV/T are compile-time tile dims (GDN_DK/GDN_DV/
// GDN_T), same convention as GEMV_{M,K,N} in gemv_int8.cc -- a MODEL supplies
// these as data (a block schedule), this file never hard-codes a model's
// head dim.
void gatedeltanet_step(bfloat16 *k, bfloat16 *v, bfloat16 *q, float *gates,
                       float *s_in, bfloat16 *o, float *s_out) {
  gatedeltanet_core<GDN_DK, GDN_DV, GDN_T>(k, v, q, gates, s_in, o, s_out);
}
}
