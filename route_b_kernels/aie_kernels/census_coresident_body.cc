//===- census_coresident_body.cc --------------------------------*- C++ -*-===//
//
// Census probe for mode-switched-multi-program-xclbin: how much program memory
// does a genuinely distinct co-resident body cost inside a GEMM-class program?
//
// The 2026-07-30 gate proved co-residency for two elementwise bodies sharing one
// topology. The GEMM-class case was left unverified. This links two separately
// authored elementwise bodies (copied verbatim from acc_add.cc / residual_add.cc)
// alongside the shipped modal epilogue, dispatched by the same rtp[0] word:
// 0/1/2 = the existing identity/silu/gelu epilogue, 3 = acc_add, 4 = residual_add.
//
// Measurement only -- not a shipped path.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// Self-contained by necessity, not by preference: AIEAssignCoreLinkFiles traces only
// direct func.call edges from the core, so an object this one CALLS is never added to
// the link line ("undefined symbol: mm_modal_epilogue_f32_f32"). Same single-object
// shape the 2026-07-30 dual_add spike used.

// Stands in for the modal epilogue's identity path.
template <int N>
void census_identity_row(const float *restrict a, float *restrict out,
                         int32_t cols) {
  event0();
  for (int i = 0; i < cols; i += N)
    ::aie::store_v(out + i, ::aie::load_v<N>(a + i));
  event1();
}

// Verbatim from acc_add.cc.
template <int N>
void census_acc_add_row(const float *restrict a, const float *restrict b,
                        float *restrict out, int32_t cols) {
  event0();
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> av = ::aie::load_v<N>(a + i);
    ::aie::vector<float, N> bv = ::aie::load_v<N>(b + i);
    ::aie::store_v(out + i, ::aie::add(av, bv));
  }
  event1();
}

// Verbatim from residual_add.cc.
template <int N>
void census_residual_add_row(const float *restrict a, const float *restrict b,
                             float *restrict out, float scale, int32_t cols) {
  event0();
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const ::aie::vector<float, N> sv = ::aie::broadcast<float, N>(scale);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> av = ::aie::load_v<N>(a + i);
    ::aie::vector<float, N> bv = ::aie::load_v<N>(b + i);
    ::aie::vector<float, N> sb = ::aie::mul(bv, sv);
    ::aie::store_v(out + i, ::aie::add(av, sb));
  }
  event1();
}

// The three bodies above are all elementwise maps over the same operand shape, so they
// would each declare the SAME topology if authored as standalone bricks. The two below
// would not, which is the point: a reduction natively wants a [rows] output rather than a
// [rows, cols] one, and a strided read natively wants different BD dims. Forcing both onto
// the shared declaration is what tests whether topology sharing survives bodies that are
// genuinely different rather than three variants of one epilogue.

// Cross-lane reduction: a different compute class (reduce tree, not a map). Its natural
// output is one value per row; it broadcasts back over the shared full-size C tile.
template <int N>
void census_row_reduce(const float *restrict a, float *restrict out, int32_t rows,
                       int32_t cols) {
  event0();
  for (int r = 0; r < rows; ++r) {
    const float *row = a + (size_t)r * cols;
    ::aie::vector<float, N> acc = ::aie::zeros<float, N>();
    for (int i = 0; i < cols; i += N)
      acc = ::aie::add(acc, ::aie::load_v<N>(row + i));
    const ::aie::vector<float, N> bv =
        ::aie::broadcast<float, N>(::aie::reduce_add(acc));
    float *orow = out + (size_t)r * cols;
    for (int i = 0; i < cols; i += N)
      ::aie::store_v(orow + i, bv);
  }
  event1();
}

// Strided read: a different ACCESS pattern over the same buffer. Standalone this would be
// expressed in the BD dims rather than in the body, so it exercises whether an access shape
// the shared topology does not describe still works when the body walks it itself.
template <int N>
void census_stride_gather(const float *restrict a, float *restrict out, int32_t rows,
                          int32_t cols) {
  event0();
  for (int r = 0; r < rows; ++r) {
    const float *src = a + (size_t)((r + 1) % rows) * cols;
    float *orow = out + (size_t)r * cols;
    for (int i = 0; i < cols; i += N)
      ::aie::store_v(orow + i, ::aie::load_v<N>(src + i));
  }
  event1();
}

extern "C" {
// Same (c_in, c_out, rtp) signature as the modal epilogue it replaces, so the
// generator swaps one Kernel declaration and nothing else moves.
void census_coresident_body(const float *__restrict c_in,
                            float *__restrict c_out,
                            const int32_t *__restrict rtp) {
  if (rtp[0] == 3) {
    census_acc_add_row<16>(c_in, c_in, c_out, EPI_M *EPI_N);
  } else if (rtp[0] == 4) {
    census_residual_add_row<16>(c_in, c_in, c_out, 0.5f, EPI_M *EPI_N);
  } else if (rtp[0] == 5) {
    census_row_reduce<16>(c_in, c_out, EPI_M, EPI_N);
  } else if (rtp[0] == 6) {
    census_stride_gather<16>(c_in, c_out, EPI_M, EPI_N);
  } else {
    census_identity_row<16>(c_in, c_out, EPI_M *EPI_N);
  }
}
}
