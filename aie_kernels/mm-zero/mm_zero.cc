//===- mm_zero.cc -----------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// The vendor mm.cc plus the zero_<type> / zero_scalar_<type> entry points it
// exported until mlir-aie #3732, which our generators still bind from the mm
// object. Compile with the vendor aie2p kernel dir on -I.

#include "mm.cc"
#include "../generic/zero.cc"

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_##mlir_type_out(ctype_out *c_out) {                                \
    zero_vectorized<ctype_out, DIM_M, DIM_N>(c_out);                           \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           r, s, t)                                            \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, DIM_M, DIM_N>(c_out);                               \
  }

extern "C" {
combos(zero_vectorized_c_func) combos(zero_scalar_c_func)
} // extern "C"
