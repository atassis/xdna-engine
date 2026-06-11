//===- dwconv1d.cc ----------------------------------------*- C++ -*-===//
//
// Depthwise conv1d, kernel size 5, 'same' padding (pad = 2), stride 1.
// One channel processed per kernel call (the per-channel taps differ, so each
// channel is its own ObjectFifo tile). Compute mirrors the AIE bf16 matmul
// convention: load bf16 -> accumulate in fp32 -> store a single bf16 round,
// so a numpy fp32-accumulate reference matches to ~1 ULP.
//
// This is the last missing GigaAM-v3 Conformer primitive (the depthwise_conv
// in every block's conv module). Scalar (correctness-first); vectorize later.
//
// TRACKED COPY — installed into mlir-aie/aie_kernels/aie2p/ by setup_route_b.sh
// (the mlir-aie tree is gitignored / re-cloned).
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

template <int T, int K, int P>
void dwconv1d_same(const bfloat16 *restrict in, const bfloat16 *restrict w,
                   bfloat16 *restrict out) {
  event0();
  float wt[K];
  for (int i = 0; i < K; i++)
    wt[i] = static_cast<float>(w[i]);
  for (int t = 0; t < T; t++) {
    float acc = 0.0f;
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < K; i++) {
      int idx = t - P + i;
      if (idx >= 0 && idx < T)
        acc += wt[i] * static_cast<float>(in[idx]);
    }
    out[t] = static_cast<bfloat16>(acc);
  }
  event1();
}

extern "C" {
// in: 400 samples, w: 16-tile (first 5 = taps), out: 400 samples
void dwconv1d_k5_bf16(bfloat16 *in, bfloat16 *w, bfloat16 *out) {
  dwconv1d_same<400, 5, 2>(in, w, out);
}
}
