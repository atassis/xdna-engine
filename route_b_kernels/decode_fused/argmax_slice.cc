// SPDX-License-Identifier: Apache-2.0
// Per-column partial argmax for the e2e/NPU lm-head. Each of the 8 whole-array columns runs this over its
// contiguous VOCAB_PAD/8 slice of the proj_out logits: it scans the slice and emits the LOCAL index of the
// max + the max value (cast to f32). The host then does the trivial 8-way reduce (global = col*slice + local;
// pick the column with the largest value) — bit-exact with the host f32 argmax (strict `>`, first-max wins,
// matches whisper.rs argmax()). Scalar scan: ~slice cycles (~6.5k → ~6 µs/core, 8 cores in parallel) — argmax
// is not the bottleneck, so favor obvious correctness over vectorization.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

// `out` is a 4×bf16 (8-byte) slot the fusion framework treats as bf16 (it is uniform-bf16); we pack the
// raw bytes [ max_val : f32 | local_idx : i32 ]. The host reinterprets the 8 bytes per column.
void argmax_slice_bf16(bfloat16 *restrict in, bfloat16 *restrict out, int32_t n) {
  event0();
  float best = -3.0e38f;
  int32_t best_i = 0;
  for (int32_t i = 0; i < n; i++) {
    float v = (float)in[i];
    if (v > best) { // strict > → first occurrence wins ties (matches host argmax)
      best = v;
      best_i = i;
    }
  }
  ((float *)out)[0] = best;        // bytes 0:4 — max value (f32)
  ((int32_t *)out)[1] = best_i;    // bytes 4:8 — LOCAL index (i32); host adds col*slice
  event1();
}

} // extern "C"
