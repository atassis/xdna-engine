// SPDX-License-Identifier: Apache-2.0
// Merge one token's V column into one slot of a column PAIR, keeping the other slot's value.
//
// A DMA cannot write one column of an adjacent pair: the innermost bf16 transfer unit is a 4-byte
// granule spanning both. So the pair is staged and written whole, and the slot this token does not
// write must survive -- read back rather than left in place, because the L1 tile is fresh per
// acquire.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void vpair_stage_bf16(bfloat16 *restrict v, bfloat16 *restrict pair_in,
                      bfloat16 *restrict pair_out, int32_t n, int32_t parity) {
  event0();
  const int keep = 1 - parity;
  for (int i = 0; i < n; ++i) {
    pair_out[2 * i + keep] = pair_in[2 * i + keep];
    pair_out[2 * i + parity] = v[i];
  }
  event1();
}

} // extern "C"
