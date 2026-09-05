// SPDX-License-Identifier: Apache-2.0
// Dead-simple identity copy: out[0:n] = in[0:n], bf16, vectorised.
//
// This exists as a PLUMBING probe, not as a useful op. If a straight copy cannot round-trip
// through a host->device->host dispatch, then nothing computed by a real kernel on that path
// can be trusted either, and there is no point debugging the real kernel. Deliberately uses
// vector load/store (no scalar float arithmetic) so the kernel itself cannot be the suspect.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void identity_copy_bf16(bfloat16 *restrict in, bfloat16 *restrict out, int32_t n) {
  event0();
  constexpr int V = 32;
  int i = 0;
  for (; i + V <= n; i += V) {
    ::aie::vector<bfloat16, V> v = ::aie::load_v<V>(in + i);
    ::aie::store_v(out + i, v);
  }
  for (; i < n; ++i)
    out[i] = in[i];
  event1();
}

} // extern "C"
