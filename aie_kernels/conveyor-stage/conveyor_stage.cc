// Trivial 2-stage CONVEYOR prototype kernels (mechanism proof, not perf).
// Two DIFFERENT functions, one per pipeline stage, each placed on its own tile:
//   stage_add1 : out[i] = in[i] + 1     (stage A)
//   stage_mul2 : out[i] = in[i] * 2     (stage B)
// A correct end-to-end result (out == (in+1)*2) proves data streamed A->B across
// two distinct cores via the inter-tile ObjectFifo belt. Scalar on purpose.
#include <cstdint>

extern "C" void stage_add1(const int32_t *__restrict in, int32_t *__restrict out,
                           int32_t n) {
  for (int i = 0; i < n; i++)
    out[i] = in[i] + 1;
}

extern "C" void stage_mul2(const int32_t *__restrict in, int32_t *__restrict out,
                           int32_t n) {
  for (int i = 0; i < n; i++)
    out[i] = in[i] * 2;
}

// f32-belt variants: A writes an f32 belt, B reads it (isolates whether an f32 ObjectFifo belt
// -- as in the attention ac belt -- is what breaks the N_QT>1 streaming).
extern "C" void stage_f32_a(const int32_t *__restrict in, float *__restrict out, int32_t n) {
  for (int i = 0; i < n; i++)
    out[i] = (float)in[i] + 1.0f;
}
extern "C" void stage_f32_b(const float *__restrict in, int32_t *__restrict out, int32_t n) {
  for (int i = 0; i < n; i++)
    out[i] = (int32_t)(in[i] * 2.0f);
}
