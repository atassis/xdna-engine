// Golden-vector generator for test_q6k_dequant.py, linked against upstream
// s2.cpp/ggml/src/ggml-quants.c so the oracle is the real ggml implementation, not
// a hand-copied re-derivation of the Q6_K bit layout (kb: hanging numbers are bugs
// until proven otherwise).
//
//   cc -DGGML_COMMON_DECL_C -I <ggml>/src -I <ggml>/include \
//      -o gen_q6k_golden gen_q6k_golden.c <ggml>/src/ggml-quants.c -lm
//   ./gen_q6k_golden <out_dir>   # writes blocks.bin (input) and golden.bin (dequant output)
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k);
void quantize_row_q6_K_ref(const float *x, block_q6_K *y, int64_t k);

// ggml-quants.c also defines quantizers for q2..q5/iq*, which reference these
// symbols; none of that code executes on the q6_K-only path this harness calls.
void ggml_abort(const char *file, int line, const char *fmt, ...) { (void)file; (void)line; (void)fmt; abort(); }
size_t ggml_row_size(int type, long long ne) { (void)type; (void)ne; return 0; }
size_t ggml_type_size(int type) { (void)type; return 0; }
const char *ggml_type_name(int type) { (void)type; return ""; }

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <out_dir>\n", argv[0]); return 2; }
    srand(12345);
    const int NB = 4;              // superblocks; NB*256 elements
    const int K = NB * 256;
    float *src = malloc(K * sizeof(float));
    for (int i = 0; i < K; i++) {
        // wide dynamic range incl. negatives, matching real weight distributions
        src[i] = ((float)(rand() % 20001) - 10000.0f) / 1000.0f;
    }
    block_q6_K *blocks = malloc(NB * sizeof(block_q6_K));
    quantize_row_q6_K_ref(src, blocks, K);

    float *out = malloc(K * sizeof(float));
    dequantize_row_q6_K(blocks, out, K);

    char path[4096];
    snprintf(path, sizeof(path), "%s/blocks.bin", argv[1]);
    FILE *fb = fopen(path, "wb");
    fwrite(blocks, sizeof(block_q6_K), NB, fb);
    fclose(fb);
    snprintf(path, sizeof(path), "%s/golden.bin", argv[1]);
    FILE *fo = fopen(path, "wb");
    fwrite(out, sizeof(float), K, fo);
    fclose(fo);
    printf("nb=%d k=%d sizeof(block_q6_K)=%zu\n", NB, K, sizeof(block_q6_K));
    return 0;
}
