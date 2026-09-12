// Scores GEMV as an mmul: c[p] = sum_d a[p][d] * b[d], p over sequence positions, d over head_dim.
//
// Drop-in for sc_matvec_rtk_bf16_bf16 (attn_block_dp/design.py:415-418) EXCEPT for the `a` tile
// order -- see the contract below. The shipped body reduces one output row per call with
// aie::mac + aie::reduce_add, and its log2(r) shift/add tree is 55 of its 61 static bundles.
// Here the accumulator is OUTPUT-shaped instead, so the head_dim reduction rides the systolic
// array and no horizontal reduce exists. N=8 is load-bearing: mmul<4,8,4> reintroduces a tree.
//
// The query is the mmul's M axis and only row 0 is live, so MM_M-1 rows are wasted MACs. That is
// the trade -- MAC issue slots are what this op has spare (1.53 MAC/cycle against a 128 peak).
//
// TILE CONTRACT, and it is NOT the shipped one. `a` must arrive hd-chunk-major:
//   a[(d/8) * m * 8 + (p/8) * 64 + (d%8) * 8 + (p%8)]
// i.e. [K/8][M/8][8 d][8 p]. The shipped stream delivers [p][d] row-major, whose 8x8 tile is 8
// runs of 8 with stride HD -- not the 64 contiguous elements load_v needs. This is an objectFIFO
// ACCESS PATTERN change on the K-cache stream into L1, not a DDR layout change: nothing under
// iron.common.kv_layout.KVLayout or its Rust mirror moves.
#define NOCPP
#include <stdint.h>
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif
#ifndef MM_M
#define MM_M 8
#endif
#ifndef MM_K
#define MM_K 8
#endif
#ifndef MM_N
#define MM_N 8
#endif

// k (head_dim) is a runtime argument, as it is on the shipped rtk body.
static inline void scores_mmul(uint32_t m, uint32_t k,
                               const bfloat16 *__restrict a,
                               const bfloat16 *__restrict b,
                               bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    using MMUL = aie::mmul<MM_M, MM_K, MM_N, bfloat16, bfloat16, accauto>;
    const uint32_t kt = k / MM_K;
    const uint32_t pt = m / MM_N;

    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (uint32_t pb = 0; pb < pt; pb++) {
        // Query tile: MM_M x MM_K, row 0 live. Its k-chunk advances with the reduction.
        const bfloat16 *__restrict pB = b;
        const bfloat16 *__restrict pA = a + pb * MM_K * MM_N;
        MMUL acc;
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (uint32_t d = 0; d < kt; d++) {
            // A = query rows, B = the [8 d][8 p] tile. The bfp16-emulated mmul transposes its B
            // operand internally (aie_api mmul_bf16_bf16.hpp:109), so no transpose is needed here.
            // The query stays the SHIPPED [HD] buffer: load its 8-wide d-chunk and replicate it
            // across all MM_M rows, so every accumulator row holds the same live query and the
            // padding rows cost MACs but no memory. Only the `a` tile order changes.
            acc.mac(aie::load_v<MM_K>(pB).template grow_replicate<MMUL::size_A>(),
                    aie::load_v<MMUL::size_B>(pA));
            pB += MM_K;
            pA += pt * MM_K * MM_N;
        }
        // Every accumulator row holds the same query, so row 0 is the answer -- and C is
        // [MM_M][MM_N] row-major, so that row is the first MM_N elements: one vector store, not
        // MM_N scalar extracts. Extracting elementwise costs 12 vextract-class instructions and
        // puts back most of the fixed block this kernel exists to delete.
        aie::vector<bfloat16, MMUL::size_C> cb = acc.template to_vector<bfloat16>();
        aie::store_v(c + pb * MM_N, cb.template extract<MM_N>(0));
    }
}

extern "C" {
void sc_matvec_mmul_bf16_bf16(uint32_t m, uint32_t row_offset, uint32_t k,
                              const bfloat16 *__restrict a_in,
                              const bfloat16 *__restrict b_in,
                              bfloat16 *__restrict c_out)
{
    scores_mmul(m, k, a_in, b_in, c_out + row_offset);
}
}
