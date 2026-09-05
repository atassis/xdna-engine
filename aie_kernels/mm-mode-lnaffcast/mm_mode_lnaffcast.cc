//===- mm_mode_lnaffcast.cc -------------------------------------*- C++ -*-===//
//
// lnaffcast as a MODE of the modal GEMM xclbin, so the op runs inside the
// resident's hardware context instead of behind a program boundary. The prize
// is the boundary, not the op: 90% of the merge's -238.0 ms/clip sits on the 96
// `lnaffcast -> modal` transitions, not on lnaffcast itself.
//
// It declares no buffer. The bf16-out resident already owns a per-core
// `cacc` of 32x128xf32 = 16384 B, which is EXACTLY the four 1024-column rows a
// C tile holds, so the staging this needs is the accumulator the GEMM was
// going to zero anyway. One x row is two A tiles, gb is exactly one B tile.
//
// NO INDEX MAP. Every fifo shuffle is undone by the mode's own shim taps, all
// three at exactly 4 dims against the shim's 4 (A [32,8,4,4]/[128,4,32,1], B
// [4,8,16,4]/[512,4,32,1], C [4,8,16,8]/[1024,8,64,1]), so the operands arrive
// in column order and every access below is contiguous at 16 lanes. Do not
// "fix" an ordering here -- if a tap is wrong the fix is the tap, and the A one
// is 4 dims only because its outer slab dimension merges with the leading digit
// (512 == 4 x 128). See lnaffcast-mode-taps-fit-the-shim-in-four-dims.
//
// THE 1x ROUTE (LNA_ONE_X on the generator side; this file needs no #if for it,
// only the extra entry point below). A_L2L1 lists EIGHT consumer cores, so with x
// on A the 8 columns of an array row compute the SAME output: 1x on the read, 8x
// on the write, 4 useful C tiles per round of 32. Swapping the operands fixes it
// -- x on B is per-column and gb on A is a weight, which is what a broadcast is
// FOR. The row axis is then broken by a skip: a column's B stream carries
// n_aie_rows C tiles' worth and each core acquires all of them and applies only
// its own, which is 8 acquire pairs per C tile against the 64 the same skip on A
// would cost, because a B tile is 4 A tiles.
//
// LNA_SCATTER_C moves the C un-permute off the tap and onto this core, which is
// the OTHER side of that choice and the one the tap derivation never priced.
// Those C strides burst at 16 B and cost 56.8% of the mode's dispatch time
// (lnaffcast-mode-contiguous-taps-recover-61-percent), and the core-side form is
// a SCATTER rather than the 993-run gather the A-side map was measured at:
// C_L2L3's innermost digit is (t, 1), so writing output element q to L1 slot
// C_L2L3[q] lands in 509 runs of 8, every base t-aligned. It must be paired with
// a CONTIGUOUS C tap (--lnaffcast-contig-taps c) -- the two are one decision, and
// either alone returns the permuted output.
//
// Self-contained because AIEAssignCoreLinkFiles traces only direct func.call
// edges from the core, so an object this one CALLED would never reach the link
// line. Same shape mm_mode_body.cc uses.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// The lnaffcast embedding dim. A build parameter, not a property of the tile:
// the tile dims fix how many rows a C tile carries, this fixes how long one is.
#ifndef LNA_COLS
#define LNA_COLS 1024
#endif

// Which side undoes C_L2L3, and at what width. 0 = the shim tap (the derived C
// strides). 1 = this core, whole write pass narrowed to t lanes. 2 = this core,
// arithmetic kept at N lanes and only the STORE split into N/t pieces -- 1 pays
// half the f32 datapath for the scatter and 2 does not, which is the difference
// lnaffcast-scatter-core-cost-is-not-a-constant left unattributed. A build
// parameter, not a property of the tile.
#ifndef LNA_SCATTER_C
#define LNA_SCATTER_C 0
#endif

// f32 elements in one A tile, and output rows in one C tile. Derived from the
// GEMM's own declared shapes so they cannot drift from the buffers they index.
static constexpr int kSlabF32 = EPI_M * EPI_K / 2;
static constexpr int kRowsPerC = EPI_M * EPI_N / LNA_COLS;

// The 1x route swaps which operand carries which thing: x rides B (per-column,
// so the 8 columns stop computing the same rows) and gb rides A (per-row and
// broadcast across columns, which is CORRECT for a weight). So the "operand is
// a whole number of output rows" property kSlabF32 gives A is what the B tile
// needs here, and gb is staged into cacc instead of x.
static constexpr int kBTileF32 = EPI_K * EPI_N / 2;
static constexpr int kRowsPerB = kBTileF32 / LNA_COLS;

static_assert(EPI_M * EPI_N == kRowsPerC * LNA_COLS,
              "C tile must hold a whole number of output rows");
static_assert(LNA_COLS % kSlabF32 == 0,
              "an x row must be a whole number of A tiles");
static_assert(LNA_COLS % 16 == 0, "the write pass walks 16 lanes at a time");
static_assert(kBTileF32 == kRowsPerB * LNA_COLS,
              "a B tile must be a whole number of x rows for the 1x route");
static_assert(kRowsPerC % kRowsPerB == 0,
              "a C tile must be a whole number of B tiles for the 1x route");
static_assert(2 * LNA_COLS <= EPI_M * EPI_N,
              "cacc must hold gb ([gamma|beta]) for the 1x route");

#if LNA_SCATTER_C
// C_L2L3's own walk, read off the generator: sizes (m/r, r, n/t, t) against
// strides (r*n, t, r*t, 1). Stream position q -> L1 slot kScatter(q), so the core
// writing output element q there is what a CONTIGUOUS C tap needs to land DDR in
// row-major order. Derived from EPI_*, not spelled, so a blocking change fails a
// static_assert instead of emitting a wrong walk.
static constexpr int kT = EPI_T;             // innermost digit: t elements at stride 1
static constexpr int kNPerT = EPI_N / EPI_T; // (n/t) digit, stride r*t
static constexpr int kRowStride = EPI_R * EPI_N;   // (m/r) digit, stride r*n
static constexpr int kGroups = LNA_COLS / kT;      // scattered stores per output row

static_assert(EPI_N % EPI_T == 0, "n must be a whole number of t-columns");
static_assert(LNA_COLS % kT == 0, "an output row must be a whole number of stores");
// The (m/r) digit is the OUTPUT ROW only when one row is exactly the r*n elements
// the other three digits span; without it the row index straddles digits and the
// closed form below is wrong rather than merely narrower.
static_assert(LNA_COLS == kRowStride,
              "LNA_SCATTER_C needs one output row == r*n; re-derive otherwise");
static_assert(kRowsPerC == EPI_M / EPI_R, "rows per C tile must be the (m/r) digit");

// L1 slot the group of kT output columns starting at g*kT must be written to.
static constexpr int scatter_base(int row, int g) {
  return row * kRowStride + (g / kNPerT) * kT + (g % kNPerT) * (EPI_R * kT);
}
#endif

// Stage one A tile into the accumulator. The A tile is DECLARED bf16 (it is the
// GEMM's operand buffer) but a mode stream fills it with f32 activations, so it
// is read through a reinterpret -- the same borrow the accadd mode already
// makes. Both buffers are objectFIFO/aie.buffer allocations, which are at least
// 32-byte aligned, so the f32 view is aligned by construction.
template <int N>
static void lnaff_stage(const bfloat16 *restrict a, float *restrict acc) {
  event0();
  const float *src = reinterpret_cast<const float *>(a);
  for (int i = 0; i < kSlabF32; i += N)
    ::aie::store_v(acc + i, ::aie::load_v<N>(src + i));
  event1();
}

// LN + affine + f32->bf16 cast over the whole staged C tile.
//
// The math is ln_affine_cast.cc's, unchanged and deliberately so: this has to
// be bit-comparable to the op it replaces or the merge cannot be gated on
// parity. That includes the rounding split -- the reductions run under the AIE
// DEFAULT mode (ctxLN never sets one) and conv_even is swapped in ONLY around
// the affine+cast write. Setting conv_even up front changes the reduction
// rounding and regressed WER 8.2 -> 8.8, compounded over 24 layers.
template <int N, int ROWS>
static void lnaff_apply_rows(const float *restrict gbf, const float *restrict xsrc,
                             bfloat16 *restrict out, int row_base) {
  event0();
  constexpr float epsilon = 1e-5f;
  constexpr int chunks = LNA_COLS / N;
  // gb is [gamma | beta], f32 like the activations. WHERE it is read from is the
  // route's choice -- a B tile in the 8x route, cacc in the 1x one -- and the
  // arithmetic below must not be able to tell, or the two routes cannot be gated
  // on one parity number.
  const float *gamma = gbf;
  const float *beta = gbf + LNA_COLS;

  for (int row = 0; row < ROWS; row++) {
    const float *x = xsrc + row * LNA_COLS;
    // The output row inside the C tile. In the 8x route xsrc IS the whole C
    // tile's worth so the two indices coincide; in the 1x route one B tile is
    // kRowsPerB of the tile's kRowsPerC rows and row_base says which.
    const int orow = row_base + row;
#if LNA_SCATTER_C == 0
    bfloat16 *y = out + orow * LNA_COLS;
#endif

    // pass 1: mean = Sx / cols
    ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
    for (int i = 0; i < chunks; i++)
      sum_v = ::aie::add(sum_v, ::aie::load_v<N>(x + i * N));
    float mean = ::aie::reduce_add(sum_v) / float(LNA_COLS);
    ::aie::vector<float, N> mean_v = ::aie::broadcast<float, N>(mean);

    // pass 2: var = S(x-mean)^2 / cols (centered, matching the host reference)
    ::aie::vector<float, N> var_v = ::aie::zeros<float, N>();
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(x + i * N), mean_v);
      // The temporary is load-bearing, not style: aie::mul yields an accum, and
      // naming it as a vector is what narrows it back. Inlining the call makes
      // add() see an accum and fail to deduce. Same spelling as ln_affine_cast.cc.
      ::aie::vector<float, N> sq = ::aie::mul(d, d);
      var_v = ::aie::add(var_v, sq);
    }
    float var = ::aie::reduce_add(var_v) / float(LNA_COLS);
#if LNA_SCATTER_C == 1
    // The two scalars presented at the store width this form walks in.
    ::aie::vector<float, kT> mean_v_t = ::aie::broadcast<float, kT>(mean);
    ::aie::vector<float, kT> inv_v_t =
        ::aie::broadcast<float, kT>(::aie::invsqrt(var + epsilon));
#else
    ::aie::vector<float, N> inv_v =
        ::aie::broadcast<float, N>(::aie::invsqrt(var + epsilon));
#endif

    // write: out = ((x - mean) * inv) * gamma + beta -> bf16.
    // The core's rounding mode is one sticky register shared by every kernel on
    // the core, so it is saved and restored rather than left swapped -- here
    // that matters more than in a standalone kernel, because the GEMM epilogue
    // runs on this same core from a different dispatch.
    const auto saved_rounding =
        ::aie::swap_rounding(::aie::rounding_mode::conv_even);
#if LNA_SCATTER_C == 1
    // Same arithmetic, narrower store. Each element is computed independently of
    // its neighbours (mean and inv are broadcasts), so kT lanes and N lanes are
    // bit-identical -- which is what lets the two forms be gated on one number.
    for (int g = 0; g < kGroups; g++) {
      const int col = g * kT;
      ::aie::vector<float, kT> d = ::aie::sub(::aie::load_v<kT>(x + col), mean_v_t);
      ::aie::vector<float, kT> norm = ::aie::mul(d, inv_v_t);
      ::aie::vector<float, kT> ng =
          ::aie::mul(norm, ::aie::load_v<kT>(gamma + col));
      ::aie::accum<accfloat, kT> acc_y;
      acc_y.from_vector(::aie::add(ng, ::aie::load_v<kT>(beta + col)));
      ::aie::store_v(out + scatter_base(orow, g),
                     acc_y.template to_vector<bfloat16>());
    }
#elif LNA_SCATTER_C == 2
    // The scatter costs one address and N/kT stores; it does not have to cost the
    // datapath. The arithmetic below is the LNA_SCATTER_C=0 loop unchanged, and
    // only the store is split -- the N lanes of chunk i are the kT-groups
    // i*kPerChunk .. +kPerChunk-1, which share the (n/t) digit, so their L1 bases
    // are one arithmetic sequence at stride r*t.
    constexpr int kPerChunk = N / kT;
    static_assert(kNPerT % kPerChunk == 0,
                  "a chunk must not straddle the (n/t) digit");
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(x + i * N), mean_v);
      ::aie::vector<float, N> norm = ::aie::mul(d, inv_v);
      ::aie::vector<float, N> ng =
          ::aie::mul(norm, ::aie::load_v<N>(gamma + i * N));
      ::aie::accum<accfloat, N> acc_y;
      acc_y.from_vector(::aie::add(ng, ::aie::load_v<N>(beta + i * N)));
      ::aie::vector<bfloat16, N> v = acc_y.template to_vector<bfloat16>();
      const int base = scatter_base(orow, i * kPerChunk);
      for (int j = 0; j < kPerChunk; j++)
        ::aie::store_v(out + base + j * (EPI_R * kT), v.template extract<kT>(j));
    }
#else
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(x + i * N), mean_v);
      ::aie::vector<float, N> norm = ::aie::mul(d, inv_v);
      ::aie::vector<float, N> ng =
          ::aie::mul(norm, ::aie::load_v<N>(gamma + i * N));
      ::aie::accum<accfloat, N> acc_y;
      acc_y.from_vector(::aie::add(ng, ::aie::load_v<N>(beta + i * N)));
      ::aie::store_v(y + i * N, acc_y.template to_vector<bfloat16>());
    }
#endif
    ::aie::set_rounding(saved_rounding);
  }
  event1();
}

extern "C" {
// One A tile -> the accumulator at `slot`. Called once per A tile; the slot is
// a compile-time constant from an unrolled generator loop, so the mode needs no
// counter and no loop-index cast.
void mm_lnaff_stage_f32(const bfloat16 *__restrict a, float *__restrict acc,
                        int32_t slot) {
  lnaff_stage<16>(a, acc + slot * kSlabF32);
}

// gb + the staged accumulator -> the bf16 C tile. The 8x route: x came in on A
// and was staged, gb is one B tile.
void mm_lnaff_apply_f32(const bfloat16 *__restrict gb, float *__restrict acc,
                        bfloat16 *__restrict out) {
  lnaff_apply_rows<16, kRowsPerC>(reinterpret_cast<const float *>(gb), acc, out, 0);
}

// The 1x route's apply: gb is already staged in cacc (kSlabF32 at a time, by the
// call above), and x is read STRAIGHT out of the B tile -- one B tile is
// kRowsPerB whole output rows, so nothing needs staging on the activation side
// and the 8x route's whole L1->L1 copy pass disappears with it.
void mm_lnaff_apply_x_f32(float *__restrict gbbuf, const bfloat16 *__restrict xb,
                          bfloat16 *__restrict out, int32_t row_base) {
  lnaff_apply_rows<16, kRowsPerB>(gbbuf, reinterpret_cast<const float *>(xb), out,
                                  row_base);
}
}
