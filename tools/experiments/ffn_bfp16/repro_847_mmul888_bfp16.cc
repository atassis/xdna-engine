// Minimal repro candidate for llvm-aie #847:
// aie::mmul<8,8,8> bf16 under BFP16 emulation -> -O0 ICE in
// AIE2PInstructionSelector::selectG_AIE_LOAD_STORE.
// Build: clang++ -O0 --target=aie2p-none-unknown-elf -std=c++20
//        -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -c repro_min.cc
#include <aie_api/aie.hpp>

extern "C" void mmul888_bfp16(const bfloat16 *__restrict a,
                              const bfloat16 *__restrict b,
                              bfloat16 *__restrict c) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  aie::vector<bfloat16, MMUL::size_A> A = aie::load_v<MMUL::size_A>(a);
  aie::vector<bfloat16, MMUL::size_B> B = aie::load_v<MMUL::size_B>(b);
  MMUL acc;
  acc.mul(A, B);
  aie::store_v(c, acc.template to_vector<bfloat16>());
}
