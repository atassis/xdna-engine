// Minimal repro of the dequant epilogue: (mi,ni) double loop, mmul, then either
// to_float+f32-store (suspect) or int32-store (bit-exact control). #include the brick
// to inherit int4/accauto/mmul_8_4 types exactly.
#include <stdint.h>
#include "gemm_int8xint4_dequant.cc"
using MMUL = aie::mmul<4,16,16,int8_t,int4,accauto>;
static constexpr unsigned kSteps=4,nTiles=4,mTiles=2,sizeC=64;
static inline MMUL accumulate(const int8_t* pA,const int4* pB,unsigned mi,unsigned ni){
  MMUL acc;
  for(unsigned ki=0;ki<kSteps;++ki){
    auto A=aie::load_v<MMUL::size_A>(pA+(mi*kSteps+ki)*MMUL::size_A);
    auto B=aie::load_v<MMUL::size_B>(pB+(ki*nTiles+ni)*MMUL::size_B);
    if(ki==0) acc.mul(A,B); else acc.mac(A,B);
  }
  return acc;
}
extern "C" void repro_f32(const int8_t* pA,const int4* pB,float* pC){   // SUSPECT
  for(unsigned mi=0;mi<mTiles;++mi) for(unsigned ni=0;ni<nTiles;++ni){
    MMUL acc=accumulate(pA,pB,mi,ni);
    auto cf=aie::to_float(acc.template to_vector<int32_t>(),0);
    aie::store_v(pC+(mi*nTiles+ni)*sizeC, cf);
  }
}
extern "C" void repro_i32(const int8_t* pA,const int4* pB,int32_t* pC){ // CONTROL (bit-exact path)
  for(unsigned mi=0;mi<mTiles;++mi) for(unsigned ni=0;ni<nTiles;++ni){
    MMUL acc=accumulate(pA,pB,mi,ni);
    aie::store_v(pC+(mi*nTiles+ni)*sizeC, acc.template to_vector<int32_t>());
  }
}
