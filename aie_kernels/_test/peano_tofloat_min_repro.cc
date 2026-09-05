// Minimal repro: to_float(<int32,64>)->f32-store in a 2x4 tile double loop.
// Isolates the dequant epilogue's SUSPECT value path (vups.2x int32->f32 + the
// magic-constant fixup) with NO mmul, so the codegen is small + inspectable.
#include <aie_api/aie.hpp>
#include <stdint.h>
static constexpr unsigned mTiles=2, nTiles=4, sizeC=64;
// (A) pure to_float->store over the grid (no accumulator source).
extern "C" void min_tofloat(const int32_t* in, float* out){
  for(unsigned mi=0;mi<mTiles;++mi) for(unsigned ni=0;ni<nTiles;++ni){
    auto v = aie::load_v<sizeC>(in + (mi*nTiles+ni)*sizeC);
    auto f = aie::to_float(v, 0);
    aie::store_v(out + (mi*nTiles+ni)*sizeC, f);
  }
}
// (B) control: pass-through int32 store (no to_float).
extern "C" void min_i32(const int32_t* in, int32_t* out){
  for(unsigned mi=0;mi<mTiles;++mi) for(unsigned ni=0;ni<nTiles;++ni){
    auto v = aie::load_v<sizeC>(in + (mi*nTiles+ni)*sizeC);
    aie::store_v(out + (mi*nTiles+ni)*sizeC, v);
  }
}
