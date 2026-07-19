#include "aie_cascade_bare.hpp"
extern "C" aie::vector<int32_t,16> t(aie::vector<int32_t,16> x){ aie::cascade_out(x); return aie::cascade_in_i32(); }
