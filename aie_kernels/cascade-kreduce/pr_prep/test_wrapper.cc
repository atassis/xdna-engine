#include "aie_cascade_bare.hpp"
using namespace aie;
// int32-vector cascade round-trip (put then get)
extern "C" vector<int32_t,16> t_i32(vector<int32_t,16> x){ cascade_out(x); return cascade_in_i32(); }
// acc32 cascade round-trip -- the fused-K-reduction target (partial sums as accumulators)
extern "C" accum<acc32,16> t_acc32(accum<acc32,16> a){ cascade_out(a); return cascade_in_acc32(); }
