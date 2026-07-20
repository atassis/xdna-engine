import numpy as np
from bricklib import GEN, iron, _build_oneshot
import ml_dtypes
N=32
shim = ('#include <aie_api/aie.hpp>\n#include <stdint.h>\n'
 '#include "%s/ramp_tables.inc"\n' % str(GEN) +
 'extern "C" void gather_probe(int8_t*restrict keys, bfloat16*restrict sout){\n'
 '  const ::aie::lut<4,bfloat16> lut(256,kRampAb,kRampAb);\n'
 '  ::aie::parallel_lookup<int8,::aie::lut<4,bfloat16>> sl(lut,0,128);\n'
 '  ::aie::vector<int8,32> k=::aie::load_v<32>(keys);\n'
 '  ::aie::store_v(sout, sl.fetch(k));\n}\n')
gshim=GEN/"gather_ramp_shim.cc"; gshim.write_text(shim)
design=_build_oneshot("gather_probe", gshim, [N], N, [np.int8], ml_dtypes.bfloat16, [])
keys=np.arange(N, dtype=np.int8)
kt=iron.tensor(np.ascontiguousarray(keys),dtype=np.int8,device="npu")
ot=iron.zeros((N,),dtype=ml_dtypes.bfloat16,device="npu"); design(kt,ot)
dev=ot.numpy().astype(np.float32)
print("keys        :", keys.tolist())
print("logical read:", dev.astype(int).tolist(), "(expect 128..159)")
