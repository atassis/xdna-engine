import numpy as np, importlib.util
from bricklib import GEN, iron, _build_oneshot
spec=importlib.util.spec_from_file_location("g","route_b_kernels/bricks/rope-lut/golden.py")
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
sin_tab,cos_tab=g.sincos_lut_tables()
N=32
shim = (
 '#include <aie_api/aie.hpp>\n#include <stdint.h>\n'
 '#include "route_b_kernels/bricks/rope-lut/rope_lut_tables.inc"\n'
 'extern "C" void gather_probe(int8_t*restrict keys, bfloat16*restrict sout){\n'
 '  const ::aie::lut<4,bfloat16> sin_lut(256,kSinLutAb,kSinLutCd);\n'
 '  ::aie::parallel_lookup<int8,::aie::lut<4,bfloat16>> sl(sin_lut,0,128);\n'
 '  ::aie::vector<int8,%d> k=::aie::load_v<%d>(keys);\n'
 '  ::aie::store_v(sout, sl.fetch(k));\n'
 '}\n' % (N,N))
gshim=GEN/"gather_probe_shim.cc"; gshim.write_text(shim)
design=_build_oneshot("gather_probe", gshim, [N], N, [np.int8], __import__("ml_dtypes").bfloat16, [])
import ml_dtypes
keys=np.arange(N, dtype=np.int8)                      # keys 0..31
kt=iron.tensor(np.ascontiguousarray(keys), dtype=np.int8, device="npu")
ot=iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
design(kt, ot)
dev=ot.numpy().astype(np.float32)
exp=np.asarray(sin_tab, np.float32)[keys.astype(int)+128]
print("keys:", keys.tolist())
print("gathered sin:", dev[:12].round(3).tolist())
print("expected sin:", exp[:12].round(3).tolist())
print("match:", np.allclose(dev, exp, atol=0.02))
