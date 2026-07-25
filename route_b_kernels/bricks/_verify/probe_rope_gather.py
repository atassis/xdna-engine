import numpy as np, ml_dtypes, importlib.util
import verify_rope_lut as vr
from bricklib import GEN, iron, _build_oneshot
g,_cc = vr.load_golden()
M,D,ROT = vr.M, vr.D, vr.ROT; half=ROT//2
# qk: x1=1, x2=0 -> out1=cos_gathered, out2=sin_gathered (isolates the gather)
qk = np.zeros((M,D), np.float32); qk[:,0:half]=1.0
qk_bf = qk.astype(ml_dtypes.bfloat16)
pos = np.arange(M, dtype=np.int32)
inv_freq = g.build_inv_freq(ROT).astype(np.float32)
cbuf = np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)
gshim=GEN/"rope_dbg_shim.cc"; gshim.write_text(f'#include <stdint.h>\n#include "{_cc}"\n{vr.SHIM_BODY}\n')
design=_build_oneshot("rope_lut_verify",gshim,[qk_bf.size,cbuf.size],M*D,[ml_dtypes.bfloat16,np.int32],ml_dtypes.bfloat16,vr.COMPILE_FLAGS)
its=[iron.tensor(np.ascontiguousarray(qk_bf.reshape(-1)),dtype=ml_dtypes.bfloat16,device="npu"),iron.tensor(np.ascontiguousarray(cbuf.reshape(-1)),dtype=np.int32,device="npu")]
out=iron.zeros((M*D,),dtype=ml_dtypes.bfloat16,device="npu"); design(*its,out)
dev=out.numpy().reshape(M,D).astype(np.float32)
cos_dev=dev[:,0:half]; sin_dev=dev[:,half:ROT]
# expected keys + cos/sin
sin_tab,cos_tab=g.sincos_lut_tables()
print("row0 pos=0 (all keys=0 -> cos=1,sin=0):")
print("  cos_dev[0,:8]:", cos_dev[0,:8].round(3).tolist())
print("  sin_dev[0,:8]:", sin_dev[0,:8].round(3).tolist())
for r in [1,2]:
    theta=pos[r]*inv_freq; key=g._quantize_key(theta); idx=key+128
    print(f"row{r} pos={pos[r]}: keys[:6]={key[:6].tolist()}")
    print(f"  cos_dev[:6]={cos_dev[r,:6].round(3).tolist()} exp_cos={cos_tab[idx][:6].round(3).tolist()}")
    print(f"  sin_dev[:6]={sin_dev[r,:6].round(3).tolist()} exp_sin={sin_tab[idx][:6].round(3).tolist()}")
