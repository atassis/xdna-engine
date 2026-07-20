import numpy as np, ml_dtypes
import verify_rope_lut as vr
from bricklib import GEN, iron, _build_oneshot
qk_bf16, cbuf, pos, invf, exp = vr.build_case()
_, cc = vr.load_golden()
gshim = GEN / "rope_dbg_shim.cc"; gshim.write_text(f'#include <stdint.h>\n#include "{cc}"\n{vr.SHIM_BODY}\n')
design = _build_oneshot("rope_lut_verify", gshim, [qk_bf16.size, cbuf.size], vr.M*vr.D, [ml_dtypes.bfloat16, np.int32], ml_dtypes.bfloat16, vr.COMPILE_FLAGS)
in_ts=[iron.tensor(np.ascontiguousarray(qk_bf16.reshape(-1)),dtype=ml_dtypes.bfloat16,device="npu"),iron.tensor(np.ascontiguousarray(cbuf.reshape(-1)),dtype=np.int32,device="npu")]
out_t=iron.zeros((vr.M*vr.D,),dtype=ml_dtypes.bfloat16,device="npu"); design(*in_ts,out_t)
dev=out_t.numpy().reshape(vr.M,vr.D).astype(np.float32); inp=np.asarray(qk_bf16,np.float32)
match=[i for i in range(32) if abs(dev[0,i]-inp[0,i])<0.03]
print("row0 (pos=0) matching indices in [0,32):", match)
print("dev[0][:16]:", dev[0,:16].round(2).tolist())
print("inp[0][:16]:", inp[0,:16].round(2).tolist())
