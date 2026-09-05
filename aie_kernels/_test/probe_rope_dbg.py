import numpy as np, ml_dtypes
import verify_rope_lut as vr
from bricklib import GEN, iron, _build_oneshot
qk_bf16, cbuf, pos, invf, exp = vr.build_case()
_, cc = vr.load_golden()
gshim = GEN / "rope_dbg_shim.cc"
gshim.write_text(f'#include <stdint.h>\n#include "{cc}"\n{vr.SHIM_BODY}\n')
design = _build_oneshot("rope_lut_verify", gshim, [qk_bf16.size, cbuf.size], vr.M*vr.D,
                        [ml_dtypes.bfloat16, np.int32], ml_dtypes.bfloat16, vr.COMPILE_FLAGS)
in_ts = [iron.tensor(np.ascontiguousarray(qk_bf16.reshape(-1)), dtype=ml_dtypes.bfloat16, device="npu"),
         iron.tensor(np.ascontiguousarray(cbuf.reshape(-1)), dtype=np.int32, device="npu")]
out_t = iron.zeros((vr.M*vr.D,), dtype=ml_dtypes.bfloat16, device="npu")
design(*in_ts, out_t)
dev = out_t.numpy().reshape(vr.M, vr.D).astype(np.float32)
inp = np.asarray(qk_bf16, np.float32)
ref = np.asarray(exp, np.float32)
print("pos:", pos[:4].tolist())
print("row0 pos=0 -> should be IDENTITY:")
print("  inp[0,:6]:", inp[0,:6].round(3).tolist())
print("  dev[0,:6]:", dev[0,:6].round(3).tolist())
print("  ref[0,:6]:", ref[0,:6].round(3).tolist())
print("row1 pos=1:")
print("  dev[1,:6]:", dev[1,:6].round(3).tolist())
print("  ref[1,:6]:", ref[1,:6].round(3).tolist())
print("row0 identity match:", np.allclose(dev[0], inp[0], atol=0.05))
