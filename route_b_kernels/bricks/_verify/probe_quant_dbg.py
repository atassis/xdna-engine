import numpy as np, ml_dtypes, bricklib
import verify_cast_quant as vc
c = vc._quant_case()
# build + run on device manually to inspect dev output
shim = c["shim"]; 
design = bricklib._build_rowwise(c["symbol"], None, c["m"], c["in_cols"], c["out_cols"], 0,
                                 [], c["in_dt"], c["out_dt"], np.float32) if False else None
# simplest: reuse verify_rowwise but capture via monkeypatch of numpy compare is hard;
# instead replicate the device run inline like verify_rowwise does.
from bricklib import GEN, iron, _build_rowwise
import bricklib as B
gshim = GEN / f'{c["name"]}_shim.cc'
gshim.write_text(f'// dbg\n#include <stdint.h>\n#include "{c["cc"]}"\n{shim}\n')
design = _build_rowwise(c["symbol"], gshim, c["m"], c["in_cols"], c["out_cols"], 0, [], c["in_dt"], c["out_dt"], np.float32)
x = c["x"]
x_t = iron.tensor(np.ascontiguousarray(x.reshape(-1)), dtype=c["in_dt"], device="npu")
out_t = iron.zeros((c["m"]*c["out_cols"],), dtype=c["out_dt"], device="npu")
design(x_t, out_t)
dev = out_t.numpy().reshape(c["m"], c["out_cols"]).astype(np.int64)
ref = np.asarray(c["exp"]).astype(np.int64)
print("scale used:", None)
print("dev[0,:12]:", dev[0,:12].tolist())
print("ref[0,:12]:", ref[0,:12].tolist())
print("dev/ref ratio (nonzero ref):", (dev[ref!=0]/ref[ref!=0])[:12])
print("x_bf16[0,:12] (f32):", np.asarray(x, np.float32)[0,:12].round(3).tolist())
