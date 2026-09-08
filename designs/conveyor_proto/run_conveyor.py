#!/usr/bin/env python3
"""Run the 2-stage conveyor prototype on-device and verify the data streamed A->B.
stage A (tile 0,2) does +1, stage B (tile 0,3) does *2, so out MUST == (in+1)*2.
A correct result proves data crossed the inter-tile ObjectFifo belt through 2 cores."""
import os, sys, numpy as np, pyxrt

EX = os.path.join(os.path.dirname(__file__), "build")
N = 256
instr = np.fromfile(f"{EX}/insts.bin", dtype=np.uint32)
xclbin = pyxrt.xclbin(f"{EX}/final.xclbin")
kname = xclbin.get_kernels()[0].get_name()
d = pyxrt.device(0); d.register_xclbin(xclbin)
ctx = pyxrt.hw_context(d, xclbin.get_uuid())
k = pyxrt.kernel(ctx, kname)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

inp = np.arange(N, dtype=np.int32)
bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
bo_in = pyxrt.bo(d, inp.nbytes, pyxrt.bo.host_only, k.group_id(3))
bo_out = pyxrt.bo(d, N * 4, pyxrt.bo.host_only, k.group_id(4))
bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
bo_in.write(inp.tobytes(), 0); bo_in.sync(TO)

r = k(3, bo_instr, instr.size, bo_in, bo_out); r.wait()
bo_out.sync(FROM)
out = np.frombuffer(bo_out.read(N * 4, 0), dtype=np.int32)
expect = (inp + 1) * 2
ok = np.array_equal(out, expect)
print(f"kernel='{kname}'  in[:6]={inp[:6]}  out[:6]={out[:6]}  expect[:6]={expect[:6]}")
print("CONVEYOR PASS (data streamed A->B through 2 tiles)" if ok else f"FAIL: {(out != expect).sum()} mismatches")
sys.exit(0 if ok else 1)
