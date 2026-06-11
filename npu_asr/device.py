"""Single shared XDNA2 device + kernel cache.

The NPU is single-tenant (a 2nd hw-context from another *process* fails CREATE_HWCTX),
and within a process the 8 columns are a shared budget. We therefore keep ONE
pyxrt.device and cache one hw_context+kernel per xclbin path, so all op engines share
the device and contexts are created at most once each. pyxrt is imported lazily so the
package is importable (for inspection) without the NPU runtime.
"""
import numpy as np


class NpuDevice:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        import pyxrt
        self.pyxrt = pyxrt
        self.d = pyxrt.device(0)
        self._kernels = {}      # xclbin_path -> kernel
        self.TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    def kernel(self, xclbin_path):
        """Register the xclbin (once) and return its kernel handle (cached)."""
        if xclbin_path not in self._kernels:
            xb = self.pyxrt.xclbin(xclbin_path)
            self.d.register_xclbin(xb)
            ctx = self.pyxrt.hw_context(self.d, xb.get_uuid())
            self._kernels[xclbin_path] = self.pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
        return self._kernels[xclbin_path]

    # --- buffer helpers ---
    def bo_in(self, kernel, gid, data_u16):
        """host_only input BO filled from a uint16 view of bf16 data, synced TO device."""
        b = self.pyxrt.bo(self.d, data_u16.nbytes, self.pyxrt.bo.host_only, kernel.group_id(gid))
        b.write(np.ascontiguousarray(data_u16).tobytes(), 0); b.sync(self.TO)
        return b

    def bo_instr(self, kernel, gid, instr_u32):
        b = self.pyxrt.bo(self.d, instr_u32.nbytes, self.pyxrt.bo.cacheable, kernel.group_id(gid))
        b.write(instr_u32.tobytes(), 0); b.sync(self.TO)
        return b

    def bo_out(self, kernel, gid, nbytes):
        return self.pyxrt.bo(self.d, nbytes, self.pyxrt.bo.host_only, kernel.group_id(gid))

    def bo_dummy(self, kernel, gid, nbytes=1):
        return self.pyxrt.bo(self.d, nbytes, self.pyxrt.bo.host_only, kernel.group_id(gid))
