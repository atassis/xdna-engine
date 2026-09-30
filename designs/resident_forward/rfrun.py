"""One hw_context over a full ELF carrying several named control codes (`main:<name>`), shared BOs."""
import os
import subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import pyxrt
import rf_paths

BUILD = Path(os.environ.get("RF_BUILD", str(rf_paths.BUILD_ROOT)))
STATUS = os.environ.get("RF_AIE_STATUS", "")   # optional external device-status tool; unset = disabled


class Ctx:
    def __init__(self, dev, build, sizes, names):
        self.elf = pyxrt.elf(str(BUILD / build / "design.elf"))
        self.ctx = pyxrt.hw_context(dev, self.elf)
        self.kern = {n: pyxrt.ext.kernel(self.ctx, f"main:{n}") for n in names}
        self.bo = [pyxrt.bo(dev, s, pyxrt.bo.host_only, 0) for s in sizes]
        self.v = [np.frombuffer(b.map(), dtype=np.uint8) for b in self.bo]
        self.runs = {}
        for n, k in self.kern.items():
            r = pyxrt.run(k)
            for i, b in enumerate(self.bo):
                r.set_arg(i, b)
            self.runs[n] = r
        if "boot" in self.runs:
            self.run("boot")

    def put(self, i, x):
        self.v[i][:len(x)] = np.frombuffer(np.ascontiguousarray(x).tobytes(), np.uint8)
        self.bo[i].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def fill(self, i, byte):
        self.v[i][:] = byte
        self.bo[i].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def get(self, i):
        self.bo[i].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return self.v[i].copy()

    def run(self, name, timeout_ms=4000, snap_ms=900, snap_file=None):
        """Dispatch and wait. If still running after snap_ms, a side thread snapshots the AIE tile
        status (a hang is only observable while the context is live)."""
        r = self.runs[name]
        done = threading.Event()

        def snap():
            if not done.wait(snap_ms / 1e3):
                with open(snap_file, "w") as f:
                    subprocess.run([sys.executable, STATUS, "--all"], stdout=f, stderr=subprocess.STDOUT)
        th = threading.Thread(target=snap) if snap_file else None
        t0 = time.perf_counter()
        r.start()
        if th:
            th.start()
        try:
            st = r.wait(timeout_ms)
        finally:
            done.set()
            if th:
                th.join()
        dt = time.perf_counter() - t0
        if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise TimeoutError(f"{name}: state {st} after {dt * 1e3:.0f} ms")
        return dt
