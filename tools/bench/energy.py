"""RAPL package-energy sampler. The counter is user-readable on this box."""
import time

PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"
MAXR = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"


def _read(p):
    with open(p) as f:
        return int(f.read())


def readable():
    try:
        _read(PKG)
        return True
    except (PermissionError, FileNotFoundError):
        return False


class EnergyMeter:
    """Times the block; adds package energy when the RAPL counter is readable.

    The counter is root-only on some kernels/distros, so energy degrades to None
    rather than failing the whole measurement -- latency stays valid either way.
    Grant it with: sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
    """

    def __enter__(self):
        self.ok = readable()
        if self.ok:
            self.max = _read(MAXR)
            self.e0 = _read(PKG)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.t = time.perf_counter() - self.t0
        self.uj = ((_read(PKG) - self.e0) % self.max) if self.ok else None

    @property
    def joules(self):
        return self.uj / 1e6 if self.uj is not None else None

    @property
    def watts(self):
        j = self.joules
        return (j / self.t if self.t else 0.0) if j is not None else None
