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
    def __enter__(self):
        self.max = _read(MAXR)
        self.e0 = _read(PKG)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.t = time.perf_counter() - self.t0
        self.uj = (_read(PKG) - self.e0) % self.max

    @property
    def joules(self):
        return self.uj / 1e6

    @property
    def watts(self):
        return self.joules / self.t if self.t else 0.0
