"""Process/system metric samplers: peak RSS + CPU-idle fraction + per-process CPU time."""
import os

_TICKS = os.sysconf("SC_CLK_TCK")


def peak_rss_kb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1])
    return 0


def proc_cpu_ms(pid):
    """utime+stime of pid AND its whole thread group, in ms.

    Divided by tokens generated this gives host CPU-ms per token -- the direct
    measure of how much work a nominally on-accelerator runtime still does on
    the CPU per step. A fully device-resident decode loop trends toward the
    cost of the submit/wait syscalls alone.
    """
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(") ", 1)[1].split()
    # after the comm field: state is [0], utime is [11], stime is [12]
    return (int(parts[11]) + int(parts[12])) * 1000.0 / _TICKS


class ProcCpu:
    """CPU-time delta of one process across a block."""

    def __init__(self, pid):
        self.pid = pid
        self.ms = 0.0

    def __enter__(self):
        self.t0 = proc_cpu_ms(self.pid)
        return self

    def __exit__(self, *a):
        try:
            self.ms = proc_cpu_ms(self.pid) - self.t0
        except (FileNotFoundError, ProcessLookupError, IndexError):
            self.ms = 0.0


def _cpu():
    f = list(map(int, open("/proc/stat").readline().split()[1:8]))
    return sum(f), f[3] + f[4]


class CpuSampler:
    def __enter__(self):
        self.t0, self.i0 = _cpu()
        return self

    def __exit__(self, *a):
        t1, i1 = _cpu()
        dt = t1 - self.t0
        self.idle_frac = (i1 - self.i0) / dt if dt else 0.0
