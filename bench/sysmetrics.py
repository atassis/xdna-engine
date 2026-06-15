"""Process/system metric samplers: peak RSS + CPU-idle fraction."""


def peak_rss_kb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1])
    return 0


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
