import os
import time

from bench.sysmetrics import CpuSampler, peak_rss_kb


def test_peak_rss_kb_positive():
    assert peak_rss_kb(os.getpid()) > 0


def test_cpu_sampler_idle_frac_in_range():
    with CpuSampler() as s:
        time.sleep(0.2)
    assert 0.0 <= s.idle_frac <= 1.0
