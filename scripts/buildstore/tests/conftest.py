# SPDX-License-Identifier: Apache-2.0
# tmp_path defaults under /tmp, which record.py's EXCLUDED treats as non-input (real recipe
# scratch); route it elsewhere so a test's own inputs aren't swallowed by that exclusion.
import os

def pytest_configure(config):
    if not config.option.basetemp:
        d = "/mnt/data/xdna/scratch/p1-impl/pytest-tmp"
        os.makedirs(d, exist_ok=True)
        config.option.basetemp = d
