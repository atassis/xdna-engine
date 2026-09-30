# SPDX-License-Identifier: Apache-2.0
# tmp_path defaults under /tmp, which record.py's EXCLUDED treats as non-input (real recipe
# scratch); route it elsewhere so a test's own inputs aren't swallowed by that exclusion.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "lib"))
from data_root import XDNA_SCRATCH  # noqa: E402


def pytest_configure(config):
    if not config.option.basetemp:
        d = str(XDNA_SCRATCH / "p1-impl" / "pytest-tmp")
        os.makedirs(d, exist_ok=True)
        config.option.basetemp = d
