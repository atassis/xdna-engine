# SPDX-License-Identifier: Apache-2.0
"""repro_check.sh must pass on a deterministic builder and fail on a nondeterministic one."""
import os
import stat
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE / "repro_check.sh"

FAKE = """#!/usr/bin/env bash
# writes full.elf into the cwd; NONDET=1 adds a per-run random byte string
printf 'ELF-%s' "$1" > full.elf
[ "${NONDET:-0}" = 1 ] && head -c 16 /dev/urandom >> full.elf
printf 'x' > aie.xclbin
mkdir -p sim/reports; printf 'r' > "sim/reports/graph report.xpe"
[ "${NONDET_SUB:-0}" = 1 ] && head -c 16 /dev/urandom >> "sim/reports/graph report.xpe"
exit 0
"""


def _setup(tmp_path):
    f = tmp_path / "aiecc"
    f.write_text(FAKE)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    src = tmp_path / "aie.mlir"
    src.write_text("module {}\n")
    return f, src


def _run(tmp_path, **env):
    fake, src = _setup(tmp_path)
    e = dict(os.environ, AIECC_BIN=str(fake), PEANO_INSTALL_DIR="/nonexistent-ok", **env)
    return subprocess.run([str(CHECK), str(src), str(tmp_path / "out"), "--", "-j1"],
                          env=e, capture_output=True, text=True)


def test_identical_builds_pass(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "IDENTICAL full.elf" in r.stdout and "IDENTICAL aie.xclbin" in r.stdout


def test_nondeterministic_build_fails(tmp_path):
    r = _run(tmp_path, NONDET="1")
    assert r.returncode == 1
    assert "DIFF full.elf" in r.stdout


def test_requires_explicit_peano(tmp_path):
    fake, src = _setup(tmp_path)
    e = {k: v for k, v in os.environ.items() if k != "PEANO_INSTALL_DIR"}
    e["AIECC_BIN"] = str(fake)
    r = subprocess.run([str(CHECK), str(src), str(tmp_path / "o"), "--"], env=e,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "PEANO_INSTALL_DIR" in r.stderr


def test_nondeterminism_in_a_subdirectory_fails(tmp_path):
    r = _run(tmp_path, NONDET_SUB="1")
    assert r.returncode == 1
    assert "DIFF sim/reports/graph report.xpe" in r.stdout
