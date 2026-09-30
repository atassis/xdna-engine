# SPDX-License-Identifier: Apache-2.0
"""Tests for the S0 build-profiling shim and summarizer. No toolchain, no device."""
import os
import stat
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM = HERE / "buildprof_shim.sh"

FAKE_AIECC = """#!/usr/bin/env bash
echo "fake-args: $*" >&2
if [ "$1" = "--version" ]; then echo "  git SHA:  8e3958b596a"; exit 0; fi
cat >&2 <<'EOF'
aiecc: profile (per-edge time and resident memory):
        ms    dRSS MiB    peak MiB  edge
       120        10.0       110.0  placed.mlir
      3000        50.0       400.0  lowered_main_core_0_2.mlir
      9000       200.0       900.0  npu_dma_lowered.mlir
        40         1.0       900.0  full.elf
     12160                   900.0  total
EOF
exit "${FAKE_RC:-0}"
"""


def _fake(tmp_path):
    p = tmp_path / "aiecc"
    p.write_text(FAKE_AIECC)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _run(tmp_path, args, **env):
    e = dict(os.environ, BUILDPROF_REAL_AIECC=str(_fake(tmp_path)),
             BUILDPROF_DIR=str(tmp_path / "logs"), **env)
    return subprocess.run([str(SHIM), *args], env=e, capture_output=True, text=True)


def test_shim_appends_profile_and_logs(tmp_path):
    r = _run(tmp_path, ["aie.mlir", "-j4"])
    assert r.returncode == 0
    assert "fake-args: aie.mlir -j4 --profile --no-progress" in r.stderr
    logs = list((tmp_path / "logs").glob("aiecc-*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert text.startswith("argv: aie.mlir -j4")
    assert "npu_dma_lowered.mlir" in text
    assert "rc: 0" in text and "wall_ms:" in text


def test_shim_passes_version_through_unlogged(tmp_path):
    r = _run(tmp_path, ["--version"])
    assert r.returncode == 0 and "git SHA:  8e3958b596a" in r.stdout
    assert not (tmp_path / "logs").exists()


def test_shim_propagates_failure(tmp_path):
    r = _run(tmp_path, ["aie.mlir"], FAKE_RC="3")
    assert r.returncode == 3
    assert "rc: 3" in next((tmp_path / "logs").glob("aiecc-*.log")).read_text()
