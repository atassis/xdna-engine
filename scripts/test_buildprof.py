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


# aiecc's --resume allow-list (CommandLineOptions.h resumePassthroughKind) does not include
# --profile: `--resume` plus any other graph-shaping flag is a hard error. This fake mirrors that.
FAKE_AIECC_RESUME_GUARD = """#!/usr/bin/env bash
resume=0 profile=0
for a in "$@"; do
  case "$a" in --resume|--resume=*) resume=1 ;; esac
  [ "$a" = "--profile" ] && profile=1
done
if [ "$resume" = 1 ] && [ "$profile" = 1 ]; then
  echo "aiecc: --resume rejects other arguments: --profile" >&2
  exit 1
fi
exit 0
"""


def _fake_resume_guard(tmp_path):
    p = tmp_path / "aiecc"
    p.write_text(FAKE_AIECC_RESUME_GUARD)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_shim_passes_resume_through_unmodified(tmp_path):
    e = dict(os.environ, BUILDPROF_REAL_AIECC=str(_fake_resume_guard(tmp_path)),
             BUILDPROF_DIR=str(tmp_path / "logs"))
    r = subprocess.run([str(SHIM), "--resume=m.json"], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


import importlib.util


def _summ():
    spec = importlib.util.spec_from_file_location("bps", HERE / "buildprof_summarize.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_stage_of_classifies_edges():
    s = _summ().stage_of
    assert s("placed.mlir") == "front"
    assert s("lowered_main_core_0_2.mlir") == "per-core"
    assert s("opted_main_core_0_2.ll") == "per-core"
    assert s("npu_dma_lowered.mlir") == "control-code"
    assert s("npu_materialized_2.mlir") == "control-code"
    assert s("full.elf") == "package"
    assert s("partition_main.json") == "package"
    assert s("something_new") == "other"


def test_parse_profile_stops_at_total_row():
    # The real `total` row has an empty dRSS column (3 tokens), so ROW never matches it; a
    # digit-led line after it must not be swept in as if it were still inside the block.
    text = (
        "aiecc: profile (per-edge time and resident memory):\n"
        "        ms    dRSS MiB    peak MiB  edge\n"
        "       120        10.0       110.0  placed.mlir\n"
        "     12160                   900.0  total\n"
        "        5         1.0         2.0  not_a_profile_row.mlir\n"
    )
    rows = _summ().parse_profile(text)
    assert rows == [("placed.mlir", 120, 110.0)]


def test_parse_profile_rows(tmp_path):
    _run(tmp_path, ["aie.mlir"])
    log = next((tmp_path / "logs").glob("aiecc-*.log"))
    rows = _summ().parse_profile(log.read_text())
    assert rows == [("placed.mlir", 120, 110.0), ("lowered_main_core_0_2.mlir", 3000, 400.0),
                    ("npu_dma_lowered.mlir", 9000, 900.0), ("full.elf", 40, 900.0)]


def test_summarize_artifact_dir(tmp_path):
    art = tmp_path / "s0" / "qwen3-0.6b-decode"
    art.mkdir(parents=True)
    _run(art, ["aie.mlir"])                       # writes art/logs/aiecc-*.log
    (art / "logs").rename(art / "aiecc-logs")
    (art / "time.txt").write_text(
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:00.00\n"
        "\tMaximum resident set size (kbytes): 2048000\nrc=0\n")
    out = _summ().summarize(tmp_path / "s0")
    lines = out.strip().splitlines()
    assert lines[0] == "artifact\tstage\tms\tpeak_mib"
    assert "qwen3-0.6b-decode\tcontrol-code\t9000\t900.0" in lines
    assert "qwen3-0.6b-decode\tper-core\t3000\t400.0" in lines
    assert "qwen3-0.6b-decode\tTOTAL_WALL\t60000\t2000.0" in lines
    assert any(l.startswith("qwen3-0.6b-decode\tOUTSIDE_AIECC\t") for l in lines)
