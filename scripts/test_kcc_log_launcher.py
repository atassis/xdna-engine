# SPDX-License-Identifier: Apache-2.0
"""kcc_log_launcher.sh records one JSON line per kernel compile and runs the compile unchanged."""
import json
import os
import stat
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "kcc_log_launcher.sh"


def test_logs_and_runs(tmp_path):
    fake_cc = tmp_path / "clang++"
    fake_cc.write_text('#!/usr/bin/env bash\necho ran "$@" > "$KCC_TEST_OUT"\n')
    fake_cc.chmod(fake_cc.stat().st_mode | stat.S_IEXEC)
    src = tmp_path / "k.cc"
    src.write_text("int f() { return 1; }\n")
    log = tmp_path / "kcc.jsonl"
    env = dict(os.environ, KCC_LOG=str(log), KCC_NEXT="", KCC_TEST_OUT=str(tmp_path / "ran"))
    r = subprocess.run([str(LAUNCHER), str(fake_cc), str(src), "-c", "-o", str(tmp_path / "k.o"),
                        "-DDIM_K=64", "-MF", str(tmp_path / "k.o.d")], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "ran").read_text().startswith("ran ")
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["source"] == str(src)
    assert rec["args"] == ["-c", "-DDIM_K=64"]          # output and depfile paths removed
    assert len(rec["source_sha256"]) == 64
    assert rec["rc"] == 0
    assert 0 <= rec["end"] - rec["start"] < 30           # wall time of the compile, not exec'd away


def test_propagates_nonzero_exit_and_logs_it(tmp_path):
    fake_cc = tmp_path / "clang++"
    fake_cc.write_text("#!/usr/bin/env bash\nexit 7\n")
    fake_cc.chmod(fake_cc.stat().st_mode | stat.S_IEXEC)
    src = tmp_path / "k.cc"
    src.write_text("int f() { return 3; }\n")
    log = tmp_path / "kcc.jsonl"
    env = dict(os.environ, KCC_LOG=str(log), KCC_NEXT="")
    r = subprocess.run([str(LAUNCHER), str(fake_cc), str(src), "-c", "-o", str(tmp_path / "k.o")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 7
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["rc"] == 7


def test_pch_flags_before_source(tmp_path):
    # Default builds splice "-mno-vitis-headers -include-pch <pch>" BEFORE the source
    # (utils.py ~585-588), so the source is not argv[2].
    fake_cc = tmp_path / "clang++"
    fake_cc.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_cc.chmod(fake_cc.stat().st_mode | stat.S_IEXEC)
    src = tmp_path / "k.cc"
    src.write_text("int f() { return 2; }\n")
    log = tmp_path / "kcc.jsonl"
    env = dict(os.environ, KCC_LOG=str(log), KCC_NEXT="")
    r = subprocess.run([str(LAUNCHER), str(fake_cc), "-mno-vitis-headers", "-include-pch",
                        str(tmp_path / "x.pch"), str(src), "-c", "-o", str(tmp_path / "k.o")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["source"] == str(src)
    assert rec["args"] == ["-mno-vitis-headers", "-c"]   # PCH path dropped, source removed
