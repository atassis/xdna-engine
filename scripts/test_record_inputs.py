# SPDX-License-Identifier: Apache-2.0
"""record_inputs.py: every input class is detected when it changes, and only then.

Each mutation test is paired with a sabotage test that disables that input class in the recorder
and shows the same mutation goes UNDETECTED -- so a green mutation test is evidence, not luck.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "record_inputs.py"

GEN = """import os, helper
os.environ.setdefault("GEN_DEFAULTED", "4")
int(os.environ["GEN_DEFAULTED"])
os.environ["GEN_ASSIGNED"] = "x"
os.environ["GEN_ASSIGNED"]
v = os.environ.get("GEN_FLAG", "0")
if os.environ.get("GEN_COPY_ENV"):
    os.environ.copy()
data = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")).read()
open(os.environ["GEN_OUT"], "w").write(helper.render(v, data))
"""


def _project(tmp_path):
    (tmp_path / "gen.py").write_text(GEN)
    (tmp_path / "helper.py").write_text("def render(v, d):\n    return f'{v}:{d}'\n")
    (tmp_path / "data.txt").write_text("hello")
    return tmp_path


def _env(tmp_path, **extra):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("GEN_", "UNRELATED"))}
    e.update(GEN_OUT=str(tmp_path / "out.mlir"), PYTHONPATH=str(tmp_path), **extra)
    return e


def _record(tmp_path, **extra):
    r = subprocess.run([sys.executable, str(REC), "record", "--manifest", str(tmp_path / "m.json"),
                        "--", str(tmp_path / "gen.py")], env=_env(tmp_path, **extra),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _check(tmp_path, **extra):
    return subprocess.run([sys.executable, str(REC), "check", "--manifest", str(tmp_path / "m.json")],
                          env=_env(tmp_path, **extra), capture_output=True, text=True)


def test_unchanged_is_a_hit(tmp_path):
    _record(_project(tmp_path))
    assert _check(tmp_path).returncode == 0


def test_read_env_change_is_a_miss(tmp_path):
    _record(_project(tmp_path))
    r = _check(tmp_path, GEN_FLAG="1")
    assert r.returncode == 1 and "GEN_FLAG" in r.stdout


def test_unread_env_change_is_a_hit(tmp_path):
    _record(_project(tmp_path))
    assert _check(tmp_path, UNRELATED="1").returncode == 0


def test_environ_copy_does_not_record_every_variable(tmp_path):
    _record(_project(tmp_path), GEN_COPY_ENV="1")
    assert _check(tmp_path, GEN_COPY_ENV="1", UNRELATED="1").returncode == 0


def test_imported_module_change_is_a_miss(tmp_path):
    _record(_project(tmp_path))
    (tmp_path / "helper.py").write_text("def render(v, d):\n    return f'{d}:{v}'\n")
    r = _check(tmp_path)
    assert r.returncode == 1 and "helper.py" in r.stdout


def test_data_file_change_is_a_miss(tmp_path):
    _record(_project(tmp_path))
    (tmp_path / "data.txt").write_text("changed")
    assert _check(tmp_path).returncode == 1


def test_written_output_is_not_an_input(tmp_path):
    _record(_project(tmp_path))
    (tmp_path / "out.mlir").write_text("edited by hand")
    assert _check(tmp_path).returncode == 0


def test_sabotaged_env_recording_misses_the_env_change(tmp_path):
    _record(_project(tmp_path), RECORD_INPUTS_SABOTAGE="env")
    assert _check(tmp_path, GEN_FLAG="1").returncode == 0


def test_sabotaged_file_recording_misses_the_data_change(tmp_path):
    _record(_project(tmp_path), RECORD_INPUTS_SABOTAGE="files")
    (tmp_path / "data.txt").write_text("changed")
    assert _check(tmp_path).returncode == 0


def test_self_defaulted_env_is_not_an_input(tmp_path):
    _record(_project(tmp_path))
    assert _check(tmp_path).returncode == 0          # GEN_DEFAULTED absent outside, as recorded


def test_externally_set_defaulted_env_is_an_input(tmp_path):
    _record(_project(tmp_path))
    r = _check(tmp_path, GEN_DEFAULTED="8")
    assert r.returncode == 1 and "GEN_DEFAULTED" in r.stdout


def test_env_assigned_before_read_is_not_an_input(tmp_path):
    _record(_project(tmp_path))
    assert _check(tmp_path, GEN_ASSIGNED="outside").returncode == 0
