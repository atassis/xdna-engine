# SPDX-License-Identifier: Apache-2.0
"""build --all --report and repin-report: two `build --report` snapshots compared by IDENTITY,
never bytes -- IDENTICAL/CHANGED/FAILED per recipe, plus a device-work summary."""
import json, os, pathlib, subprocess, sys
CLI = pathlib.Path(__file__).resolve().parents[2] / "buildstore.py"


def cli(tmp, *args):
    env = dict(os.environ, BUILDSTORE_CAS=str(tmp / "cas"), BUILDSTORE_REPO=str(tmp))
    return subprocess.run([sys.executable, str(CLI), *args], env=env, capture_output=True,
                          text=True, check=True).stdout


def test_build_all_report_and_repin_report(tmp_path):
    (tmp_path / "in_same.txt").write_text("1")
    (tmp_path / "in_changed.txt").write_text("1")
    tsv = tmp_path / "r.tsv"
    tsv.write_text(
        "same\tcat \"$REPO/in_same.txt\" > \"$OUT/o\"\n"
        "changed\tcat \"$REPO/in_changed.txt\" > \"$OUT/o\"\n"
        "broken\texit 3\n"
    )
    out_a = tmp_path / "oa"; out_b = tmp_path / "ob"
    ra = tmp_path / "a.json"; rb = tmp_path / "b.json"
    cli(tmp_path, "build", "--all", "--recipes", str(tsv), "--out-root", str(out_a),
        "--report", str(ra))
    rep_a = json.loads(ra.read_text())
    assert rep_a["same"]["status"] == "BUILT"
    assert rep_a["broken"]["status"] == "FAILED"

    (tmp_path / "in_changed.txt").write_text("2")           # only "changed" moves
    cli(tmp_path, "build", "--all", "--recipes", str(tsv), "--out-root", str(out_b),
        "--report", str(rb))
    rep_b = json.loads(rb.read_text())

    out = cli(tmp_path, "repin-report", str(ra), str(rb)).strip().splitlines()
    assert "IDENTICAL same" in out
    assert any(line.startswith("CHANGED changed ") and "o" in line for line in out)
    assert "FAILED broken" in out
    assert out[-1] == "device work: changed"


def test_build_requires_recipe_or_all(tmp_path):
    tsv = tmp_path / "r.tsv"; tsv.write_text("fake\techo hi\n")
    env = dict(os.environ, BUILDSTORE_CAS=str(tmp_path / "cas"), BUILDSTORE_REPO=str(tmp_path))
    r = subprocess.run([sys.executable, str(CLI), "build", "--recipes", str(tsv),
                       "--out-root", str(tmp_path / "o")], env=env, capture_output=True, text=True)
    assert r.returncode != 0
