import os, pathlib, subprocess, sys, time
CLI = pathlib.Path(__file__).resolve().parents[2] / "buildstore.py"

def cli(tmp, *args):
    env = dict(os.environ, BUILDSTORE_CAS=str(tmp / "cas"), BUILDSTORE_REPO=str(tmp))
    return subprocess.run([sys.executable, str(CLI), *args], env=env, capture_output=True,
                          text=True, check=True).stdout

def test_second_run_hits_and_input_change_misses(tmp_path):
    (tmp_path / "in.txt").write_text("1")
    tsv = tmp_path / "r.tsv"; tsv.write_text("fake\tcat \"$REPO/in.txt\" > \"$OUT/o\"\n")
    a = cli(tmp_path, "build", "fake", "--recipes", str(tsv), "--out-root", str(tmp_path / "o"))
    assert a.startswith("BUILT fake ")
    b = cli(tmp_path, "build", "fake", "--recipes", str(tsv), "--out-root", str(tmp_path / "o"))
    assert b.startswith("HIT fake ") and b.split()[2] == a.split()[2]
    (tmp_path / "in.txt").write_text("2")
    c = cli(tmp_path, "build", "fake", "--recipes", str(tsv), "--out-root", str(tmp_path / "o"))
    assert c.startswith("BUILT fake ") and c.split()[2] != a.split()[2]
    assert (tmp_path / "o" / "fake" / "o").read_text() == "2"
