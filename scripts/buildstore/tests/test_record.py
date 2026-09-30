import json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import record

def build(tmp, script):
    out = tmp / "out"; out.mkdir(exist_ok=True)
    return record.run(["bash", "-c", script], cwd=tmp, env={"PATH": "/usr/bin:/bin"},
                      out_roots=[out], work_roots=[], cache_roots=[])

def test_inputs_recorded_and_outputs_excluded(tmp_path):
    (tmp_path / "a.h").write_text("1")
    m = build(tmp_path, "cat a.h > out/o; cat out/o > /dev/null; test -e missing.h; true")
    assert str(tmp_path / "a.h") in m["reads"]
    assert not any(p.startswith(str(tmp_path / "out")) for p in m["reads"])
    assert str(tmp_path / "missing.h") in m["absent"]
    assert any(p.endswith("/bash") for p in m["reads"])        # tools by content
    assert record.check(m) == []
    assert m["cwd"] == str(tmp_path) and m["writable"] == [str(tmp_path / "out")]

# One mutation per input class. Each must make check() report exactly that change.
def test_each_class_goes_red(tmp_path):
    (tmp_path / "a.h").write_text("1"); (tmp_path / "d").mkdir()
    m = build(tmp_path, "cat a.h > out/o; ls d > /dev/null; test -e missing.h; true")
    (tmp_path / "a.h").write_text("2")
    assert record.check(m) == [f"file {tmp_path}/a.h"]
    (tmp_path / "a.h").write_text("1")
    (tmp_path / "d" / "new").write_text("")
    assert record.check(m) == [f"dir {tmp_path}/d"]
    os.remove(tmp_path / "d" / "new")
    (tmp_path / "missing.h").write_text("")
    assert record.check(m) == [f"appeared {tmp_path}/missing.h"]
    os.remove(tmp_path / "missing.h")
    m2 = dict(m, env={"PATH": "/bin"})
    assert record.check(m2, env={"PATH": "/usr/bin:/bin"}) == ["env PATH"]
    # A recipe's inline parameters (WINDOW_RUNGS=..., PRECISION=...) live in argv, not in a file.
    assert record.check(m, argv=m["argv"][:-1] + [m["argv"][-1] + " X=1"]) == ["argv"]

def test_git_metadata_is_not_an_input(tmp_path):
    (tmp_path / ".git").mkdir(); (tmp_path / ".git" / "HEAD").write_text("x")
    m = build(tmp_path, "cat .git/HEAD > /dev/null")
    assert not any("/.git/" in p for p in m["reads"])

def test_sabotaged_classes_are_caught(tmp_path, monkeypatch):
    """The mutation test above is only a gate if dropping a class makes it pass wrongly."""
    (tmp_path / "a.h").write_text("1")
    monkeypatch.setenv("BUILDSTORE_SABOTAGE", "reads")
    m = build(tmp_path, "cat a.h > out/o")
    (tmp_path / "a.h").write_text("2")
    assert record.check(m) == []          # sabotaged: the change is invisible, as it must be

def test_hermetic_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KNOB", "1")
    env = record.hermetic_env(os.environ, ["PATH", "HOME"])
    assert "SECRET_KNOB" not in env and env["PATH"] == os.environ["PATH"]
