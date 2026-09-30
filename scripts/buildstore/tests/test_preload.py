import os, subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
SO = subprocess.run([str(HERE.parent / "build_preload.sh")], check=True,
                    capture_output=True, text=True).stdout.strip()

def run_logged(tmp, argv, cwd=None):
    rec = tmp / "rec"; rec.mkdir(exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "LD_PRELOAD": SO, "REC_DIR": str(rec)}
    subprocess.run(argv, env=env, cwd=cwd or tmp, check=True)
    events = set()
    for f in rec.iterdir():
        for line in f.read_text().splitlines():
            events.add((line[0], line[2:]))
    return events

def test_bash_child_python_reads_and_writes(tmp_path):
    (tmp_path / "in.txt").write_text("x")
    os.symlink("in.txt", tmp_path / "link.txt")
    script = (f"cat link.txt > out.txt; "
              f"{sys.executable} -c \"import os; os.path.exists('nope'); os.listdir('.')\"")
    ev = run_logged(tmp_path, ["bash", "-c", script])
    t = str(tmp_path)
    assert ("R", f"{t}/in.txt") in ev                     # read through a symlink, by a child
    assert ("L", f"{t}/link.txt\t{t}/in.txt") in ev
    assert ("W", f"{t}/out.txt") in ev
    assert ("A", f"{t}/nope") in ev                       # stat() miss from Python
    assert ("D", t) in ev                                 # os.listdir
    assert any(k == "M" and p.endswith("/bash") for k, p in ev)   # the executable itself
    assert any(k == "M" and "libc.so" in p for k, p in ev)        # a DT_NEEDED library

def test_openat_relative_to_dirfd(tmp_path):
    (tmp_path / "sub").mkdir(); (tmp_path / "sub" / "f").write_text("y")
    code = "import os; fd=os.open('sub', os.O_RDONLY); os.close(os.open('f', os.O_RDONLY, dir_fd=fd))"
    ev = run_logged(tmp_path, [sys.executable, "-c", code])
    assert ("R", f"{tmp_path}/sub/f") in ev
