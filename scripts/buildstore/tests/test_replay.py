import json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); import record

def test_replay_passes_on_full_manifest_and_fails_on_a_dropped_read(tmp_path):
    (tmp_path / "a.h").write_text("1"); out = tmp_path / "out"; out.mkdir()
    cmd = ["bash", "-c", f"cat {tmp_path}/a.h > {out}/o"]
    m = record.run(cmd, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, out_roots=[out],
                   work_roots=[], cache_roots=[])
    mf = tmp_path / "m.json"; mf.write_text(json.dumps(m))
    r = subprocess.run([str(HERE / "replay_bwrap.sh"), str(mf), str(out), str(tmp_path / "r")])
    assert r.returncode == 0
    del m["reads"][str(tmp_path / "a.h")]; mf.write_text(json.dumps(m))
    r = subprocess.run([str(HERE / "replay_bwrap.sh"), str(mf), str(out), str(tmp_path / "r2")])
    assert r.returncode != 0


def test_replay_scales_past_bwrap_9000_arg_limit(tmp_path):
    # bwrap 0.13.0 aborts at ~9000 args (one --ro-bind pair per recorded path); a real recipe
    # (gemma3-270m-decode) records ~9.6k. Synthesize >9000 recorded reads and check replay
    # still succeeds.
    many = tmp_path / "many"; many.mkdir()
    (tmp_path / "a.h").write_text("1"); out = tmp_path / "out"; out.mkdir()
    cmd = ["bash", "-c", f"cat {tmp_path}/a.h > {out}/o"]
    m = record.run(cmd, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, out_roots=[out],
                   work_roots=[], cache_roots=[])
    for i in range(9200):
        p = many / f"f{i}"
        p.write_text(str(i))
        m["reads"][str(p)] = record.sha_file(p)
    mf = tmp_path / "m.json"; mf.write_text(json.dumps(m))
    r = subprocess.run([str(HERE / "replay_bwrap.sh"), str(mf), str(out), str(tmp_path / "r")])
    assert r.returncode == 0
