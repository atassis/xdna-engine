import json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); import identity, record

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


def test_replay_compares_by_identity_not_bytes(tmp_path):
    """meta.json's generator provenance is git state, not reproducible in the sandbox --
    identity.py already drops it. Replay must pass on a run whose meta.json differs ONLY in
    that field, and still catch a real content change (non-provenance key)."""
    out = tmp_path / "out"; out.mkdir()
    script = (f'echo -n fixed > {out}/f; '
              f'printf \'{{"toolchain": {{"x": %s}}, "k": 1}}\' "$RANDOM" > {out}/meta.json')
    cmd = ["bash", "-c", script]
    m = record.run(cmd, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, out_roots=[out],
                   work_roots=[], cache_roots=[])
    ref_id = identity.of(out)
    mf = tmp_path / "m.json"; mf.write_text(json.dumps(m))
    r = subprocess.run([str(HERE / "replay_bwrap.sh"), str(mf), str(out), str(tmp_path / "r")])
    assert r.returncode == 0
    assert identity.of(out) == ref_id            # ref dir is untouched by replay

    # A real content change (not provenance) must still be caught.
    (out / "f").write_text("mutated")
    assert identity.of(out) != ref_id
    r = subprocess.run([str(HERE / "replay_bwrap.sh"), str(mf), str(out), str(tmp_path / "r3")])
    assert r.returncode != 0


def test_replay_furnishes_venv_python_and_mlir_distro(tmp_path):
    """Two unhooked-probe classes (existence-tested, never opened, so no manifest records them):
    the .venv-iron fallback and toolchain_up.sh's `[ -e mlir-tblgen ]` cached-instance gate. A
    recipe that itself probes for them must see them in the sandbox too."""
    repo = tmp_path / "repo"; venv = repo / ".venv-iron" / "bin"; venv.mkdir(parents=True)
    (venv / "python").write_text("#!/bin/sh\necho ok\n"); (venv / "python").chmod(0o755)
    cache = tmp_path / "cache"; tblgen_dir = cache / "mlir-distro" / "x" / "mlir" / "bin"
    tblgen_dir.mkdir(parents=True); (tblgen_dir / "mlir-tblgen").write_text("bin"); (tblgen_dir / "mlir-tblgen").chmod(0o755)
    out = tmp_path / "out"; out.mkdir()
    script = (f'[ -x "$REPO/.venv-iron/bin/python" ] && '
              f'[ -e "$XDNA_CACHE/mlir-distro/x/mlir/bin/mlir-tblgen" ] && '
              f'echo ok > {out}/o')
    cmd = ["bash", "-c", script]
    m = record.run(cmd, cwd=tmp_path, env={"PATH": "/usr/bin:/bin", "REPO": str(repo),
                                          "XDNA_CACHE": str(cache)},
                   out_roots=[out], work_roots=[], cache_roots=[])
    mf = tmp_path / "m.json"; mf.write_text(json.dumps(m))
    r = subprocess.run([str(HERE / "replay_bwrap.sh"), str(mf), str(out), str(tmp_path / "r")])
    assert r.returncode == 0


def test_replay_furnishes_whole_mlir_aie_instance(tmp_path):
    """The instance is content-addressed (buildstore.py keys it via MLIR_AIE_INSTANCE), so
    replay furnishes its WHOLE tree rather than tracking each internal existence probe --
    there are too many (aie-translate, vendored symlinks, backfill markers) to name one by
    one, unlike the single-file venv-python/mlir-tblgen cases."""
    inst = tmp_path / "instance" / "build" / "bin"; inst.mkdir(parents=True)
    (inst / "aie-translate").write_text("bin")
    out = tmp_path / "out"; out.mkdir()
    # toolchain_up.sh's _link_vendored_tools backfills a missing symlink into the shared
    # instance in place, even on the cached path -- the instance must be WRITABLE, not RO.
    script = (f'[ -e "$MLIR_AIE_INSTANCE/build/bin/aie-translate" ] && '
              f'ln -sfn /nonexistent "$MLIR_AIE_INSTANCE/build/bin/backfilled" && '
              f'echo ok > {out}/o')
    cmd = ["bash", "-c", script]
    m = record.run(cmd, cwd=tmp_path, env={"PATH": "/usr/bin:/bin",
                                          "MLIR_AIE_INSTANCE": str(tmp_path / "instance")},
                   out_roots=[out], work_roots=[], cache_roots=[])
    mf = tmp_path / "m.json"; mf.write_text(json.dumps(m))
    r = subprocess.run([str(HERE / "replay_bwrap.sh"), str(mf), str(out), str(tmp_path / "r")])
    assert r.returncode == 0


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
