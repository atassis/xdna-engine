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

def test_verify_passes_on_a_deterministic_hit_and_catches_a_nondeterministic_one(tmp_path):
    """--verify is the P1 sufficiency gate: cold-rebuild a HIT into scratch and require the
    same identity.of(). A deterministic recipe passes; one whose output varies per run (what
    the stored manifest can't see) must fail the gate, not silently HIT."""
    tsv = tmp_path / "r.tsv"
    tsv.write_text("fake\techo fixed > \"$OUT/o\"\n"
                   "flaky\techo \"$RANDOM\" > \"$OUT/o\"\n")
    out = tmp_path / "o"
    cli(tmp_path, "build", "fake", "flaky", "--recipes", str(tsv), "--out-root", str(out))
    cli(tmp_path, "build", "fake", "--recipes", str(tsv), "--out-root", str(out), "--verify")
    r = subprocess.run([sys.executable, str(CLI), "build", "flaky", "--recipes", str(tsv),
                       "--out-root", str(out), "--verify"],
                       env=dict(os.environ, BUILDSTORE_CAS=str(tmp_path / "cas"),
                               BUILDSTORE_REPO=str(tmp_path)),
                       capture_output=True, text=True)
    assert r.returncode != 0 and "cold identity" in r.stderr


def test_replay_subcommand_dispatches_to_replay_py(tmp_path):
    """`buildstore.py replay` is the diagnostic surface for Task 3's mechanism, not a gate --
    just check it dispatches and reports success on a trivial manifest."""
    import json
    HERE = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(HERE)); import record  # noqa: E402
    (tmp_path / "a.h").write_text("1"); out = tmp_path / "out"; out.mkdir()
    cmd = ["bash", "-c", f"cat {tmp_path}/a.h > {out}/o"]
    m = record.run(cmd, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, out_roots=[out],
                   work_roots=[], cache_roots=[])
    mf = tmp_path / "m.json"; mf.write_text(json.dumps(m))
    r = subprocess.run([sys.executable, str(CLI), "replay", str(mf), str(out),
                       str(tmp_path / "r")])
    assert r.returncode == 0


def test_iron_pin_verified_is_a_key_input(tmp_path):
    """A repo whose scripts/amd_paths.sh gates on IRON_PIN_VERIFIED: build injects the verified
    sha, so a stale pin (lock changed under it) is a cache MISS, not a silent stale hit."""
    (tmp_path / "scripts").mkdir()
    amd_paths = tmp_path / "scripts" / "amd_paths.sh"
    amd_paths.write_text(
        'iron_require_pin() {\n'
        '  want=$(sed -n \'s/^IRON_FORK_COMMIT=\\([0-9a-f]\\{7,\\}\\).*/\\1/p\' toolchain.lock | head -1)\n'
        '  [ -n "${IRON_PIN_VERIFIED:-}" ] && { [ "$IRON_PIN_VERIFIED" = "$want" ] && return 0 || return 1; }\n'
        '  return 0\n'
        '}\n'
    )
    (tmp_path / "toolchain.lock").write_text("IRON_FORK_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
    tsv = tmp_path / "r.tsv"
    tsv.write_text('fake\t. "$REPO/scripts/amd_paths.sh" && echo "$IRON_PIN_VERIFIED" > "$OUT/o"\n')
    a = cli(tmp_path, "build", "fake", "--recipes", str(tsv), "--out-root", str(tmp_path / "o"))
    assert a.startswith("BUILT fake ")
    assert (tmp_path / "o" / "fake" / "o").read_text().strip() == "a" * 40
    (tmp_path / "toolchain.lock").write_text("IRON_FORK_COMMIT=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n")
    b = cli(tmp_path, "build", "fake", "--recipes", str(tsv), "--out-root", str(tmp_path / "o"))
    assert b.startswith("BUILT fake ") and b.split()[2] != a.split()[2]
    assert (tmp_path / "o" / "fake" / "o").read_text().strip() == "b" * 40

def test_xdna_cache_resolved_and_baked_in(tmp_path):
    """cache_env.sh's own default resolves XDNA_CACHE via the worktree's .git, which a replay
    sandbox never has (same class as the IRON pin). build() must resolve it once, outside any
    sandbox, and put a CONCRETE path in the manifest env."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "cache_env.sh").write_text(
        'export XDNA_CACHE="${XDNA_CACHE:-/resolved/by/cache_env}"\n')
    tsv = tmp_path / "r.tsv"
    tsv.write_text('fake\t. "$REPO/scripts/cache_env.sh" && echo "$XDNA_CACHE" > "$OUT/o"\n')
    a = cli(tmp_path, "build", "fake", "--recipes", str(tsv), "--out-root", str(tmp_path / "o"))
    assert a.startswith("BUILT fake ")
    assert (tmp_path / "o" / "fake" / "o").read_text().strip() == "/resolved/by/cache_env"

def test_mlir_aie_instance_resolved_and_baked_in(tmp_path):
    """toolchain_up.sh's cached branch is gated on [ -e ]/[ -L ] probes that succeed against
    real files a replay sandbox can't see (same class as the other two). build() must resolve
    the instance dir once and put its CONCRETE path in the manifest env."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "toolchain_up.sh").write_text('#!/bin/bash\necho /resolved/instance\n')
    tsv = tmp_path / "r.tsv"
    tsv.write_text('fake\techo "$MLIR_AIE_INSTANCE" > "$OUT/o"\n')
    a = cli(tmp_path, "build", "fake", "--recipes", str(tsv), "--out-root", str(tmp_path / "o"))
    assert a.startswith("BUILT fake ")
    assert (tmp_path / "o" / "fake" / "o").read_text().strip() == "/resolved/instance"
