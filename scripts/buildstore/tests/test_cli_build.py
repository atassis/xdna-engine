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
