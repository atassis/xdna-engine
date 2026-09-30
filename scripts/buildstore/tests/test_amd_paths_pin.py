# SPDX-License-Identifier: Apache-2.0
"""iron_require_pin's IRON_PIN_VERIFIED shortcut (amd_paths.sh): a replay sandbox never has
.git, so the orchestrator verifies the pin once and hands the sha down instead of the function
re-running git itself."""
import pathlib, subprocess

AMD_PATHS = pathlib.Path(__file__).resolve().parents[2] / "amd_paths.sh"
WANT = "deadbeefdeadbeef00000000000000000000000"


def run(tmp_path, extra_env):
    lock = tmp_path / "toolchain.lock"
    lock.write_text(f"IRON_FORK_COMMIT={WANT}   # fake, for the test\n")
    env = dict(extra_env, PATH="/usr/bin:/bin", IRON_LOCK=str(lock), IRON_DIR="/nonexistent")
    script = f'. "{AMD_PATHS}"; iron_require_pin'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)


def test_matching_verified_pin_short_circuits_without_git(tmp_path):
    r = run(tmp_path, {"IRON_PIN_VERIFIED": WANT})
    assert r.returncode == 0, r.stderr


def test_mismatched_verified_pin_is_an_error(tmp_path):
    r = run(tmp_path, {"IRON_PIN_VERIFIED": "stalestalestale"})
    assert r.returncode != 0
    assert "stale" in r.stderr.lower()


def test_unset_verified_pin_falls_back_to_git_and_fails_on_nonexistent_dir(tmp_path):
    r = run(tmp_path, {})
    assert r.returncode != 0
