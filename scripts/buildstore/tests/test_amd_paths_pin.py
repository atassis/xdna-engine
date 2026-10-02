# SPDX-License-Identifier: Apache-2.0
"""Ambient identity values never replace IRON source validation."""
import pathlib, subprocess

AMD_PATHS = pathlib.Path(__file__).resolve().parents[2] / "amd_paths.sh"
WANT = "a" * 40


def run(tmp_path, extra_env):
    lock = tmp_path / "toolchain.lock"
    lock.write_text(f"IRON_SOURCE_COMMIT={WANT}   # fake, for the test\n")
    env = dict(extra_env, PATH="/usr/bin:/bin", IRON_LOCK=str(lock), IRON_DIR="/nonexistent")
    script = f'. "{AMD_PATHS}"; iron_require_source'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)


def test_ambient_pinned_identity_cannot_bypass_a_missing_source(tmp_path):
    r = run(tmp_path, {"IRON_SOURCE_IDENTITY": f"pinned:{WANT}"})
    assert r.returncode != 0


def test_ambient_mismatched_identity_is_an_error(tmp_path):
    r = run(tmp_path, {"IRON_SOURCE_IDENTITY": "pinned:stalestalestale"})
    assert r.returncode != 0
    assert "missing or uninitialized" in r.stderr


def test_unset_identity_fails_on_a_missing_source(tmp_path):
    r = run(tmp_path, {})
    assert r.returncode != 0


def test_ambient_development_override_cannot_bypass_a_missing_source(tmp_path):
    identity = "override:" + "b" * 40
    rejected = run(tmp_path, {"IRON_SOURCE_IDENTITY": identity})
    assert rejected.returncode != 0
    accepted = run(tmp_path, {"IRON_SOURCE_IDENTITY": identity, "IRON_ALLOW_UNPINNED": "1"})
    assert accepted.returncode != 0


def test_ambient_dirty_override_cannot_bypass_a_missing_source(tmp_path):
    identity = "dirty:" + "b" * 40 + ":" + "c" * 64
    rejected = run(tmp_path, {"IRON_SOURCE_IDENTITY": identity, "IRON_ALLOW_DIRTY": "1"})
    assert rejected.returncode != 0
    bypass = run(tmp_path, {
        "IRON_SOURCE_IDENTITY": identity,
        "IRON_ALLOW_DIRTY": "1",
        "IRON_ALLOW_UNPINNED": "1",
    })
    assert bypass.returncode != 0
