# SPDX-License-Identifier: Apache-2.0
"""gate-record/gate-status: device gate results keyed by artifact IDENTITY (provenance-blind)
plus driver srcversion and firmware version, so a pin whose bytes are unchanged reuses a gate
without re-running it, and a real driver/firmware move is detected. Off-device via
BUILDSTORE_DRIVER_FILE/BUILDSTORE_FW_FILE."""
import json, os, pathlib, subprocess, sys
CLI = pathlib.Path(__file__).resolve().parents[2] / "buildstore.py"


def cli(tmp, env, *args):
    full = dict(os.environ, BUILDSTORE_CAS=str(tmp / "cas"), **env)
    return subprocess.run([sys.executable, str(CLI), *args], env=full, capture_output=True,
                          text=True, check=True).stdout


def art(tmp, name, elf=b"E", toolchain="a"):
    d = tmp / name; d.mkdir(exist_ok=True)
    (d / "decode.elf").write_bytes(elf)
    (d / "meta.json").write_text(json.dumps({"sha256": "s", "toolchain": {"instance": toolchain}}))
    return d


def test_gate_record_and_status(tmp_path):
    driver = tmp_path / "driver"; driver.write_text("drv1\n")
    fw = tmp_path / "fw"; fw.write_text("fw1\n")
    env = {"BUILDSTORE_DRIVER_FILE": str(driver), "BUILDSTORE_FW_FILE": str(fw)}
    a = art(tmp_path, "a")
    tj = tmp_path / "tier1.json"; tj.write_text('{"ok": true}')
    cli(tmp_path, env, "gate-record", str(a), "tier1", "0", str(tj))

    assert cli(tmp_path, env, "gate-status", str(a)).strip() == "VALID tier1"

    b = art(tmp_path, "a2", elf=b"OTHER")           # different device bytes -> different identity
    assert cli(tmp_path, env, "gate-status", str(b)).strip() == "NONE"

    c = art(tmp_path, "a3", toolchain="b")           # provenance-only diff -> SAME identity
    assert cli(tmp_path, env, "gate-status", str(c)).strip() == "VALID tier1"

    fw.write_text("fw2\n")                           # firmware moved under this identity
    assert cli(tmp_path, env, "gate-status", str(a)).strip() == "STALE tier1 firmware"


def test_gate_record_never_fails(tmp_path):
    """A bogus artifact dir or an unreadable driver/fw path must not make gate-record itself
    fail a gate (it records, it does not judge)."""
    env = {"BUILDSTORE_DRIVER_FILE": "/nonexistent/driver", "BUILDSTORE_FW_FILE": "/nonexistent/fw"}
    missing = tmp_path / "does-not-exist"
    r = cli(tmp_path, env, "gate-record", str(missing), "tier1", "1", str(tmp_path / "no.json"))
    assert r is not None  # cli() already asserts returncode 0 via check=True
