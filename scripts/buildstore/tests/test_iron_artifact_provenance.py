"""Release artifact IRON provenance must identify the clean pinned source."""
import json
import pathlib
import subprocess
import sys


CHECK = pathlib.Path(__file__).resolve().parents[2] / "check_iron_artifact_provenance.py"
WANT = "a" * 40


def check(tmp_path, iron):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"iron": iron}))
    return subprocess.run(
        [sys.executable, str(CHECK), "--expected", WANT, str(meta)],
        capture_output=True,
        text=True,
    )


def test_release_artifact_requires_explicit_clean_pinned_identity(tmp_path):
    accepted = check(tmp_path, {"commit": WANT, "dirty": False, "identity": f"pinned:{WANT}"})
    assert accepted.returncode == 0, accepted.stderr

    for provenance in (
        {},
        {"commit": WANT, "dirty": True, "identity": f"dirty:{WANT}:" + "b" * 64},
        {"commit": WANT, "dirty": False, "identity": "override:" + WANT},
        {"commit": WANT, "dirty": False},
        {"commit": "b" * 40, "dirty": False, "identity": "pinned:" + "b" * 40},
    ):
        rejected = check(tmp_path, provenance)
        assert rejected.returncode != 0
