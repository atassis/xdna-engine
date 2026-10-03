"""Release artifact IRON provenance must identify the clean pinned source."""
import json
import ast
import pathlib
import subprocess
import sys


CHECK = pathlib.Path(__file__).resolve().parents[2] / "check_iron_artifact_provenance.py"
WANT = "a" * 40


def test_prefill_metadata_records_the_resolved_iron_source(tmp_path):
    repo = CHECK.parents[1]
    tree = ast.parse((repo / "designs/decode_fused/gen_llm_prefill.py").read_text())
    metadata = next(node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                    and any(isinstance(target, ast.Name) and target.id == "meta" for target in node.targets))
    entry = next((value for key, value in zip(metadata.keys, metadata.values)
                  if isinstance(key, ast.Constant) and key.value == "iron"), None)
    assert entry is not None, "prefill producer omits the provenance required by installation"
    expected = {"commit": WANT, "dirty": False, "identity": f"pinned:{WANT}"}
    recorded = eval(compile(ast.Expression(entry), "prefill-metadata", "eval"),
                    {"iron_provenance": lambda: expected})
    assert recorded == expected
    assert check(tmp_path, recorded).returncode == 0


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
