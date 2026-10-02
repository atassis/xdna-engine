"""Resident artifacts preserve the same IRON provenance schema as decode artifacts."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load(relative):
    spec = importlib.util.spec_from_file_location(relative.stem, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resident_pack_and_package_preserve_iron_identity(tmp_path, monkeypatch):
    fwd_pack = load(Path("designs/resident_forward/fwd_pack.py"))
    package_artifact = load(Path("designs/resident_forward/package_artifact.py"))
    want = "a" * 40
    monkeypatch.setenv("IRON_SOURCE_IDENTITY", f"pinned:{want}")
    assert fwd_pack.provenance()["iron_identity"] == f"pinned:{want}"
    assert fwd_pack.provenance()["iron_dirty"] is False

    build = tmp_path / "build"; build.mkdir()
    for name in ("fwd_layout.json", "params.txt", "gen_args.txt"):
        (build / name).write_text("{}\n")
    (build / "design.elf").write_bytes(b"elf")
    (build / "pack.json").write_text(json.dumps({"provenance": {
        "iron_commit": want,
        "iron_dirty": False,
        "iron_identity": f"pinned:{want}",
    }}))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"kind": "resident_forward_ladder", "elf": "design.elf", "nlayer": 0}))
    weights = tmp_path / "weights"; weights.mkdir(); (weights / "w_head.npy").write_bytes(b"w")
    store = tmp_path / "store"; (store / "blobs").mkdir(parents=True)
    (store / "manifest.json").write_text(json.dumps({"embedding": {"blob": "e"}}))
    (store / "blobs" / "e.bin").write_bytes(b"e")

    out = tmp_path / "out"
    package_artifact.package(out, build, meta, weights, store)
    assert json.loads((out / "meta.json").read_text())["iron"] == {
        "commit": want, "dirty": False, "identity": f"pinned:{want}",
    }
