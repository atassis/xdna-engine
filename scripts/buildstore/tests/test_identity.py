import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); import identity

def art(tmp, meta, elf=b"E"):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "decode.elf").write_bytes(elf); (tmp / "meta.json").write_text(json.dumps(meta))
    return tmp

def test_provenance_ignored_device_bytes_not(tmp_path):
    base = {"elf": "decode.elf", "sha256": "s", "toolchain": {"instance": "a"}, "iron": {"tree": "/x"}}
    a = identity.of(art(tmp_path / "a", base))
    b = identity.of(art(tmp_path / "b", dict(base, toolchain={"instance": "b"}, iron={"tree": "/y"})))
    assert a == b
    assert identity.of(art(tmp_path / "c", dict(base, sha256="t"))) != a
    assert identity.of(art(tmp_path / "d", base, elf=b"F")) != a

def test_sabotage_dropping_a_device_key_is_caught(tmp_path, monkeypatch):
    base = {"elf": "decode.elf", "sha256": "s"}
    monkeypatch.setattr(identity, "PROVENANCE_KEYS", identity.PROVENANCE_KEYS + ("sha256",))
    assert identity.of(art(tmp_path / "a", base)) == identity.of(art(tmp_path / "b", dict(base, sha256="t")))
