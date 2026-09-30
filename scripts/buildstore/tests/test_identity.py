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


def make_decode(tmp, buf=b"W"):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "buffers").mkdir(exist_ok=True)
    (tmp / "buffers" / "w.bin").write_bytes(buf)
    (tmp / "meta.json").write_text(json.dumps({"elf": "x"}))
    return tmp


def make_prefill(tmp, decode_dir):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "prefill.elf").write_bytes(b"P")
    meta = {"elf": "prefill.elf",
            "decode_artifact": {"meta": str(decode_dir / "meta.json")},
            "weights_from": str(decode_dir / "buffers")}
    (tmp / "meta.json").write_text(json.dumps(meta))
    return tmp


def test_referenced_artifact_hashed_by_identity_not_checkout_path(tmp_path):
    d1 = make_decode(tmp_path / "a" / "decode")
    d2 = make_decode(tmp_path / "b" / "decode")
    p1 = make_prefill(tmp_path / "a" / "prefill", d1)
    p2 = make_prefill(tmp_path / "b" / "prefill", d2)
    assert identity.of(p1) == identity.of(p2)
    (d2 / "buffers" / "w.bin").write_bytes(b"X")
    assert identity.of(p1) != identity.of(p2)


def test_self_referencing_meta_does_not_loop(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    (a / "x.bin").write_bytes(b"1")
    (a / "meta.json").write_text(json.dumps({"self": str(a)}))
    identity.of(a)  # must terminate
