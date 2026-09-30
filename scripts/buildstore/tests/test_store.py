import os, pathlib, stat, sys, threading
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); import store

def test_put_tree_materialize_roundtrip_and_dedup(tmp_path):
    s = store.Store(tmp_path / "cas")
    src = tmp_path / "src"; (src / "sub").mkdir(parents=True)
    (src / "a").write_bytes(b"x"); (src / "sub" / "b").write_bytes(b"x")
    os.chmod(src / "a", 0o755)
    tid = s.put_tree(src)
    assert len(list((tmp_path / "cas" / "objects").iterdir())) == 1      # one object for both
    dst = tmp_path / "dst"; s.materialize(tid, dst)
    assert (dst / "sub" / "b").read_bytes() == b"x"
    assert os.stat(dst / "a").st_mode & stat.S_IXUSR
    obj = next((tmp_path / "cas" / "objects").iterdir())
    assert not os.stat(obj).st_mode & 0o222                              # read-only object
    assert s.put_tree(src) == tid                                        # deterministic id

def test_action_roundtrip(tmp_path):
    s = store.Store(tmp_path / "cas")
    s.put_action("r", "k1", {"tree": "t", "manifest": {}, "timings": {}})
    assert [a["tree"] for a in s.actions("r")] == ["t"]

def test_single_flight(tmp_path):
    s = store.Store(tmp_path / "cas"); order = []
    def work(i):
        with s.lock("r"):
            order.append(("in", i)); order.append(("out", i))
    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all(order[j][0] == "in" and order[j + 1] == ("out", order[j][1])
               for j in range(0, len(order), 2))

def test_materialize_falls_back_to_copy_when_hardlink_unavailable(tmp_path, monkeypatch):
    s = store.Store(tmp_path / "cas")
    src = tmp_path / "src"; src.mkdir()
    (src / "a").write_bytes(b"x")
    tid = s.put_tree(src)
    real_link = os.link
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError("cross-device")))
    dst = tmp_path / "dst"
    s.materialize(tid, dst)
    assert (dst / "a").read_bytes() == b"x"
    assert os.stat(dst / "a").st_mode & 0o777 == 0o444
    monkeypatch.setattr(os, "link", real_link)
