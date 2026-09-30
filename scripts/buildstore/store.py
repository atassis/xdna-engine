# SPDX-License-Identifier: Apache-2.0
"""Content-addressed store: immutable objects, tree manifests, recipe actions, per-recipe lock."""
import contextlib, fcntl, hashlib, json, os, pathlib, shutil, stat, threading


def _atomic_write(path, data: bytes, mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)


class Store:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        for d in ("objects", "trees", "actions", "locks", "gates"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def put_file(self, path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        obj = self.root / "objects" / sha
        if not obj.exists():
            tmp = obj.with_name(f".{sha}.tmp{os.getpid()}.{threading.get_ident()}")
            shutil.copyfile(path, tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            os.chmod(tmp, 0o444)
            os.replace(tmp, obj)
        return sha

    def put_tree(self, src):
        src = pathlib.Path(src)
        entries = {}
        for p in sorted(src.rglob("*")):
            if p.is_file() and not p.is_symlink():
                entries[str(p.relative_to(src))] = [self.put_file(p),
                                                   bool(os.stat(p).st_mode & stat.S_IXUSR)]
        data = json.dumps(entries, sort_keys=True).encode()
        tid = hashlib.sha256(data).hexdigest()
        tp = self.root / "trees" / f"{tid}.json"
        if not tp.exists():
            _atomic_write(tp, data)
        return tid

    def tree(self, tid):
        return json.loads((self.root / "trees" / f"{tid}.json").read_bytes())

    def materialize(self, tid, dst):
        """Put tree `tid` at `dst` (replaced whole): hardlink when on one filesystem, else copy."""
        dst = pathlib.Path(dst)
        tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        for rel, (sha, x) in self.tree(tid).items():
            out = tmp / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            obj = self.root / "objects" / sha
            try:
                if x:  # an object is 0444; an executable output gets its own copy
                    raise OSError
                os.link(obj, out)
            except OSError:
                shutil.copyfile(obj, out)
                os.chmod(out, 0o555 if x else 0o444)
        old = dst.with_name(f".{dst.name}.old{os.getpid()}")
        if dst.exists():
            os.replace(dst, old)
        os.replace(tmp, dst)
        shutil.rmtree(old, ignore_errors=True)

    def put_action(self, recipe, key, record):
        _atomic_write(self.root / "actions" / recipe / f"{key}.json",
                      json.dumps(record, sort_keys=True).encode(), mode=0o644)

    def actions(self, recipe):
        d = self.root / "actions" / recipe
        files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) \
            if d.is_dir() else []
        return [json.loads(p.read_bytes()) for p in files]

    @contextlib.contextmanager
    def lock(self, recipe):
        with open(self.root / "locks" / f"{recipe}.lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
