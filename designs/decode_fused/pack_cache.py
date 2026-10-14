#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed cache with explicit pruning for decode quantization and prefill packing.

`repack_gemm_weight` is a pure byte permutation, so a dump and its packing params always yield the
same bytes. The key names every input that changes them: the dump's content, each repack parameter,
and digests of `quant.py` (IRON is ancestry-pinned, so the packer can move without the lock) and of
the generator.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tempfile

CACHE_FORMAT_VERSION = 1
DEFAULT_MAX_BYTES = 16 * (1 << 30)  # Policy: object admission and explicit shared-budget pruning.

_dr_spec = importlib.util.spec_from_file_location(
    "data_root", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
                               "scripts", "lib", "data_root.py"))
_data_root = importlib.util.module_from_spec(_dr_spec)
_dr_spec.loader.exec_module(_data_root)


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def content_key(entry, src_digest, packer_digest, generator_digest):
    # `buf` and the source path are identity, not content: equal bytes and params must collide.
    payload = {
        "v": CACHE_FORMAT_VERSION,
        "op": "repack_gemm_weight",
        "src_digest": src_digest,
        "N": entry["N"], "K": entry["K"],
        "tile_k": entry["tile_k"], "tile_n": entry["tile_n"],
        "group_size": entry["group_size"], "weight_dtype": entry["weight_dtype"],
        "cols": entry["cols"], "scale_dtype": entry["scale_dtype"],
        "mmul": list(entry["mmul"]),
        "packer_digest": packer_digest,
        "generator_digest": generator_digest,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class ContentStore:
    """Objects at `<root>/objects/<key[:2]>/<key>.bin`; `put` publishes atomically.

    Producers must replace materialized inodes, not write through hardlinks into cached content.
    `put` never evicts: the store may exceed its budget until explicit `prune`, which requires
    all readers sharing the store to participate in leases.
    """

    def __init__(self, root, max_bytes=None):
        self.root = root
        self.objects = os.path.join(root, "objects")
        self.max_bytes = int(os.environ.get("PACK_CACHE_MAX_BYTES", DEFAULT_MAX_BYTES)
                             if max_bytes is None else max_bytes)
        if self.max_bytes < 0:
            raise ValueError("PACK_CACHE_MAX_BYTES must be nonnegative")

    def _path(self, key):
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("packed-cache key must be 64 lowercase hexadecimal characters")
        return os.path.join(self.objects, key[:2], f"{key}.bin")

    def get(self, key):
        """Unleased lookup; eviction-capable readers must use `open` or `materialize`."""
        with self._lock(fcntl.LOCK_SH):
            return self._get(key)

    def _get(self, key):
        path = self._path(key)
        if not os.path.isfile(path):
            return None
        os.utime(path, None)
        return path

    @contextmanager
    def _lock(self, mode):
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, ".lock"), "a+b") as lock:
            fcntl.flock(lock, mode)
            yield

    @contextmanager
    def open(self, key):
        """Pin a cached reader against eviction until its context exits."""
        with self._lock(fcntl.LOCK_SH):
            path = self._get(key)
            if path is None:
                yield None
            else:
                with open(path, "rb") as cached:
                    yield cached

    def put(self, key, src_path):
        """Cache an object, bypassing oversized inputs; returned paths are unleased."""
        self._path(key)
        if os.path.getsize(src_path) > self.max_bytes or self.max_bytes == 0:
            return src_path
        with self._lock(fcntl.LOCK_EX):
            return self._put(key, src_path)

    def _put(self, key, src_path):
        dst = self._path(key)
        if os.path.isfile(dst):
            return dst
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as out, open(src_path, "rb") as inp:
                shutil.copyfileobj(inp, out)
            os.replace(tmp, dst)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return dst

    def prune(self):
        """Enforce the total budget explicitly; all shared-store readers must use leases."""
        with self._lock(fcntl.LOCK_EX):
            self._prune()

    def _prune(self):
        objects = []
        if not os.path.isdir(self.objects):
            return
        for shard in os.scandir(self.objects):
            if not re.fullmatch(r"[0-9a-f]{2}", shard.name) or not shard.is_dir(follow_symlinks=False):
                continue
            for obj in os.scandir(shard.path):
                if (re.fullmatch(r"[0-9a-f]{64}\.bin", obj.name)
                        and obj.name[:2] == shard.name and obj.is_file(follow_symlinks=False)):
                    info = obj.stat(follow_symlinks=False)
                    objects.append((info.st_mtime_ns, obj.path, info.st_size))
        total = sum(size for _, _, size in objects)
        for _, path, size in sorted(objects):
            if total <= self.max_bytes:
                break
            os.unlink(path)
            total -= size

    def materialize(self, key, dest_path):
        """Atomically publish a hardlink (copy across filesystems); False on a miss."""
        with self.open(key) as cached:
            if cached is None:
                return False
            fd, temporary = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dest_path)),
                                            prefix=".materialize-")
            os.close(fd)
            try:
                os.unlink(temporary)
                try:
                    os.link(cached.name, temporary)
                except OSError:
                    shutil.copy2(cached.name, temporary)
                os.replace(temporary, dest_path)
            finally:
                if os.path.lexists(temporary):
                    os.unlink(temporary)
            return True


def default_cache_dir():
    return os.environ.get("PACK_CACHE_DIR", os.environ.get(
        "PREFILL_PACK_CACHE_DIR", str(_data_root.XDNA_CACHE / "packed-weights")))


def cache_enabled():
    return os.environ.get("PREFILL_PACK_CACHE", "1") == "1"
