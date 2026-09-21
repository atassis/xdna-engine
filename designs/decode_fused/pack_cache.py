#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed cache for prefill's packed quantized weight buffers.

`repack_gemm_weight` is a pure byte permutation, so a dump and its packing params always yield the
same bytes. The key names every input that changes them: the dump's content, each repack parameter,
and digests of `quant.py` (IRON is ancestry-pinned, so the packer can move without the lock) and of
the generator.
"""
import hashlib
import json
import os
import shutil
import tempfile

CACHE_FORMAT_VERSION = 1


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
    """Objects at `<root>/objects/<key[:2]>/<key>.bin`. `put` is atomic, so a killed build never
    leaves a truncated object; nothing opens a materialized buffer for write, so hardlinks are safe."""

    def __init__(self, root):
        self.root = root
        self.objects = os.path.join(root, "objects")

    def _path(self, key):
        return os.path.join(self.objects, key[:2], f"{key}.bin")

    def get(self, key):
        p = self._path(key)
        return p if os.path.isfile(p) else None

    def put(self, key, src_path):
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

    def materialize(self, key, dest_path):
        """Hardlink the object to `dest_path` (copy across filesystems); False on a miss."""
        src = self.get(key)
        if src is None:
            return False
        if os.path.lexists(dest_path):
            os.unlink(dest_path)
        try:
            os.link(src, dest_path)
        except OSError:
            shutil.copy2(src, dest_path)
        return True


def default_cache_dir():
    return os.environ.get("PREFILL_PACK_CACHE_DIR", "/mnt/data/xdna/cache/prefill_pack")


def cache_enabled():
    return os.environ.get("PREFILL_PACK_CACHE", "1") == "1"
