#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed cache for prefill's packed quantized weight buffers.

`iron.common.quant.repack_gemm_weight` is a pure byte-permutation of one row-packed `.npy` dump
into GEMM's tile-planar layout (see gen_llm_prefill.py's quant_pack loop) -- no requantization, so
every build from the same dump + packing params reproduces the same bytes. Every build re-derives
them anyway: a 48-layer gemma4-12b build repacks 480 buffers (~6.5 GB) every time, ~2.6 min of a
19 min build.

Keyed on content, not path/mtime: the key must name every input that changes the bytes, or a
stale hit is possible. Here that is the dumped weight's own bytes, every repack_gemm_weight
parameter, and a digest of the two Python sources that decide them (this module's packer identity
is `quant.py`'s own content, not a git SHA -- IRON is ancestry-pinned, not exact-SHA-pinned, so a
local commit can move `repack_gemm_weight` without moving toolchain.lock).
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
    """entry: one gen_llm_prefill.py `quant_pack` dict (buf/src_dir/src_file/N/K/tile_k/tile_n/
    group_size/weight_dtype/cols/scale_dtype/mmul).

    Every field `repack_gemm_weight` reads is in the key, plus the two source files whose bytes
    its OUTPUT depends on: the dumped weight (content, not path) and the packer module. `buf` and
    `src_dir`/`src_file` are identity, not content, and are deliberately excluded -- two different
    files with the same bytes and params must collide (that is the win), and a renamed buffer with
    unchanged content must still hit.
    """
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
    """Flat content-addressed store: `<root>/objects/<key[:2]>/<key>.bin`.

    `put` is atomic (mkstemp + os.replace) so a killed build cannot leave a truncated object that
    a later `get` would then serve. Nothing downstream of a materialized buffer opens it for write
    (rust/npu-engine only ever `std::fs::read`s `buffers/*.bin`, verified against every call site
    in llm/artifact.rs and llm/npu_decode.rs), so `materialize`'s hardlink cannot corrupt the store.
    """

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
        """Populate `dest_path` from the cached object at `key`. Hardlink when the store and
        `dest_path` share a filesystem, copy otherwise. Returns False on a miss (caller packs)."""
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
