"""Blob writer for `buffers/<name>.bin` that leaves runs of zeros unallocated."""

import hashlib
import os

_CHUNK = 1 << 20
_POOL_DIRNAME = "blobs"


def _pool_root(path):
    """`<model>/blobs` for a blob at `<model>/<arm>/buffers/<name>.bin`, or None.

    Arms of ONE model are what pack identical weights, so that is the level the pool sits at:
    above it the entries would span models whose blobs never coincide, below it an arm would
    have no sibling to share with. A path that is not inside a `buffers/` directory gets no
    pool, which keeps this scoped to exactly what scripts/dedup_artifacts.sh treats.

    XDNA_BLOB_POOL=0 turns pooling off; any other value names a pool directory to use instead.
    """
    setting = os.environ.get("XDNA_BLOB_POOL", "")
    if setting == "0":
        return None
    if setting:
        return setting
    buffers = os.path.dirname(os.path.abspath(path))
    if os.path.basename(buffers) != "buffers":
        return None
    return os.path.join(os.path.dirname(os.path.dirname(buffers)), _POOL_DIRNAME)


def _adopt(tmp, pooled, path, size):
    """Put `path` on the pooled inode for these bytes, creating the entry from `tmp` if new.

    Ordered so `tmp` survives every failure until the final rename: a raise here has to leave
    the caller something to fall back on.
    """
    os.makedirs(os.path.dirname(pooled), exist_ok=True)
    try:
        os.link(tmp, pooled)
    except FileExistsError:
        have = os.path.getsize(pooled)
        if have != size:
            raise OSError(f"pooled blob {pooled} is {have} B, these bytes are {size} B")
        os.unlink(tmp)
        tmp = path + ".lnk"
        if os.path.exists(tmp):
            os.unlink(tmp)
        os.link(pooled, tmp)
    os.replace(tmp, path)


def write_blob(path, data):
    """Write `data` to `path`, storing whole zero chunks as holes.

    A KV cache is registered in a generator's `weights` dict to get a layout entry and a
    `meta.json` name; its VALUE is filler, since the host zeroes the region at load rather
    than trust the blob. Dense, that filler cost 35 GB over 3792 files (2026-09-11) --
    `gen_llm_decode.py`/`gen_llm_prefill.py` now skip this call for a cache buffer entirely
    (no blob, not even a sparse one; `meta["weights"]` omits the name and `layout` still
    carries it) rather than pay even the sparse form.

    Zero-run rather than by-name, because a cache blob is not always zero here: `gen_decode.py`
    seeds a random past segment into `kc`/`vc` when P>0, and still calls this for every name in
    `meta["weights"]` -- `npu-dev fused-elf` and `npu-dev prefill-golden` read those blobs back
    by name.

    Identical bytes land on ONE inode, shared with whatever sibling arm packed them first
    (see `_pool_root`): an arm's own bytes are its ELF and a few buffers, tens of MB against
    the 8-12 GB of weights every arm re-packs. Whatever happens, the blob is renamed into
    place, so a rebuild gives `path` a NEW inode and its sharers keep the bytes they had.
    """
    mv = memoryview(data).cast("B")
    zero = bytes(_CHUNK)
    digest = hashlib.blake2b(str(len(mv)).encode(), digest_size=16)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.truncate(len(mv))
        for off in range(0, len(mv), _CHUNK):
            chunk = mv[off:off + _CHUNK]
            digest.update(chunk)
            if chunk == zero[:len(chunk)]:
                continue
            f.seek(off)
            f.write(chunk)

    pool = _pool_root(path)
    if pool is not None:
        try:
            _adopt(tmp, os.path.join(pool, digest.hexdigest() + ".bin"), path, len(mv))
            return path
        except OSError:
            # A pool on another filesystem, or read-only, or holding a wrong-sized entry.
            # None of that may stop the blob from landing where it was asked for.
            if not os.path.exists(tmp):
                raise
    os.replace(tmp, path)
    return path
