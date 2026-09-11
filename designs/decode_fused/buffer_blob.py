"""Blob writer for `buffers/<name>.bin` that leaves runs of zeros unallocated."""

_CHUNK = 1 << 20


def write_blob(path, data):
    """Write `data` to `path`, storing whole zero chunks as holes.

    A KV cache is registered in a generator's `weights` dict to get a layout entry and a
    `meta.json` name; its VALUE is filler, since the host zeroes the region at load rather
    than trust the blob. Dense, that filler cost 35 GB over 3792 files (2026-09-11).

    Zero-run rather than by-name, because a cache blob is not always zero: `gen_decode.py`
    seeds a random past segment into `kc`/`vc` when P>0. Sparse rather than absent, because
    `arena_share_probe`, `fused_elf_probe` and `prefill_golden_probe` each read every
    `meta["weights"]` blob by name.
    """
    mv = memoryview(data).cast("B")
    zero = bytes(_CHUNK)
    with open(path, "wb") as f:
        f.truncate(len(mv))
        for off in range(0, len(mv), _CHUNK):
            chunk = mv[off:off + _CHUNK]
            if chunk == zero[:len(chunk)]:
                continue
            f.seek(off)
            f.write(chunk)
    return path
