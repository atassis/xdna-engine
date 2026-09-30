"""Write a control-code ELF as `<name>.elf.zst` only (host loader
`npu-models::llm::artifact::read_elf_bytes` decompresses it transparently).
"""

import hashlib
import os
import subprocess

# The design note's measured level: `zstd -3 --long=27` took the served gemma4 decode ELF from
# 18.8 MB to 215 KB and the prefill ring ELF from 114.4 MB to 714 KB, byte-identical, 10-20 ms
# decode.
_ZSTD_ARGS = ["-q", "-f", "-3", "--long=27"]


def write_elf(path, elf_bytes):
    """Write `path + ".zst"` only; return provenance for `meta.json`.

    `{"sha256": <uncompressed hex>, "elf_zst_sha256": <compressed hex>, "elf_zst_bytes": <int>}`.
    `sha256` is the artifact_hash the host must reproduce from the decompressed bytes -- see
    `npu-models::llm::npu_decode::open`'s `artifact_hash` comment. Verifies the `.zst` decompresses
    byte-identical before returning; a stale plain `path` from an older build is removed so the
    output dir is unambiguous.
    """
    uncompressed_sha256 = hashlib.sha256(elf_bytes).hexdigest()

    with open(path, "wb") as f:
        f.write(elf_bytes)

    zst_path = path + ".zst"
    subprocess.run(["zstd", *_ZSTD_ARGS, "-o", zst_path, path], check=True)
    with open(zst_path, "rb") as f:
        zst_bytes = f.read()

    roundtrip = subprocess.run(["zstd", "-q", "-d", "--long=27", "-c", zst_path], check=True,
                                stdout=subprocess.PIPE).stdout
    if hashlib.sha256(roundtrip).hexdigest() != uncompressed_sha256:
        raise RuntimeError(f"{zst_path}: round-trip mismatch, zstd decompression did not "
                            "reproduce the original ELF bytes")

    os.unlink(path)

    return {
        "sha256": uncompressed_sha256,
        "elf_zst_sha256": hashlib.sha256(zst_bytes).hexdigest(),
        "elf_zst_bytes": len(zst_bytes),
    }


def read_elf(path):
    """Read an ELF written by `write_elf`: `path` if present (stale plain file), else decompress
    `path + ".zst"`."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    zst_path = path + ".zst"
    try:
        import zstandard
    except ImportError:
        return subprocess.run(["zstd", "-q", "-d", "--long=27", "-c", zst_path], check=True,
                               stdout=subprocess.PIPE).stdout
    with open(zst_path, "rb") as f:
        return zstandard.ZstdDecompressor().stream_reader(f).read()
