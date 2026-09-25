"""Write a control-code ELF plus its compressed `.elf.zst` sibling. The host loader
(`npu-engine::llm::artifact::read_elf_bytes`) reads whichever exists, plain first.

`ELF_ZST_ONLY=1` drops the plain file after compressing -- opt-in, so nothing switches format
silently.
"""

import hashlib
import os
import subprocess

# The design note's measured level: `zstd -3 --long=27` took the served gemma4 decode ELF from
# 18.8 MB to 215 KB and the prefill ring ELF from 114.4 MB to 714 KB, byte-identical, 10-20 ms
# decode.
_ZSTD_ARGS = ["-q", "-f", "-3", "--long=27"]


def write_elf(path, elf_bytes):
    """Write `path` and `path + ".zst"`; return provenance for `meta.json`.

    `{"sha256": <uncompressed hex>, "elf_zst_sha256": <compressed hex>, "elf_zst_bytes": <int>}`.
    `sha256` is the artifact_hash the host must reproduce from either form -- see
    `npu-engine::llm::npu_decode::open`'s `artifact_hash` comment.
    """
    with open(path, "wb") as f:
        f.write(elf_bytes)
    uncompressed_sha256 = hashlib.sha256(elf_bytes).hexdigest()

    zst_path = path + ".zst"
    subprocess.run(["zstd", *_ZSTD_ARGS, "-o", zst_path, path], check=True)
    with open(zst_path, "rb") as f:
        zst_bytes = f.read()

    if os.environ.get("ELF_ZST_ONLY") == "1":
        os.unlink(path)

    return {
        "sha256": uncompressed_sha256,
        "elf_zst_sha256": hashlib.sha256(zst_bytes).hexdigest(),
        "elf_zst_bytes": len(zst_bytes),
    }
