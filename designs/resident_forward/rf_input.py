"""Stage the single compiler input, aie.mlir, on a fresh inode.

Callers hold work/.build.lock across input replacement and compilation.
"""
import hashlib
import os
from pathlib import Path
import tempfile


def prepare_input(work, text):
    work = Path(work)
    source = work / "aie.mlir"
    redundant = work / "design.mlir"
    if source.is_symlink() or (source.exists() and not source.is_file()):
        raise ValueError(f"Not a regular compiler input: {source}")
    if redundant.is_symlink():
        if os.readlink(redundant) != source.name:
            raise ValueError(f"Unexpected input symlink: {redundant}")
    elif redundant.exists():
        if not redundant.is_file() or not source.is_file():
            raise ValueError(f"Unpaired design input: {redundant}")
        with redundant.open("rb") as a, source.open("rb") as b:
            if hashlib.file_digest(a, "sha256").digest() != hashlib.file_digest(b, "sha256").digest():
                raise ValueError(f"Divergent design input: {redundant}")
    redundant.unlink(missing_ok=True)
    # aiecc's wrapper rewrites this file; detach any shared inode first.
    with tempfile.NamedTemporaryFile(mode="w", dir=work, prefix=".input-", delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(text)
            f.flush()
            os.replace(temporary, source)
        finally:
            temporary.unlink(missing_ok=True)
