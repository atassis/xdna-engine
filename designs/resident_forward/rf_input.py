"""One physical MLIR input with a relative design.mlir inspection alias.

Callers hold work/.build.lock across input replacement and compilation.
"""
import hashlib
import os
from pathlib import Path
import tempfile


def alias_input(work):
    work = Path(work)
    source = work / "aie.mlir"
    alias = work / "design.mlir"
    if source.is_symlink() or (source.exists() and not source.is_file()):
        raise ValueError(f"Not a regular compiler input: {source}")
    if alias.is_symlink():
        if os.readlink(alias) != source.name:
            raise ValueError(f"Unexpected input alias: {alias}")
        return
    if alias.exists():
        if not alias.is_file() or not source.is_file():
            raise ValueError(f"Unpaired design input: {alias}")
        with alias.open("rb") as a, source.open("rb") as b:
            if hashlib.file_digest(a, "sha256").digest() != hashlib.file_digest(b, "sha256").digest():
                raise ValueError(f"Divergent design input: {alias}")
    with tempfile.TemporaryDirectory(prefix=".input-alias-", dir=work) as tmp:
        link = Path(tmp) / alias.name
        link.symlink_to(source.name)
        os.replace(link, alias)


def prepare_input(work, text):
    work = Path(work)
    alias_input(work)
    source = work / "aie.mlir"
    # aiecc's wrapper rewrites this file; detach any shared inode first.
    with tempfile.NamedTemporaryFile(mode="w", dir=work, prefix=".input-", delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(text)
            f.flush()
            os.replace(temporary, source)
        finally:
            temporary.unlink(missing_ok=True)
