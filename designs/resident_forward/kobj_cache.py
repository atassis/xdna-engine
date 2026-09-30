"""Content-addressed kernel-object cache shared by one build run's parts/calls.

F-b design, mirrored from IRON's KernelCompilationRule (iron/common/compilation/base.py,
_kernel_object_key/_compile_raw_kernel_object): key = sha256(source bytes, flags, target); the
raw object is written via a private temp file + os.replace() under <run_dir>/kobj/, then linked
into each caller's own object path. IRON scopes kobj/<run_id>/ to one compile() call (one
process); here <run_dir> is the caller's RF_BUILD, so the cache is shared across a whole ladder's
parts (same rlayer_design flags recompile the same ~19 kernels once per part otherwise) -- a
fresh RF_BUILD per recipe invocation still means it is never reused across builds. A flock per
key serializes parts racing the same kernel now that fwd_ladder.sh runs them concurrently.
"""
import fcntl
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def kobj_dir(run_dir):
    # One ladder invocation (RF_KOBJ_RUN, set by fwd_ladder.sh) or one process: the key has no
    # compiler identity or include closure, so objects must never outlive the run that built them.
    run = os.environ.get("RF_KOBJ_RUN") or f"pid{os.getpid()}"
    d = Path(run_dir) / "kobj" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def kobj_key(sources):
    """sources: [(path, [flags...]), ...] -- one entry per source feeding the object (>1 for a
    partial-linked object). Order matters (partial links are order-sensitive)."""
    h = hashlib.sha256()
    for src, flags in sources:
        h.update(Path(src).read_bytes())
        h.update(b"\0".join(f.encode() for f in flags))
        h.update(b"\xff")
    return h.hexdigest()[:24]


def cached_compile(run_dir, key, build_fn):
    """Return kobj/<key>.o, compiling it via build_fn(tmp_path) first if absent. build_fn must
    write a complete object to tmp_path. Blocks on a per-key flock so concurrent callers wanting
    the same kernel serialize instead of racing the compile."""
    d = kobj_dir(run_dir)
    cached = d / f"{key}.o"
    with open(d / f"{key}.lock", "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            if not cached.exists():
                fd, tmp = tempfile.mkstemp(dir=d, suffix=".o.tmp")
                os.close(fd)
                try:
                    build_fn(tmp)
                    os.replace(tmp, cached)
                except BaseException:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
    return cached


def place(cached, dest):
    """Hardlink cached -> dest (content-addressed, so sharing the inode is safe); falls back to
    a copy if dest is on a different filesystem."""
    dest = Path(dest)
    if dest.exists():
        dest.unlink()
    try:
        os.link(cached, dest)
    except OSError:
        shutil.copy(cached, dest)
