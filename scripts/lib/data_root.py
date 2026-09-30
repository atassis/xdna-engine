# SPDX-License-Identifier: Apache-2.0
"""Single resolver for every generated-data path this repo writes: downloaded
checkpoints, built model artifacts, toolchain/build caches, scratch work,
buildstore CAS, qlab corpora, logs. Mirrors scripts/lib/data_root.sh.

    sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "scripts" / "lib"))
    from data_root import XDNA_SCRATCH, XDNA_QLAB  # noqa: E402

Point XDNA_DATA at a bigger disk via config/local.env (gitignored; copy
config/local.env.example) instead of exporting it in every shell. Each
derived path is also individually overridable through its own env var.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # scripts/lib/data_root.py -> repo root


def _main_checkout():
    """The main checkout a linked worktree belongs to (its .git is a file), else None."""
    if not (REPO / ".git").is_file():
        return None
    import subprocess
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                       capture_output=True, text=True)
    return Path(r.stdout.strip()).parent if r.returncode == 0 and r.stdout.strip() else None


_MAIN = _main_checkout()


def _load_local_env():
    # A worktree has no config/local.env of its own (gitignored): use the main checkout's.
    path = REPO / "config" / "local.env"
    if not path.is_file() and _MAIN is not None:
        path = _MAIN / "config" / "local.env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_local_env()


def _env_path(name, default):
    return Path(os.environ.get(name, str(default)))


_default_data = REPO / "data"
if not _default_data.exists() and _MAIN is not None and (_MAIN / "data").is_dir():
    _default_data = _MAIN / "data"
XDNA_DATA = _env_path("XDNA_DATA", _default_data)
XDNA_MODELS = _env_path("XDNA_MODELS", XDNA_DATA / "models")
XDNA_ARTIFACTS = _env_path("XDNA_ARTIFACTS", XDNA_DATA / "artifacts")
XDNA_CACHE = _env_path("XDNA_CACHE", XDNA_DATA / "cache")
XDNA_BUILD = _env_path("XDNA_BUILD", XDNA_DATA / "build")
XDNA_SCRATCH = _env_path("XDNA_SCRATCH", XDNA_DATA / "scratch")
XDNA_CAS = _env_path("XDNA_CAS", XDNA_DATA / "cas")
XDNA_QLAB = _env_path("XDNA_QLAB", XDNA_DATA / "qlab")
XDNA_LOGS = _env_path("XDNA_LOGS", XDNA_DATA / "logs")

# Names scripts already used before this root existed; default now derives
# from XDNA_DATA instead of a hardcoded path.
BUILDSTORE_CAS = _env_path("BUILDSTORE_CAS", XDNA_CAS)
QLAB_WORK = _env_path("QLAB_WORK", XDNA_QLAB)
XDNA_ARTIFACT_STORE = _env_path("XDNA_ARTIFACT_STORE", XDNA_ARTIFACTS)
XDNA_MODEL_STORE = _env_path("XDNA_MODEL_STORE", XDNA_MODELS)
XDNA_BUILD_ROOT = _env_path("XDNA_BUILD_ROOT", XDNA_BUILD)


def xdna_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
