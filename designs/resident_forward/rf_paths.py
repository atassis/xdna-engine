"""Path defaults shared by the resident-forward build and data-prep scripts.

XDNA_DATA overrides where generated build/scratch/artifact trees live; default is this repo's own
`build/` and `artifacts/` (already gitignored, same convention `designs/decode_fused` uses).
IRON defaults to the repository's pinned `third_party/iron` submodule. Build recipes verify the
exact ref before compilation.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("XDNA_DATA", str(REPO)))
ARTIFACTS = DATA_ROOT / "artifacts" / "gemma4-12b"
BUILD_ROOT = DATA_ROOT / "build" / "resident_forward"


def iron_dir():
    return os.environ.get("IRON_DIR", str(REPO / "third_party" / "iron"))


def iron_kernels_root():
    return os.path.join(iron_dir(), "aie_kernels")


def iron_kernel_dir():
    return os.path.join(iron_kernels_root(), "aie2p")


def iron_kernel(*parts):
    return os.path.join(iron_kernel_dir(), *parts)
