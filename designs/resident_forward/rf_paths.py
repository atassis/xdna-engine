"""Path defaults shared by the resident-forward build and data-prep scripts.

XDNA_DATA overrides where generated build/scratch/artifact trees live; default is this repo's own
`build/` and `artifacts/` (already gitignored, same convention `designs/decode_fused` uses).
IRON_DIR must be set by the caller (see `recipes/rf48C.sh`) -- it is not defaulted here because a
silently-wrong IRON tree changes the generated MLIR/ELF bytes.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("XDNA_DATA", str(REPO)))
ARTIFACTS = DATA_ROOT / "artifacts" / "gemma4-12b"
BUILD_ROOT = DATA_ROOT / "build" / "resident_forward"


def iron_dir():
    d = os.environ.get("IRON_DIR")
    if not d:
        raise SystemExit("ERROR: IRON_DIR is not set (point it at an IRON checkout on the pinned commit)")
    return d


def iron_kernels_root():
    return os.path.join(iron_dir(), "aie_kernels")


def iron_kernel_dir():
    return os.path.join(iron_kernels_root(), "aie2p")


def iron_kernel(*parts):
    return os.path.join(iron_kernel_dir(), *parts)


def fused_attn_dir():
    """aie_kernels/aie2p/fused_attn.cc, needed by rattnh.kernels()'s rf_attn_qk_k.cc/rf_attn_pv_k.cc,
    postdates IRON_PIN in the served rf48C build (it was built against a second, one-commit-ahead
    checkout of the same IRON fork). Defaults to IRON_DIR; set RF_FUSED_ATTN_DIR explicitly to
    reproduce the exact historical pin if IRON_DIR does not yet carry fused_attn.cc."""
    return os.environ.get("RF_FUSED_ATTN_DIR", iron_kernel_dir())
