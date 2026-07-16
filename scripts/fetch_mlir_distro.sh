#!/usr/bin/env bash
# Provision the pinned MLIR core distro -- a prebuilt LLVM/MLIR wheel from the Xilinx/mlir-aie
# `mlir-distro` release -- into a content-addressed cache. This is a DEPENDENCY-FETCH step (network),
# deliberately SEPARATE from the local aiecc build (scripts/toolchain_up.sh, which stays local-only).
# Run it once per MLIR_DISTRO_WHEEL pin, or whenever the cache is missing.
#
#   scripts/fetch_mlir_distro.sh [MLIR_DISTRO_WHEEL]   # defaults to the pin in toolchain.lock
#
# Prints the resolved distro dir (the CMAKE_PREFIX_PATH toolchain_up uses) on stdout.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL="${1:-}"
if [ -z "$WHEEL" ]; then
  set -a; . "$REPO/toolchain.lock"; set +a
  WHEEL="${MLIR_DISTRO_WHEEL:-}"
fi
[ -n "$WHEEL" ] || { echo "fetch_mlir_distro: no MLIR_DISTRO_WHEEL (pass as arg or set in toolchain.lock)" >&2; exit 2; }

. "$REPO/scripts/cache_env.sh"   # -> XDNA_CACHE (in-workspace build cache)
CACHE="$XDNA_CACHE/mlir-distro/${WHEEL#mlir-}"
if [ ! -e "$CACHE/mlir/bin/mlir-tblgen" ]; then
  echo "[fetch_mlir_distro] fetching $WHEEL from Xilinx/mlir-aie mlir-distro release ..." >&2
  mkdir -p "$CACHE"
  _whl="$CACHE/pkg.whl"
  gh release download mlir-distro --repo Xilinx/mlir-aie \
    --pattern "${WHEEL}-py3-none-manylinux*x86_64.whl" --output "$_whl" --clobber >&2
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$_whl" "$CACHE"
  # zipfile drops the exec bit -- restore it on the distro binaries (mlir-tblgen, llvm-tblgen, ...).
  find "$CACHE/mlir/bin" -type f -exec chmod +x {} + 2>/dev/null || true
  rm -f "$_whl"
fi
echo "$CACHE/mlir"
