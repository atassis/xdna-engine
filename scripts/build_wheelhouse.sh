#!/usr/bin/env bash
# Rebuild the local wheelhouse (vendor/wheelhouse/) from the uv archive cache.
#
# vendor/ is gitignored, so a from-scratch checkout has NO wheelhouse. This script reconstructs the
# one wheel that is not reliably fetchable from the network find-links indexes -- mlir_aie -- by
# repacking its already-unpacked tree from ~/.cache/uv/archive-v0 into a proper cp314 wheel.
# scripts/setup_route_b.sh auto-invokes this when the wheel is missing. Idempotent: a fresh repack
# each run (cheap, deterministic; overwrites any prior wheel of the same version).
#
# NOTE: the llvm-aie/Peano wheel is deliberately NOT handled here -- its cp310 tag cannot install into
# the py3.14 venv, so setup_route_b.sh provides it by copying the unpacked archive tree straight into
# site-packages, never via a wheel.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MLIR_AIE_VER='0.0.1.2026033104+e4f35d6'
CACHE="${UV_CACHE_DIR:-$HOME/.cache/uv}/archive-v0"
DEST="$REPO/vendor/wheelhouse"

# Locate the unpacked mlir_aie tree in the uv archive cache (the dir that holds its .dist-info).
# Guard the search so a missing/empty cache falls through to the loud error below rather than
# tripping `set -e`/pipefail on the find/command-substitution.
MAR=""
if [ -d "$CACHE" ]; then
  MAR=$(find "$CACHE" -maxdepth 2 -name "mlir_aie-${MLIR_AIE_VER}.dist-info" 2>/dev/null \
          | head -1 | xargs -r dirname || true)
fi
if [ -z "${MAR:-}" ] || [ ! -d "$MAR/mlir_aie" ]; then
  echo "ERROR: mlir_aie ${MLIR_AIE_VER} is not in the uv archive cache ($CACHE)." >&2
  echo "       Populate it first, then re-run this script:" >&2
  echo "         uv pip install --python .venv-iron mlir_aie==${MLIR_AIE_VER} \\" >&2
  echo "           --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4" >&2
  exit 1
fi

mkdir -p "$DEST"

# Pick a python that can pack a wheel. Prefer the toolchain venv (it carries `wheel`); otherwise run
# a throwaway interpreter via uv with `wheel` injected, so this works before .venv-iron has wheel.
if [ -x "$REPO/.venv-iron/bin/python" ] && "$REPO/.venv-iron/bin/python" -c 'import wheel' 2>/dev/null; then
  PACK=("$REPO/.venv-iron/bin/python" -m wheel pack)
else
  PACK=(uv run --with wheel --python 3.14 -- python -m wheel pack)
fi

echo "Repacking mlir_aie ${MLIR_AIE_VER} from ${MAR} -> ${DEST}"
"${PACK[@]}" "$MAR" --dest-dir "$DEST"
ls -1 "$DEST"/mlir_aie-*.whl
echo "Wheelhouse ready at ${DEST}."
