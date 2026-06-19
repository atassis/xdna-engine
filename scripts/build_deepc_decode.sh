#!/usr/bin/env bash
# Build the deep-C (constant ELF + runtime scratchpad params) fused Whisper decode.
#
# Reproduces internal notes. Compile-only (no NPU needed).
#
#   bash scripts/build_deepc_decode.sh [LAYERS] [OUT_DIR]
#   LAYERS   default 12
#   OUT_DIR  default <repo>/artifacts/fused_decode12   (the dir the engine loads)
#
# Prerequisites (paths overridable via env):
#   VENV_IRON  = vendored mlir-aie 1.3.2 venv (py3.14) WITH torch installed:
#                  uv venv .venv-iron --python 3.14 ; uv pip install --python .venv-iron <mlir_aie wheel> torch
#                default: <repo>/.venv-iron
#   IRON       = amd/IRON operator-library checkout. The deep-C patch (patches/amd-IRON-deepc.patch)
#                is applied to it if not already present.  default: ~/repositories/ns/amd/IRON
#   AIEBU_DIR  = dir containing aiebu-asm (from the XRT aiebu submodule build)
#                default: ~/repositories/ns/amd/XRT-src/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm
#   WEIGHTS    = whisper decoder weights dir.  default: <repo>/artifacts/whisper-small/whisper_decoder
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAYERS="${1:-12}"
OUT="${2:-$REPO/artifacts/fused_decode12}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
IRON="${IRON:-~/repositories/ns/amd/IRON}"
AIEBU_DIR="${AIEBU_DIR:-~/repositories/ns/amd/XRT-src/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/whisper-small/whisper_decoder}"
PATCH="$REPO/patches/amd-IRON-deepc.patch"
GEN="$REPO/route_b_kernels/decode_fused/gen_decode.py"

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: $VENV_IRON/bin/python missing (see prerequisites)"; exit 1; }
[ -d "$IRON/iron" ] || { echo "ERROR: amd/IRON not at $IRON"; exit 1; }
command -v "$AIEBU_DIR/aiebu-asm" >/dev/null || [ -x "$AIEBU_DIR/aiebu-asm" ] || { echo "ERROR: aiebu-asm not at $AIEBU_DIR"; exit 1; }

# Apply the deep-C amd/IRON patch (fuse_mlir hoist + StridedCopy/Softmax scratchpad) if not present.
if [ -n "${SKIP_IRON_PATCH:-}" ]; then
  echo "[build] SKIP_IRON_PATCH=1 — assuming deep-C already present in the shared IRON tree (do NOT re-apply over stacked patches)"
elif git -C "$IRON" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "[build] amd/IRON deep-C patch already applied"
else
  echo "[build] applying amd/IRON deep-C patch"
  git -C "$IRON" apply "$PATCH"
fi

export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_DIR:$PATH"
export PEANO_INSTALL_DIR="$VENV_IRON/lib/python3.14/site-packages/llvm-aie"
export PYTHONPATH="$IRON${PYTHONPATH:+:$PYTHONPATH}"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT   # amd/IRON writes build/ intermediates under CWD
mkdir -p "$OUT"
echo "[build] gen_decode --layers $LAYERS -> $OUT (work=$WORK) ${GEN_EXTRA:+extra: $GEN_EXTRA}"
( cd "$WORK" && "$VENV_IRON/bin/python" "$GEN" --weights "$WEIGHTS" --layers "$LAYERS" --out "$OUT" ${GEN_EXTRA:-} )
echo "[build] done. decode.elf=$(du -h "$OUT/decode.elf" | cut -f1); params:"
cat "$OUT/params.txt"
