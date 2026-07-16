#!/usr/bin/env bash
# Build the standalone proj_out (lm-head) GEMV ELF — the e2e/NPU wide-dispatch path.
# Usage: [SKIP_IRON_PATCH=1] scripts/build_projout_elf.sh [OUT_DIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/artifacts/projout_elf}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
. "$REPO/scripts/amd_paths.sh"       # -> IRON_DIR, AIEBU_ASM_DIR (relocatable; env-overridable)
IRON="${IRON:-$IRON_DIR}"
AIEBU_DIR="${AIEBU_DIR:-$AIEBU_ASM_DIR}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/whisper-small/whisper_decoder}"
GEN="$REPO/route_b_kernels/decode_fused/gen_projout.py"
[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: $VENV_IRON/bin/python missing"; exit 1; }
[ -d "$IRON/iron" ] || { echo "ERROR: amd/IRON not at $IRON"; exit 1; }
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_DIR:$PATH"
export PEANO_INSTALL_DIR="$VENV_IRON/lib/python3.14/site-packages/llvm-aie"
export PYTHONPATH="$IRON${PYTHONPATH:+:$PYTHONPATH}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"
# Default = fuse the on-NPU argmax (complete e2e/NPU lm-head: ELF returns a token id). PROJOUT_NO_ARGMAX=1
# builds the logits-only ELF (host argmax) instead.
ARGMAX_FLAG="--argmax"; [ -n "${PROJOUT_NO_ARGMAX:-}" ] && ARGMAX_FLAG=""
echo "[build] gen_projout ${ARGMAX_FLAG} -> $OUT (work=$WORK)"
( cd "$WORK" && "$VENV_IRON/bin/python" "$GEN" --weights "$WEIGHTS" --out "$OUT" ${ARGMAX_FLAG} )
echo "[build] done. projout.elf=$(du -h "$OUT/projout.elf" | cut -f1)"
