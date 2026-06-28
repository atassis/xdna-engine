#!/usr/bin/env bash
# Peano CPU compile-check for the subsample patch-embed mmul kernel + run the
# numpy golden. CPU/build-only -- no NPU. (Phase: brick-rebuild, task A5.)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/route_b_kernels/subsample_conv2d"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT"

source "$REPO/scripts/air_env.sh"
AIEOPT_DIR="$(python3 -c 'import mlir_aie; print(list(mlir_aie.__path__)[0])')"
WARN="-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-unknown-attributes"

echo "[build] golden (numpy, gates rel-L2 <= 0.08 vs block_in)"
python3 "$SRC/golden_subsample.py"

echo "[build] Peano compile-check: default mmul 4x8x8 (bf16 emul)"
"$PEANO_INSTALL_DIR/bin/clang++" -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  $WARN -DNDEBUG -I "$AIEOPT_DIR/include" \
  -c "$SRC/subsample_patch_embed.cc" -o "$OUT/subsample_patch_embed.o"

echo "[build] Peano compile-check: bfp16 8x8x8 (true systolic, the ~4x format lever)"
"$PEANO_INSTALL_DIR/bin/clang++" -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  $WARN -DNDEBUG -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -I "$AIEOPT_DIR/include" \
  -c "$SRC/subsample_patch_embed.cc" -o "$OUT/subsample_patch_embed_bfp16.o"

echo "[build] OK. symbols:"
"$PEANO_INSTALL_DIR/bin/llvm-nm" "$OUT/subsample_patch_embed.o" | grep ' T '
