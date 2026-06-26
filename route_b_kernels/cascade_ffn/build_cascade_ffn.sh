#!/usr/bin/env bash
# Build the single-launch bf16 Whisper-FFN cascade (Task 4) to an xclbin.
# CPU-side compile-only; no device run (that is a later batched device window).
#
#   bash route_b_kernels/cascade_ffn/build_cascade_ffn.sh
#
# Steps: (1) source the mlir-air airenv, (2) compile the Task-3 kernel
# mv_bf16_gelu.cc -> mv_bf16_gelu.o (Peano, aie2p), (3) run ffn_cascade.py to
# lower the air module and emit air.xclbin + air.insts.bin + air.mlir into
# artifacts/cascade_ffn/. aircc resolves link_with="mv_bf16_gelu.o" from CWD,
# so the .o is placed in the output dir where ffn_cascade.py chdirs to compile.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/route_b_kernels/cascade_ffn"
OUT="$REPO/artifacts/cascade_ffn"

# (1) toolchain isolation: mlir-air airenv only (NOT .venv-iron).
source "$REPO/scripts/air_env.sh"

AIEOPT_DIR="$(python3 -c 'import mlir_aie; print(list(mlir_aie.__path__)[0])')"
WARN="-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body"

mkdir -p "$OUT"

# (2) compile the kernel into OUT (link_with resolves it from the compile CWD).
# -O2 first; fall back to -Os if the core ELF overflows ~16KB program memory.
echo "[build] compiling kernel mv_bf16_gelu.cc -> mv_bf16_gelu.o (-O2)"
"$PEANO_INSTALL_DIR/bin/clang++" -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  $WARN -DNDEBUG -I "$AIEOPT_DIR/include" \
  -c "$SRC/mv_bf16_gelu.cc" -o "$OUT/mv_bf16_gelu.o"

# (3) lower + build the xclbin (ffn_cascade.py chdirs to --out to compile).
echo "[build] lowering ffn_cascade.py -> air.xclbin + air.insts.bin + air.mlir"
python3 "$SRC/ffn_cascade.py" \
  --compile-mode compile-only --output-format xclbin --out "$OUT"

echo "[build] done. artifacts in $OUT:"
ls -la "$OUT/air.xclbin" "$OUT/air.insts.bin" "$OUT/air.mlir" 2>/dev/null || true
