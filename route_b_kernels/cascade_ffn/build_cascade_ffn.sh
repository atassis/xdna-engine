#!/usr/bin/env bash
# Build the single-launch bf16 Whisper-FFN cascade (Task 4) to an xclbin OR elf.
# CPU-side compile-only; no device run (that is a later batched device window).
#
#   bash route_b_kernels/cascade_ffn/build_cascade_ffn.sh           # xclbin (default)
#   OUTPUT_FORMAT=elf bash route_b_kernels/cascade_ffn/build_cascade_ffn.sh
#   M_TILE=64 OUTPUT_FORMAT=elf bash .../build_cascade_ffn.sh        # M-tiled ELF
#
# Steps: (1) source the mlir-air airenv, (2) compile the Task-3 kernel
# mv_bf16_gelu.cc -> mv_bf16_gelu.o (Peano, aie2p), (3) run ffn_cascade.py to
# lower the air module and emit air.{xclbin|elf} + air.mlir into the out dir.
# aircc resolves link_with="mv_bf16_gelu.o" from CWD, so the .o is placed in the
# output dir where ffn_cascade.py chdirs to compile.
#
# Env knobs:
#   OUTPUT_FORMAT  xclbin (default) | elf. ELF is the RE-ENTRANT path: the
#                  mlir-air load_pdi cascade-lock reset is gated on outputElf, so
#                  re-dispatch only survives on the ELF build (xclbin aborts on
#                  dispatch #2). ELF needs aiebu-asm on PATH (ironenv) -- added below.
#   M_TILE         FFN activation rows per launch trip (default = generator default).
#   OUT            output dir (default artifacts/cascade_ffn, or _elf for ELF).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/route_b_kernels/cascade_ffn"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-xclbin}"
if [[ "$OUTPUT_FORMAT" == "elf" ]]; then
  OUT="${OUT:-$REPO/artifacts/cascade_ffn_elf}"
else
  OUT="${OUT:-$REPO/artifacts/cascade_ffn}"
fi

# (1) toolchain isolation: mlir-air airenv only (NOT .venv-iron).
source "$REPO/scripts/air_env.sh"

# ELF output needs the aiebu-asm assembler (only in IRON ironenv).
if [[ "$OUTPUT_FORMAT" == "elf" ]]; then
  export PATH="$HOME/repositories/ns/amd/IRON/ironenv/bin:$PATH"
  command -v aiebu-asm >/dev/null || { echo "[build] ERROR: aiebu-asm not on PATH (need ironenv)"; exit 1; }
fi

AIEOPT_DIR="$(python3 -c 'import mlir_aie; print(list(mlir_aie.__path__)[0])')"
WARN="-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body"

mkdir -p "$OUT"

# (2) compile the kernel into OUT (link_with resolves it from the compile CWD).
# -O2 first; fall back to -Os if the core ELF overflows ~16KB program memory.
echo "[build] compiling kernel mv_bf16_gelu.cc -> mv_bf16_gelu.o (-O2)"
"$PEANO_INSTALL_DIR/bin/clang++" -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  $WARN -DNDEBUG -I "$AIEOPT_DIR/include" \
  -c "$SRC/mv_bf16_gelu.cc" -o "$OUT/mv_bf16_gelu.o"

# (3) lower + build the artifact (ffn_cascade.py chdirs to --out to compile).
MTILE_ARG=()
[[ -n "${M_TILE:-}" ]] && MTILE_ARG=(--m-tile "$M_TILE")
echo "[build] lowering ffn_cascade.py -> air.$OUTPUT_FORMAT + air.mlir ${M_TILE:+(M_TILE=$M_TILE)}"
python3 "$SRC/ffn_cascade.py" \
  --compile-mode compile-only --output-format "$OUTPUT_FORMAT" "${MTILE_ARG[@]}" --out "$OUT"

echo "[build] done. artifacts in $OUT:"
if [[ "$OUTPUT_FORMAT" == "elf" ]]; then
  ls -la "$OUT/air.elf" "$OUT/air.mlir" 2>/dev/null || true
else
  ls -la "$OUT/air.xclbin" "$OUT/air.insts.bin" "$OUT/air.mlir" 2>/dev/null || true
fi
