#!/usr/bin/env bash
# CPU smoke gate (NO NPU): with the wired instance, the modal generator must emit logical_tile, the
# place-tiles pass must produce physical aie.tile, and a full xclbin must build. Catches version/placement/
# binding breaks before any device window. Exit 0 = pass.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/toolchain.lock"; set +a
INST="$("$REPO/scripts/toolchain_up.sh")"
export PEANO_INSTALL_DIR="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie"
export PATH="$REPO/.venv-iron/bin:$PATH"   # make's `python3` = venv python (deps: ml_dtypes etc.)...
export PYTHONPATH="$INST/python:${PYTHONPATH:-}"   # ...while `aie` resolves to the fork instance (place-tiles)
export AIECC_PATH="$INST/bin/aiecc"
MMW="$REPO/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
cp "$REPO/route_b_kernels/whole_array_fused/whole_array_modal_iron.py" "$MMW/"   # bare-resolve_program generator
cp "$REPO/route_b_kernels/whole_array_fused/Makefile.modal" "$MMW/"
rm -f "$MMW"/build/mm_silu_epilogue_64x32x96.o "$MMW"/build/mm_64x32x96.o "$MMW"/build/aie_512x800x3072_64x32x96_8c_modalsilu.mlir
WA_C_DEPTH=1 "${REPO}/.venv-iron/bin/python" -c 'pass'   # ensure venv active path
GEN="$MMW/build/aie_512x800x3072_64x32x96_8c_modalsilu.mlir"          # generator output (pre-placement)
PLACED="$GEN.prj/input_with_addresses.mlir"                          # after the place-tiles pass
XCLBIN="$MMW/build/final_512x800x3072_64x32x96_8c_modalsilu.xclbin"
# NON-DESTRUCTIVE: the gate rebuilds the production-named xclbin to validate the toolchain, but must never
# LEAVE a CPU-only artifact in build/ (this session's #1 lesson). Save the existing (device-validated) xclbin
# and restore it on exit, pass or fail. The .prj/ placement artifacts the assertions read persist regardless.
GATEBAK=""
if [ -f "$XCLBIN" ]; then GATEBAK="$XCLBIN.gatebak"; cp -f "$XCLBIN" "$GATEBAK"; fi
trap '[ -n "$GATEBAK" ] && [ -f "$GATEBAK" ] && mv -f "$GATEBAK" "$XCLBIN"' EXIT
( cd "$REPO" && WA_C_DEPTH=1 make -C "$MMW" -f Makefile.modal NPU2=1 M=512 K=800 N=3072 m=64 k=32 n=96 \
    n_aie_cols=8 emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 \
    build/final_512x800x3072_64x32x96_8c_modalsilu.xclbin )
# Gate signals (place-tiles model): the GENERATOR emits LOGICAL tiles, aiecc's place-tiles PASS turns them
# into PHYSICAL aie.tile (in the .prj), and the full xclbin builds end-to-end.
[ -f "$GEN" ] || { echo "SMOKE FAIL: no generator MLIR ($GEN)"; exit 1; }
# 1) the fork place-tiles-model generator emitted LOGICAL tiles (the wheel/Python-placer model would not)
logical=$(grep -c 'aie\.logical_tile' "$GEN" 2>/dev/null || true); logical=${logical:-0}
[ "$logical" -gt 0 ] || { echo "SMOKE FAIL: generator emitted no logical_tile -- wrong toolchain model"; exit 1; }
# 2) the place-tiles pass placed them: physical aie.tile present, none left unplaced
[ -f "$PLACED" ] || { echo "SMOKE FAIL: no placed MLIR ($PLACED) -- place-tiles did not run"; exit 1; }
phys=$(grep -c '\baie\.tile\b' "$PLACED" 2>/dev/null || true); phys=${phys:-0}
unpl=$(grep -c '(?, ?)' "$PLACED" 2>/dev/null || true); unpl=${unpl:-0}
[ "$phys" -gt 0 ] && [ "$unpl" -eq 0 ] || { echo "SMOKE FAIL: place-tiles did not place (phys=$phys unpl=$unpl)"; exit 1; }
# 3) the full xclbin built end-to-end
[ -f "$XCLBIN" ] || { echo "SMOKE FAIL: no xclbin"; exit 1; }
echo "SMOKE PASS (CPU): logical_tile=$logical -> placed aie.tile=$phys (0 unplaced), xclbin built"
