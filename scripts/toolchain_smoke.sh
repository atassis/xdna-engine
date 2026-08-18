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
export MLIR_AIE_INSTANCE="$INST"   # route_b_override.mk hard-requires this; without it the gate
                                   # dies in make before compiling anything, which reads as a
                                   # toolchain failure rather than the missing export it is.
MMW="$REPO/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
# This gate COPIES generator + Makefile into the mlir-aie submodule and builds there, which is the
# exact drift surface verify_kernel_source.sh exists to catch: a correctly-bumped toolchain.lock can
# still compile stale kernel source with no error. Check before spending the build, same as the four
# build_*.sh scripts. (Left unwired until now only because another session held this file modified.)
"$REPO/scripts/verify_kernel_source.sh" Makefile.modal
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
# With no prior copy there is nothing to restore, so the gate's own CPU build would be what stays --
# the artifact this whole dance exists to keep out of build/. Delete it in that case instead.
trap 'if [ -n "$GATEBAK" ] && [ -f "$GATEBAK" ]; then mv -f "$GATEBAK" "$XCLBIN"; else rm -f "$XCLBIN"; fi' EXIT
# The goal must not already be satisfied, or make answers "Nothing to be done", never re-emits the
# generator .mlir, and the gate decides the toolchain on an artifact the toolchain did not build.
# GATEBAK holds the device-validated copy the trap restores.
rm -f "$XCLBIN"
( cd "$REPO" && WA_C_DEPTH=1 make -C "$MMW" -f Makefile.modal NPU2=1 M=512 K=800 N=3072 m=64 k=32 n=96 \
    n_aie_cols=8 emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 \
    build/final_512x800x3072_64x32x96_8c_modalsilu.xclbin )
# Gate signals (place-tiles model): the GENERATOR emits LOGICAL tiles, aiecc's place-tiles PASS assigns
# PHYSICAL coordinates, and the full xclbin builds end-to-end.
[ -f "$GEN" ] || { echo "SMOKE FAIL: no generator MLIR ($GEN)"; exit 1; }
# 1) the fork place-tiles-model generator emitted LOGICAL tiles (the wheel/Python-placer model would not)
logical=$(grep -c 'aie\.logical_tile' "$GEN" 2>/dev/null || true); logical=${logical:-0}
[ "$logical" -gt 0 ] || { echo "SMOKE FAIL: generator emitted no logical_tile -- wrong toolchain model"; exit 1; }

# 2) the place-tiles pass actually placed them. Read the placement evidence LAYOUT-TOLERANTLY: the
#    declarative-aiecc rewrite (mlir-aie #3271) stopped writing a placed .mlir into the .prj entirely --
#    that dir now holds only .ll/.script/.json/.pdi plus per-core artifact dirs. Asserting on the old
#    hardcoded `<gen>.mlir.prj/input_with_addresses.mlir` therefore false-FAILed on a perfectly good
#    toolchain that had just built the xclbin, which either blocks a good re-pin or trains the operator
#    to ignore the gate. Accept either form of proof, and only fail when NEITHER is present:
#      (a) a placed .mlir anywhere in the .prj (older layouts) -> physical aie.tile present, none unplaced;
#      (b) per-core artifact dirs named by PHYSICAL coords, e.g. elfs_main_core_<col>_<row> (declarative
#          layout) -> those names cannot exist unless place-tiles assigned real coordinates.
placed_how=""; phys=0
PLACED_MLIR="$(ls "$GEN.prj"/*.mlir "$PLACED" 2>/dev/null | head -1 || true)"
if [ -n "$PLACED_MLIR" ] && [ -f "$PLACED_MLIR" ]; then
  phys=$(grep -c '\baie\.tile\b' "$PLACED_MLIR" 2>/dev/null || true); phys=${phys:-0}
  unpl=$(grep -c '(?, ?)' "$PLACED_MLIR" 2>/dev/null || true); unpl=${unpl:-0}
  [ "$phys" -gt 0 ] && [ "$unpl" -eq 0 ] \
    || { echo "SMOKE FAIL: place-tiles did not place (phys=$phys unpl=$unpl) in $PLACED_MLIR"; exit 1; }
  placed_how="placed aie.tile=$phys (0 unplaced) in $(basename "$PLACED_MLIR")"
else
  phys=$(find "$GEN.prj" -maxdepth 1 -type d -regextype posix-extended \
           -regex '.*/(elfs|objects)_main_core_[0-9]+_[0-9]+$' 2>/dev/null | wc -l)
  [ "$phys" -gt 0 ] \
    || { echo "SMOKE FAIL: no placement evidence in $GEN.prj -- no placed .mlir AND no physical per-core dirs"; exit 1; }
  placed_how="physical per-core artifact dirs=$phys"
fi

# 3) the full xclbin built end-to-end
[ -f "$XCLBIN" ] || { echo "SMOKE FAIL: no xclbin"; exit 1; }

# 4) the ARCHIVER works. Compiling and archiving fail independently: clang links libLLVM.so.<ver> by
#    exact name while llvm-ar resolves the libLLVM.so SONAME symlink, so a drifted symlink leaves the
#    compiler green and silently breaks every multi-kernel `llvm-ar rcs` (the IRON fused-epilogue
#    kernels). Self-heals the symlink when it can, fails loud when it cannot.
"$REPO/scripts/install_peano_local.sh" --check >/dev/null 2>&1 \
  || { echo "SMOKE FAIL: Peano archiver check failed -- run scripts/install_peano_local.sh --check"; exit 1; }

echo "SMOKE PASS (CPU): logical_tile=$logical -> $placed_how, xclbin built, llvm-ar OK"
