#!/usr/bin/env bash
# Rebuild mlir-aie's `aiecc` compiler driver (and the libs it links, incl. the AIEX transforms)
# from the VENDORED submodule SOURCE, against the prebuilt MLIR/LLVM distro wheel — into an ISOLATED
# build prefix. Does NOT touch the shipped wheel aiecc (.venv-iron/.../mlir_aie/bin/aiecc).
#
# Why: deploy the getOrCreateDataMemref O(n^2)->O(n) fix
# (route_b_kernels/patches/mlir-aie-getorcreatedatamemref-on2.patch) that unblocks large-B (8-column)
# ELF builds. Point a build at the result with:  AIECC_PATH=<repo>/mlir-aie/build-on2/bin/aiecc
#
#   bash scripts/build_mlir_aie_aiecc.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLIR_AIE="$REPO/mlir-aie"
VENV="$REPO/.venv-iron"
PY="$VENV/bin/python"
PEANO="$VENV/lib/python3.14/site-packages/llvm-aie"
BUILD="${BUILD:-build-on2}"            # isolated, gitignored build dir under mlir-aie/
cd "$MLIR_AIE"

# 0. Nested submodules needed to BUILD aiecc (the vendored mlir-aie ships only as a wheel, so these
#    are normally absent): aie-rt = the CDO backend, bootgen = PDI generation, modulesXilinx = cmake
#    Find modules, aie_api = kernel headers. Idempotent (no-op once present).
git submodule update --init --recursive \
  cmake/modulesXilinx third_party/aie-rt third_party/bootgen third_party/aie_api

# 1. Prebuilt MLIR/LLVM distro wheel (idempotent fetch+unzip).
WV="$(bash utils/clone-llvm.sh --get-wheel-version)"
mkdir -p my_install
if [ ! -d my_install/mlir ]; then
  ( cd my_install && "$PY" -m pip -q download "mlir==$WV" \
      -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro \
    && unzip -q -u mlir-*.whl )
fi
WHL_MLIR="$(realpath my_install/mlir)"
echo "[build] WHL_MLIR=$WHL_MLIR  PEANO=$PEANO  BUILD=$BUILD"

# 2. Configure. Python bindings OFF (we only need the aiecc tool; the IRON frontend keeps using the
#    wheel bindings). RTTI OFF to match the MLIR distro wheel. Vitis components emptied (no v++ here;
#    we compile via PEANO/llvm-aie, not xchesscc — aiecc/CDO do not need Vitis).
CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_PREFIX_PATH="$WHL_MLIR"
  -DCMAKE_MODULE_PATH="$MLIR_AIE/cmake/modulesXilinx"
  -DLLVM_EXTERNAL_LIT="$(command -v lit)"
  -DCMAKE_BUILD_TYPE=Release
  -DLLVM_ENABLE_ASSERTIONS=ON
  -DAIE_ENABLE_BINDINGS_PYTHON=OFF
  -DLLVM_ENABLE_RTTI=OFF
  # Skip test/example subdirectories — their check-* lit targets depend on AIEPythonModules (absent
  # with bindings off) and are irrelevant to building the aiecc compiler. (Gated by our build-enabling
  # patch mlir-aie-build-aiecc-standalone.patch.)
  -DAIE_INCLUDE_TESTS_AND_EXAMPLES=OFF
  -DLLVM_INCLUDE_TESTS=OFF
  -DAIE_INCLUDE_INTEGRATION_TESTS=OFF
  -DAIE_RUNTIME_TARGETS=x86_64
  -DAIE_VITIS_COMPONENTS=
  -DPEANO_INSTALL_DIR="$PEANO"
  -DPython3_EXECUTABLE="$PY"
  # XRT/hsa are the NPU *runtime*, not needed to build the aiecc *compiler*. Skip them — the
  # system XRT cmake package here is broken (references a missing /usr/lib/libxilinxopencl.a).
  -DCMAKE_DISABLE_FIND_PACKAGE_XRT=ON
  -DCMAKE_DISABLE_FIND_PACKAGE_hsa-runtime64=ON
  # System AIEBU cmake package is partial (header aiebu/aiebu.h missing). Skip it so aiecc falls back
  # to the aiebu-asm subprocess (the decode build already provides aiebu-asm via AIEBU_DIR).
  -DCMAKE_DISABLE_FIND_PACKAGE_aiebu=ON
)
command -v lld >/dev/null && CMAKE_ARGS+=(-DLLVM_USE_LINKER=lld)

cmake -B "$BUILD" "${CMAKE_ARGS[@]}" . 2>&1 | tee "$BUILD-cmake.log"

# 3. Build only the aiecc tool (pulls in libAIEX with the fix). Parallelism via ninja default.
ninja -C "$BUILD" aiecc 2>&1 | tee "$BUILD-ninja.log"
echo "[build] DONE -> $MLIR_AIE/$BUILD/bin/aiecc"
ls -la "$MLIR_AIE/$BUILD/bin/aiecc"
