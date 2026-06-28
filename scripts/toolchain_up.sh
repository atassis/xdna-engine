#!/usr/bin/env bash
# Build (or locate, if already built) the mlir-aie-with-bindings toolchain INSTANCE for the current
# toolchain.lock, into a content-addressed dir keyed by the lock hash. Prints the instance dir on stdout.
# Self-consistent: fork IRON (place-tiles) + fork aiecc + the kernel aie_api headers, one version.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/toolchain.lock"; set +a
source "$REPO/scripts/fast_build_env.sh"   # ccache + lld (no-ops if absent)
LOCKHASH="$(sha256sum "$REPO/toolchain.lock" | cut -c1-12)"
INST="${TOOLCHAIN_HOME:-$HOME/.cache/xdna2-build/instances}/$LOCKHASH"
PYPKG="$INST/python/aie/iron/program.py"
WHEEL_BIN="$REPO/.venv-iron/lib/python3.14/site-packages/mlir_aie/bin"

# Fill the instance bin with the vendored prebuilt tools it does NOT build itself (bootgen PDI packager,
# aie-translate, etc.). aiecc + aie-opt are built from the fork source (version-sensitive, place-tiles);
# these others are version-agnostic vendored binaries, taken from the wheel. Idempotent.
_link_vendored_tools() {
  local t
  for t in "$WHEEL_BIN"/*; do
    local b; b="$(basename "$t")"
    [ -e "$INST/build/bin/$b" ] || ln -sfn "$t" "$INST/build/bin/$b"
  done
}

if [ -f "$PYPKG" ] && grep -q "def resolve_program(self, device_name" "$PYPKG"; then
  _link_vendored_tools   # backfill vendored tools into already-built instances
  echo "$INST"; exit 0   # cached, self-consistent
fi
echo "[toolchain_up] building instance $LOCKHASH ..." >&2
"$REPO/.venv-iron/bin/python" -m pip install -q "nanobind==$NANOBIND"
mkdir -p "$INST"
# Source = a CLEAN checkout of the fork integration-branch commit (NO dirty working tree); the route_b kernels
# are overlaid by sync_kernels (policy B). The prebuilt MLIR distro + cmake helpers come from the submodule.
SRC="$INST/src"
if [ ! -e "$SRC/tools/aiecc/aiecc.cpp" ]; then
  rm -rf "$SRC"; git -C "$REPO/mlir-aie" worktree prune
  git -C "$REPO/mlir-aie" cat-file -e "${MLIR_AIE_FORK_COMMIT}^{commit}" 2>/dev/null \
    || git -C "$REPO/mlir-aie" fetch -q fork "$MLIR_AIE_FORK_COMMIT"
  git -C "$REPO/mlir-aie" worktree add -q --detach "$SRC" "$MLIR_AIE_FORK_COMMIT" >&2
  # The worktree has empty nested-submodule dirs; point them at the submodule's populated, validated versions
  # (version-stable build deps: cmake helpers, aie-rt/xaiengine, bootgen, aie_api). Symlink avoids worktree+
  # submodule object-store quirks and reuses exactly what the validated build linked.
  for nested in cmake/modulesXilinx third_party/aie-rt third_party/bootgen third_party/aie_api; do
    [ -e "$REPO/mlir-aie/$nested" ] && { rm -rf "$SRC/$nested"; ln -sfn "$REPO/mlir-aie/$nested" "$SRC/$nested"; }
  done
  bash "$REPO/scripts/sync_kernels.sh" "$SRC" >&2
fi
cmake -G Ninja -B "$INST/build" -S "$SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$REPO/.venv-iron/bin/python" \
  -DCMAKE_PREFIX_PATH="$REPO/$MLIR_DISTRO_DIR" \
  -DMLIR_DIR="$REPO/$MLIR_DISTRO_DIR/lib/cmake/mlir" \
  -DCMAKE_MODULE_PATH="$REPO/mlir-aie/cmake/modulesXilinx" \
  -DAIE_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_USE_LINKER=lld \
  -DCMAKE_DISABLE_FIND_PACKAGE_XRT=ON -DCMAKE_DISABLE_FIND_PACKAGE_hsa-runtime64=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_aiebu=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache >&2
ninja -C "$INST/build" AIEPythonModules aiecc aie-opt >&2
ln -sfn "$INST/build/python" "$INST/python"
ln -sfn "$REPO/.venv-iron/lib/python3.14/site-packages/mlir_aie/include/aie_api" "$INST/build/include/aie_api"
ln -sfn "$INST/build/bin" "$INST/bin"
_link_vendored_tools
echo "$INST"
