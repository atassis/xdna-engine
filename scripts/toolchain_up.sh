#!/usr/bin/env bash
# Build (or locate, if already built) the mlir-aie-with-bindings toolchain INSTANCE for the current
# toolchain.lock, into a content-addressed dir keyed by the lock hash. Prints the instance dir on stdout.
# Self-consistent: fork IRON (place-tiles) + fork aiecc + the kernel aie_api headers, one version.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/toolchain.lock"; set +a
source "$REPO/scripts/fast_build_env.sh"   # ccache + lld (no-ops if absent)
source "$REPO/scripts/toolchain_gc.sh"

# Resolve the MLIR core distro (the LLVM/MLIR framework aiecc is built ON -- NOT Peano). It is a
# prebuilt dependency provisioned SEPARATELY by scripts/fetch_mlir_distro.sh (network); this script
# stays local-only and just resolves the path. Prefer the single-file pin MLIR_DISTRO_WHEEL (bumping
# the MLIR core = one line in toolchain.lock); fall back to a repo-relative MLIR_DISTRO_DIR for old locks.
if [ -n "${MLIR_DISTRO_WHEEL:-}" ]; then
  MLIR_DISTRO_ABS="$XDNA_CACHE/mlir-distro/${MLIR_DISTRO_WHEEL#mlir-}/mlir"
  [ -e "$MLIR_DISTRO_ABS/bin/mlir-tblgen" ] || {
    echo "[toolchain_up] MLIR distro $MLIR_DISTRO_WHEEL not provisioned. Run: scripts/fetch_mlir_distro.sh" >&2
    exit 1
  }
else
  MLIR_DISTRO_ABS="$REPO/$MLIR_DISTRO_DIR"
fi

LOCKHASH="$(sha256sum "$REPO/toolchain.lock" | cut -c1-12)"
INST="${TOOLCHAIN_HOME:-$XDNA_CACHE/instances}/$LOCKHASH"
PYPKG="$INST/python/aie/iron/program.py"
WHEEL_BIN="$REPO/.venv-iron/lib/python3.14/site-packages/mlir_aie/bin"

# Fill the instance bin with the vendored prebuilt tools it does NOT build itself. aiecc, aie-opt and
# aie-translate are built from the fork source (version-sensitive, place-tiles); bootgen / aie-lsp-server /
# aie-reset / aie-visualize come from the wheel, since they consume no aie-opt output and are genuinely
# version-independent. Idempotent.
#
# History: until 2026-08-24 this comment called the whole vendored set "version-agnostic", which was
# false for aie-translate -- it CONSUMES aie-opt's output, so it cannot be version-independent by
# construction. Measured that day: the wheel was mlir_aie 0.0.1.2026033104 (31 Mar) against a fork pin
# of 17 Aug, and the two disagreed about aiex.npu.address_patch (aie-opt prints it with an operand, the
# vendored aie-translate answered "requires zero operands"), failing 51 Targets/ lit tests. aiecc
# translates IN-PROCESS against the fork-built library and never execs this binary, so no xclbin was
# ever affected -- the skew only ever reached `check-aie`. Fixed by building aie-translate from the
# fork (see _build_aie_translate below) instead of vendoring it.
_link_vendored_tools() {
  local t
  for t in "$WHEEL_BIN"/*; do
    local b; b="$(basename "$t")"
    [ "$b" = "aie-translate" ] && continue   # fork-built by _build_aie_translate, never the wheel copy
    [ -e "$INST/build/bin/$b" ] || ln -sfn "$t" "$INST/build/bin/$b"
  done
}

# Refresh the include/ symlinks aie.iron + the kernel headers resolve against. Run on BOTH the cold
# build and the warm early-return so a plain re-run against any instance is self-healing (the warm
# path does not rebuild, so these would otherwise never be recreated if removed).
_link_include_dirs() {
  ln -sfn "$REPO/.venv-iron/lib/python3.14/site-packages/mlir_aie/include/aie_api" "$INST/build/include/aie_api"
  ln -sfn "$REPO/mlir-aie/aie_kernels" "$INST/build/include/aie_kernels"   # aie.iron _default_source_path resolves kernel .cc here (aie2p/mm.cc etc.)
}

# Point the generated lit config at the Peano we actually run. Without PEANO_INSTALL_DIR at configure
# time it is "<unset>", detect_peano fails, and the 126 tests that say `REQUIRES: peano` report
# UNSUPPORTED instead of running -- a local `lit` then finds 1211 tests where CI finds 1219 and says
# nothing about the difference. That is how Xilinx/mlir-aie#3461 shipped a routing regression that
# two aie2 unit tests catch: they were never executed here. Same self-healing contract as
# _link_include_dirs, since the warm path never re-runs cmake.
_wire_peano_lit() {
  local cfg="$INST/build/test/lit.site.cfg.py"
  local peano="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie"
  [ -f "$cfg" ] && [ -x "$peano/bin/clang" ] || return 0
  grep -q "^config.peano_install_dir = r\"\"\"$peano\"\"\"$" "$cfg" && return 0
  sed -i "s|^config\.peano_install_dir = r\"\"\".*\"\"\"$|config.peano_install_dir = r\"\"\"$peano\"\"\"|" "$cfg"
}

# Teach the instance that this box's NPU is an npu2. The 2026-08-20 driver update renamed it by
# silicon revision (npu4_regs.c:159, AIE2_DEV_REVISION_GPT1 -> "NPU Gorgon Point 1"); mlir-aie
# substring-matches that against a hardcoded NPU_MODELS allowlist, so every python device runner
# raises `Unknown device type` in the constructor, before any dispatch. Same silicon (1022:17f0
# rev 0x10, 8 columns) -- Gorgon Point IS npu2, confirmed by execution.
#
# No pinned commit carries the entry: 677319c935c sits on the unmerged branch
# fix/npu-device-name-gorgon-point. Until it lands and we re-pin, this is a tethered patch, applied
# to $INST/src (build/python symlinks it) on both the cold and warm paths. Self-retiring: once a pin
# carries "Gorgon Point" the grep guard makes it a no-op.
_recognise_gorgon_point() {
  local f
  for f in "$INST/src/python/utils/hostruntime/xrtruntime/hostruntime.py" \
           "$INST/src/python/aie_lit_utils/lit_config_helpers.py"; do
    [ -f "$f" ] || continue
    grep -q '"Gorgon Point"' "$f" && continue
    grep -q '^\( *\)"npu2": \[' "$f" || {
      echo "[toolchain_up] WARN: no NPU_MODELS npu2 entry in $f -- device runners may not open this NPU" >&2
      continue
    }
    sed -i 's|^\( *\)"npu2": \[\(.*\)\],$|\1"npu2": [\2, "Gorgon Point"],|' "$f"
  done
  # A hand-patch on 2026-08-21 replaced build/python/.../hostruntime.py with a REAL file, silently
  # diverging it from the src/ it used to symlink. Re-point it so there is one copy to reason about.
  local b="$INST/build/python/aie/utils/hostruntime/xrtruntime/hostruntime.py"
  local t="$INST/src/python/utils/hostruntime/xrtruntime/hostruntime.py"
  [ -f "$b" ] && [ ! -L "$b" ] && [ -f "$t" ] && ln -sfn "$t" "$b"
  return 0
}

# Build the scratchpad host binding into an instance that predates the flag flip. Same self-healing
# contract as _link_include_dirs: the warm path never re-runs cmake, so an instance configured with
# AIE_ENABLE_XRT_PYTHON_BINDINGS=OFF would keep skipping every scratchpad test forever. Self-retiring
# once the .so exists. Checks pybind11 FIRST -- reconfiguring without it is a CMake FATAL_ERROR, which
# on the warm path would leave a half-updated cache on an instance that was working.
_build_parameter_scratchpad() {
  compgen -G "$INST/build/python/aie/_mlir_libs/_parameter_scratchpad*.so" >/dev/null 2>&1 && return 0
  [ -f "$INST/build/CMakeCache.txt" ] || return 0
  "$REPO/.venv-iron/bin/python" -c "import pybind11" 2>/dev/null || {
    echo "[toolchain_up] WARN: pybind11 missing in .venv-iron -- scratchpad host binding not built;" >&2
    echo "[toolchain_up]       aiecc still emits params.txt, but every scratchpad test will SKIP." >&2
    return 0
  }
  echo "[toolchain_up] backfilling _parameter_scratchpad into $LOCKHASH ..." >&2
  cmake -B "$INST/build" -S "$INST/src" -DAIE_ENABLE_XRT_PYTHON_BINDINGS=ON >&2 \
    && ninja -C "$INST/build" _parameter_scratchpad >&2 \
    || echo "[toolchain_up] WARN: scratchpad binding backfill failed; scratchpad tests will skip" >&2
}

# Build aie-translate from the fork instead of vendoring the wheel's five-months-older copy (see the
# _link_vendored_tools comment above for why the skew matters). A regular file here means an instance
# already has the fork build -- skip. `rm -f` before the link is defensive: aie-opt/aiecc were already
# proven fork-built by this point, so the only remaining vendored entry at this path is the symlink
# _link_vendored_tools used to create; deleting it first means the link step creates a fresh file rather
# than writing through whatever is already there, regardless of the linker's own symlink handling.
_build_aie_translate() {
  [ -e "$INST/build/bin/aie-translate" ] && [ ! -L "$INST/build/bin/aie-translate" ] && return 0
  rm -f "$INST/build/bin/aie-translate"
  ninja -C "$INST/build" aie-translate >&2
}

if [ -f "$PYPKG" ] && grep -q "def resolve_program(self, device_name" "$PYPKG"; then
  _build_aie_translate  # backfill the fork-built aie-translate (else it stays the stale wheel symlink)
  _link_vendored_tools   # backfill vendored tools into already-built instances
  _link_include_dirs     # backfill include/ symlinks (aie_api + aie_kernels)
  _wire_peano_lit        # backfill the lit peano path (else `REQUIRES: peano` tests silently skip)
  _recognise_gorgon_point  # backfill the npu2 device-name entry (else every device runner raises)
  _build_parameter_scratchpad  # backfill the scratchpad host binding (else scratchpad tests silently skip)
  touch "$INST"          # record last-used (for gc_instances keep-newest-N); warm path never GCs
  echo "$INST"; exit 0   # cached, self-consistent
fi
echo "[toolchain_up] building instance $LOCKHASH ..." >&2
# nanobind builds the MLIR bindings; pybind11 builds _parameter_scratchpad (below). Both are
# configure-time hard requirements -- cmake FATAL_ERRORs without them.
"$REPO/.venv-iron/bin/python" -m pip install -q "nanobind==$NANOBIND" pybind11
mkdir -p "$INST"
# Source = a CLEAN checkout of the fork integration-branch commit (NO dirty working tree); the route_b kernels
# are overlaid by sync_kernels (policy B). The prebuilt MLIR distro + cmake helpers come from the submodule.
SRC="$INST/src"
if [ ! -e "$SRC/tools/aiecc/aiecc.cpp" ]; then
  rm -rf "$SRC"; git -C "$REPO/mlir-aie" worktree prune
  git -C "$REPO/mlir-aie" cat-file -e "${MLIR_AIE_FORK_COMMIT}^{commit}" 2>/dev/null \
    || git -C "$REPO/mlir-aie" fetch -q fork "$MLIR_AIE_FORK_COMMIT"
  git -C "$REPO/mlir-aie" worktree add -q --detach "$SRC" "$MLIR_AIE_FORK_COMMIT" >&2
  # Point the worktree's empty nested-submodule dirs at the main checkout's populated versions -- but FIRST
  # pin each to the exact commit MLIR_AIE_FORK_COMMIT records for it. These deps are NOT version-stable:
  # bumping the pin can bump a submodule (e.g. aie-rt 6a15e48 -> e2aca220), and mlir-aie's own vendor
  # patches (third_party/patches/aie-rt/*.patch) only apply to the pinned version -- symlinking a stale
  # main-checkout submodule then fails `apply_aie_rt_vendor_patches` at CMake time. The pin is authoritative.
  for nested in cmake/modulesXilinx third_party/aie-rt third_party/bootgen third_party/aie_api; do
    git -C "$REPO/mlir-aie" ls-tree "$MLIR_AIE_FORK_COMMIT" "$nested" >/dev/null 2>&1 || continue
    want=$(git -C "$REPO/mlir-aie" rev-parse "${MLIR_AIE_FORK_COMMIT}:$nested" 2>/dev/null) || continue
    # fresh clone: the submodule dir is empty/uninitialized -> init it so it has an object store to pin
    [ -e "$REPO/mlir-aie/$nested/.git" ] || git -C "$REPO/mlir-aie" submodule update --init -- "$nested" >/dev/null 2>&1 || true
    cur=$(git -C "$REPO/mlir-aie/$nested" rev-parse HEAD 2>/dev/null || echo none)
    if [ "$want" != "$cur" ]; then
      git -C "$REPO/mlir-aie/$nested" cat-file -e "${want}^{commit}" 2>/dev/null \
        || git -C "$REPO/mlir-aie/$nested" fetch -q origin "$want" 2>/dev/null || true
      git -C "$REPO/mlir-aie/$nested" checkout -q --force --detach "$want" 2>/dev/null \
        || echo "[toolchain_up] WARN: could not pin $nested to ${want:0:10} (have ${cur:0:10}); build may fail" >&2
    fi
    [ -e "$REPO/mlir-aie/$nested" ] && { rm -rf "$SRC/$nested"; ln -sfn "$REPO/mlir-aie/$nested" "$SRC/$nested"; }
  done
  bash "$REPO/scripts/sync_kernels.sh" "$SRC" >&2
fi
# AIE_ENABLE_XRT_PYTHON_BINDINGS=ON builds _parameter_scratchpad, the host side of runtime
# scratchpad params. Despite the option name the module is XRT-free (TEST_UTILS_USE_XRT=0, raw
# buffer), and DISABLE_FIND_PACKAGE_XRT below does not suppress it -- XRT_COREUTIL/UUID come from
# find_library, so the cmake_dependent_option guarding it holds. OFF emitted params.txt with
# nothing able to read it, so every scratchpad test skipped rather than failed.
cmake -G Ninja -B "$INST/build" -S "$SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$REPO/.venv-iron/bin/python" \
  -DCMAKE_PREFIX_PATH="$MLIR_DISTRO_ABS" \
  -DMLIR_DIR="$MLIR_DISTRO_ABS/lib/cmake/mlir" \
  -DCMAKE_MODULE_PATH="$REPO/mlir-aie/cmake/modulesXilinx" \
  -DAIE_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_USE_LINKER=lld \
  -DCMAKE_DISABLE_FIND_PACKAGE_XRT=ON -DCMAKE_DISABLE_FIND_PACKAGE_hsa-runtime64=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_aiebu=ON \
  -DAIE_ENABLE_XRT_PYTHON_BINDINGS=ON \
  -DPEANO_INSTALL_DIR="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache >&2
ninja -C "$INST/build" AIEPythonModules aiecc aie-opt >&2
ln -sfn "$INST/build/python" "$INST/python"
_link_include_dirs
_wire_peano_lit
_recognise_gorgon_point
ln -sfn "$INST/build/bin" "$INST/bin"
_build_aie_translate
_link_vendored_tools
touch "$INST"                                          # record last-used before GC (protects it as newest)
gc_instances "${TOOLCHAIN_HOME:-$XDNA_CACHE/instances}" "${TOOLCHAIN_KEEP:-4}" "$INST"
echo "$INST"
