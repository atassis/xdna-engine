#!/usr/bin/env bash
# =============================================================================
# fast_build_env.sh -- cross-cutting fast-build levers for the AIE toolchain.
#
# SOURCE this (do not execute) from any component build to pick up ccache, a
# fast linker (lld/mold), ninja, and -j(nproc). It also exposes per-component
# cmake flag fragments as shell functions so a from-scratch fork-rebuild can
# consume one consistent profile instead of re-deriving flags per repo.
#
#   source scripts/fast_build_env.sh        # sets env + defines fb_* helpers
#   cmake -B build $(fb_ccache_flags) $(fb_llvm_fast_flags) ...
#
# Levers (all CPU-only, none touch the NPU):
#   1. ccache    -- the single biggest lever; LLVM cold-hours -> warm-minutes.
#   2. lld/mold  -- parallel linker (LLVM has thousands of link steps).
#   3. ninja     -- parallel build graph (vs serial make).
#   4. -j nproc  -- saturate all cores.
#
# Idempotent + side-effect-light: only exports vars + prepends a ccache shim
# dir to PATH. Safe to source multiple times.
# =============================================================================

# ---- 0. locate ccache (system first, then the user-local static binary) -----
. "$(dirname "${BASH_SOURCE[0]}")/cache_env.sh"   # -> XDNA_CACHE (in-workspace build cache)
_fb_cache_home="$XDNA_CACHE"
if command -v ccache >/dev/null 2>&1; then
  FB_CCACHE="$(command -v ccache)"
elif [ -x "$_fb_cache_home/bin/ccache" ]; then
  FB_CCACHE="$_fb_cache_home/bin/ccache"
  # make bare `ccache` resolvable for sub-builds that hardcode the name
  case ":$PATH:" in *":$_fb_cache_home/bin:"*) : ;; *) PATH="$_fb_cache_home/bin:$PATH";; esac
  export PATH
else
  FB_CCACHE=""   # not available; callers fall back to no launcher
fi
export FB_CCACHE

# ---- shared ccache store (one cache across ALL components/rebuilds) ----------
export CCACHE_DIR="${CCACHE_DIR:-$_fb_cache_home/ccache}"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-40G}"   # an LLVM tree alone is ~10-15G of objects
export CCACHE_COMPRESS="${CCACHE_COMPRESS:-1}"
mkdir -p "$CCACHE_DIR" 2>/dev/null || true

# ---- 1. pick the fastest available linker -----------------------------------
# mold > lld > default. (-DLLVM_USE_LINKER takes the bare name.)
if command -v mold >/dev/null 2>&1; then
  FB_LINKER="mold"
elif command -v ld.lld >/dev/null 2>&1 || command -v lld >/dev/null 2>&1; then
  FB_LINKER="lld"
else
  FB_LINKER=""
fi
export FB_LINKER

# ---- 2. parallelism ----------------------------------------------------------
export FB_NPROC="${FB_NPROC:-$(nproc)}"

# =============================================================================
# cmake flag fragments (echo space-separated flags; use unquoted in cmake call)
# =============================================================================

# ccache as the C/C++ (and optionally CUDA) compiler launcher. No-op if absent.
fb_ccache_flags() {
  [ -n "$FB_CCACHE" ] || return 0
  printf -- '-DCMAKE_C_COMPILER_LAUNCHER=%s -DCMAKE_CXX_COMPILER_LAUNCHER=%s ' \
    "$FB_CCACHE" "$FB_CCACHE"
}

# fast linker flag (cmake LLVM-style projects honor LLVM_USE_LINKER).
fb_linker_flag() {
  [ -n "$FB_LINKER" ] || return 0
  printf -- '-DLLVM_USE_LINKER=%s ' "$FB_LINKER"
}

# For non-LLVM cmake projects (XRT, aiebu): inject the linker via flags instead.
fb_linker_flag_generic() {
  [ -n "$FB_LINKER" ] || return 0
  printf -- '-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=%s -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=%s ' \
    "$FB_LINKER" "$FB_LINKER"
}

# -----------------------------------------------------------------------------
# Peano / llvm-aie (the heavy one). AIE backend + host only, no tests/docs/
# examples/benchmarks, Release, optimized tablegen, ccache + lld.
# Add LLVM_ENABLE_ASSERTIONS=ON only for a debug/bisect build (build-asserts).
# -----------------------------------------------------------------------------
fb_llvm_fast_flags() {
  printf -- '%s' \
    "-G Ninja \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_TARGETS_TO_BUILD=AIE \
-DLLVM_ENABLE_ASSERTIONS=OFF \
-DLLVM_OPTIMIZED_TABLEGEN=ON \
-DLLVM_INCLUDE_TESTS=OFF \
-DLLVM_INCLUDE_EXAMPLES=OFF \
-DLLVM_INCLUDE_BENCHMARKS=OFF \
-DLLVM_INCLUDE_DOCS=OFF \
-DLLVM_ENABLE_PROJECTS= \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ "
  fb_linker_flag
  fb_ccache_flags
}

# -----------------------------------------------------------------------------
# mlir-aie / mlir-air: build against a PREBUILT MLIR distro wheel (never
# recompile LLVM/MLIR). ninja + lld + ccache, tests off.
# Pass -DCMAKE_PREFIX_PATH / -DLLVM_DIR / -DMLIR_DIR to the prebuilt wheel.
# -----------------------------------------------------------------------------
fb_mlir_aie_fast_flags() {
  printf -- '%s' \
    "-G Ninja \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_INCLUDE_TESTS=OFF "
  fb_linker_flag
  fb_ccache_flags
}

# -----------------------------------------------------------------------------
# XRT / aiebu: generic cmake. ninja + ccache + lld-via-flags, Release.
# -----------------------------------------------------------------------------
fb_xrt_fast_flags() {
  printf -- '%s' \
    "-G Ninja -DCMAKE_BUILD_TYPE=Release "
  fb_linker_flag_generic
  fb_ccache_flags
}

# Convenience: one-line status banner.
fb_print_status() {
  echo "[fast_build_env] ccache=${FB_CCACHE:-<none>}  CCACHE_DIR=$CCACHE_DIR (max $CCACHE_MAXSIZE)"
  echo "[fast_build_env] linker=${FB_LINKER:-<default>}  nproc=$FB_NPROC  generator=Ninja"
}

# When sourced interactively / for a quick check, print status.
case "${1:-}" in status|--status) fb_print_status;; esac
