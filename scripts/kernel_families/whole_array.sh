#!/usr/bin/env bash
# Adapter for the declared-kernel build driver. Contract: `whole_array.sh build <stem>
# <mlir-aie-root>` builds the artifact into its normal mlir-aie build dir and stamps
# that dir (via ensure_fresh_sandbox), then exits 0, or exits non-zero on failure. It
# does NOT copy anything anywhere -- publish_kernels.sh, which the caller already runs
# right after any build attempt, is the sole writer into the install-owned kernels
# dir; an adapter doing its own copy on top was redundant and is gone.
# `<mlir-aie-root>` is where mlir-aie lives, so the adapter can find its own build dir
# under it instead of assuming a fixed path relative to this repo -- the rest of this
# system (npu-dev kernels-build's CLI, publish_kernels.sh) already takes that root as
# an argument rather than hardcoding it. Parakeet's K=1024 resident family (below) and the plain bf16, M=512, tile 32x32x32,
# 8-column matmul family are understood -- the latter is the one uniform loop in
# scripts/build_kernels.sh (`for KN in 768x768 3072x768 ...; do make -C $MMW NPU2=1
# M=512 K=$K N=$N dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1; done`), generic
# in K and N with no extra flags. Every other whole_array variant (modal/silu/int8/
# turbo) picks a DIFFERENT Makefile and different extra flags per shape in
# build_kernels.sh, and none of that is safely inferable from the stem alone --
# refusing an unrecognized stem is the honest behaviour; a wrong guess would build
# something that matches the stem's NAME while being compiled with the wrong flags,
# which is exactly the silent-wrong-rebuild failure this whole design exists to
# catch, not reintroduce.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Spawned as a subprocess (Command::new, not a shell built-in), so it inherits no ambient
# PYTHONPATH/AIECC_PATH from an interactively-sourced iron_env.sh -- unlike build_kernels.sh,
# which is normally run from a shell where a developer already sourced it. Source it here so
# the adapter is correct when invoked exactly as its own contract describes, standalone.
# shellcheck source=../iron_env.sh
source "$REPO/scripts/iron_env.sh"
# shellcheck source=../kernel_sandbox.sh
source "$REPO/scripts/kernel_sandbox.sh"

cmd="${1:?usage: whole_array.sh build <stem> <mlir-aie-root>}"
stem="${2:?usage: whole_array.sh build <stem> <mlir-aie-root>}"
mlir_aie_root="${3:?usage: whole_array.sh build <stem> <mlir-aie-root>}"
MMW="$mlir_aie_root/programming_examples/basic/matrix_multiplication/whole_array"

case "$cmd" in
  build) ;;
  *) echo "[whole_array] unsupported command '$cmd' -- only 'build' exists today" >&2; exit 2 ;;
esac

# Parakeet's resident K=1024 family, both tiles: scripts/build_parakeet_kernels.sh's recipe
# (Makefile.resident, which writes the insts_*.txt the engine reads; the stock Makefile writes .bin).
if [[ "$stem" =~ ^512x1024x([0-9]+)_(64x32x128|32x32x32)_8c$ ]]; then
  N="${BASH_REMATCH[1]}"
  tile="${BASH_REMATCH[2]}"
  cd "$REPO"
  bash scripts/sync_kernels.sh "$mlir_aie_root" >/dev/null
  scripts/verify_kernel_source.sh Makefile.resident
  ensure_fresh_sandbox "$MMW/build"
  rm -f "$MMW/build/mm_${tile}.o" "$MMW/build/aie_512x1024x${N}_${tile}_8c.mlir"
  if [ "$tile" = 64x32x128 ]; then
    WA_C_DEPTH=1 make -f Makefile.resident -C "$MMW" NPU2=1 M=512 K=1024 N="$N" m=64 k=32 n=128 \
      dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 \
      emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 \
      "build/final_512x1024x${N}_64x32x128_8c.xclbin"
  else
    make -f Makefile.resident -C "$MMW" NPU2=1 M=512 K=1024 N="$N" m=32 k=32 n=32 \
      dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 \
      "build/final_512x1024x${N}_32x32x32_8c.xclbin"
  fi
  for f in "final_512x1024x${N}_${tile}_8c.xclbin" "insts_512x1024x${N}_${tile}_8c.txt"; do
    [ -f "$MMW/build/$f" ] || { echo "[whole_array] build did not produce $f" >&2; exit 1; }
  done
  echo "[whole_array] built $stem in $MMW/build (publish_kernels.sh will collect it)"
  exit 0
fi

if [[ ! "$stem" =~ ^512x([0-9]+)x([0-9]+)_32x32x32_8c$ ]]; then
  echo "[whole_array] refusing '$stem': no known recipe (plain 512x{K}x{N}_32x32x32_8c, or 512x1024x{N}_{64x32x128,32x32x32}_8c)" >&2
  exit 1
fi
K="${BASH_REMATCH[1]}"
N="${BASH_REMATCH[2]}"

# Stamp the build dir with the current toolchain hash BEFORE building, same as
# build_kernels.sh does via ensure_fresh_sandbox. Without this, publish_kernels.sh's own
# pin-consistency check finds no .toolchain-stamp and refuses to publish ANY family's
# build, so a rebuild lands on disk but never gets a kernel_manifest.json entry --
# confirmed by direct observation: a real end-to-end run built and staged a correct
# xclbin, and still came back PresentUnverified rather than Present, for exactly this
# reason. `ensure_fresh_sandbox` reads `$REPO` from this same shell (sourced, not a
# subprocess) -- it's already set above, correctly, to this script's own repo root.
ensure_fresh_sandbox "$MMW/build"

# Tolerate the exit, then REQUIRE the xclbin -- makefile-common's `all` target is
# `${xclbin_target} ${targetname}.exe`, and the `.exe` half needs a working system XRT
# cmake config; on this box `xrt-config.cmake` fails to find `libxilinxopencl.a` and
# `all` dies there AFTER the xclbin is already built. Confirmed by direct observation:
# a real build here produced a correct xclbin+insts pair and still exited non-zero from
# the unrelated .exe step. `build_kernels.sh` already carries this exact tolerance for
# the same reason; naive `set -e` here would report a false failure on every build.
make -C "$MMW" NPU2=1 M=512 K="$K" N="$N" dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 || true

built="$MMW/build/final_512x${K}x${N}_32x32x32_8c.xclbin"
[ -f "$built" ] || { echo "[whole_array] make reported success but $built is missing" >&2; exit 1; }

# makefile-common's insts_target is `.bin`, not `.txt` -- the adapter doesn't need to know or
# care anymore, since it no longer touches insts files itself; publish_kernels.sh finds
# whichever extension is on disk.
echo "[whole_array] built $stem in $MMW/build (publish_kernels.sh will collect it)"
