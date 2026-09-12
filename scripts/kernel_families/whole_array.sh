#!/usr/bin/env bash
# Adapter for the declared-kernel build driver. Contract: `whole_array.sh build <stem>
# <dest-dir>` must leave `<dest-dir>/final_<stem>.xclbin` (+ insts, when the family
# produces one) in place, or exit non-zero. Only the plain bf16, M=512, tile 32x32x32,
# 8-column matmul family is understood -- that is the one uniform loop in
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
MMW="$REPO/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
# shellcheck source=../kernel_sandbox.sh
source "$REPO/scripts/kernel_sandbox.sh"

cmd="${1:?usage: whole_array.sh build <stem> <dest-dir>}"
stem="${2:?usage: whole_array.sh build <stem> <dest-dir>}"
dest="${3:?usage: whole_array.sh build <stem> <dest-dir>}"

case "$cmd" in
  build) ;;
  *) echo "[whole_array] unsupported command '$cmd' -- only 'build' exists today" >&2; exit 2 ;;
esac

if [[ ! "$stem" =~ ^512x([0-9]+)x([0-9]+)_32x32x32_8c$ ]]; then
  echo "[whole_array] refusing '$stem': only 512x{K}x{N}_32x32x32_8c (plain bf16, no variant) has a known recipe" >&2
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

mkdir -p "$dest"
cp -f "$built" "$dest/final_${stem}.xclbin.tmp"
mv -f "$dest/final_${stem}.xclbin.tmp" "$dest/final_${stem}.xclbin"
# makefile-common's insts_target is `.bin`, not `.txt` -- verified against the built
# artifact names on disk and against makefile-common's `insts_target?=build/insts_${target_suffix}.bin`.
insts="$MMW/build/insts_512x${K}x${N}_32x32x32_8c.bin"
if [ -f "$insts" ]; then
  cp -f "$insts" "$dest/insts_${stem}.bin.tmp"
  mv -f "$dest/insts_${stem}.bin.tmp" "$dest/insts_${stem}.bin"
fi
echo "[whole_array] built and staged $stem -> $dest/final_${stem}.xclbin"
