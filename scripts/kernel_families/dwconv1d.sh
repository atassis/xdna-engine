#!/usr/bin/env bash
# Adapter for the declared-kernel build driver. Contract: `dwconv1d.sh build <stem>
# <mlir-aie-root>` builds the artifact into its normal mlir-aie build dir and stamps
# that dir (via ensure_fresh_sandbox), then exits 0, or exits non-zero on failure. Does
# NOT copy anything anywhere -- publish_kernels.sh is the sole writer into the
# install-owned kernels dir (same contract as whole_array.sh).
#
# Both declared stems are FIXED-shape, one-shot lookups -- no shape parsing needed, unlike
# whole_array's generic K/N. Unlike whole_array's makefile-common `all` (xclbin + a host
# .exe that fails to link on this box), Makefile.dwsilu/dwsilu_t's own `all` target is
# JUST the xclbin, so no `.exe`-tolerance is needed here.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Spawned as a subprocess (Command::new), so it inherits no ambient PYTHONPATH/AIECC_PATH --
# same reasoning as whole_array.sh, source both here rather than assume a caller's shell did.
# shellcheck source=../iron_env.sh
source "$REPO/scripts/iron_env.sh"
# shellcheck source=../kernel_sandbox.sh
source "$REPO/scripts/kernel_sandbox.sh"

cmd="${1:?usage: dwconv1d.sh build <stem> <mlir-aie-root>}"
stem="${2:?usage: dwconv1d.sh build <stem> <mlir-aie-root>}"
mlir_aie_root="${3:?usage: dwconv1d.sh build <stem> <mlir-aie-root>}"
DWML="$mlir_aie_root/programming_examples/ml/dwconv1d"

case "$cmd" in
  build) ;;
  *) echo "[dwconv1d] unsupported command '$cmd' -- only 'build' exists today" >&2; exit 2 ;;
esac

case "$stem" in
  dwconv_silu_1024x400)   makefile=Makefile.dwsilu ;;
  dwconv_silu_t_1024x400) makefile=Makefile.dwsilu_t ;;
  *)
    echo "[dwconv1d] refusing '$stem': only dwconv_silu_1024x400 / dwconv_silu_t_1024x400 have a known recipe" >&2
    exit 1
    ;;
esac

# Stamp the build dir with the current toolchain hash BEFORE building -- same reason as
# whole_array.sh: without it, publish_kernels.sh's pin-consistency check finds no
# .toolchain-stamp and refuses to publish, so a correct rebuild never gets a manifest entry.
ensure_fresh_sandbox "$DWML/build"

make -C "$DWML" -f "$makefile" NPU2=1 cols=8 "build/final_${stem}.xclbin"

built="$DWML/build/final_${stem}.xclbin"
[ -f "$built" ] || { echo "[dwconv1d] make reported success but $built is missing" >&2; exit 1; }

echo "[dwconv1d] built $stem in $DWML/build (publish_kernels.sh will collect it)"
