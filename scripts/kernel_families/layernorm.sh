#!/usr/bin/env bash
# Adapter for the declared-kernel build driver. Contract: `layernorm.sh build <stem>
# <mlir-aie-root>` builds the artifact into its normal mlir-aie build dir and stamps
# that dir (via ensure_fresh_sandbox), then exits 0, or exits non-zero on failure. Does
# NOT copy anything anywhere -- publish_kernels.sh is the sole writer into the
# install-owned kernels dir (same contract as whole_array.sh/dwconv1d.sh).
#
# Despite the family name this directory holds a heterogeneous bag of small ops, one
# Makefile per op-type, `rows`/`cols` parameterizing each from its `<R>x<C>` core. Every
# stem gets its own case arm rather than a shared regex (unlike whole_array's one generic
# recipe) -- the extra flags (dtype=bf16b, scale/stag) vary per stem, and a wrong guess
# would build the right-NAMED xclbin under the wrong flags. resadd's (scale,stag) pair is
# hardcoded per stem rather than derived from scale arithmetically -- stag is a
# caller-supplied filename tag the Makefile does not compute from scale either. Every
# invocation below requests the specific xclbin file as its make target (never `all`),
# which is also what keeps `make` from ever reaching makefile-common's host-`.exe`+CMake
# leg (the leg whole_array.sh needs `|| true` for, because it builds `all` implicitly).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=../iron_env.sh
source "$REPO/scripts/iron_env.sh"
# shellcheck source=../kernel_sandbox.sh
source "$REPO/scripts/kernel_sandbox.sh"

cmd="${1:?usage: layernorm.sh build <stem> <mlir-aie-root>}"
stem="${2:?usage: layernorm.sh build <stem> <mlir-aie-root>}"
mlir_aie_root="${3:?usage: layernorm.sh build <stem> <mlir-aie-root>}"
LNML="$mlir_aie_root/programming_examples/ml/layernorm"

case "$cmd" in
  build) ;;
  *) echo "[layernorm] unsupported command '$cmd' -- only 'build' exists today" >&2; exit 2 ;;
esac

extra=()
case "$stem" in
  ctxln_512x1024)             makefile=Makefile.ctxln;      rows=512;  cols=1024 ;;
  ctxln_512x768)              makefile=Makefile.ctxln;      rows=512;  cols=768  ;;
  affcast_512x1024)           makefile=Makefile.affinecast; rows=512;  cols=1024 ;;
  cast_512x1024)              makefile=Makefile.cast;       rows=512;  cols=1024 ;;
  cast_512x4096)              makefile=Makefile.cast;       rows=512;  cols=4096 ;;
  lnaffcast_512x1024)         makefile=Makefile.lnaffcast;  rows=512;  cols=1024 ;;
  deint_512x4096)             makefile=Makefile.deint;      rows=512;  cols=4096 ;;
  glu_512x1024)               makefile=Makefile.glu;        rows=512;  cols=1024 ;;
  accadd_512x1024)            makefile=Makefile.accadd;     rows=512;  cols=1024 ;;
  accadd_512x1024_bf16b)      makefile=Makefile.accadd;     rows=512;  cols=1024; extra=(dtype=bf16b) ;;
  resadd_512x1024_s050)       makefile=Makefile.resadd;     rows=512;  cols=1024; extra=(scale=0.5 stag=050) ;;
  resadd_512x1024_s050_bf16b) makefile=Makefile.resadd;     rows=512;  cols=1024; extra=(scale=0.5 stag=050 dtype=bf16b) ;;
  resadd_512x1024_s100)       makefile=Makefile.resadd;     rows=512;  cols=1024; extra=(scale=1.0 stag=100) ;;
  resadd_512x1024_s100_bf16b) makefile=Makefile.resadd;     rows=512;  cols=1024; extra=(scale=1.0 stag=100 dtype=bf16b) ;;
  silu_1024x400)              makefile=Makefile.silu2;      rows=1024; cols=400  ;;
  *)
    echo "[layernorm] refusing '$stem': no known recipe (only the 15 declared ctxln/affcast/cast/lnaffcast/deint/glu/accadd/resadd/silu stems)" >&2
    exit 1
    ;;
esac

# Stamp the build dir with the current toolchain hash BEFORE building -- same reason as
# whole_array.sh/dwconv1d.sh: without it, publish_kernels.sh's pin-consistency check finds
# no .toolchain-stamp and refuses to publish, so a correct rebuild never gets a manifest entry.
ensure_fresh_sandbox "$LNML/build"

make -C "$LNML" -f "$makefile" NPU2=1 rows="$rows" cols="$cols" "${extra[@]}" "build/final_${stem}.xclbin"

built="$LNML/build/final_${stem}.xclbin"
[ -f "$built" ] || { echo "[layernorm] make reported success but $built is missing" >&2; exit 1; }

echo "[layernorm] built $stem in $LNML/build (publish_kernels.sh will collect it)"
