#!/usr/bin/env bash
# Reproduce the Route B (open mlir-aie/Peano) build environment on this CachyOS box.
# Idempotent: safe to re-run. mlir-aie is a PINNED git submodule (see docs/11); .venv-iron is
# .gitignored. This script is the durable record of the CachyOS fixes (3-file patch + gcc-13
# shims + pinned toolchain) needed to build/run on Arch/CachyOS.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# The exact upstream mlir-aie commit our build is pinned to. This MUST match both the
# submodule gitlink (recorded in our history) and the toolchain wheel below (the wheel
# version string embeds +g<short-sha>). Bumping = change all three together + re-fit the
# patch + re-run scripts/test_repro_vendoring.sh. See internal notes.
MLIR_AIE_SHA=8373e49165649644f1ec414c2e406c0abbbf51cf

# 1. Python 3.14 venv with system pyxrt visible (matches the system pyxrt 3.14 .so)
[ -d .venv-iron ] || uv venv --python 3.14 --system-site-packages .venv-iron

# 2. Toolchain wheels (~1.8 GB: llvm-aie/Peano + mlir_aie). PINNED to the versions that
#    match MLIR_AIE_SHA (mlir_aie 1.3.2.dev126+g8373e49 was BUILT from that commit). Skip if
#    already present. NOTE: these wheels live in rotating release assets; if a pin 404s, the
#    upstream has rotated it out -> follow the bump procedure in docs/11 (it's the same
#    deliberate op as bumping the submodule).
if ! { .venv-iron/bin/python -c 'import aie' 2>/dev/null && \
       ls .venv-iron/lib/python3.14/site-packages/llvm-aie/bin/clang >/dev/null 2>&1; }; then
  uv pip install --python .venv-iron \
    --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 \
    --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
    'mlir_aie==1.3.2.dev126+g8373e49' 'llvm-aie==21.0.0.2026052701+9e603b76'
fi

# 3. gcc-13/g++-13 shims → real gcc (makefile-common hardcodes CC?=gcc-13; we have gcc16)
mkdir -p .venv-iron/cc-shim
ln -sf "$(command -v gcc)" .venv-iron/cc-shim/gcc-13
ln -sf "$(command -v g++)" .venv-iron/cc-shim/g++-13

# 4. Init the pinned mlir-aie submodule (records MLIR_AIE_SHA in our history; replaces the
#    old gitignored `git clone --depth 1 main` which had no version pinning). Try a fast
#    shallow-by-SHA init first; fall back to a full submodule clone if git/server refuses.
if [ ! -e mlir-aie/.git ]; then
  git submodule update --init --depth 1 mlir-aie 2>/dev/null \
    || git submodule update --init mlir-aie
fi
# Pin to the exact recorded commit (the submodule gitlink already points here; this is a
# belt-and-suspenders checkout in case a shallow fetch landed on a branch tip).
if [ "$(git -C mlir-aie rev-parse HEAD)" != "$MLIR_AIE_SHA" ]; then
  git -C mlir-aie fetch --depth 1 origin "$MLIR_AIE_SHA"
  git -C mlir-aie checkout -q "$MLIR_AIE_SHA"
fi

PE="$REPO/mlir-aie/programming_examples"

# PATCH: the 3 CachyOS build fixes to upstream files (common.cmake env XRT dirs; common.h
# <cassert> for gcc16; layernorm Makefile redirect space), as ONE patch tethered to
# MLIR_AIE_SHA. Replaces the old inline seds/python. Idempotent: if it reverse-applies
# cleanly the patch is already in place, so skip. See route_b_kernels/patches/.
PATCHFILE="$REPO/route_b_kernels/patches/mlir-aie-cachyos.patch"
if git -C mlir-aie apply --reverse --check "$PATCHFILE" 2>/dev/null; then
  echo "  mlir-aie-cachyos.patch already applied"
else
  git -C mlir-aie apply "$PATCHFILE" && echo "  applied mlir-aie-cachyos.patch"
fi

# PATCH: parallelize aiecc per-core compilation on the make-based whole_array path (Parakeet
# build-perf; AIECC_JOBS, default 1 = no behaviour change). Mirrors the IRON full-elf
# AIECC_JOBS lever for the make path IRON's base.py patch can't reach. Applies AFTER cachyos
# (shares makefile-common). Idempotent (reverse-check). See route_b_kernels/patches/.
JOBSPATCH="$REPO/route_b_kernels/patches/mlir-aie-aiecc-jobs.patch"
if git -C mlir-aie apply --reverse --check "$JOBSPATCH" 2>/dev/null; then
  echo "  mlir-aie-aiecc-jobs.patch already applied"
else
  git -C mlir-aie apply "$JOBSPATCH" && echo "  applied mlir-aie-aiecc-jobs.patch"
fi

# INSTALL D: our custom kernels/designs. route_b_kernels/ (tracked) is the single source of
# truth; copy them FORWARD into the gitignored mlir-aie build sandbox (one-directional => no
# drift; real files so relative-path Makefiles/includes work). See docs/08-10 + sync_kernels.sh.
bash "$REPO/scripts/sync_kernels.sh"

echo "Route B env ready. Use:  source scripts/iron_env.sh  then  make NPU2=1 run  in an example dir."
echo "Build dwconv1d:  make -C mlir-aie/programming_examples/ml/dwconv1d NPU2=1"
