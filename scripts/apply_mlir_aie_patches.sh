#!/usr/bin/env bash
# =============================================================================
# Apply ALL mlir-aie SOURCE patches to the vendored submodule, in stacking order.
#
# WHY: the vendored mlir-aie submodule (pinned at MLIR_AIE_SHA) is patched in-tree
# to carry our O(n^2) fixes + build-speed levers, but NO single script re-applied
# them after a clean submodule checkout (setup_route_b.sh applies ONLY cachyos +
# aiecc-jobs; build_mlir_aie_aiecc.sh ASSUMES the rest are present). So a true
# from-scratch build of build-on2/bin/aiecc would silently miss every compiler fix.
# This script closes that gap.
#
# CONTRACT: run on a CLEAN mlir-aie submodule (right after `git submodule update`,
# i.e. BEFORE setup_route_b.sh applies cachyos+jobs — this script applies those
# too). It is NOT idempotent: several patches stack on the SAME files (aiecc.cpp,
# makefile-common), so a per-patch reverse-check can't reliably tell "already
# applied" from "out of order". To RE-apply, reset the submodule first:
#   git -C mlir-aie checkout . && git -C mlir-aie clean -fd <patched paths>
# (the order here is the one the fork branch atassis/mlir-aie:xdna2-asr was built
# with — it applies cleanly to the pinned base by construction.)
#
#   bash scripts/apply_mlir_aie_patches.sh
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLIR_AIE="$REPO/mlir-aie"
P="$REPO/route_b_kernels/patches"

[ -e "$MLIR_AIE/.git" ] || { echo "FATAL: mlir-aie submodule not checked out (run: git submodule update --init mlir-aie)"; exit 1; }

# Stacking order (mirrors route_b_kernels/patches/README.md + the atassis/mlir-aie
# xdna2-asr fork branch). The aiecc.cpp chain (slice-once -> kernel-cache ->
# phase-timers -> materialize-lower-once -> pass-timing -> slice-strip ->
# core-elf-cache) MUST be in this relative order; the on2 / datawords / shimdma /
# standalone patches touch disjoint files and slot in where shown.
PATCHES=(
  mlir-aie-cachyos                       # CachyOS build fixes + bf16 microkernel (foundational)
  mlir-aie-aiecc-jobs                    # -j AIECC_JOBS for the make whole_array path (after cachyos)
  mlir-aie-getorcreatedatamemref-on2     # O(n^2)->O(n) getOrCreateDataMemref (the reason build-on2 exists)
  mlir-aie-aietoconfiguration-expandpdis-on2  # expand-load-pdis O(n^2) fix
  mlir-aie-build-aiecc-standalone        # CMake: build the aiecc tool standalone
  mlir-aie-compilecores-slice-once       # aiecc.cpp: slice module once (base of the aiecc chain)
  mlir-aie-compilecores-kernel-cache     # aiecc.cpp: per-core kernel-cache dedup (after slice-once)
  mlir-aie-aiecc-phase-timers            # aiecc.cpp: byte-neutral phase timers (after kernel-cache)
  mlir-aie-aiecc-materialize-lower-once  # aiecc.cpp: O(devices^2) lower-once (after phase-timers)
  mlir-aie-aiecc-pass-timing             # aiecc.cpp: byte-neutral pass timing (after lower-once)
  mlir-aie-aiecc-slice-strip-runtimeseq  # aiecc.cpp: strip runtime_seq from per-core slices
  mlir-aie-aietargetnpu-datawords-cache  # AIETargetNPU.cpp: getDataWords symbol/data cache (independent)
  mlir-aie-shimdma-symbol-cache          # AIESubstituteShimDMAAllocations O(n^2)->O(1) (independent)
  mlir-aie-aiecc-core-elf-cache          # aiecc.cpp peanoLlcAndLink: opt-in CORE_CACHE_DIR (top of the chain)
)

# Guard: refuse to run on a non-clean-base tree (the first patch must apply).
if ! git -C "$MLIR_AIE" apply --check "$P/${PATCHES[0]}.patch" >/dev/null 2>&1; then
  echo "[apply_mlir_aie_patches] ${PATCHES[0]} does not apply — the submodule is NOT at a clean base"
  echo "  (it looks already-patched, or modified). To re-apply from clean:"
  echo "    git -C mlir-aie checkout . && git -C mlir-aie clean -fd lib include tools programming_examples aie_kernels CMakeLists.txt"
  echo "  then re-run this script. Refusing to apply onto a dirty tree."
  exit 1
fi

for name in "${PATCHES[@]}"; do
  f="$P/$name.patch"
  [ -f "$f" ] || { echo "FATAL: missing patch $f"; exit 2; }
  if git -C "$MLIR_AIE" apply "$f"; then
    echo "  [apply] $name"
  else
    echo "FATAL: $name failed to apply (out of order, or tree drifted from the pinned base)"; exit 3
  fi
done
echo "[apply_mlir_aie_patches] done: applied all ${#PATCHES[@]} mlir-aie source patches."
echo "[apply_mlir_aie_patches] NOTE: custom kernels (route_b_kernels/) are synced separately by scripts/sync_kernels.sh (run by setup_route_b.sh)."
echo "[apply_mlir_aie_patches] NEXT: bash scripts/build_mlir_aie_aiecc.sh"
