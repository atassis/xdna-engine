#!/usr/bin/env bash
# Verify the kernel C++ source the whole_array Makefile family (Makefile.modal,
# Makefile.modal.int8, Makefile.resident -- the family route_b_override.mk's own header names)
# will ACTUALLY compile matches its ground truth. Run this BEFORE the first `make -f
# Makefile.modal[.int8]` / `-f Makefile.resident` invocation in a kernel-build script; a
# mismatch exits non-zero and must abort the build (this script sets -e; do not swallow its
# exit code with `|| true`).
#
# WHY THIS EXISTS (2026-07-30): hashing toolchain.lock is NOT sufficient. route_b_override.mk's
#   kernels_dir=${srcdir}/../../../../aie_kernels/aie2p
# resolves relative to wherever the mlir-aie SUBMODULE happens to be checked out -- never the
# pinned toolchain INSTANCE that toolchain_up.sh resolves the compiler from. So a correctly
# re-pinned toolchain.lock can be built against stale kernel source with zero error. Confirmed at
# both the source and the binary level: the shipped modal xclbin's mm.cc object carried zero
# trace of upstream #3442 even though toolchain.lock pinned a commit that has it -- reproduced
# directly and independently re-verified before this script was written.
#
# HOW IT RESOLVES "the actually-compiled source" (never hardcodes a path):
#  1. kernels_dir: asks GNU Make itself, via `--eval` injecting a throwaway print-only target into
#     the REAL Makefile with the REAL vars (NPU2=1, same as every call site). Hand-deriving the
#     same ${srcdir}/../../../../aie_kernels/... arithmetic here would just be a SECOND copy of
#     route_b_override.mk's own logic -- exactly the kind of duplicated source of truth that
#     drifts. Asking Make means a future edit to that arithmetic is still caught correctly.
#  2. Which files: reads the ${kernels_dir}/*.cc prerequisites straight out of the Makefile's own
#     `.o:` rules (grep on the tracked text Make itself parses) -- not an assumed list, not a
#     directory walk (so untracked/generated files sitting in kernels_dir, e.g. .o/.stamp/.mlir
#     build output, are never touched or flagged).
#
# GROUND TRUTH is picked PER FILE, by CONTENT-comparison, never by comparing paths (comparing
# paths would flag "kernels_dir points at the submodule" even when that submodule happens to be
# correctly pinned -- content hashing is what actually answers "did the build get the right
# bytes"):
#  - VENDOR kernels (no route_b_kernels/ override -- e.g. mm.cc, the plain upstream matmul):
#    ground truth = the PINNED TOOLCHAIN INSTANCE's own worktree source,
#    $(scripts/toolchain_up.sh)/src/aie_kernels/{aie2p,aie2}/<file> -- what toolchain.lock
#    actually bought. A mismatch here is exactly tonight's bug class.
#  - ROUTE_B kernels (route_b_kernels/aie_kernels/<file> exists -- intentionally vendored-and-
#    patched or net-new, e.g. mm_silu_epilogue.cc): ground truth = that TRACKED file. Comparing
#    these against the toolchain instance would be a false positive (upstream may not even have
#    the file, or has the unpatched original) -- sync_kernels.sh's own contract is the copy at
#    kernels_dir must equal route_b_kernels/, so that is what we check.
#
# Usage: scripts/verify_kernel_source.sh [Makefile.modal|Makefile.modal.int8|Makefile.resident ...]
#   No args = check the full family. Exit 0 = every resolved source matches its ground truth and
#   an OK line is printed per file; also stamps build/.kernel_source_manifest (verified-hash
#   record, informational -- the exit code is the gate, not the file). Exit 1 = drift or a
#   resolution failure; message names the Makefile, the file, both paths, both hashes, and how to
#   fix it. Never mutates kernels_dir or the pin -- read-only except for the manifest stamp.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
RB_KERNELS=route_b_kernels/aie_kernels

family=("$@")
[ ${#family[@]} -gt 0 ] || family=(Makefile.modal Makefile.modal.int8 Makefile.resident)

# The only var route_b_override.mk's kernels_dir branches on is devicename (from NPU2); every
# real call site (build_kernels.sh, build_parakeet_kernels.sh, build_parakeet_modal_kernels.sh)
# passes NPU2=1. Passed through --eval alongside the real Makefile, so if a future edit makes
# kernels_dir depend on something else, asking Make (not this list) is still what answers it.
npu2_var="${NPU2:-1}"

resolve_kernels_dir() {
  local mk="$1"
  make -s -C "$MMW" -f "$mk" "NPU2=$npu2_var" \
    --eval='__pf_kdir__: ; @echo __KD__=$(kernels_dir)' __pf_kdir__ 2>/dev/null \
    | sed -n 's/^__KD__=//p'
}

echo "[verify_kernel_source] resolving pinned toolchain instance (toolchain_up.sh) ..." >&2
inst="$(scripts/toolchain_up.sh)"
[ -n "$inst" ] || { echo "[verify_kernel_source] FAIL: toolchain_up.sh returned no instance path" >&2; exit 1; }

fail=0
checked=0
manifest="$MMW/build/.kernel_source_manifest"
manifest_lines=()

for mk in "${family[@]}"; do
  mkpath="$MMW/$mk"
  if [ ! -f "$mkpath" ]; then
    echo "[verify_kernel_source] FAIL: $mk not present at $mkpath -- run scripts/sync_kernels.sh first" >&2
    fail=1; continue
  fi

  kdir_raw="$(resolve_kernels_dir "$mk")"
  if [ -z "$kdir_raw" ]; then
    echo "[verify_kernel_source] FAIL: could not resolve kernels_dir for $mk (make --eval produced nothing -- Makefile shape changed?)" >&2
    fail=1; continue
  fi
  kdir="$(realpath -m "$kdir_raw")"

  mapfile -t srcs < <(grep -oP '\$\{kernels_dir\}/\K[A-Za-z0-9_./]+\.cc' "$mkpath" | sort -u)
  if [ ${#srcs[@]} -eq 0 ]; then
    echo "[verify_kernel_source] FAIL: $mk declares no \${kernels_dir}/*.cc prerequisite -- Makefile shape changed, nothing to verify against" >&2
    fail=1; continue
  fi

  for f in "${srcs[@]}"; do
    checked=$((checked + 1))
    resolved="$kdir/$f"
    if [ ! -f "$resolved" ]; then
      echo "[verify_kernel_source] FAIL $mk: $f -- kernels_dir resolved to $kdir but $f is not there" >&2
      fail=1; continue
    fi
    got_hash="$(sha256sum "$resolved" | cut -d' ' -f1)"

    rb_path="$RB_KERNELS/$f"
    if [ -f "$rb_path" ]; then
      class="route_b"; truth="$rb_path"
      want_hash="$(sha256sum "$rb_path" | cut -d' ' -f1)"
    else
      class="vendor"
      truth="$inst/src/aie_kernels/aie2p/$f"
      [ -f "$truth" ] || truth="$inst/src/aie_kernels/aie2/$f"
      if [ ! -f "$truth" ]; then
        echo "[verify_kernel_source] FAIL $mk: $f -- not found in the pinned instance ($inst/src/aie_kernels/{aie2p,aie2}/$f) or in route_b_kernels/aie_kernels/; cannot establish ground truth" >&2
        fail=1; continue
      fi
      want_hash="$(sha256sum "$truth" | cut -d' ' -f1)"
    fi

    if [ "$got_hash" != "$want_hash" ]; then
      echo "[verify_kernel_source] FAIL $mk/$f ($class kernel): the build will compile
    $resolved   sha256=$got_hash
  which does NOT match its ground truth
    $truth   sha256=$want_hash
  -- the build would silently ship the WRONG $f (this is the failure class that shipped a
  resident modal xclbin without upstream #3442's rounding fix on 2026-07-29/30). FIX:
  $([ "$class" = vendor ] \
      && echo "the mlir-aie submodule at $(dirname "$(dirname "$kdir")") is not checked out at toolchain.lock's MLIR_AIE_FORK_COMMIT ($inst was built from that commit) -- resync it, or point kernels_dir at \$INST/src/aie_kernels (owner call, out of scope here)." \
      || echo "re-run scripts/sync_kernels.sh -- the sandbox copy of $f is stale vs route_b_kernels/aie_kernels/$f.")" >&2
      fail=1
    else
      echo "[verify_kernel_source] OK   $mk/$f ($class) == $truth (sha256=$got_hash)"
      manifest_lines+=("$mk $f $class $got_hash")
    fi
  done
done

if [ "$checked" -eq 0 ]; then
  echo "[verify_kernel_source] FAIL: nothing was checked (empty family?)" >&2
  exit 1
fi

if [ "$fail" -ne 0 ]; then
  echo "[verify_kernel_source] kernel-source provenance check FAILED -- refusing to build stale/drifted kernels." >&2
  exit 1
fi

# Informational stamp only (mirrors kernel_sandbox.sh's .toolchain-stamp convention) -- records
# what THIS run verified, for a human/CI to inspect after the fact. The gate is the exit code
# above, not this file; nothing reads it back as a shortcut (no "manifest says green, skip the
# recompute" path -- that would recreate exactly the "asserted not verified" bug this exists to
# close).
mkdir -p "$(dirname "$manifest")"
{
  echo "# kernel-source provenance, verified $(date -u +%FT%TZ)"
  echo "# toolchain.lock sha256(12)=$(sha256sum toolchain.lock | cut -c1-12)  instance=$inst"
  printf '%s\n' "${manifest_lines[@]}"
} > "$manifest"

echo "[verify_kernel_source] kernel-source provenance OK ($checked file(s) verified against ground truth)."
