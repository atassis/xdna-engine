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
#  3. KERNEL_CC (the compiler that actually turns those .cc files into .o -- route_b_override.mk:
#     `KERNEL_CC=${PEANO_INSTALL_DIR}/bin/clang++`): resolved the SAME way, via the SAME --eval
#     call, never re-derived. This is a SECOND, independent provenance axis found by the
#     FindPeano lane 2026-07-30: a Peano `clang`/`clang++` embeds its own build SHA in
#     `--version`'s VCS banner (LLVM_REPOSITORY/LLVM_REVISION, clang/lib/Basic/Version.cpp), e.g.
#       clang version 21.0.0git (https://github.com/Xilinx/llvm-aie.git dfadd0855266e8898cd675c2eba2ea0df32baa9f)
#       clang version 21.0.0git (git@github.com:atassis/llvm-aie.git 0c8fe2df9d36aaa1ba4666f5f3721397701aa8a8)
#     confirmed to match toolchain.lock's PEANO_FORK_COMMIT exactly on the pinned install. Kernel
#     SOURCE and COMPILER IDENTITY are independent failure axes -- correct source through a wrong
#     or unresolved compiler is just as silent a miscompile as wrong source through the right
#     compiler -- so both are gated here. Two caveats the FindPeano lane confirmed and this script
#     therefore does NOT rely on: `llc`/`opt`/`llvm-ar --version` print only a bare "LLVM version
#     21.0.0git" with no repo/SHA (clang/clang++ specifically is the one carrying provenance, so
#     that is the only binary this checks); and no Peano install ships a CMake package config
#     (no `lib/cmake/*Config.cmake` to read instead). Separately, the same lane found
#     mlir-aie's CMakeLists.txt:71-76 takes PEANO_INSTALL_DIR from the environment into a CACHE
#     STRING with zero existence check -- this script does not depend on that path at all (it
#     invokes the resolved clang/clang++ binary directly and fails loud itself if absent), so it
#     is unaffected whether or not that lane's fix (branch cmake/validate-peano-install-dir,
#     unmerged) is present.
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
#  - KERNEL_CC (compiler identity): ground truth = toolchain.lock's PEANO_FORK_COMMIT, checked
#    against the resolved clang/clang++'s own --version VCS banner (see point 3 above). If
#    toolchain.lock has no PEANO_FORK_COMMIT (a future lock shape using PEANO_DIST only), this
#    sub-check is SKIPPED with a loud note rather than failed -- PEANO_FORK_COMMIT is today's
#    "active Peano" field per toolchain.lock's own header comment, not guaranteed forever, and an
#    inapplicable axis should not block the kernel-source axis it is additive to.
#
# Usage: scripts/verify_kernel_source.sh [Makefile.modal|Makefile.modal.int8|Makefile.resident ...]
#   No args = check the full family. Exit 0 = every resolved source AND the resolved compiler
#   match their ground truth; an OK line is printed per check; also stamps
#   build/.kernel_source_manifest (verified-hash record, informational -- the exit code is the
#   gate, not the file). Exit 1 = drift or a resolution failure; message names the Makefile, the
#   file/binary, both paths, both hashes/SHAs, and how to fix it. Never mutates kernels_dir,
#   PEANO_INSTALL_DIR, or the pin -- read-only except for the manifest stamp.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
RB_KERNELS=route_b_kernels/aie_kernels
# The TRACKED originals of the Makefiles sync_kernels.sh copies into $MMW. Ground truth for the
# sandbox copies, which are untracked inside the submodule working tree.
RB_MAKEFILES=route_b_kernels/whole_array_fused

# Transitive closure of quoted `#include "..."` starting from the Makefile's declared prerequisites,
# printed one relative path per line (input order first, then discovered).
#
# Quoted includes resolve relative to the INCLUDING file's own directory, which is what this walks.
# Angle-bracket includes are toolchain headers -- they come from the pinned instance and are covered
# by the KERNEL_CC identity axis, not by hashing files in kernels_dir. A quoted include that does not
# resolve under kernels_dir is likewise found via -I in the toolchain, so it is skipped rather than
# failed: a gate that cries wolf on legitimate toolchain headers gets switched off, and this one has
# to survive being run on every build.
include_closure() {
  local kdir="$1"; shift
  local -a work=("$@")
  local -A seen=()
  local rel path base inc next
  while [ ${#work[@]} -gt 0 ]; do
    rel="${work[0]}"; work=("${work[@]:1}")
    [ -n "${seen[$rel]:-}" ] && continue
    seen[$rel]=1
    printf '%s\n' "$rel"
    path="$kdir/$rel"
    [ -f "$path" ] || continue      # missing declared file: the caller's existence check reports it
    base="$(dirname "$rel")"
    while IFS= read -r inc; do
      [ -n "$inc" ] || continue
      if [ "$base" = "." ]; then next="$inc"; else next="$base/$inc"; fi
      # Normalise ./ and ../ so the same file cannot enter the set under two spellings.
      next="$(realpath -m --relative-to="$kdir" "$kdir/$next" 2>/dev/null || printf '%s' "$next")"
      case "$next" in
        ../*|/*) continue ;;        # escapes kernels_dir -> not our ground truth
      esac
      [ -f "$kdir/$next" ] && work+=("$next")
    done < <(grep -oP '^\s*#\s*include\s+"\K[^"]+' "$path" 2>/dev/null || true)
  done
}

family=("$@")
[ ${#family[@]} -gt 0 ] || family=(Makefile.modal Makefile.modal.int8 Makefile.resident)

# The only var route_b_override.mk's kernels_dir branches on is devicename (from NPU2); every
# real call site (build_kernels.sh, build_parakeet_kernels.sh, build_parakeet_modal_kernels.sh)
# passes NPU2=1. Passed through --eval alongside the real Makefile, so if a future edit makes
# kernels_dir depend on something else, asking Make (not this list) is still what answers it.
npu2_var="${NPU2:-1}"

# PEANO_INSTALL_DIR drives KERNEL_CC (see point 3 above). Every real call site already exports it
# via `source scripts/iron_env.sh` before invoking this script, so it is normally just inherited;
# self-source as a fallback only for standalone/manual invocation, so this script works on its own.
if [ -z "${PEANO_INSTALL_DIR:-}" ]; then
  echo "[verify_kernel_source] PEANO_INSTALL_DIR not in env -- sourcing scripts/iron_env.sh to resolve it (same as every build script does before calling this)" >&2
  # shellcheck disable=SC1091
  source scripts/iron_env.sh >/dev/null
fi

# toolchain.lock's own "active Peano" field (see its header comment); empty on an older/future
# lock shape that doesn't use it, in which case the KERNEL_CC sub-check is skipped, not failed.
peano_fork_commit="$(sed -n 's/^PEANO_FORK_COMMIT=\([0-9a-f]*\).*/\1/p' toolchain.lock | head -1)"

# Resolves kernels_dir AND KERNEL_CC in ONE make process (both are plain variable reads off the
# same real Makefile + real vars -- no reason to pay two process-spawns for two answers). Combines
# stdout+stderr (sed below only keeps the __KD__/__CC__ lines, so make's own warnings are silently
# dropped on success) so a genuine make failure -- which `set -e` would otherwise abort on with NO
# diagnostic, since the caller checks the exit status explicitly via `||` -- is still visible.
resolve_make_vars() {
  local mk="$1"
  make -s -C "$MMW" -f "$mk" "NPU2=$npu2_var" \
    --eval='__pf_print__: ; @echo __KD__=$(kernels_dir); echo __CC__=$(KERNEL_CC)' \
    __pf_print__ 2>&1
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

  # HOLE 1 (closed): this Makefile is an untracked `cp` target INSIDE the mlir-aie submodule working
  # tree -- invisible to `git status` in both the outer repo and the submodule. Checking only that it
  # EXISTS anchored the entire gate (kernels_dir, KERNEL_CC, the prerequisite list) on a file that
  # could be stale or hand-edited with no trail anywhere, so the gate would report green fast and
  # consistently against a drifted world. That is the tool's own failure class pointed at itself: a
  # gate whose ground truth is untracked has MOVED the drift, not removed it.
  canon="$RB_MAKEFILES/$mk"
  if [ ! -f "$canon" ]; then
    echo "[verify_kernel_source] FAIL: $mk has no canonical source at $canon -- cannot establish that the sandbox copy is current" >&2
    fail=1; continue
  fi
  if ! cmp -s "$canon" "$mkpath"; then
    echo "[verify_kernel_source] FAIL $mk: the sandbox copy differs from its canonical source
    sandbox   $mkpath   sha256=$(sha256sum "$mkpath" | cut -d' ' -f1)
    canonical $canon   sha256=$(sha256sum "$canon" | cut -d' ' -f1)
  Everything below is resolved THROUGH this file (kernels_dir, KERNEL_CC, which .cc files are
  prerequisites), so a stale or hand-edited copy silently narrows what the gate checks. FIX:
  re-run scripts/sync_kernels.sh; if the sandbox copy is the one you meant to change, edit
  $canon instead -- it is the tracked one." >&2
    fail=1; continue
  fi
  manifest_lines+=("$mk makefile $(sha256sum "$canon" | cut -d' ' -f1)")

  if ! vars_out="$(resolve_make_vars "$mk")"; then
    echo "[verify_kernel_source] FAIL: make itself failed resolving kernels_dir/KERNEL_CC for $mk:" >&2
    echo "$vars_out" >&2
    fail=1; continue
  fi
  kdir_raw="$(sed -n 's/^__KD__=//p' <<<"$vars_out")"
  cc_raw="$(sed -n 's/^__CC__=//p' <<<"$vars_out")"
  if [ -z "$kdir_raw" ]; then
    echo "[verify_kernel_source] FAIL: could not resolve kernels_dir for $mk (make --eval produced nothing -- Makefile shape changed?)" >&2
    fail=1; continue
  fi
  kdir="$(realpath -m "$kdir_raw")"

  # --- axis 2: compiler identity (independent of kernel source; see point 3 in the header) ---
  checked=$((checked + 1))
  if [ -z "$cc_raw" ]; then
    echo "[verify_kernel_source] FAIL $mk: could not resolve KERNEL_CC (make --eval produced nothing -- Makefile shape changed?)" >&2
    fail=1
  elif [ ! -x "$cc_raw" ]; then
    echo "[verify_kernel_source] FAIL $mk: KERNEL_CC resolved to $cc_raw, which is not an executable file -- PEANO_INSTALL_DIR (\"${PEANO_INSTALL_DIR:-<unset>}\") is wrong or the Peano install is missing." >&2
    fail=1
  else
    cc_ver="$("$cc_raw" --version 2>/dev/null | head -1)"
    # Any remote spelling, not just https://. The banner records whatever URL the build used, and
    # the ACTIVE Peano is a local build of a fork cloned over SSH -- `(git@github.com:atassis/
    # llvm-aie.git <sha>)`. The old https-only pattern therefore matched nothing and the gate FAILED
    # on a correctly-pinned toolchain, which is a false positive that gets a gate switched off. The
    # SHA is what we compare; the URL is not evidence of anything.
    cc_sha="$(grep -oP '\(\S+ \K[0-9a-f]{40}(?=\))' <<<"$cc_ver" || true)"
    if [ -z "$cc_sha" ]; then
      echo "[verify_kernel_source] FAIL $mk: KERNEL_CC=$cc_raw produced no VCS/SHA provenance banner (--version: \"$cc_ver\") -- this is not (or no longer looks like) the pinned Peano fork clang; llc/opt/llvm-ar never carry this banner, only clang/clang++ do, so an empty result here means the wrong binary, not the wrong tool." >&2
      fail=1
    elif [ -z "$peano_fork_commit" ]; then
      echo "[verify_kernel_source] NOTE $mk: KERNEL_CC=$cc_raw self-reports $cc_sha, but toolchain.lock has no PEANO_FORK_COMMIT to compare against -- skipping this sub-check (not a failure)." >&2
    elif [ "$cc_sha" != "$peano_fork_commit" ]; then
      echo "[verify_kernel_source] FAIL $mk: KERNEL_CC=$cc_raw self-reports build SHA
    $cc_sha
  which does NOT match toolchain.lock's PEANO_FORK_COMMIT
    $peano_fork_commit
  -- the build would compile with the WRONG compiler (right or wrong kernel source either way).
  FIX: PEANO_INSTALL_DIR (\"$PEANO_INSTALL_DIR\") is not the toolchain.lock-pinned Peano build --
  re-run scripts/toolchain_up.sh's Peano provisioning (install_peano_local.sh) for the current pin." >&2
      fail=1
    else
      echo "[verify_kernel_source] OK   $mk KERNEL_CC=$cc_raw self-reports $cc_sha == toolchain.lock PEANO_FORK_COMMIT"
      manifest_lines+=("$mk KERNEL_CC compiler $cc_sha")
    fi
  fi

  mapfile -t declared < <(grep -oP '\$\{kernels_dir\}/\K[A-Za-z0-9_./]+\.cc' "$mkpath" | sort -u)
  if [ ${#declared[@]} -eq 0 ]; then
    echo "[verify_kernel_source] FAIL: $mk declares no \${kernels_dir}/*.cc prerequisite -- Makefile shape changed, nothing to verify against" >&2
    fail=1; continue
  fi
  # HOLE 2 (closed): the Makefile NAMES prerequisites, but the compiler READS the transitive closure.
  # `mm.cc` does `#include "zero.cc"`, which appeared in no prerequisite list, so the entire
  # modal/resident family could report green while zero.cc drifted and changed every compiled object.
  mapfile -t srcs < <(include_closure "$kdir" "${declared[@]}")

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
  echo "[verify_kernel_source] kernel-source / compiler-identity provenance check FAILED -- refusing to build stale/drifted kernels." >&2
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

echo "[verify_kernel_source] kernel-source + compiler-identity provenance OK ($checked check(s) verified against ground truth)."
