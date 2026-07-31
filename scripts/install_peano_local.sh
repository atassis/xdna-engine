#!/usr/bin/env bash
# install_peano_local.sh -- install a locally-built Peano (llvm-aie) into a usable dist dir, and keep
# the libLLVM.so SONAME symlink pointed at a COMPLETE dylib.
#
# WHY THIS SCRIPT EXISTS
# build_peano_fast.sh produces a COMPILER-ONLY tree: its target list is "llc opt clang lld", so the
# build dir has no per-triple newlib runtimes, no headers, no drivers, and none of the binutils --
# notably NO llvm-ar. Symlinking a venv straight at it therefore fails (missing stdio.h/algorithm, and
# no archiver at all). The working procedure is instead:
#
#   1. `cp -al` (hardlink-copy) a KNOWN-GOOD install -- this carries runtimes, headers, drivers and
#      the full binutils set, at near-zero disk cost;
#   2. swap in ONLY the freshly-built libLLVM.so.<ver> + libclang-cpp.so.<ver>, which is where the new
#      AIE codegen actually lives;
#   3. repoint the .venv-iron/.../llvm-aie symlink. Rollback = repoint it back.
#
# That is ABI-safe only while the delta is AIE-backend-only (no core-LLVM ABI change).
#
# THE TRAP THIS SCRIPT CLOSES
# Step 2 leaves TWO libLLVM dylibs in lib/: the base install's complete one (SONAME `libLLVM.so`) and
# the newly-swapped-in build-fast one (SONAME `libLLVM.so.<ver>`). They are NOT interchangeable -- the
# build-fast dylib is a partial build and does not export everything the full one does. Tools split by
# HOW they link it:
#
#   - clang / clang++ NEED `libLLVM.so.<ver>` by exact name  -> always get the new AIE codegen;
#   - llvm-ar NEEDs the `libLLVM.so` SONAME symlink          -> gets whatever that symlink points at.
#
# So if the `libLLVM.so` symlink drifts onto the build-fast dylib, compiling keeps working while
# ARCHIVING dies at load with `undefined symbol: _ZTIN4llvm18format_object_baseE`. That silently breaks
# every build step that archives multiple kernel objects into a .a (e.g. the IRON fused-epilogue
# kernels: `llvm-ar rcs gemv_..._gelu_kernels.a gemv_....o gelu.o`) -- and because clang is unaffected,
# it does not look like a toolchain problem. Done by hand this is easy to get wrong; `--check` below
# makes it self-healing and assertable.
#
# Usage:
#   scripts/install_peano_local.sh --from <good-install> --build <build-fast-dir> --tag <name>
#                                     # hardlink-copy <good-install> -> .cache/peano-local/<name>,
#                                     # swap in the new dylibs, guard the SONAME, assert the archiver
#   scripts/install_peano_local.sh --check [<install-dir>]
#                                     # guard ONLY: verify/repair the symlink + assert llvm-ar runs.
#                                     # Idempotent; safe to re-run any time. Defaults to the install
#                                     # the repo venv currently points at.
#   scripts/install_peano_local.sh --activate <install-dir>
#                                     # repoint .venv-iron/.../llvm-aie at <install-dir> (+ guard)
#
# Env overrides:
#   PEANO_LOCAL_HOME   install root   (default: <workspace>/.cache/peano-local)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
PEANO_LOCAL_HOME="${PEANO_LOCAL_HOME:-$WS/.cache/peano-local}"
VENV_PKGS="$REPO/.venv-iron/lib/python3.14/site-packages"
VENV_LINK="$VENV_PKGS/llvm-aie"

# The RTTI typeinfo for llvm::format_object_base. Chosen as the probe because it is exported by a
# COMPLETE libLLVM and absent from the partial build-fast dylib, and because it is precisely the
# symbol whose absence aborts llvm-ar. Name is mangling-stable across LLVM versions.
PROBE_SYM='_ZTIN4llvm18format_object_baseE'

log() { echo "[install_peano_local] $*" >&2; }
die() { echo "[install_peano_local] ERROR: $*" >&2; exit 1; }

# Read a dylib's SONAME. Uses system binutils (objdump/readelf), never the toolchain's own llvm-*
# tools -- those may themselves be the thing that is broken, which would make the check circular.
_soname() {
  if command -v objdump >/dev/null 2>&1; then
    objdump -p "$1" 2>/dev/null | awk '/SONAME/{print $2; exit}'
  elif command -v readelf >/dev/null 2>&1; then
    readelf -d "$1" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p' | head -1
  fi
}

# Does this dylib export the probe symbol? Same rule: system binutils only.
#
# NOTE the `|| true` on each pipeline: these dylibs are >80 MB and dump ~10^5 symbols, so a `grep -q`
# (or -m1) exits the moment it matches and the upstream nm/objdump then dies of SIGPIPE -> 141. Under
# `set -o pipefail` that failure becomes the pipeline's status and a SUCCESSFUL match reads as "symbol
# absent" -- i.e. the guard would condemn a perfectly good install. Capture the match instead of
# relying on the pipeline's exit status.
_exports_probe() {
  local hit
  if command -v nm >/dev/null 2>&1; then
    hit="$(nm -D --defined-only "$1" 2>/dev/null | grep -m1 -F -- "$PROBE_SYM" || true)"
  elif command -v objdump >/dev/null 2>&1; then
    hit="$(objdump -T "$1" 2>/dev/null | grep -m1 -F -- "$PROBE_SYM" || true)"
  else
    die "need nm or objdump (system binutils) to validate libLLVM"
  fi
  [ -n "$hit" ]
}

# Pick the libLLVM in <libdir> that llvm-ar can actually load: it must export the probe symbol AND
# declare SONAME `libLLVM.so` (that is the name llvm-ar records in its DT_NEEDED). Prints a basename.
_complete_libllvm() {
  local libdir="$1" f base
  for f in "$libdir"/libLLVM*.so*; do
    [ -f "$f" ] || continue          # skip the symlink itself and any dangling entry
    [ -L "$f" ] && continue
    base="$(basename "$f")"
    [ "$(_soname "$f")" = "libLLVM.so" ] || continue
    _exports_probe "$f" || continue
    echo "$base"; return 0
  done
  return 1
}

# Idempotent guard: make lib/libLLVM.so point at a dylib that can actually serve llvm-ar.
_guard_soname() {
  local inst="$1" libdir="$1/lib" want cur
  [ -d "$libdir" ] || die "no lib/ in $inst"
  want="$(_complete_libllvm "$libdir")" \
    || die "no libLLVM in $libdir both exports $PROBE_SYM and declares SONAME libLLVM.so.
       This install cannot serve llvm-ar. Re-copy from a known-good install (--from)."
  cur="$(readlink "$libdir/libLLVM.so" 2>/dev/null || true)"
  if [ "$cur" = "$want" ]; then
    log "OK   libLLVM.so -> $want (complete, exports probe symbol)"
  else
    ln -sfn "$want" "$libdir/libLLVM.so"
    log "REPAIRED libLLVM.so: ${cur:-<missing>} -> $want"
  fi
}

# Prove the thing we actually care about: the archiver loads and runs. This is the gate -- a green
# symlink check is necessary but this is what the failure mode was.
_assert_archiver() {
  local inst="$1" ar="$1/bin/llvm-ar"
  [ -x "$ar" ] || die "no llvm-ar at $ar -- this install did not come from a full dist.
       build_peano_fast.sh does NOT build llvm-ar; hardlink-copy a complete install with --from."
  "$ar" --version >/dev/null 2>&1 \
    || die "llvm-ar present but fails to run:
$("$ar" --version 2>&1 | head -3)"
  log "OK   llvm-ar runs clean ($ar)"
}

# Round-trip the archiver on real objects. `--version` only proves the dynamic loader resolved the
# symbols needed at STARTUP; `rcs` on two objects is the operation that was actually broken (the IRON
# fused-epilogue kernel archive), so that is what we assert.
#
# Peano's clang is an AIE cross-compiler with NO host target ("unknown target triple 'unknown'"), so
# the objects must be built for an AIE triple, not for the build machine.
_assert_archive_roundtrip() {
  local inst="$1" tmp triple built=""
  [ -x "$inst/bin/clang" ] || { log "SKIP archive round-trip (no clang in this install)"; return 0; }
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  echo 'int a(void){return 1;}' > "$tmp/a.c"
  echo 'int b(void){return 2;}' > "$tmp/b.c"
  for triple in aie2p-none-unknown-elf aie2-none-unknown-elf; do
    if "$inst/bin/clang" --target="$triple" -c "$tmp/a.c" -o "$tmp/a.o" 2>/dev/null \
    && "$inst/bin/clang" --target="$triple" -c "$tmp/b.c" -o "$tmp/b.o" 2>/dev/null; then
      built="$triple"; break
    fi
  done
  [ -n "$built" ] || { log "SKIP archive round-trip (no usable AIE triple)"; return 0; }
  "$inst/bin/llvm-ar" rcs "$tmp/t.a" "$tmp/a.o" "$tmp/b.o" \
    || die "llvm-ar rcs FAILED on two $built objects -- this is the multi-kernel archive break."
  [ -s "$tmp/t.a" ] || die "llvm-ar rcs produced an empty archive"
  log "OK   llvm-ar rcs round-trip on 2 $built objects"
}

_activate() {
  local inst="$1"
  [ -d "$inst" ] || die "no such install: $inst"
  # Absolutize: ln resolves a relative target against the SYMLINK's dir, not the cwd, so a
  # relative --activate arg passes the -d check above and still lands on a dangling link.
  inst="$(cd "$inst" && pwd -P)"
  [ -d "$VENV_PKGS" ] || die "no venv site-packages at $VENV_PKGS"
  ln -sfn "$inst" "$VENV_LINK"
  [ -x "$VENV_LINK/bin/clang++" ] || die "activated link is broken: $VENV_LINK -> $inst"
  log "activated: $VENV_LINK -> $inst"
}

_resolve_default_install() {
  local p
  p="$(readlink -f "$VENV_LINK" 2>/dev/null || true)"
  [ -n "$p" ] && [ -d "$p" ] || die "no active Peano install (venv symlink $VENV_LINK is missing/dangling); pass a dir explicitly"
  echo "$p"
}

# ---------------------------------------------------------------------------------------------
FROM=""; BUILD=""; TAG=""; MODE="install"; CHECK_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from)     FROM="${2:?--from needs a dir}"; shift 2 ;;
    --build)    BUILD="${2:?--build needs a dir}"; shift 2 ;;
    --tag)      TAG="${2:?--tag needs a name}"; shift 2 ;;
    --check)    MODE="check"; shift; if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then CHECK_DIR="$1"; shift; fi ;;
    --activate) MODE="activate"; CHECK_DIR="${2:?--activate needs a dir}"; shift 2 ;;
    -h|--help)  sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)          die "unknown arg: $1" ;;
  esac
done

case "$MODE" in
  check)
    INST="${CHECK_DIR:-$(_resolve_default_install)}"
    log "checking $INST"
    _guard_soname "$INST"
    _assert_archiver "$INST"
    _assert_archive_roundtrip "$INST"
    log "PASS"
    ;;
  activate)
    _guard_soname "$CHECK_DIR"
    _assert_archiver "$CHECK_DIR"
    _activate "$CHECK_DIR"
    ;;
  install)
    [ -n "$FROM" ] && [ -n "$BUILD" ] && [ -n "$TAG" ] \
      || die "need --from <good-install> --build <build-fast-dir> --tag <name> (or use --check)"
    [ -d "$FROM/lib" ] || die "--from $FROM does not look like a Peano install (no lib/)"
    [ -d "$BUILD/lib" ] || die "--build $BUILD has no lib/ -- did build_peano_fast.sh run?"
    DEST="$PEANO_LOCAL_HOME/$TAG"
    [ -e "$DEST" ] && die "$DEST already exists -- pick another --tag, or rm it first"
    mkdir -p "$PEANO_LOCAL_HOME"

    log "hardlink-copy $FROM -> $DEST"
    cp -al "$FROM" "$DEST"

    # Swap in ONLY the dylibs carrying the new AIE codegen. Copy through a temp name + mv so a
    # half-written dylib can never be left in place, and break the hardlink to $FROM first --
    # writing in place would corrupt the source install we copied from.
    swapped=0
    for so in "$BUILD"/lib/libLLVM.so.*git "$BUILD"/lib/libclang-cpp.so.*git; do
      [ -f "$so" ] || continue
      base="$(basename "$so")"
      rm -f "$DEST/lib/$base"                      # break the hardlink before writing
      cp "$so" "$DEST/lib/.$base.tmp"
      mv -f "$DEST/lib/.$base.tmp" "$DEST/lib/$base"
      log "swapped in $base"
      swapped=$((swapped + 1))
    done
    [ "$swapped" -gt 0 ] || die "no libLLVM.so.*git / libclang-cpp.so.*git found in $BUILD/lib -- nothing to install"

    # Refresh the driver binaries the fast build DOES produce (clang, llc, opt, lld and friends).
    # Everything else -- llvm-ar included -- stays as hardlink-copied from $FROM.
    for b in "$BUILD"/bin/*; do
      [ -f "$b" ] && [ -x "$b" ] || continue
      base="$(basename "$b")"
      [ -e "$DEST/bin/$base" ] || continue         # only refresh what the base install already had
      rm -f "$DEST/bin/$base"
      cp "$b" "$DEST/bin/.$base.tmp"
      mv -f "$DEST/bin/.$base.tmp" "$DEST/bin/$base"
    done
    log "refreshed driver bins from $BUILD/bin"

    _guard_soname "$DEST"
    _assert_archiver "$DEST"
    _assert_archive_roundtrip "$DEST"
    log "installed: $DEST"
    log "activate with: scripts/install_peano_local.sh --activate $DEST"
    ;;
esac
