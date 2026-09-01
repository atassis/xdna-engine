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
#   scripts/install_peano_local.sh --resolve
#                                     # print the install matching toolchain.lock's PEANO_FORK_COMMIT
#                                     # (exit 1 if we do not have it) -- answers "must we re-pin?"
#   scripts/install_peano_local.sh --list
#                                     # identity, LLVM major, last-used, why-protected, aliases
#   scripts/install_peano_local.sh --gc [--keep N] [--dry-run]
#                                     # keep newest N (default 2) PER LLVM MAJOR; never removes the
#                                     # active, the pinned, or anything toolchain.lock names
#   scripts/install_peano_local.sh --migrate [--dry-run]
#                                     # rename legacy --tag dirs to their identity, old name -> alias
#   scripts/install_peano_local.sh --add-root <name> <install-dir>
#                                     # symlink <name> under roots/ -> <install-dir>; --gc treats any
#                                     # rooted install as protected, no toolchain.lock prose needed
#
# NAMING: an install directory is named for the build it CONTAINS (`llvm<major>-<sha12>`), read from
# `clang++ --version`, not for the day it was made. A hand-typed name is an unverified claim; this
# one is re-derivable from the artifact and collapses duplicates automatically. `--tag` survives as
# a human alias symlink, which is also what keeps toolchain.lock's by-path rollback prose valid.
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

# ---------------------------------------------------------------------------------------------
# IDENTITY -- an install is named by what it IS, not by when it was made.
#
# A hand-typed --tag is an unverified claim about a directory's contents. It cost a full manual
# audit on 2026-09-01: six dirs whose names implied six builds actually held FIVE (cint-test and
# integ-2026-07-31 were the same inode), and the only way to learn which was which was to run
# clang++ in each one. The build SHA is not merely computable, it is SELF-REPORTED by the artifact:
#
#   clang version 22.0.0git (git@github.com:atassis/llvm-aie.git fd3c04a086a1...)
#
# so the canonical name is derived from the binary, and can be re-verified against it at any time.
# A key hashed from build INPUTS cannot do that -- it still lies if the tree was dirty.
#
# The LLVM major is in the name (not just the SHA) because it is the axis that makes a rollback
# expensive: crossing 21 -> 22 needs a matching PEANO_DIST seed, so retention is per-major.
# Deliberately NOT keyed on sha256(toolchain.lock): Peano is determined by PEANO_FORK_COMMIT +
# PEANO_DIST and moves independently of MLIR_AIE_FORK_COMMIT ("PEANO WAS STAGED SEPARATELY" in the
# lock), so a lock-wide key would mint a fresh dir for a byte-identical Peano.
_peano_identity() {
  local inst="$1" v maj sha
  v="$("$inst/bin/clang++" --version 2>/dev/null | head -1)" || return 1
  [ -n "$v" ] || return 1
  maj="$(printf '%s' "$v" | sed -n 's/.*clang version \([0-9][0-9]*\)\..*/\1/p')"
  sha="$(printf '%s' "$v" | grep -oE '[0-9a-f]{40}' | head -1)"
  [ -n "$maj" ] && [ -n "$sha" ] || return 1
  echo "llvm${maj}-${sha:0:12}"
}

# Read a bare KEY=value from toolchain.lock (values there carry trailing '# ...' prose).
_lock_field() {
  sed -n "s/^$1=\([^ #]*\).*/\1/p" "$REPO/toolchain.lock" | head -1
}

# Real installs only -- aliases (symlinks) are skipped so nothing is counted or GC'd twice.
_real_installs() {
  local d
  for d in "$PEANO_LOCAL_HOME"/*/; do
    [ -d "$d" ] || continue
    d="${d%/}"
    [ -L "$d" ] && continue
    [ "$(basename "$d")" = "roots" ] && continue
    echo "$d"
  done
  return 0
}

# Is this install referenced by anything that must keep working? Protects, in order:
#   active venv symlink | toolchain.lock's PEANO_FORK_COMMIT | any name toolchain.lock names in prose
# The last one matters because the lock documents rollback by PATH, and a mechanical age/count GC
# cannot see a prose pointer. (The lock is not edited to fix that: LOCKHASH is sha256 of the WHOLE
# file, so touching even a comment orphans the built mlir-aie instance and forces a rebuild.)
_is_protected() {
  local inst="$1" id pin active a rooted
  id="$(_peano_identity "$inst" 2>/dev/null || true)"
  active="$(readlink -f "$VENV_LINK" 2>/dev/null || true)"
  [ -n "$active" ] && [ "$active" = "$(cd "$inst" && pwd -P)" ] && { echo "active"; return 0; }
  if rooted="$(_is_rooted "$inst")"; then echo "$rooted"; return 0; fi
  pin="$(_lock_field PEANO_FORK_COMMIT)"
  [ -n "$pin" ] && [ -n "$id" ] && [ "${id#*-}" = "${pin:0:12}" ] && { echo "pinned"; return 0; }
  # the install itself, or any alias pointing at it, named in the lock
  for a in "$(basename "$inst")" $(_aliases_of "$inst"); do
    grep -q -- "peano-local/$a" "$REPO/toolchain.lock" 2>/dev/null && { echo "lock-ref:$a"; return 0; }
  done
  return 1
}

# A GC root is a REFERENCE, not a sentence. The llvm21 rollback target was reachable only through a
# line of prose in toolchain.lock, so _is_protected had to grep the lock to avoid collecting it.
PEANO_ROOTS="$PEANO_LOCAL_HOME/roots"
_add_root() {
  local name="$1" target="$2"
  [ -d "$PEANO_LOCAL_HOME/$target" ] || die "no such install: $target"
  mkdir -p "$PEANO_ROOTS"
  ln -sfn "../$target" "$PEANO_ROOTS/$name"
  log "root: $name -> $target"
}
_is_rooted() {
  local inst="$1" r
  [ -d "$PEANO_ROOTS" ] || return 1
  for r in "$PEANO_ROOTS"/*; do
    [ -L "$r" ] || continue
    [ "$(readlink -f "$r")" = "$(cd "$inst" && pwd -P)" ] && { echo "root:$(basename "$r")"; return 0; }
  done
  return 1
}

_aliases_of() {
  local inst="$1" l
  for l in "$PEANO_LOCAL_HOME"/*; do
    [ -L "$l" ] || continue
    [ "$(readlink -f "$l")" = "$(cd "$inst" && pwd -P)" ] && basename "$l"
  done
  return 0   # a non-match on the LAST entry would otherwise be this function's status, and
             # `set -o pipefail` would carry it into the caller's assignment (set -e abort)
}

# Shows the directory NAME beside the identity its binary actually reports. A legacy hand-tagged
# dir makes the two disagree, which is the whole point: the name is a claim, the identity is evidence.
_list() {
  local d id prot mtime aliases flag
  printf '%-22s %-22s %-9s %-12s %-26s %s\n' NAME CONTAINS NAME-VS LAST-USED PROTECTED ALIASES
  for d in $(_real_installs); do
    id="$(_peano_identity "$d" 2>/dev/null || echo '?UNPROBEABLE')"
    prot="$(_is_protected "$d" || true)"
    mtime="$(date -d "@$(stat -c %Y "$d")" +%Y-%m-%d 2>/dev/null)"
    aliases="$(_aliases_of "$d" | tr '\n' ',' | sed 's/,$//')"
    # NB: an `[ test ] && var=x` STATEMENT returns 1 when the test is false, which under `set -e`
    # aborts the loop. It only showed up once --migrate made the names match.
    if [ "$(basename "$d")" = "$id" ]; then flag=ok; else flag=MISNAMED; fi
    printf '%-22s %-22s %-9s %-12s %-26s %s\n' \
      "$(basename "$d")" "$id" "$flag" "$mtime" "${prot:--}" "${aliases:--}"
  done
}

# Which install matches the CURRENT pin? This is the question that had to be answered by hand on
# 2026-09-01 (six clang++ invocations) to find out whether a re-pin was needed at all.
_resolve_pin() {
  local pin d id
  pin="$(_lock_field PEANO_FORK_COMMIT)"
  [ -n "$pin" ] || die "no PEANO_FORK_COMMIT in $REPO/toolchain.lock"
  for d in $(_real_installs); do
    id="$(_peano_identity "$d" 2>/dev/null || true)"
    [ -n "$id" ] && [ "${id#*-}" = "${pin:0:12}" ] && { echo "$d"; return 0; }
  done
  die "no install matches PEANO_FORK_COMMIT=$pin
       have: $(for d in $(_real_installs); do _peano_identity "$d" 2>/dev/null; done | tr '\n' ' ')
       build one, or re-pin toolchain.lock to an install you have."
}

# Keep the newest KEEP per LLVM MAJOR, never touching a protected install. Per-major because a
# flat keep-N deletes the cross-major rollback: on 2026-09-01 the set was 4x llvm21 + 2x llvm22,
# so "keep newest 3" would have dropped every llvm21 but one -- including the dir the lock names
# as THE llvm21 rollback target, whose rebuild is a full LLVM build across a major bump.
_gc() {
  local keep="$1" dry="$2" d id maj prot seen line
  declare -A count=()
  # Newest first, grouped by major. mtime alone does NOT order this set: it is only bumped on
  # --activate, so installs copied in the same session tie exactly (all three llvm21 installs
  # here share one mtime). Ties therefore break on the identity, so the victim is deterministic
  # rather than whatever order the glob happened to produce.
  for line in $(for d in $(_real_installs); do
                  id="$(_peano_identity "$d" 2>/dev/null || echo 'llvm?-unknown')"
                  echo "$(stat -c %Y "$d")|${id%%-*}|$d"
                done | sort -t'|' -k1,1rn -k3,3r); do
    maj="$(echo "$line" | cut -d'|' -f2)"; d="$(echo "$line" | cut -d'|' -f3)"
    count[$maj]=$(( ${count[$maj]:-0} + 1 ))
    # An install we cannot identify is one we cannot prove is redundant. Never delete it.
    if [ "$maj" = "llvm?" ]; then
      log "KEEP  $(basename "$d")  (identity unreadable -- refusing to delete what cannot be verified)"
      continue
    fi
    if prot="$(_is_protected "$d")"; then
      log "KEEP  $(basename "$d")  ($maj, $prot)"; continue
    fi
    if [ "${count[$maj]}" -le "$keep" ]; then
      log "KEEP  $(basename "$d")  ($maj, newest-$keep)"
    elif [ "$dry" = "1" ]; then
      log "WOULD REMOVE $(basename "$d")  ($maj, #${count[$maj]} of its major)"
    else
      log "REMOVE $(basename "$d")  ($maj, #${count[$maj]} of its major)"
      for a in $(_aliases_of "$d"); do rm -f "$PEANO_LOCAL_HOME/$a"; done
      rm -rf "$d"
    fi
  done
}

# Rename legacy hand-tagged dirs to their identity, leaving the OLD NAME AS AN ALIAS symlink --
# which keeps toolchain.lock's by-path rollback prose working without editing the lock (see
# _is_protected). Duplicates collapse: two names for one build become two aliases on one dir.
_migrate() {
  local dry="$1" d id dest
  # A dry run must predict the REAL outcome, so it has to model the state it would create: the
  # second name for one build becomes an alias, not a second rename. Without this the preview
  # reports two RENAMEs to one destination -- a plan that cannot happen.
  declare -A planned=()
  for d in $(_real_installs); do
    id="$(_peano_identity "$d" 2>/dev/null || true)"
    [ -n "$id" ] || { log "SKIP $(basename "$d") -- cannot probe clang++"; continue; }
    [ "$(basename "$d")" = "$id" ] && { log "OK   $(basename "$d") already identity-named"; continue; }
    dest="$PEANO_LOCAL_HOME/$id"
    if [ "$dry" = "1" ]; then
      if [ -e "$dest" ] || [ -n "${planned[$id]:-}" ]; then
        log "WOULD ALIAS  $(basename "$d") -> $id (duplicate of ${planned[$id]:-existing}, same build)"
      else
        log "WOULD RENAME $(basename "$d") -> $id"; planned[$id]="$(basename "$d")"
      fi
      continue
    fi
    if [ -e "$dest" ]; then
      # same build under a second name: drop the duplicate dir, keep the name as an alias
      rm -rf "$d"; ln -sfn "$id" "$d"
      log "ALIAS  $(basename "$d") -> $id (was a duplicate of the same build)"
    else
      mv "$d" "$dest"; ln -sfn "$id" "$d"
      log "RENAME $(basename "$d") -> $id (old name kept as alias)"
    fi
  done
}

_resolve_default_install() {
  local p
  p="$(readlink -f "$VENV_LINK" 2>/dev/null || true)"
  [ -n "$p" ] && [ -d "$p" ] || die "no active Peano install (venv symlink $VENV_LINK is missing/dangling); pass a dir explicitly"
  echo "$p"
}

# ---------------------------------------------------------------------------------------------
FROM=""; BUILD=""; TAG=""; MODE="install"; CHECK_DIR=""; KEEP=2; DRY=0; ROOT_NAME=""; ROOT_TGT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from)     FROM="${2:?--from needs a dir}"; shift 2 ;;
    --build)    BUILD="${2:?--build needs a dir}"; shift 2 ;;
    --tag)      TAG="${2:?--tag needs a name}"; shift 2 ;;
    --check)    MODE="check"; shift; if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then CHECK_DIR="$1"; shift; fi ;;
    --activate) MODE="activate"; CHECK_DIR="${2:?--activate needs a dir}"; shift 2 ;;
    --list)     MODE="list"; shift ;;
    --resolve)  MODE="resolve"; shift ;;
    --gc)       MODE="gc"; shift ;;
    --migrate)  MODE="migrate"; shift ;;
    --add-root) MODE="addroot"; ROOT_NAME="${2:?--add-root needs a name}"; ROOT_TGT="${3:?--add-root needs a target}"; shift 3 ;;
    --keep)     KEEP="${2:?--keep needs a number}"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    -h|--help)  sed -n '2,52p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
    touch "$CHECK_DIR"          # last-used, for --gc's newest-per-major ranking
    ;;
  list)     _list ;;
  resolve)  _resolve_pin ;;
  gc)       _gc "$KEEP" "$DRY" ;;
  migrate)  _migrate "$DRY" ;;
  addroot)  _add_root "$ROOT_NAME" "$ROOT_TGT" ;;
  install)
    [ -n "$FROM" ] && [ -n "$BUILD" ] \
      || die "need --from <good-install> --build <build-fast-dir> (--tag is optional; the install is
       named by its own build SHA). Or use --check/--list/--resolve/--gc."
    [ -d "$FROM/lib" ] || die "--from $FROM does not look like a Peano install (no lib/)"
    [ -d "$BUILD/lib" ] || die "--build $BUILD has no lib/ -- did build_peano_fast.sh run?"
    mkdir -p "$PEANO_LOCAL_HOME"

    # Stage first: the identity can only be read off the FINISHED artifact (post dylib+driver swap),
    # never off the inputs, so the final directory name is not known until the build is assembled.
    DEST="$(mktemp -d "$PEANO_LOCAL_HOME/.staging-XXXXXX")"
    rmdir "$DEST"
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

    # Name it for what it is, now that there is an artifact to ask.
    ID="$(_peano_identity "$DEST")" || { rm -rf "$DEST"; die "installed tree does not self-report a build SHA"; }
    FINAL="$PEANO_LOCAL_HOME/$ID"
    if [ -e "$FINAL" ]; then
      rm -rf "$DEST"
      log "identical build already installed as $ID -- reusing it, nothing new written"
      DEST="$FINAL"
    else
      mv "$DEST" "$FINAL"; DEST="$FINAL"
      log "installed: $DEST"
    fi
    # A --tag is provenance ("why did I build this"), never the key. Kept as an alias symlink.
    if [ -n "$TAG" ]; then
      [ -e "$PEANO_LOCAL_HOME/$TAG" ] && [ ! -L "$PEANO_LOCAL_HOME/$TAG" ] \
        && die "--tag $TAG collides with a real install dir"
      ln -sfn "$ID" "$PEANO_LOCAL_HOME/$TAG"; log "alias: $TAG -> $ID"
    fi
    log "activate with: scripts/install_peano_local.sh --activate $DEST"
    ;;
esac
