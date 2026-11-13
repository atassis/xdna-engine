#!/usr/bin/env bash
# Exercise Peano resource-header assembly without an LLVM build or live install.
set -euo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/install_peano_local.sh"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
export PEANO_LOCAL_HOME="$T/store"
mkdir -p "$T/bin" "$T/seed/bin" "$T/seed/lib/clang/22/include" "$T/build/bin" "$T/build/lib/clang/22"
export FIXTURE="$T"
cat > "$T/bin/objdump" <<'SH'
#!/bin/sh
echo '  SONAME               libLLVM.so'
SH
cat > "$T/bin/nm" <<'SH'
#!/bin/sh
echo '00000000 T _ZTIN4llvm18format_object_baseE'
SH
cat > "$T/bin/cmake" <<'SH'
#!/bin/sh
prefix=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --prefix) prefix=$2; shift 2;;
    --prefix=*) prefix=${1#--prefix=}; shift;;
    *) shift;;
  esac
done
[ -n "$prefix" ] || exit 2
[ "${FAIL_HEADER_INSTALL:-0}" != 1 ] || exit 1
mkdir -p "$prefix/lib/clang/22/include"
[ ! -f "$FIXTURE/candidate/current.h" ] || cp "$FIXTURE/candidate/current.h" "$prefix/lib/clang/22/include/current.h"
[ ! -f "$FIXTURE/candidate/generated.h" ] || cp "$FIXTURE/candidate/generated.h" "$prefix/lib/clang/22/include/generated.h"
SH
chmod +x "$T/bin/objdump" "$T/bin/nm" "$T/bin/cmake"
export PATH="$T/bin:$PATH"
cat > "$T/seed/bin/clang++" <<'SH'
#!/bin/sh
echo 'clang version 22.0.0git (seed aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)'
SH
cat > "$T/seed/bin/clang" <<'SH'
#!/bin/sh
while [ "$#" -gt 0 ]; do
  if [ "$1" = -o ]; then shift; printf object > "$1"; exit 0; fi
  shift
done
SH
cat > "$T/seed/bin/llvm-ar" <<'SH'
#!/bin/sh
[ "$1" != rcs ] || printf archive > "$2"
exit 0
SH
chmod +x "$T/seed/bin/clang++" "$T/seed/bin/clang" "$T/seed/bin/llvm-ar"
echo seed-current > "$T/seed/lib/clang/22/include/current.h"
printf 'seed-generated\n' > "$T/seed/lib/clang/22/include/generated.h"
echo old > "$T/seed/lib/libLLVM.so.22git"
echo old > "$T/seed/lib/libclang-cpp.so.22git"
cat > "$T/build/bin/clang++" <<'SH'
#!/bin/sh
echo 'clang version 22.0.0git (candidate bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)'
SH
cat > "$T/build/bin/clang" <<'SH'
#!/bin/sh
if [ "$1" = -print-resource-dir ]; then echo "${RESOURCE_OVERRIDE:-$FIXTURE/build/lib/clang/22}"; exit 0; fi
while [ "$#" -gt 0 ]; do
  if [ "$1" = -o ]; then shift; printf object > "$1"; exit 0; fi
  shift
done
SH
cat > "$T/build/bin/llvm-ar" <<'SH'
#!/bin/sh
[ "$1" != rcs ] || printf archive > "$2"
exit 0
SH
chmod +x "$T/build/bin/clang++" "$T/build/bin/clang" "$T/build/bin/llvm-ar"
echo new > "$T/build/lib/libLLVM.so.22git"
echo new > "$T/build/lib/libclang-cpp.so.22git"
echo seed > "$T/seed/lib/libLLVM.so.22"
echo seed > "$T/seed/lib/libclang-cpp.so.22"
mkdir -p "$T/candidate"
printf 'candidate-current\n' > "$T/candidate/current.h"
printf 'candidate-generated\n' > "$T/candidate/generated.h"
# Record seed bytes so hardlink-copy mistakes cannot hide behind matching paths.
seed_before="$(find "$T/seed" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)"
run_install() { bash "$SCRIPT" --from "$T/seed" --build "$T/build" --tag candidate; }
run_install_build_path() { bash "$SCRIPT" --from "$T/seed" --build "$1" --tag candidate; }
assert_no_install_staging() {
  local root="${1:-$PEANO_LOCAL_HOME}" leaks
  leaks="$(find "$root" -mindepth 1 -maxdepth 1 \( -name '.headers-*' -o -name '.staging-*' \) -print)"
  [ -z "$leaks" ] || fail "install leaked owned staging paths: $leaks"
}
pass() { echo "  PASS $1"; }
fail() { echo "  FAIL $1" >&2; exit 1; }

# Candidate resource headers replace the seed copy without mutating seed bytes.
run_install
ID="llvm22-bbbbbbbbbbbb"
[ "$(cat "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include/current.h")" = candidate-current ] \
  && pass 'candidate current resource header installed' || fail 'candidate current resource header missing'
[ "$(cat "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include/generated.h")" = candidate-generated ] \
  && pass 'generated resource header installed through CMake component' || fail 'generated header missing'
seed_after="$(find "$T/seed" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)"
[ "$seed_before" = "$seed_after" ] && pass 'seed bytes unchanged' || fail 'seed hardlinks overwritten'

# Absolute directory symlinks in the seed must not redirect header replacement out of staging.
mv "$T/seed/lib/clang/22" "$T/seed/lib/clang/real-22"
ln -s "$T/seed/lib/clang/real-22" "$T/seed/lib/clang/22"
mkdir -p "$T/symlink-store"
if PEANO_LOCAL_HOME="$T/symlink-store" run_install >"$T/seed-symlink.log" 2>&1; then
  pass 'seed header symlink assembled safely'
else
  grep -Fq 'seed resource-header destination escapes staging' "$T/seed-symlink.log" \
    || { cat "$T/seed-symlink.log" >&2; fail 'seed symlink failed for an unexpected reason'; }
  pass 'out-of-staging seed header symlink rejected safely'
fi
[ "$(cat "$T/seed/lib/clang/real-22/include/current.h")" = seed-current ] \
  && [ "$(cat "$T/seed/lib/clang/real-22/include/generated.h")" = seed-generated ] \
  && pass 'absolute seed header target remains unchanged' || fail 'absolute seed header target was modified'
assert_no_install_staging "$T/symlink-store"
rm "$T/seed/lib/clang/22"
mv "$T/seed/lib/clang/real-22" "$T/seed/lib/clang/22"

# A relative seed link that resolves inside staging remains usable.
mv "$T/seed/lib/clang/22" "$T/seed/lib/clang/real-22"
ln -s real-22 "$T/seed/lib/clang/22"
mkdir -p "$T/relative-symlink-store"
if PEANO_LOCAL_HOME="$T/relative-symlink-store" run_install; then
  pass 'internal relative seed header symlink assembled'
else
  fail 'safe internal relative seed header symlink rejected'
fi
[ "$(cat "$T/seed/lib/clang/real-22/include/current.h")" = seed-current ] \
  && [ "$(cat "$T/seed/lib/clang/real-22/include/generated.h")" = seed-generated ] \
  && pass 'internal relative seed target remains unchanged' || fail 'internal relative seed target was modified'
[ "$(cat "$T/relative-symlink-store/llvm22-bbbbbbbbbbbb/lib/clang/22/include/current.h")" = candidate-current ] \
  && pass 'candidate headers installed through internal relative link' || fail 'candidate headers missing through internal relative link'
assert_no_install_staging "$T/relative-symlink-store"
rm "$T/seed/lib/clang/22"
mv "$T/seed/lib/clang/real-22" "$T/seed/lib/clang/22"

# Relative and symlinked build arguments must resolve against Clang's absolute resource path.
(cd "$T" && run_install_build_path build)
pass 'relative build path accepted'
ln -s "$T/build" "$T/build-link"
run_install_build_path "$T/build-link"
pass 'symlinked build path accepted'
mkdir -p "$T/build/lib/evil"
if RESOURCE_OVERRIDE="$T/build/lib/clang/../evil" run_install; then
  fail 'resource path traversal outside clang resource tree accepted'
else
  pass 'resource path traversal rejected'
fi
assert_no_install_staging

# Failures after header installation must remove only this invocation's staging trees.
rm -f "$T/build/lib/libLLVM.so.22git" "$T/build/lib/libclang-cpp.so.22git"
if run_install; then fail 'missing candidate dylibs accepted'; else pass 'missing candidate dylibs rejected'; fi
assert_no_install_staging
echo new > "$T/build/lib/libLLVM.so.22git"
echo new > "$T/build/lib/libclang-cpp.so.22git"
# A later-stage failure after header staging must also clean both temporary trees.
mv "$T/seed/bin/llvm-ar" "$T/seed/bin/llvm-ar.saved"
if run_install; then fail 'missing seed archiver accepted'; else pass 'missing seed archiver rejected'; fi
mv "$T/seed/bin/llvm-ar.saved" "$T/seed/bin/llvm-ar"
assert_no_install_staging

# A missing candidate payload must not silently leave seed-owned resource headers in place.
rm -rf "$PEANO_LOCAL_HOME" "$T/candidate"; mkdir -p "$PEANO_LOCAL_HOME"
if run_install; then fail 'missing resource-header payload accepted'; else pass 'missing resource-header payload rejected'; fi
mkdir -p "$T/candidate"
printf 'candidate-current\n' > "$T/candidate/current.h"
printf 'candidate-generated\n' > "$T/candidate/generated.h"

# Same identity with stale header tree: fail without replacing or mutating the existing store.
rm -rf "$PEANO_LOCAL_HOME"; mkdir -p "$PEANO_LOCAL_HOME/$ID/bin" "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include" "$PEANO_LOCAL_HOME/$ID/lib"
cp "$T/build/bin/clang++" "$PEANO_LOCAL_HOME/$ID/bin/clang++"
printf 'stale\n' > "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include/current.h"
printf 'unchanged marker\n' > "$PEANO_LOCAL_HOME/$ID/marker"
if run_install; then fail 'stale same-ID store accepted'; else pass 'stale same-ID store rejected'; fi
[ "$(cat "$PEANO_LOCAL_HOME/$ID/marker")" = 'unchanged marker' ] \
  && [ "$(cat "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include/current.h")" = stale ] \
  && pass 'stale existing store remains unchanged' || fail 'existing store was changed'

# Good existing reuse remains a no-op, but only when its expected resource tree matches.
rm -rf "$PEANO_LOCAL_HOME"; mkdir -p "$PEANO_LOCAL_HOME/$ID/bin" "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include" "$PEANO_LOCAL_HOME/$ID/lib"
cp "$T/build/bin/clang++" "$PEANO_LOCAL_HOME/$ID/bin/clang++"
cp "$T/candidate/current.h" "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include/current.h"
cp "$T/candidate/generated.h" "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include/generated.h"
run_install
[ "$(cat "$PEANO_LOCAL_HOME/$ID/lib/clang/22/include/current.h")" = candidate-current ] \
  && pass 'matching same-ID store reused' || fail 'matching same-ID reuse failed'
[ "$seed_before" = "$(find "$T/seed" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)" ] \
  && pass 'seed remains unchanged after reuse' || fail 'seed changed after reuse'
