#!/usr/bin/env bash
# Provision the declared Peano seed and activate codegen built from the pinned fork.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$REPO/scripts/lib/data_root.sh"
set -a; . "$REPO/toolchain.lock"; set +a

: "${PEANO_DIST:?toolchain.lock must set PEANO_DIST}"
: "${PEANO_DIST_SHA256:?toolchain.lock must set PEANO_DIST_SHA256}"
: "${PEANO_FORK_COMMIT:?toolchain.lock must set PEANO_FORK_COMMIT}"

SEED_ROOT="${PEANO_SEED_HOME:-$XDNA_CACHE/peano-seed}"
SEED="$SEED_ROOT/$PEANO_DIST"
SOURCE="${PEANO_SOURCE_DIR:-$XDNA_CACHE/peano-source/$PEANO_FORK_COMMIT}"
BUILD="${PEANO_BUILD_DIR:-$XDNA_CACHE/peano-build/$PEANO_FORK_COMMIT}"
VENV_LINK="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie"
ASSET="${PEANO_DIST}-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
UV_ARCHIVE="${UV_CACHE_DIR:-$HOME/.cache/uv}/archive-v0"

seed_ready() {
  [ -x "$SEED/llvm-aie/bin/clang" ] && [ -f "$SEED/.xdna-seed" ] \
    && [ "$(cat "$SEED/.xdna-seed")" = "$PEANO_DIST $PEANO_DIST_SHA256" ]
}

stage_seed_from_archive() {
  local archive stage d
  [ -d "$UV_ARCHIVE" ] || return 1
  archive="$(find "$UV_ARCHIVE" -maxdepth 2 -name "llvm_aie-${PEANO_DIST#llvm_aie-}.dist-info" \
    -print -quit 2>/dev/null | xargs -r dirname)"
  [ -n "$archive" ] && [ -d "$archive/llvm-aie" ] || return 1
  stage="$(mktemp -d "$SEED_ROOT/.${PEANO_DIST}.XXXXXX")"
  cp -a "$archive/llvm-aie" "$stage/"
  for d in "$archive"/llvm_aie-*.dist-info; do
    [ -d "$d" ] && cp -a "$d" "$stage/"
  done
  printf '%s %s\n' "$PEANO_DIST" "$PEANO_DIST_SHA256" > "$stage/.xdna-seed"
  mv "$stage" "$SEED"
}

stage_seed_from_release() {
  local stage wheel
  stage="$(mktemp -d "$SEED_ROOT/.${PEANO_DIST}.XXXXXX")"
  wheel="$stage/$ASSET"
  gh release download nightly --repo Xilinx/llvm-aie --pattern "$ASSET" --output "$wheel" --clobber
  printf '%s  %s\n' "$PEANO_DIST_SHA256" "$wheel" | sha256sum -c -
  python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$wheel" "$stage"
  rm -f "$wheel"
  [ -x "$stage/llvm-aie/bin/clang" ] || {
    echo "[provision_peano] ERROR: $ASSET did not contain llvm-aie/bin/clang" >&2
    exit 1
  }
  printf '%s %s\n' "$PEANO_DIST" "$PEANO_DIST_SHA256" > "$stage/.xdna-seed"
  mv "$stage" "$SEED"
}

mkdir -p "$SEED_ROOT" "$(dirname "$SOURCE")" "$(dirname "$BUILD")"
if ! seed_ready; then
  [ ! -e "$SEED" ] || {
    echo "[provision_peano] ERROR: incomplete seed directory: $SEED" >&2
    exit 1
  }
  if stage_seed_from_archive; then
    echo "[provision_peano] seed restored from $UV_ARCHIVE" >&2
  else
    echo "[provision_peano] fetching $ASSET" >&2
    stage_seed_from_release
  fi
fi

if [ ! -d "$SOURCE/.git" ]; then
  mkdir -p "$SOURCE"
  git -C "$SOURCE" init -q
  git -C "$SOURCE" remote add fork "${PEANO_FORK_URL:-https://github.com/atassis/llvm-aie}"
elif [ -n "$(git -C "$SOURCE" status --porcelain)" ]; then
  echo "[provision_peano] ERROR: pinned Peano source is dirty: $SOURCE" >&2
  exit 1
fi
git -C "$SOURCE" remote get-url fork >/dev/null 2>&1 \
  || git -C "$SOURCE" remote add fork "${PEANO_FORK_URL:-https://github.com/atassis/llvm-aie}"
git -C "$SOURCE" cat-file -e "${PEANO_FORK_COMMIT}^{commit}" 2>/dev/null \
  || git -C "$SOURCE" fetch --depth 1 fork "$PEANO_FORK_COMMIT"
git -C "$SOURCE" checkout --detach "$PEANO_FORK_COMMIT"
[ "$(git -C "$SOURCE" rev-parse HEAD)" = "$PEANO_FORK_COMMIT" ] || {
  echo "[provision_peano] ERROR: source checkout is not $PEANO_FORK_COMMIT" >&2
  exit 1
}

LLVM_AIE_SRC="$SOURCE" BUILD_DIR="$BUILD" bash "$REPO/scripts/build_peano_fast.sh"
bash "$REPO/scripts/install_peano_local.sh" --from "$SEED" --build "$BUILD" --tag "$PEANO_FORK_COMMIT"
INSTALL="$(bash "$REPO/scripts/install_peano_local.sh" --resolve)"
bash "$REPO/scripts/install_peano_local.sh" --activate "$INSTALL"
"$VENV_LINK/bin/clang++" --version | grep -Fq "$PEANO_FORK_COMMIT" || {
  echo "[provision_peano] ERROR: activated compiler does not report $PEANO_FORK_COMMIT" >&2
  exit 1
}
echo "[provision_peano] active fork compiler: $PEANO_FORK_COMMIT" >&2
