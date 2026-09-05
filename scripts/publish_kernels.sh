#!/usr/bin/env bash
# Publish the built kernels into ONE install-owned directory, by COPY.
#
# Why this exists. A production install should hold the artifacts a running engine needs -- config,
# kernels, weights, binaries -- and not a compiler tree. Today `install.sh` symlinks the whole
# mlir-aie checkout into the install and the runtime resolves xclbins out of
# `programming_examples/*/build`, so the install depends on a developer's working tree existing at a
# fixed path, and a shipped service reads a dev-tree `toolchain.lock` for its freshness gate.
#
# One install-owned ROOT with one subdirectory per kernel family, not one flat directory. Flat was
# the first attempt and it does not work: the runtime loads BARE `final.xclbin` / `insts.bin` from
# dwconv1d (`engines.rs:453-454`) and from parakeet (`npu.rs:997,1060`), distinguishing them by
# DIRECTORY rather than by name -- and layernorm publishes files of exactly those names too. The
# collision is real and it appeared within the hour, as soon as layernorm was rebuilt.
#
# The point of the move is served either way: the install stops containing a compiler tree, stops
# depending on a developer checkout at a fixed path, and stops sharing a sandbox that a re-pin
# purges. Making everything stem-addressed so a single flat directory works is a follow-up that has
# to change those runtime call sites, which is a bigger change than moving files.
#
# The collision check stays. Within a family it cannot fire today, which is the point: it is there
# for the next family added, and a silent overwrite in a published set is indistinguishable from a
# complete one.
#
#   publish_kernels.sh <dest-dir> [<mlir-aie-root>]
#
# The source root is an argument because `install.sh` exposes ENGINE_MLIR_AIE as a knob; a publisher
# that hardcoded $REPO/mlir-aie would silently ignore an operator who set it.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:?usage: publish_kernels.sh <dest-dir> [<mlir-aie-root>]}"
MLIR_AIE_ROOT="${2:-$REPO/mlir-aie}"
[ -d "$MLIR_AIE_ROOT" ] || { echo "[publish-kernels] ERROR: no mlir-aie root at '$MLIR_AIE_ROOT'" >&2; exit 1; }
PE="$MLIR_AIE_ROOT/programming_examples"

# The dirs a SERVING engine resolves from. Every one must carry a .toolchain-stamp: publishing an
# artifact whose pin is unknown is how a re-pin goes unnoticed, which cost five days once.
SRC_DIRS=(
  "$PE/basic/matrix_multiplication/whole_array/build"
  "$PE/ml/dwconv1d/build"
  "$PE/ml/layernorm/build"
)
# NOT published, and named here rather than silently absent:
#   ml/mha_decode/build -- carries no .toolchain-stamp and is not covered by
#     check_kernel_artifact_freshness.sh either. It is reached only from ctx_decode.rs, the opt-in
#     NPU_DECODE per-op backend that NPU_DECODE_FUSED takes precedence over, so nothing the service
#     selects by default needs it. Add it here once it is stamped; do not add it unstamped.

fail=0
note() { printf '[publish-kernels] %s\n' "$*" >&2; }
err()  { printf '[publish-kernels] ERROR: %s\n' "$*" >&2; fail=1; }

# One pin, or none. All four directories are built from the same toolchain; if they disagree, the
# published set would mix vintages and the single stamp below would name only one of them -- exactly
# the silent-staleness failure the stamp exists to prevent.
stamp=""; stamp_src=""
for d in "${SRC_DIRS[@]}"; do
  [ -d "$d" ] || { note "skip (absent): ${d#$REPO/}"; continue; }
  s="$(cat "$d/.toolchain-stamp" 2>/dev/null || true)"
  [ -n "$s" ] || { err "${d#$REPO/} has no .toolchain-stamp -- refusing to publish an artifact whose pin is unknown"; continue; }
  if [ -z "$stamp" ]; then stamp="$s"; stamp_src="$d"
  elif [ "$s" != "$stamp" ]; then
    err "pin mismatch: ${stamp_src#$REPO/} is $stamp but ${d#$REPO/} is $s -- rebuild both before publishing"
  fi
done
[ "$fail" -eq 0 ] || { note "refusing to publish"; exit 1; }
[ -n "$stamp" ] || { err "no source directory had anything to publish"; exit 1; }

mkdir -p "$DEST"
# Collision check BEFORE copying anything, so a refusal leaves the destination untouched rather than
# half-written -- a partially published set reads as a complete one.
declare -A owner=()
for d in "${SRC_DIRS[@]}"; do
  [ -d "$d" ] || continue
  fam="$(basename "$(dirname "$d")")"          # .../<family>/build -> <family>
  while IFS= read -r f; do
    key="$fam/$(basename "$f")"
    if [ -n "${owner[$key]:-}" ]; then
      err "name collision: $key is in both ${owner[$key]#$REPO/} and ${d#$REPO/}"
    else
      owner[$key]="$d"
    fi
  done < <(find "$d" -maxdepth 1 -type f \( -name '*.xclbin' -o -name 'insts*.txt' -o -name 'insts*.bin' \))
done
[ "$fail" -eq 0 ] || { note "refusing to publish"; exit 1; }

n=0
for key in "${!owner[@]}"; do
  mkdir -p "$DEST/$(dirname "$key")"
  cp -f "${owner[$key]}/$(basename "$key")" "$DEST/$key" || err "copy failed: $key"
  n=$((n+1))
done
printf '%s' "$stamp" > "$DEST/.toolchain-stamp"
[ "$fail" -eq 0 ] || exit 1
note "published $n file(s) -> $DEST  (pin $stamp)"
