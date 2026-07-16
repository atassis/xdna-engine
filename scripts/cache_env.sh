#!/usr/bin/env bash
# cache_env.sh -- single relocatable anchor for the build CACHE (toolchain instances,
# fetched MLIR distro, ccache, worktrees, goldens). Mirrors amd_paths.sh but for
# regenerable BUILD ARTIFACTS, which live INSIDE the workspace (not ~/.cache) so the
# whole tree is self-contained -- nothing surprising left in the system.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/cache_env.sh"
#   ... use "$XDNA_CACHE"/{mlir-distro,instances,ccache,goldens,...}
#
# Override XDNA_CACHE in the env to point the cache elsewhere (e.g. a fast scratch disk).

XDNA_WS="${XDNA_WS:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
export XDNA_WS
export XDNA_CACHE="${XDNA_CACHE:-$XDNA_WS/.cache}"
mkdir -p "$XDNA_CACHE" 2>/dev/null || true
