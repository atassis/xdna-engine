#!/usr/bin/env bash
# cache_env.sh -- back-compat name for the build CACHE (toolchain instances, fetched
# MLIR distro, ccache, worktrees, goldens). XDNA_CACHE now derives from XDNA_DATA (see
# scripts/lib/data_root.sh, the single resolver for every generated-data path); this
# file stays around because callers already `source scripts/cache_env.sh` for just it.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/cache_env.sh"
#   ... use "$XDNA_CACHE"/{mlir-distro,instances,ccache,goldens,...}
#
# Point it elsewhere via config/local.env's XDNA_DATA (shared by every generated-data
# path) or XDNA_CACHE directly (this one only). See data_root.sh for the worktree
# fallback that lets a linked worktree inherit the main checkout's root.
. "$(dirname "${BASH_SOURCE[0]:-$0}")/lib/data_root.sh"
mkdir -p "$XDNA_CACHE" 2>/dev/null || true
