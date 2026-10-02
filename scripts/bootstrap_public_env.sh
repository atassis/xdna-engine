#!/usr/bin/env bash
# Bootstrap the declared public source and Python environments.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

git submodule update --init --recursive
. "$REPO/scripts/amd_paths.sh"
iron_require_source
iron_require_fused_attn

scripts/setup_kernel_env.sh
scripts/fetch_mlir_distro.sh
scripts/toolchain_up.sh
scripts/setup_export_venv.sh
