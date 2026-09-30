#!/usr/bin/env bash
# verify_installed_artifacts.sh <engine_root> -- recompute every staged artifact's identity
# and compare against $engine_root/artifacts.manifest.json (written by install.sh's
# stage_served_artifacts). Exit 0 = every artifact still matches what was staged; non-zero
# names each mismatch (source changed, or the symlink is gone). Run any time to catch a
# build that rewrote an artifact IN PLACE under $ENGINE_ARTIFACTS after install.
set -euo pipefail
ENGINE_ROOT="${1:?usage: verify_installed_artifacts.sh <engine_root>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ENGINE_ROOT/artifacts.manifest.json"
[ -f "$MANIFEST" ] || { echo "[verify] no manifest at $MANIFEST -- nothing to check" >&2; exit 1; }

PY="${VERIFY_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python)"
exec "$PY" "$REPO/scripts/lib/artifact_manifest.py" verify "$ENGINE_ROOT" "$MANIFEST"
