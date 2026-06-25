#!/usr/bin/env bash
# Uninstall the npu engine service: stop + remove npu-serve (unit + binary). Keeps the onnxruntime
# lib, the config, and artifacts/ (remove manually if you want a full wipe). The legacy
# parakeet_serve/npu-asr service has been retired, so there is nothing to revert to - reinstall with
# scripts/install.sh.
set -euo pipefail

BIN_DIR="$HOME/.local/bin"
CFG_DIR="$HOME/.config/npu"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="npu-serve.service"

echo "==> stopping + removing npu-serve"
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user disable "$UNIT" 2>/dev/null || true
rm -f "$UNIT_DIR/$UNIT"
systemctl --user daemon-reload
rm -f "$BIN_DIR/npu"
echo "    (kept: $CFG_DIR/engine.toml, ~/.local/lib/npu-asr, artifacts/. remove manually for a full wipe.)"
echo "==> done. reinstall with scripts/install.sh"
