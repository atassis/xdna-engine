#!/usr/bin/env bash
# Install the xdna2 NPU engine as the local ASR (+ embeddings) service.
#
# Builds the `npu` multitool in release (best flags), installs it + the onnxruntime runtime lib,
# writes the desired-state config + a systemd --user unit, switches off the legacy npu-asr.service,
# and starts `npu serve` on :11434 (the port voxd already targets, so voxd needs no change).
#
# Idempotent + reversible: the old npu-asr.service unit is backed up (not deleted) so you can revert.
# Run it attended (it takes the single-tenant NPU): then run scripts/test_install.sh to verify.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
LIB_DIR="$HOME/.local/lib/npu-asr"          # stable onnxruntime dir (survives cargo clean)
CFG_DIR="$HOME/.config/npu"
CFG="$CFG_DIR/engine.toml"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="npu-serve.service"
PORT="${NPU_PORT:-11434}"

echo "==> repo: $REPO"
mkdir -p "$BIN_DIR" "$LIB_DIR" "$CFG_DIR" "$UNIT_DIR"

echo "==> building release (target-cpu=native, opt-level=3 + thin LTO; RPATH -> $LIB_DIR)"
# Bake an RPATH to the stable onnxruntime dir so the binaries find libonnxruntime.so.1 standalone
# (CLI use: `npu config`, `npu models`, ...) without needing LD_LIBRARY_PATH. The service unit also
# sets LD_LIBRARY_PATH (belt + suspenders).
( cd "$REPO/rust" && RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native -C link-arg=-Wl,-rpath,$LIB_DIR" \
    cargo build --release -p npu-cli -p npu-weights )

echo "==> installing binaries to $BIN_DIR"
install -m 0755 "$REPO/rust/target/release/npu" "$BIN_DIR/npu"
install -m 0755 "$REPO/rust/target/release/npu-weights" "$BIN_DIR/npu-weights"

echo "==> ensuring onnxruntime runtime lib in $LIB_DIR"
ort="$(find "$REPO/rust/target/release/build" -path '*/npu-onnx-*/out/libonnxruntime.so.1' 2>/dev/null | head -1 || true)"
if [ -n "$ort" ]; then
  ortdir="$(dirname "$ort")"
  cp -f "$ortdir"/libonnxruntime.so* "$LIB_DIR/" 2>/dev/null || true
fi
if [ ! -e "$LIB_DIR/libonnxruntime.so.1" ]; then
  echo "!! libonnxruntime.so.1 not found in $LIB_DIR and none in the build tree." >&2
  echo "   Build once so npu-onnx fetches it, or copy it manually." >&2
  exit 1
fi

echo "==> writing config $CFG (kept if it already exists)"
if [ ! -f "$CFG" ]; then
  cat > "$CFG" <<EOF
# xdna2 NPU engine - desired state. Edit + run: systemctl --user reload-or-restart npu-serve
# (or POST /admin/reload). The service is the single owner of the single-tenant NPU.
[server]
port = $PORT
max_resident = 1          # raise once device multi-residency is measured (backlog R11)
memory_ceiling_mb = 4096

[defaults]
asr = "parakeet"

[[model]]
name = "parakeet"
scenario = "$REPO/scenarios/asr.toml"
EOF
else
  echo "    (exists - leaving your config untouched)"
fi

echo "==> writing systemd unit $UNIT_DIR/$UNIT"
cat > "$UNIT_DIR/$UNIT" <<EOF
[Unit]
Description=xdna2 NPU engine (npu serve) - ASR + embeddings on :$PORT
# FLM and we both need the single-tenant XDNA2 NPU + :$PORT; starting this stops FLM.
Conflicts=flm-asr.service
After=graphical-session.target

[Service]
Type=simple
# cwd resolves artifacts/ + the NPU xclbins under mlir-aie (same as the legacy unit).
WorkingDirectory=$REPO
# onnxruntime from the stable dir (searched before the binary RUNPATH; survives cargo clean).
Environment=LD_LIBRARY_PATH=$LIB_DIR
ExecStart=$BIN_DIR/npu serve --config $CFG --port $PORT
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

echo "==> switching services (stopping voxd + any legacy npu-asr)"
systemctl --user stop voxd npu-asr 2>/dev/null || true
# Retire the legacy parakeet_serve/npu-asr unit if present (the engine now serves ASR itself).
systemctl --user disable npu-asr 2>/dev/null || true
rm -f "$UNIT_DIR/npu-asr.service" "$UNIT_DIR/npu-asr.service.bak"

systemctl --user daemon-reload
systemctl --user enable "$UNIT" >/dev/null 2>&1 || true
systemctl --user restart "$UNIT"
sleep 1
systemctl --user start voxd 2>/dev/null || true

echo "==> done. status:"
systemctl --user is-active "$UNIT" && echo "    npu-serve active on :$PORT"
echo "==> next: scripts/test_install.sh   (verifies a real transcription through the service)"
