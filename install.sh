#!/usr/bin/env bash
#
# install.sh — build & install the open GigaAM-v3 NPU ASR backend (Route B encoder)
#              as a systemd --user service, replacing the FLM-Whisper ASR endpoint.
#
# What this does (idempotent):
#   1. Resolve & sanity-check the repo.
#   2. Preflight: cargo, onnx-asr venv (import onnx_asr), XRT headers/libs.
#   3. Build the Rust workspace (--release) -> rust/target/release/encode_server.
#   4. Ensure encoder/asr artifacts exist (generate only if missing).
#   5. Install the systemd --user unit at ~/.config/systemd/user/npu-asr.service,
#      with absolute REPO/venv paths substituted in.
#   6. daemon-reload + verify the unit.
#   7. Print "Next steps" — does NOT start/stop/enable anything or touch the NPU,
#      because the NPU is single-tenant and may be in use right now.
#
# The service itself: rust/target/release/asr_serve — a single self-contained Rust
#   binary (no Python at runtime) exposing POST /v1/audio/transcriptions
#   (multipart WAV -> {"text":...}). It runs the mel preprocessor + RNNT decoder/joint
#   ONNX via the system onnxruntime (linked from the onnx-asr venv), and the encoder on
#   the NPU in-process. (A Python equivalent, scripts/asr_service.py, also exists.)
#
# It listens on :11434 — the SAME port FLM serves on. Both FLM and we need the
# single-tenant XDNA2 NPU, so they are mutually exclusive anyway; reusing the port
# means the voxd dictation client (default endpoint http://127.0.0.1:11434/...)
# needs NO config change.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Configuration / overridable env
# ---------------------------------------------------------------------------

# REPO = directory containing this script (resolve symlinks).
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(dirname "$SCRIPT_PATH")"

# onnx-asr runtime venv (has onnx_asr, onnxruntime, numpy, py3.12).
# Used to RUN the service and to generate the artifacts/asr/ ONNX models.
ONNX_ASR_VENV="${ONNX_ASR_VENV:-$HOME/npuvox-asr-bench/.venv}"

# Repo export venv (has onnx + onnxruntime). Used to (re)generate the
# artifacts/encoder/ encoder weights via extract_encoder.py.
EXPORT_VENV="${EXPORT_VENV:-$REPO/.venv}"

# XRT build/runtime environment (XDNA driver). Needed by the cargo build and at runtime.
XRT_INC_DIR="${XRT_INC_DIR:-/usr/include}"
XRT_LIB_DIR="${XRT_LIB_DIR:-/usr/lib}"

# Fixed listen port — intentionally the same one FLM uses (see header).
PORT=11434

UNIT_DIR="$HOME/.config/systemd/user"
UNIT_PATH="$UNIT_DIR/npu-asr.service"

# Pretty logging helpers.
info() { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Resolve & sanity-check the repo
# ---------------------------------------------------------------------------
info "Repo:            $REPO"
[ -f "$REPO/scripts/asr_service.py" ] || die "scripts/asr_service.py not found — is \$REPO ($REPO) really the asr-engine repo?"
[ -f "$REPO/rust/Cargo.toml" ]       || die "rust/Cargo.toml not found — is \$REPO ($REPO) really the asr-engine repo?"
ok "Repo layout looks correct."

# ---------------------------------------------------------------------------
# 2. Preflight checks (clear errors, no side effects)
# ---------------------------------------------------------------------------
info "Preflight checks..."

# 2a. cargo
command -v cargo >/dev/null 2>&1 || die "cargo not found on PATH. Install Rust (https://rustup.rs)."
ok "cargo: $(command -v cargo)"

# 2b. onnx-asr venv + onnx_asr importable
ONNX_ASR_PY="$ONNX_ASR_VENV/bin/python"
[ -x "$ONNX_ASR_PY" ] || die "onnx-asr venv python not found at $ONNX_ASR_PY (override with ONNX_ASR_VENV=...)."
"$ONNX_ASR_PY" -c "import onnx_asr" 2>/dev/null \
  || die "'import onnx_asr' failed in $ONNX_ASR_VENV — venv is missing onnx_asr."
ok "onnx-asr venv: $ONNX_ASR_VENV (import onnx_asr OK)"

# 2c. XRT headers + libs
[ -f "$XRT_INC_DIR/xrt/xrt_bo.h" ] || [ -f "$XRT_INC_DIR/xrt.h" ] \
  || die "XRT headers not found under $XRT_INC_DIR (expected xrt/xrt_bo.h). Override with XRT_INC_DIR=..."
ls "$XRT_LIB_DIR"/libxrt_coreutil.so* >/dev/null 2>&1 \
  || die "XRT libs not found under $XRT_LIB_DIR (expected libxrt_coreutil.so*). Override with XRT_LIB_DIR=..."
ok "XRT: inc=$XRT_INC_DIR lib=$XRT_LIB_DIR"

# Export venv is only needed if we have to (re)generate encoder artifacts;
# check it lazily below so a missing repo .venv doesn't block the common path.

# ---------------------------------------------------------------------------
# 3. Build the Rust workspace
# ---------------------------------------------------------------------------
info "Building Rust workspace (cargo build --release)..."
(
  cd "$REPO/rust"
  XRT_INC_DIR="$XRT_INC_DIR" XRT_LIB_DIR="$XRT_LIB_DIR" cargo build --release
)
ASR_SERVE="$REPO/rust/target/release/asr_serve"
[ -x "$ASR_SERVE" ] || die "Build finished but $ASR_SERVE is missing/not executable."
ok "Built asr_serve: $ASR_SERVE"

# ---------------------------------------------------------------------------
# 4. Artifacts
# ---------------------------------------------------------------------------
# A directory "exists with content" check (non-empty).
dir_has_content() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

ENCODER_DIR="$REPO/artifacts/encoder"
ASR_DIR="$REPO/artifacts/asr"

# 4a. Encoder weights (artifacts/encoder/) — generated by extract_encoder.py with EXPORT_VENV.
#     NOTE: extract_encoder.py may need the NPU free. If the dir is missing we do NOT
#     stop services ourselves; we print an instruction and exit so we don't disrupt
#     whatever is currently holding the single-tenant NPU.
if dir_has_content "$ENCODER_DIR"; then
  ok "Encoder artifacts present: $ENCODER_DIR (skipping extract_encoder.py)"
else
  cat >&2 <<EOF
[fail] Encoder artifacts missing/empty: $ENCODER_DIR

  These are generated by scripts/extract_encoder.py, which may require the
  single-tenant XDNA2 NPU to be FREE. This installer will NOT stop any running
  service or touch the NPU on its own.

  To generate them manually (ensure the NPU is free first), run:

    cd "$REPO" && \\
    XRT_INC_DIR="$XRT_INC_DIR" XRT_LIB_DIR="$XRT_LIB_DIR" \\
    "$EXPORT_VENV/bin/python" scripts/extract_encoder.py

  Then re-run install.sh.
EOF
  exit 1
fi

# 4b. ASR models (artifacts/asr/: preprocessor/decoder/joint ONNX + vocab) — via asr_oracle.py
#     with the onnx-asr venv. This is pure ONNX export (no NPU), safe to run here.
if dir_has_content "$ASR_DIR"; then
  ok "ASR artifacts present: $ASR_DIR (skipping asr_oracle.py)"
else
  info "ASR artifacts missing — generating via scripts/asr_oracle.py (onnx-asr venv)..."
  (
    cd "$REPO"
    "$ONNX_ASR_PY" scripts/asr_oracle.py
  )
  dir_has_content "$ASR_DIR" || die "asr_oracle.py ran but $ASR_DIR is still empty."
  ok "Generated ASR artifacts: $ASR_DIR"
fi

# ---------------------------------------------------------------------------
# 5. Install the systemd --user unit
# ---------------------------------------------------------------------------
info "Installing systemd --user unit -> $UNIT_PATH"
mkdir -p "$UNIT_DIR"

# Absolute paths are baked in at install time (no $-expansion at runtime).
#   Conflicts=flm-asr.service  -> starting ours auto-stops FLM, freeing NPU + :11434.
#   WorkingDirectory=$REPO     -> asr_service.py spawns encode_server via the relative
#                                 path rust/target/release/encode_server, so cwd matters.
cat > "$UNIT_PATH" <<EOF
[Unit]
Description=NPU GigaAM-v3 ASR (open Route B encoder) on :$PORT
# FLM and we both need the single-tenant XDNA2 NPU and both bind :$PORT.
# Conflicts makes starting this unit auto-stop FLM (freeing the NPU and the port).
Conflicts=flm-asr.service
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$REPO
# Pure-Rust single binary: runs onnx preproc/decode (system onnxruntime, baked rpath) + the
# NPU encoder in-process. No Python or env needed at runtime; cwd resolves artifacts/.
ExecStart=$ASR_SERVE $PORT
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
ok "Wrote unit."

# ---------------------------------------------------------------------------
# 6. daemon-reload + verify
# ---------------------------------------------------------------------------
info "systemctl --user daemon-reload"
systemctl --user daemon-reload

info "Verifying unit (warnings are acceptable)..."
# systemd-analyze verify exits non-zero on hard errors; warnings about
# WorkingDirectory/After are fine here. Don't let a warning abort the script.
if systemd-analyze --user verify "$UNIT_PATH"; then
  ok "Unit verified clean."
else
  info "systemd-analyze reported warnings (see above) — these are acceptable."
fi

# ---------------------------------------------------------------------------
# 7. Next steps (we deliberately do NOT start/stop/enable anything)
# ---------------------------------------------------------------------------
cat <<EOF

============================================================================
 Done. Built encode_server + installed npu-asr.service (NOT started).
 Nothing touched the NPU or any running service.
============================================================================

 Next steps
 ----------
 Activate now (stops flm-asr automatically, frees the NPU + :$PORT;
 voxd keeps running and now transcribes via our backend on :$PORT):

     systemctl --user start npu-asr.service

 Make it the default at login:

     systemctl --user enable npu-asr.service
     systemctl --user disable flm-asr.service

 Revert to FLM:

     systemctl --user stop npu-asr.service
     systemctl --user start flm-asr.service

 Notes
 -----
 * voxd needs NO config change: its default endpoint is
   http://127.0.0.1:$PORT/v1/audio/transcriptions, which we now serve.
 * FLM also served an LLM (qwen); this backend is ASR-only. If voxd relies
   on that LLM (e.g. for post-processing), that capability is not provided here.
 * Check status / logs:
     systemctl --user status npu-asr.service
     journalctl --user -u npu-asr.service -f
============================================================================
EOF
