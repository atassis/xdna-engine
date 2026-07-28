#!/usr/bin/env bash
#
# install.sh — build & install xdna-engine (ASR + embeddings on the XDNA2 NPU)
#              as a systemd --user service, replacing the FLM-Whisper ASR endpoint.
#
# What this does (idempotent):
#   1. Resolve & sanity-check the repo.
#   2. Preflight: cargo, onnx-asr venv (import onnx_asr), XRT headers/libs.
#   3. Build the Rust workspace (--release) -> rust/target/release/npu.
#  3a. INSTALL that binary to ~/.local/bin/npu  <- the step whose absence used to make
#      every "successful" install a silent no-op against a stale binary.
#   4. Ensure model artifacts exist (generate only if missing), then PREFLIGHT the engine
#      config: every scenario it references must resolve to weights on disk, or we refuse.
#   5. Install the systemd --user unit at ~/.config/systemd/user/xdna-engine.service and
#      retire the superseded npu-asr.service / npu-serve.service.
#   6. daemon-reload + verify the unit.
#   7. Print "Next steps" — does NOT start/stop/enable anything or touch the NPU,
#      because the NPU is single-tenant and may be in use right now.
#
# The service itself: `npu serve` — the engine multitool, one process owning the
#   single-tenant NPU and serving every model in ~/.config/npu/engine.toml. Exposes
#   POST /v1/audio/transcriptions (multipart WAV -> {"text":...}), /v1/embeddings,
#   /v1/models and the /admin/* control plane. It runs the mel preprocessor + decoder/joint
#   ONNX via the system onnxruntime (linked from the onnx-asr venv), and the encoder on
#   the NPU in-process. No Python at runtime.
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
SERVICE="xdna-engine"
UNIT_PATH="$UNIT_DIR/$SERVICE.service"
DESC="xdna-engine -- ASR + embeddings on the XDNA2 NPU, :$PORT"

# Units this supersedes. Both named the same thing badly: npu-asr.service was what this script
# used to write (a bare per-model ASR binary), npu-serve.service was what was actually deployed
# by hand. One service, one name now. They are stopped/disabled on install (see step 5b).
LEGACY_UNITS="npu-asr.service npu-serve.service"

# The engine is driven by the `npu` multitool, not a per-model binary: one process owns the
# single-tenant NPU and serves every configured model (ASR + embeddings today) from a
# desired-state config. The binary keeps its own name; the SERVICE is the product name.
ENGINE_BIN_DIR="${ENGINE_BIN_DIR:-$HOME/.local/bin}"
ENGINE_BIN="$ENGINE_BIN_DIR/npu"
BUILT_BIN="$REPO/rust/target/release/npu"
ENGINE_CONFIG="${ENGINE_CONFIG:-$HOME/.config/npu/engine.toml}"

# Stable runtime dir for libonnxruntime, decoupled from the volatile cargo target/ build tree
# (which `cargo clean` wipes). The unit puts this on LD_LIBRARY_PATH so the service resolves
# its .so here regardless of build-tree churn. Override with STABLE_LIB_DIR=...
STABLE_LIB_DIR="${STABLE_LIB_DIR:-$HOME/.local/lib/xdna-engine}"

# Model selection drives only the ARTIFACT preflight below; which models the service actually
# loads comes from $ENGINE_CONFIG. parakeet (multilingual RU+EN, DEFAULT) | gigaam (RU-only).
MODEL="${MODEL:-parakeet}"
case "$MODEL" in
  parakeet|gigaam) ;;
  *) printf '[fail] unknown MODEL=%s (use parakeet|gigaam)\n' "$MODEL" >&2; exit 1 ;;
esac

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
# Bake an RPATH to the stable onnxruntime dir so the binary resolves libonnxruntime.so.1 on its
# own. Without it only the SERVICE works (its unit sets LD_LIBRARY_PATH) and plain CLI use --
# `npu --help`, `npu models`, `npu transcribe` -- dies with
#   npu: error while loading shared libraries: libonnxruntime.so.1
# NB: setting RUSTFLAGS here OVERRIDES rust/.cargo/config.toml's build.rustflags rather than
# appending to it, so -C target-cpu=native must be repeated or the build silently loses AVX-512.
(
  cd "$REPO/rust"
  XRT_INC_DIR="$XRT_INC_DIR" XRT_LIB_DIR="$XRT_LIB_DIR" \
  RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native -C link-arg=-Wl,-rpath,$STABLE_LIB_DIR" \
    cargo build --release
)
[ -x "$BUILT_BIN" ] || die "Build finished but $BUILT_BIN is missing/not executable."
ok "Built engine binary: $BUILT_BIN"

# ---------------------------------------------------------------------------
# 3a. INSTALL the engine binary
# ---------------------------------------------------------------------------
# This is the step whose absence made every previous "successful" install a no-op: the unit
# runs $ENGINE_BIN, but nothing ever copied the freshly built binary there. A stale binary in
# ~/.local/bin would keep serving while the build reported green.
info "Installing engine binary -> $ENGINE_BIN"
mkdir -p "$ENGINE_BIN_DIR"
if [ -x "$ENGINE_BIN" ] && cmp -s "$BUILT_BIN" "$ENGINE_BIN"; then
  ok "Engine binary already current: $ENGINE_BIN"
else
  # NB: `[ ... ] && info ...` would abort under `set -e` when the test is false (first install).
  if [ -x "$ENGINE_BIN" ]; then
    info "  replacing existing binary (was $(date -r "$ENGINE_BIN" '+%Y-%m-%d %H:%M'))"
  fi
  # install(1) replaces atomically-ish and preserves the mode; a running service keeps its
  # open inode until restarted, so this is safe to do while the old one is serving.
  install -m 0755 "$BUILT_BIN" "$ENGINE_BIN"
  ok "Installed: $ENGINE_BIN ($(date -r "$ENGINE_BIN" '+%Y-%m-%d %H:%M'))"
fi

# ---------------------------------------------------------------------------
# 3b. Stable onnxruntime .so (decouple the runtime from the cargo build tree)
# ---------------------------------------------------------------------------
# asr_serve links libonnxruntime via a RUNPATH into rust/target/.../build/npu-onnx-*/out,
# whose symlinks point into the onnx-asr venv. `cargo clean` wipes that dir -> the service
# can no longer find libonnxruntime.so.1. Copy the real versioned .so into a STABLE dir with
# the SONAME symlink the loader needs, and put that dir on LD_LIBRARY_PATH in the unit. Because
# the binary uses DT_RUNPATH (searched AFTER LD_LIBRARY_PATH), the stable dir wins -> the
# service survives `cargo clean` and any build-tree churn.
info "Hardening onnxruntime .so -> $STABLE_LIB_DIR"
ORT_REAL_SO="$(find "$ONNX_ASR_VENV" -path '*/onnxruntime/capi/libonnxruntime.so.*' 2>/dev/null \
                 | grep -E 'libonnxruntime\.so\.[0-9]' | head -n1 || true)"
[ -n "$ORT_REAL_SO" ] || die "no libonnxruntime.so.* under $ONNX_ASR_VENV (onnx-asr venv layout changed?)."
mkdir -p "$STABLE_LIB_DIR"
cp -f "$ORT_REAL_SO" "$STABLE_LIB_DIR/"
ORT_SO_BASE="$(basename "$ORT_REAL_SO")"                        # e.g. libonnxruntime.so.1.26.0
ln -sf "$ORT_SO_BASE" "$STABLE_LIB_DIR/libonnxruntime.so.1"     # SONAME the loader needs at runtime
ln -sf "$ORT_SO_BASE" "$STABLE_LIB_DIR/libonnxruntime.so"       # link-name (completeness)
ok "Stable .so: $STABLE_LIB_DIR/$ORT_SO_BASE (+ SONAME symlink)"

# ---------------------------------------------------------------------------
# 4. Artifacts
# ---------------------------------------------------------------------------
# A directory "exists with content" check (non-empty).
dir_has_content() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

# ---- Parakeet artifacts (MODEL=parakeet) ----
if [ "$MODEL" = parakeet ]; then
  PK="$REPO/artifacts/parakeet"
  WA="$REPO/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
  dir_has_content "$PK/encoder" \
    || die "Parakeet encoder weights missing: $PK/encoder — run: $EXPORT_VENV/bin/python scripts/extract_parakeet_encoder.py (needs models/parakeet/encoder-model.onnx)."
  for f in preprocessor.onnx decoder_joint.onnx vocab.txt; do
    [ -f "$PK/$f" ] || die "Parakeet artifact missing: $PK/$f (copy nemo128.onnx / decoder_joint-model.onnx / vocab.txt from the cached istupakov repo + onnx-asr)."
  done
  # Accept EITHER the modal build or the plain one, in the runtime's own preference order
  # (npu.rs picks modalsilu > modalid > plain and falls back silently). Checking only for the
  # plain fallback failed a tree that has just the modal xclbins the engine actually loads.
  for n in 1024 2048 4096; do
    have=""
    for v in "_modalsilu" "_modalid" ""; do
      [ -f "$WA/final_512x1024x${n}_64x32x128_8c${v}.xclbin" ] && { have="$v"; break; }
    done
    [ -n "${have+x}" ] && [ -f "$WA/final_512x1024x${n}_64x32x128_8c${have}.xclbin" ] \
      || die "Parakeet NPU xclbin missing: final_512x1024x${n}_64x32x128_8c{_modalsilu,_modalid,}.xclbin — run scripts/build_parakeet_kernels.sh (needs the mlir-aie toolchain)."
  done
  ok "Parakeet artifacts present: $PK (encoder weights + preproc/decoder_joint/vocab) + NPU xclbins"
  # skip the GigaAM artifact generation below
  PARAKEET_DONE=1
fi

ENCODER_DIR="$REPO/artifacts/encoder"
ASR_DIR="$REPO/artifacts/asr"

if [ "$MODEL" = gigaam ]; then
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
fi  # end MODEL=gigaam artifact block

# ---------------------------------------------------------------------------
# 4c. Engine-config preflight — FAIL LOUD, never install a service that cannot serve
# ---------------------------------------------------------------------------
# The service loads $ENGINE_CONFIG, not this script's MODEL variable. A config pointing at a
# missing scenario, or a scenario pointing at missing weights, produces a unit that installs
# clean and then dies (or worse, serves nothing) at runtime. A re-pin silently broke the shipped
# ASR service for five days exactly this way. So: resolve the whole chain here and refuse.
info "Preflighting engine config: $ENGINE_CONFIG"
[ -f "$ENGINE_CONFIG" ] || die "engine config missing: $ENGINE_CONFIG
  Create it (see the [server]/[defaults]/[[model]] example in the README), or point
  ENGINE_CONFIG=... at an existing one."

# Every scenario the config references must exist, parse, and have its weights on disk.
scen_count=0
while IFS= read -r scen; do
  [ -n "$scen" ] || continue
  scen_count=$((scen_count + 1))
  # Relative scenario paths resolve against the repo (WorkingDirectory at runtime).
  case "$scen" in /*) scen_abs="$scen" ;; *) scen_abs="$REPO/$scen" ;; esac
  [ -f "$scen_abs" ] || die "engine config references a missing scenario: $scen_abs
  (from $ENGINE_CONFIG)"

  kind=$(grep -oP '^\s*kind\s*=\s*"\K[^"]+' "$scen_abs" | head -1)
  wdir=$(grep -oP '^\s*weights\s*=\s*"\K[^"]+' "$scen_abs" | head -1)
  [ -n "$kind" ] || die "scenario has no [scenario].kind: $scen_abs"
  if [ -n "$wdir" ]; then
    case "$wdir" in /*) wabs="$wdir" ;; *) wabs="$REPO/$wdir" ;; esac
    [ -d "$wabs" ] && [ -n "$(ls -A "$wabs" 2>/dev/null)" ] \
      || die "scenario '$scen_abs' points at missing/empty weights: $wabs"
  fi
  ok "  scenario OK: $(basename "$scen_abs") (kind=$kind, weights=${wdir:-<none>})"
done < <(grep -oP '^\s*scenario\s*=\s*"\K[^"]+' "$ENGINE_CONFIG")

[ "$scen_count" -gt 0 ] || die "engine config declares no [[model]] scenario: $ENGINE_CONFIG
  The service would start and serve nothing."
ok "Engine config preflight passed ($scen_count model(s) resolvable)."

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
Description=$DESC
# FLM and we both need the single-tenant XDNA2 NPU and both bind :$PORT.
# Conflicts makes starting this unit auto-stop FLM (freeing the NPU and the port).
Conflicts=flm-asr.service
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$REPO
# Resolve libonnxruntime.so.1 from the STABLE dir (not the volatile cargo build tree). The
# binary's DT_RUNPATH is searched AFTER LD_LIBRARY_PATH, so this wins and survives cargo clean.
Environment=LD_LIBRARY_PATH=$STABLE_LIB_DIR
# Pure-Rust single binary: runs onnx preproc/decode (system onnxruntime) + the NPU encoder
# in-process. No Python needed at runtime; cwd resolves artifacts/. (Parakeet: cwd also resolves
# the NPU xclbins under mlir-aie/.../whole_array/build via NpuMatmul root=".".)
ExecStart=$ENGINE_BIN serve --config $ENGINE_CONFIG --port $PORT
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
ok "Wrote unit."

# ---------------------------------------------------------------------------
# 5b. Retire the superseded units
# ---------------------------------------------------------------------------
# npu-asr.service (what this script used to write) and npu-serve.service (what was actually
# deployed by hand) are the same service under two names. Leaving either enabled means two
# units racing for the single-tenant NPU and for :$PORT.
for old in $LEGACY_UNITS; do
  [ "$old" = "$SERVICE.service" ] && continue
  if [ -f "$UNIT_DIR/$old" ] || systemctl --user list-unit-files "$old" >/dev/null 2>&1; then
    if systemctl --user is-active --quiet "$old" 2>/dev/null; then
      info "  stopping superseded unit: $old"
      systemctl --user stop "$old" || true
    fi
    if systemctl --user is-enabled --quiet "$old" 2>/dev/null; then
      info "  disabling superseded unit: $old"
      systemctl --user disable "$old" || true
    fi
    if [ -f "$UNIT_DIR/$old" ]; then
      mv -f "$UNIT_DIR/$old" "$UNIT_DIR/$old.superseded-by-$SERVICE"
      info "  archived $old -> $old.superseded-by-$SERVICE"
    fi
  fi
done
ok "Superseded units retired."

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
 Done. Built + installed the engine and $SERVICE.service (NOT started).
 The NPU was not touched. Superseded units (npu-asr/npu-serve) WERE stopped and
 disabled, since they contend for the same device and port.
============================================================================

 Installed
 ---------
   engine binary : $ENGINE_BIN
   unit          : $UNIT_PATH
   config        : $ENGINE_CONFIG
   onnxruntime   : $STABLE_LIB_DIR

 Next steps
 ----------
 Activate now (stops flm-asr automatically, freeing the NPU + :$PORT;
 voxd keeps running and now transcribes via our backend on :$PORT):

     systemctl --user start $SERVICE.service

 Make it the default at login:

     systemctl --user enable $SERVICE.service
     systemctl --user disable flm-asr.service

 Revert to FLM:

     systemctl --user stop $SERVICE.service
     systemctl --user start flm-asr.service

 Notes
 -----
 * The service is the single owner of the single-tenant NPU. It serves every model
   in $ENGINE_CONFIG (ASR + embeddings); edit that file and
   \`systemctl --user reload-or-restart $SERVICE\` (or POST /admin/reload).
 * voxd needs NO config change: its default endpoint is
   http://127.0.0.1:$PORT/v1/audio/transcriptions, which we now serve.
 * Re-run this script after any \`git pull\`: it rebuilds AND reinstalls the binary.
   Without the reinstall step the service keeps running whatever was in
   $ENGINE_BIN_DIR, however old.
 * Check status / logs:
     systemctl --user status $SERVICE.service
     journalctl --user -u $SERVICE.service -f
============================================================================
EOF
