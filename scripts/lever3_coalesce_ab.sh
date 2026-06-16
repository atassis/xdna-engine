#!/usr/bin/env bash
# =============================================================================================
# LEVER #3 — coalesced-dispatch A/B  (run with ALL other NPU sessions stopped; single-tenant box)
#
#   bash scripts/lever3_coalesce_ab.sh
#
# Tests whether collapsing the per-head V transposes (612 -> 336 micro-launches/token) cuts the
# fused-decode NPU dispatch — i.e. confirms the decode is launch-OVERHEAD-bound, not bandwidth-bound
# ([[lever3-dispatch-overhead-bound]]). POST-DEEP-C (new mlir-aie 1.3.2 stack, constant resident ELF):
# baseline moved to ~75 ms/tok (~CPU floor), so the single-stream win is now SMALL — this run re-baselines
# and quantifies it (and any J/token benefit). Fully unattended: builds, runs, ALWAYS restarts
# npu-asr/voxd on exit, beeps when done. Everything -> artifacts/lever3_ab_<ts>.log.
#
# WHAT IT MEASURES (3 backends; all use the NPU encoder, decode differs):
#   onnx        — CPU decode (reference, ~74 ms/tok)
#   fused       — deep-C resident fused decode, BASELINE artifacts/fused_decode12       (612 launches/tok)
#   coalesced   — deep-C resident + lever-3 (i)+(1), artifacts/fused_decode12_coalesced (336 launches/tok)
# Both ELFs are resident-scratchpad format (new stack) → identical engine path; only the transposes differ.
# For each: per-phase decode breakdown (FUSED_PHASE: dispatch / load_elf / lm_head / patch / io),
# e2e + ms/token (WHISPER_TIMING), RAPL pkg energy J/transcription (WHISPER_ENERGY), pooled 17-clip WER.
#
# CORRECTNESS GATE (the load-bearing check): coalesced WER must == baseline (0.1172) AND the en_01
# transcription text must be IDENTICAL baseline-vs-coalesced (argmax-identity => WER-safe, the canonical
# gate [[decode-fused-elf-wer-safe]]). A finer rel-L2 via verify_fused_decode.py is NOT run here — it
# rebuilds its own runlist and would need syncing to the coalesced structure (follow-up if WER disagrees).
#
# NOTE: FUSED_PHASE cannot split WITHIN the single 12-layer 'dispatch' (one opaque NPU run); the A/B
# measures that dispatch as a whole. Per-launch attribution needs AIE hardware trace (out of scope).
#
# RAPL: if energy shows N/A, run ONCE first:  sudo chmod -R a+r /sys/class/powercap/intel-rapl*/
# =============================================================================================
set -u

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WT"
IRON="${IRON:-$HOME/repositories/ns/amd/IRON}"
TPATCH="$WT/route_b_kernels/patches/iron-transpose-num-batches.patch"  # lever-3 transpose num_batches
LDLIB=~/.local/lib/npu-asr
W3="$WT/rust/target/release/whisper_e2e_timing"
SERVE="$WT/rust/target/release/engine_serve"
SCEN="$WT/scenarios/asr-whisper-small.toml"
CLIP="$WT/artifacts/wer_clips/en_01.wav"
WEIGHTS="$WT/artifacts/whisper-small/whisper_decoder"
BASE_DIR="$WT/artifacts/fused_decode12"
COAL_DIR="$WT/artifacts/fused_decode12_coalesced"
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$WT/artifacts/lever3_ab_${TS}.log"
WERDIR="$WT/artifacts/lever3_wer_${TS}"
PORT=11434
URL="http://127.0.0.1:${PORT}/v1/audio/transcriptions"
mkdir -p "$WT/artifacts" "$WERDIR"; : > "$LOG"

log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; echo "[svc] npu services restarted" | tee -a "$LOG"; }
beep(){ ( speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 & local p=$!; sleep 2; kill -9 "$p" >/dev/null 2>&1 ); }
trap 'restart; beep; echo "[done] log: $LOG"' EXIT

log "================ LEVER-3 COALESCE A/B  $TS ================"
log "host: $(uname -srm)"
log "uptime:$(uptime)"
log "-- top CPU procs (should be near-idle) --"
ps -eo comm,pcpu,pmem --sort=-pcpu 2>/dev/null | head -6 | tee -a "$LOG"

# ---- RAPL readability ----
if cat "$RAPL" >/dev/null 2>&1; then
  log "[rapl] energy_uj readable — energy will be measured"
else
  sudo -n chmod -R a+r /sys/class/powercap/intel-rapl*/ 2>/dev/null || true
  cat "$RAPL" >/dev/null 2>&1 && log "[rapl] readable after sudo -n chmod" \
    || log "[rapl] NOT readable — ENERGY = N/A. Enable once: sudo chmod -R a+r /sys/class/powercap/intel-rapl*/"
fi

# =========================================================================================
# STEP 0 — build the coalesced ELF on the new 1.3.2 stack via build_deepc_decode.sh
#          (compile-only; done BEFORE the timed section so build heat doesn't bias energy).
#          Baseline = deep-C's resident ELF (artifacts/fused_decode12, already built). Both ELFs are
#          resident-scratchpad format → run through the same deep-C whisper_decoder.rs path; the only
#          difference is the coalesced transposes (gen_decode.py has both deep-C params + lever-3 (i)+(1)).
# =========================================================================================
log "\n################  STEP 0 — build coalesced ELF (new 1.3.2 stack)  ################"
if [ ! -f "$BASE_DIR/decode.elf" ]; then
  log "[ERR] baseline ELF missing: $BASE_DIR/decode.elf (deep-C resident baseline — build via scripts/build_deepc_decode.sh 12). Aborting."; exit 1
fi
# Ensure the lever-3 transpose num_batches change is in the shared IRON (orthogonal to amd-IRON-deepc.patch,
# which build_deepc_decode.sh applies). Idempotent: skip if already present.
if git -C "$IRON" apply --reverse --check "$TPATCH" >/dev/null 2>&1; then
  log "[iron] lever-3 transpose num_batches already applied"
elif git -C "$IRON" apply --check "$TPATCH" >/dev/null 2>&1; then
  git -C "$IRON" apply "$TPATCH" && log "[iron] applied lever-3 transpose num_batches patch"
else
  log "[ERR] lever-3 transpose patch neither applies nor is already present in $IRON. Resolve manually."; exit 1
fi
log "[build] building coalesced 12-layer ELF -> $COAL_DIR via build_deepc_decode.sh (compile-only) ..."
if bash "$WT/scripts/build_deepc_decode.sh" 12 "$COAL_DIR" >>"$LOG" 2>&1; then
  log "[build] coalesced ELF OK: $(ls -la "$COAL_DIR/decode.elf" | awk '{print $5" bytes"}')"
else
  log "[ERR] coalesced ELF build FAILED — see log above. Aborting."; exit 1
fi

# ---- rust binaries (MUST rebuild — whisper.rs adds NPU_DECODE_FUSED_DIR override) ----
log "\n[build] building release binaries in this worktree (first build can take several minutes) ..."
if ( cd "$WT/rust" && cargo build -p npu-engine --release --bin whisper_e2e_timing --bin engine_serve ) >>"$LOG" 2>&1; then
  log "[build] binaries OK"
else
  log "[ERR] cargo build FAILED — see log above. Aborting."; exit 1
fi

# ---- preflight: encoder xclbin must resolve (the engine loads it via ./mlir-aie relative to CWD). ----
# In a fresh worktree, mlir-aie is a real (artifact-less) submodule checkout — it MUST be a symlink to a
# checkout that has the BUILT xclbins, else every backend panics in ctx2.rs (encoder), not the decode.
ENC_XCLBIN="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/final_512x800x3072_64x32x96_8c_modalsilu.xclbin"
if [ ! -f "$ENC_XCLBIN" ]; then
  log "[ERR] encoder xclbin missing: $ENC_XCLBIN"
  log "      Worktree mlir-aie must symlink a checkout WITH built xclbins:"
  log "        rm -rf mlir-aie && ln -s <main-repo>/mlir-aie mlir-aie"
  exit 1
fi
log "[preflight] encoder xclbin present"
log "[cooldown] 15s so the build heat doesn't bias the idle energy numbers ..."; sleep 15

# =========================================================================================
# single-tenant
# =========================================================================================
log "\n[svc] stopping npu-asr + voxd for single-tenant NPU ..."
systemctl --user stop npu-asr.service voxd.service; sleep 2
if fuser /dev/accel/accel0 2>/dev/null; then
  log "[ERR] /dev/accel/accel0 still busy after stopping services — aborting (other session still running?)."; exit 1
fi
log "[svc] device clear — single-tenant"

# =========================================================================================
# STEP 1 — E2E latency + per-phase breakdown + RAPL energy  (en_01, warmup + 3 passes)
# =========================================================================================
log "\n################  STEP 1 — TIMING / PER-PHASE / ENERGY (en_01)  ################"
run_e2e(){  # $1 = label ; $2.. = extra env KV pairs
  local label="$1"; shift
  log "\n----------------- [$label] -----------------"
  env WHISPER_TIMING=1 FUSED_PHASE_TIMING=1 "$@" LD_LIBRARY_PATH="$LDLIB" "$W3" "$CLIP" 2>&1 \
    | grep -E "FUSED_PHASE\] (steps|per-token)|  [a-z_]+ +[0-9]|WHISPER_TIMING|WHISPER_ENERGY|warmup text|fused decode ELF dir" \
    | tee -a "$LOG"
}
run_e2e onnx
run_e2e fused      NPU_DECODE_FUSED=1
run_e2e coalesced  NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="$COAL_DIR"

# =========================================================================================
# STEP 2 — correctness: pooled WER + en_01 transcription-text identity (baseline vs coalesced)
# =========================================================================================
log "\n################  STEP 2 — POOLED WER (17 clips) + TEXT-IDENTITY GATE  ################"
run_wer(){  # $1 = label ; $2.. = extra env KV pairs
  local label="$1"; shift
  local serve_log="$WERDIR/serve_${label}.log"
  log "\n----------------- WER [$label] -----------------"
  env LD_LIBRARY_PATH="$LDLIB" "$@" "$SERVE" "$SCEN" "$PORT" >"$serve_log" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || { log "[wer:$label] serve died early:"; tail -15 "$serve_log" | tee -a "$LOG"; return 1; }
    sleep 1
  done
  [ "$up" = 1 ] || { log "[wer:$label] serve never healthy"; kill "$pid" 2>/dev/null; return 1; }
  python3 "$WT/scripts/whisper_decode_wer.py" --url "$URL" --label "$label" --out "$WERDIR/wer_${label}.json" 2>&1 | tee -a "$LOG"
  # capture the en_01 transcription for the text-identity gate
  curl -sf -F "file=@${CLIP}" -F "response_format=json" "$URL" 2>/dev/null > "$WERDIR/text_${label}.json" || true
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 2
}
run_wer onnx
run_wer fused      NPU_DECODE_FUSED=1
run_wer coalesced  NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="$COAL_DIR"

log "\n-- text-identity gate (baseline-fused vs coalesced, en_01) --"
TF="$WERDIR/text_fused.json"; TC="$WERDIR/text_coalesced.json"
if [ -s "$TF" ] && [ -s "$TC" ]; then
  if diff <(python3 -c "import json,sys;print(json.load(open('$TF')).get('text','').strip())") \
          <(python3 -c "import json,sys;print(json.load(open('$TC')).get('text','').strip())") >/dev/null 2>&1; then
    log "  PASS — coalesced transcription is IDENTICAL to baseline (argmax-identical => WER-safe)"
  else
    log "  ⚠ DIFFER — coalesced text != baseline. Coalescing changed the output; inspect:"
    log "    baseline:  $(python3 -c "import json;print(json.load(open('$TF')).get('text',''))" 2>/dev/null)"
    log "    coalesced: $(python3 -c "import json;print(json.load(open('$TC')).get('text',''))" 2>/dev/null)"
  fi
else
  log "  (text capture missing — rely on pooled WER below)"
fi

# =========================================================================================
# SUMMARY
# =========================================================================================
log "\n################  SUMMARY — does coalescing cut the dispatch?  ################"
SNAP="$WERDIR/_snapshot.txt"; cp "$LOG" "$SNAP"
log "-- per-backend e2e (encoder / decode / ms-per-token / dispatches) --"
grep -E "WHISPER_TIMING" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"
log "-- per-backend energy (RAPL pkg J/transcription, avg W, RTF) --"
grep -E "WHISPER_ENERGY" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"
log "-- fused per-phase breakdown (mean ms/token; KEY ROW = 'dispatch') --"
grep -E "FUSED_PHASE\] steps" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"
log "-- pooled WER (onnx ~0.1136; fused baseline 0.1172; coalesced MUST match 0.1172) --"
grep -E "ALL pooled WER|=== label=" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"
log "\n[interpret] launch count 612 (fused) -> 336 (coalesced), -45%. If 'dispatch' ms drops with it,"
log "            the 56.9 ms is OVERHEAD-bound (lever-3 thesis confirmed) and trending toward CPU's 74.1 ms/tok."
log "[done] full log: $LOG"
