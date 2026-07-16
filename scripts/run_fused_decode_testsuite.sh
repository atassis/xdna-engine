#!/usr/bin/env bash
# =============================================================================
# Comprehensive fused-decode + full-ASR-e2e test suite.
# RUN ON AN IDLE MACHINE (single-tenant NPU; the script stops/starts npu-asr/voxd itself).
#
#   bash scripts/run_fused_decode_testsuite.sh
#
# Logs EVERY step to a fixed file:  artifacts/fused_testsuite.log  (printed at the end).
# Covers, with known-fact comparisons:
#   0. Known KB facts (baseline numbers).
#   1. Fused-decode CORRECTNESS — every block + whole 12-layer (rel-L2) + argmax parity (WER-safety).
#   2. Fused-decode PER-TOKEN timing (1 decode step).
#   3. Whisper FULL-TRANSCRIPTION e2e (whisper_e2e_timing): ONNX vs NPU-step1 vs NPU+on-chip-attn
#      — per-stage breakdown (encoder/decode/#tokens/ms-per-token) on en + ru clips.
#   4. Whisper decode argmax PARITY vs ONNX (verify_whisper_decode host/npu/npu-attn).
#   5. Whisper WER (full transcription accuracy, 17 clips): ONNX vs NPU pooled WER vs known 0.1136.
#   6. Other ASR / models (best-effort): Parakeet WER, embeddings, ESM latency.
#   7. Projected fused FULL-TRANSCRIPTION latency (encoder + #tokens x measured per-token).
#
# "token" = ONE decode step (one BPE word-piece). A full transcription = encoder once + N tokens.
# A beep sounds when the whole suite finishes.
# =============================================================================
set -u
. "$(dirname "${BASH_SOURCE[0]}")/amd_paths.sh"   # -> IRON_DIR (relocatable; env-overridable)
IRON="$IRON_DIR"
# WT = repo root (derived from this script's location), so the suite runs from any worktree —
# including main, which (unlike the fresh fused worktree) HAS the built encoder xclbins, so the
# full-transcription e2e + WER sections actually run instead of panicking on a missing xclbin.
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LDLIB=~/.local/lib/npu-asr
WDIR=$WT/artifacts/whisper-small/whisper_decoder
ENC=$WT/artifacts/whisper-small/refs/encoded.npy
GEN=$WT/route_b_kernels/decode_fused
PROBE=$WT/rust/target/debug/fused_elf_probe
CLIP_EN=$WT/artifacts/wer_clips/en_01.wav
CLIP_RU=$WT/artifacts/wer_clips/ru_01.wav
LOG=$WT/artifacts/fused_testsuite.log
mkdir -p "$WT/artifacts"; : > "$LOG"

section(){ echo -e "\n\n========================================================================\n## $*\n========================================================================" | tee -a "$LOG"; }
run(){ echo -e "\n\$ $*" | tee -a "$LOG"; { eval "$@"; } >>"$LOG" 2>&1; local rc=$?; tail -n 40 "$LOG" | sed 's/^/    /'; echo "  [exit $rc]" | tee -a "$LOG"; }
note(){ echo -e "# $*" | tee -a "$LOG"; }

restart_svc(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; }
beep(){ ( speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 & local p=$!; sleep 3; kill -9 $p >/dev/null 2>&1 ) ; }
trap 'restart_svc; echo "[services restarted on exit]" | tee -a "$LOG"; beep' EXIT

{ echo "FUSED-DECODE + ASR-E2E TEST SUITE"; date; echo "host: $(uname -srm)"; echo "log: $LOG"; } | tee -a "$LOG"

# ---------------------------------------------------------------------------
section "0. KNOWN KB FACTS (baseline to compare against)"
for s in whisper-npu-decode-e2e whisper-npu-decode-dispatches whisper-npu-decode-wer per-dispatch-floor \
         decode-attn-block-dispatch-latency decode-attn-overhead-bound energy-offload context-switch-cost \
         asr-serve-e2e-parakeet embeddings-e2e-latency esm-native-latency \
         decode-fused-elf-self-attn decode-fused-elf-cross-attn decode-fused-elf-whole-decode decode-fused-elf-wer-safe; do
  echo -e "\n--- $s ---" | tee -a "$LOG"; "$WT/scripts/kb.sh" show "$s" 2>/dev/null | grep -E "^value:|^conditions:" | tee -a "$LOG"
done

# ---------------------------------------------------------------------------
section "RAPL energy enable (package energy_uj is root-only on this box; you run this script, so sudo works)"
note "energy needs world-readable RAPL; harmless if it fails (energy lines just say 'unreadable')."
run "sudo -n chmod -R a+r /sys/class/powercap/intel-rapl*/ 2>/dev/null && echo 'RAPL readable' || echo 'RAPL NOT enabled — run: sudo chmod -R a+r /sys/class/powercap/intel-rapl*/  (energy will be N/A)'"

section "BUILD (probes debug; engine bins release)"
source "$IRON/ironenv/bin/activate" 2>/dev/null
run "which aiebu-asm || echo MISSING_aiebu-asm"
run "cd $WT/rust && cargo build -p npu-asr --bin fused_elf_probe 2>&1 | tail -2"
run "cd $WT/rust && cargo build -p npu-engine --release --bin verify_whisper_decode --bin engine_serve --bin whisper_e2e_timing --bin verify_embeddings --bin verify_esm 2>&1 | tail -3"

# ---------------------------------------------------------------------------
section "REGENERATE FUSED ARTIFACTS (host-only IRON compile; build/ cleaned per-op)"
gen(){ local name=$1; shift; run "cd $IRON && rm -rf build/${name}* build/*.prj 2>/dev/null; python $GEN/$* "; }
gen spike2gemv  "spike_gen_2gemv.py --out $WT/artifacts/fused_spike"
gen ln_qkv      "gen_ln_qkv.py --weights $WDIR --layer 0 --out $WT/artifacts/fused_ln_qkv"
gen ffn         "gen_ffn.py --weights $WDIR --layer 0 --out $WT/artifacts/fused_ffn"
gen self_attn   "gen_self_attn.py --weights $WDIR --layer 0 --prompt-len 448 --num-preceding 5 --out $WT/artifacts/fused_self_attn"
gen cross_attn  "gen_cross_attn.py --weights $WDIR --layer 0 --out $WT/artifacts/fused_cross_attn"
gen layer       "gen_layer.py --weights $WDIR --layer 0 --out $WT/artifacts/fused_layer"
gen decode      "gen_decode.py --weights $WDIR --layers 2 --out $WT/artifacts/fused_decode2"
gen decode      "gen_decode.py --weights $WDIR --layers 12 --out $WT/artifacts/fused_decode12"

# ===========================================================================
section "DEVICE TESTS — stopping npu-asr/voxd (single-tenant)"
run "systemctl --user stop npu-asr.service voxd.service; sleep 2"
run "fuser -v /dev/accel/accel0 2>&1 || echo device-free"

section "1a. Fused block CORRECTNESS (rel-L2, gate <=0.08)"
note "decode12 expected ~0.093 (kernel-approx compounding over 12 layers; WER-safe per parity below)"
for d in fused_spike fused_ln_qkv fused_ffn fused_self_attn fused_cross_attn fused_layer fused_decode2 fused_decode12; do
  run "$PROBE $WT/artifacts/$d"
done

section "1b. Whole-decode argmax PARITY vs f32 ideal (WER-safety; full greedy chain = a short transcription)"
run "python $GEN/verify_fused_decode.py --weights $WDIR --encoded $ENC --layers 12 --steps 32"

section "2. Fused whole-decode PER-TOKEN timing (1 decode step; trailing rel-fail is a cache-dirty artifact)"
run "FUSED_TIME=1 $PROBE $WT/artifacts/fused_decode12"

section "3. Whisper FULL-TRANSCRIPTION e2e — whisper_e2e_timing (encoder + N tokens), 3 timed passes"
note "this is the FULL transcription (not 1 token); reports e2e/encoder/decode ms, #tokens, ms/token, dispatches/token"
W3=$WT/rust/target/release/whisper_e2e_timing
for clip in "$CLIP_EN" "$CLIP_RU"; do
  run "cd $WT && WHISPER_TIMING=1 LD_LIBRARY_PATH=$LDLIB $W3 $clip                          # ONNX decode"
  run "cd $WT && WHISPER_TIMING=1 NPU_DECODE=1 LD_LIBRARY_PATH=$LDLIB $W3 $clip             # NPU step-1 decode"
  run "cd $WT && WHISPER_TIMING=1 NPU_DECODE=1 NPU_DECODE_ATTN=1 LD_LIBRARY_PATH=$LDLIB $W3 $clip  # NPU on-chip self-attn"
  run "cd $WT && WHISPER_TIMING=1 NPU_DECODE_FUSED=1 LD_LIBRARY_PATH=$LDLIB $W3 $clip             # FULL-NPU fused whole-decode (1 dispatch/token)"
done
note "[WHISPER_ENERGY] lines above give per-transcription RAPL package Joules + avg W per backend (NPU-vs-CPU energy)."

section "4. Whisper decode argmax PARITY vs ONNX (verify_whisper_decode)"
run "cd $WT/rust && WHISPER_ROOT=$WT LD_LIBRARY_PATH=$LDLIB cargo run -q -p npu-engine --release --bin verify_whisper_decode -- --host"
run "cd $WT/rust && WHISPER_ROOT=$WT LD_LIBRARY_PATH=$LDLIB cargo run -q -p npu-engine --release --bin verify_whisper_decode -- --npu"
run "cd $WT/rust && WHISPER_ROOT=$WT LD_LIBRARY_PATH=$LDLIB cargo run -q -p npu-engine --release --bin verify_whisper_decode -- --npu-attn"

# restart services for the harnesses that manage their own engine_serve
run "systemctl --user start npu-asr.service voxd.service; sleep 2"

# ===========================================================================
section "5. Whisper WER (FULL transcription accuracy, 17 clips) — ONNX vs NPU pooled WER vs known 0.1136"
run "bash $WT/scripts/_whisper_decode_wer_run.sh"
note "on-chip self-attn WER variant:"
run "test -f $WT/scripts/_whisper_decode_attn_wer_run.sh && bash $WT/scripts/_whisper_decode_attn_wer_run.sh || echo 'attn-wer script absent — skipped'"

section "6. OTHER MODELS (best-effort; skipped cleanly if env/deps absent)"
note "Parakeet full-transcription WER (needs onnx_asr venv):"
run "test -x ~/npuvox-asr-bench/.venv/bin/python && (cd $WT && ~/npuvox-asr-bench/.venv/bin/python scripts/parakeet_npu_wer.py npu artifacts/wer_clips) || echo 'parakeet venv/onnx_asr absent — skipped'"
note "Embeddings (bge-base) e2e latency:"
run "test -f $WT/scripts/_esm_latency.sh && bash $WT/scripts/_esm_latency.sh scenarios/bge-base.toml 11436 bge || echo 'embeddings harness skipped'"
note "ESM-2 native e2e latency:"
run "test -f $WT/scripts/_esm_latency.sh && bash $WT/scripts/_esm_latency.sh scenarios/esm2-35m-native.toml 11437 esm35 || echo 'esm harness skipped'"

# ===========================================================================
section "7. SUMMARY (grep key numbers from this run)"
echo "-- fused block rel-L2 --" | tee -a "$LOG"; grep -E "rel-L2 = |PASS|FAIL" "$LOG" | tail -20 | sed 's/^/  /'
echo "-- argmax parity --" | tee -a "$LOG"; grep -E "argmax parity:|PARITY" "$LOG" | tail -5 | sed 's/^/  /'
echo "-- fused per-token timing --" | tee -a "$LOG"; grep -E "re-registration|dispatch alone|FULL per-token" "$LOG" | tail -3 | sed 's/^/  /'
echo "-- whisper full-transcription timing (per backend) --" | tee -a "$LOG"; grep -E "WHISPER_TIMING" "$LOG" | sed 's/^/  /'
echo "-- whisper energy (RAPL pkg J/transcription, avg W, RTF) --" | tee -a "$LOG"; grep -E "WHISPER_ENERGY" "$LOG" | sed 's/^/  /'
echo "-- pooled WER (onnx / npu / fused vs known 0.1136) --" | tee -a "$LOG"; grep -iE "ALL pooled WER|label=" "$LOG" | tail -12 | sed 's/^/  /'

section "DONE — full log at $LOG"
echo "Suite finished $(date). Beeping." | tee -a "$LOG"
