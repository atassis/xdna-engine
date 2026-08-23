#!/usr/bin/env bash
# END-TO-END gate for the lnaffcast mode wired into npu.rs (PARAKEET_LN_MODE=1) --
# task mode-switched-multi-program-xclbin, its `next:` item 1, second half ("WIRE npu.rs").
#
# THE QUESTION. lnaffcast_merge_device.sh gated the mode IN ISOLATION: three streams on one loaded
# xclbin, one host reference, rel-L2 1.7849e-05. That says the mode computes affine_LN. It does NOT
# say the ENCODER still transcribes, because in the encoder the mode is reached through npu.rs's own
# BO plumbing -- operands swapped (gb on A, x on B), a C tap that walks [PAD_M,KRES] row-major into a
# C the GEMM stream declares [PAD_M,DFF] panel-major, and a `_dev` variant whose B is the previous
# brick's output BO rather than a host-written one. Any of those can be wired wrong and still leave
# the isolated parity number untouched.
#
# THE GATE IS TOKEN PARITY, not WER. The 17-clip WER is chaotic at ~1e-5, so it cannot resolve a
# change this small in either direction; what it CAN do is answer "did the transcript move", and that
# is a comparison against our own baseline arm, not against refs.json. Both arms decode through the
# same onnx-asr TDT decoder at temp 0, so identical encoder semantics must give identical text.
#
# ARMS, same binary, same mels, same decoder -- one env var apart:
#   base   PARAKEET_LN_MODE unset  lnaffcast on its own `lnaffcast_512x1024` xclbin (shipped)
#   mode   PARAKEET_LN_MODE=1      lnaffcast as a mode of the fc1 panel -- one program with fc1
#
# It reports three things, in increasing order of what they cost to move:
#   (1) per-clip rel-L2 between the two encoder outputs        -- a NOTE, never a gate
#   (2) TOKEN PARITY: hypotheses identical clip-for-clip       -- THE GATE
#   (3) both arms' WER against refs.json                       -- context, so a shared regression
#       against the reference is still visible when the two arms agree with each other
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies free with fuser + pgrep (never
# `systemctl is-active` -- it prints inactive for stopped, absent AND running-as-a-plain-process
# alike), and ALWAYS restores on exit including on abort.
#
# MELS must be a set refs.json can score -- the decode step keys off refs.json, so a mel dir with
# clips it does not name contributes encodings to (1) but nothing to (2)/(3).
#
#   bash scripts/lnaffcast_mode_encoder_gate.sh                          # artifacts/wer_mels
#   MELS=... WER_CLIPS=... bash scripts/lnaffcast_mode_encoder_gate.sh   # a wider set
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$WT/artifacts/lnmode_gate"
LOG="$OUT/gate_$TS.log"
mkdir -p "$OUT"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

BIN="$WT/rust/target/release/parakeet_encode_npu"
MELS="${MELS:-$WT/artifacts/wer_mels}"
PY="${PY:-$HOME/npuvox-asr-bench/.venv/bin/python}"

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

[ -x "$BIN" ] || { log "FATAL missing (prebuild with: cargo build -p npu-probes --release --bin parakeet_encode_npu): $BIN"; exit 1; }
[ -d "$MELS" ] || { log "FATAL: mel dir $MELS does not exist"; exit 1; }
[ -x "$PY" ]  || { log "FATAL: decoder python $PY not executable"; exit 1; }
NCLIP=$(ls "$MELS"/*.npy 2>/dev/null | wc -l)
[ "$NCLIP" -ge 1 ] || { log "FATAL: no .npy mels in $MELS"; exit 1; }

log "===== lnaffcast mode, end to end: does the encoder still transcribe?  $(date -Is) ====="
log "mels: $MELS ($NCLIP clips)   bin: $BIN"

log "[svc] stopping xdna-engine + npu-vox"
systemctl --user stop xdna-engine.service npu-vox.service >/dev/null 2>&1
sleep 2
held=0
fuser /dev/accel/accel0 >/dev/null 2>&1 && held=1
pgrep -af '(^|/)npu serve' >/dev/null 2>&1 && held=1
if [ "$held" = 1 ]; then
  log "FATAL: device still held -- another session has the NPU. Aborting (single-tenant)."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"
  pgrep -af '(^|/)npu serve' | tee -a "$LOG"
  exit 75
fi
log "[svc] device clear"

rc=0
encode(){  # $1 = arm name, rest = env assignments
  local arm="$1"; shift
  local dir="$OUT/$arm"
  rm -rf "$dir"; mkdir -p "$dir"
  log "\n---------- arm $arm ($*) ----------"
  env NPU_XCLBIN_ROOT="$WT" "$@" "$BIN" "$MELS" "$dir" 2>&1 | tee -a "$LOG"
  local r=${PIPESTATUS[0]}
  [ "$r" = 0 ] || { log "arm $arm FAILED rc=$r"; rc=$r; return $r; }
  local n; n=$(ls "$dir"/*.npy 2>/dev/null | wc -l)
  [ "$n" = "$NCLIP" ] || { log "arm $arm wrote $n/$NCLIP encodings"; rc=1; return 1; }
}

encode base                     || exit $rc
encode mode PARAKEET_LN_MODE=1  || exit $rc

# (1) NUMERIC NOTE. Not a gate -- see error-metrics-are-notes-not-gates. It is here to say HOW the
# two arms differ when the token gate fails, and to catch the degenerate pass where an arm wrote
# zeros or NaN and the decoder emitted the empty string for both.
log "\n---------- encoder-output delta (a note, not the gate) ----------"
# The heredoc must be redirected onto $PY, NOT onto the tee at the end of the pipeline -- attach it
# to the pipeline and `tee` gets the script on stdin, prints it, and python reads an empty program
# and exits 0. That reads exactly like a passing check.
"$PY" - "$OUT/base" "$OUT/mode" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, os, glob, numpy as np
b, m = sys.argv[1], sys.argv[2]
worst, nan = 0.0, []
for p in sorted(glob.glob(os.path.join(b, "*.npy"))):
    c = os.path.basename(p)
    q = os.path.join(m, c)
    if not os.path.exists(q):
        print(f"  {c:<12s} MISSING in mode arm"); nan.append(c); continue
    x, y = np.load(p).astype(np.float64), np.load(q).astype(np.float64)
    if x.shape != y.shape:
        print(f"  {c:<12s} SHAPE {x.shape} vs {y.shape}"); nan.append(c); continue
    if not np.isfinite(y).all():
        print(f"  {c:<12s} NON-FINITE in mode arm"); nan.append(c); continue
    d = np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30)
    ex = int(np.count_nonzero(x == y))
    worst = max(worst, d)
    print(f"  {c:<12s} rel-L2 {d:.4e}  exact {ex}/{x.size}")
print(f"  worst rel-L2 {worst:.4e}" + (f"   BAD CLIPS: {nan}" if nan else ""))
sys.exit(1 if nan else 0)
PYEOF
[ ${PIPESTATUS[0]} = 0 ] || rc=1

# (2) + (3). --per-clip prints the hypothesis per clip, which is what makes the two arms diffable;
# the aggregate WER alone would hide a transcript that moved without moving the mean.
for arm in base mode; do
  log "\n---------- decode: $arm ----------"
  "$PY" scripts/parakeet_npu_wer.py decode-wer "$OUT/$arm" --per-clip 2>&1 | tee "$OUT/decode_$arm.txt" | tee -a "$LOG"
done

log "\n---------- TOKEN PARITY (the gate) ----------"
# Drop the one line that names the encode dir -- it differs between the arms BY CONSTRUCTION and
# would fail the diff on the arm labels rather than on the transcripts.
strip(){ grep -v 'swapped encoder from' "$1"; }
if diff -u <(strip "$OUT/decode_base.txt") <(strip "$OUT/decode_mode.txt") > "$OUT/decode_diff.txt"; then
  log "PASS: hypotheses IDENTICAL across all $NCLIP clips"
else
  log "FAIL: the transcript moved -- see $OUT/decode_diff.txt"
  sed -n '1,60p' "$OUT/decode_diff.txt" | tee -a "$LOG"
  rc=1
fi

log "\nlog: $LOG   rc=$rc"
exit $rc
