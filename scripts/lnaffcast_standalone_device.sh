#!/usr/bin/env bash
# Device wrapper: price the lnaffcast MODE against the SHIPPED STANDALONE op at the encoder's own
# row count (task mode-switched-multi-program-xclbin, the last item before the merge verdict).
#
# THE QUESTION. Merging lnaffcast's context into the modal GEMM's is priced at -240.0 ms/clip of
# transition tax, but that ledger prices BOUNDARIES only. The merge's net is
#
#     -240.0 ms/clip  +  dispatches_per_clip x (mode - standalone)
#
# and the right-hand term has never been measured. The mode and the standalone op do the same
# arithmetic on the same bytes but on different topologies -- 32 cores off the GEMM's borrowed bf16
# fifos against 8 cores with real f32 operands -- so neither the sign nor the size of that
# difference is predictable from either side alone.
#
# 512 ROWS IS NOT A CHOICE. npu.rs pads to PAD_M = 512 and dispatches one baked stream, so the
# shipped op runs 512 rows on every clip whatever T is; and 512 is the only count that xclbin can
# run, its sequence_length being baked into the taps. The mode's 512-row arm is already banked at
# 590.9 us (artifacts/lnaffcast_1x_1x512.json), and it is re-run here rather than quoted so that
# both sides come from ONE quiesced session.
#
# THE CONTROL IS A BRACKET, NOT A COMPANION STREAM. Every prior arm on this task rode a modal
# xclbin, so the GEMM stream ran alongside it as a within-group control. The standalone op's xclbin
# has no GEMM stream. So the GEMM control runs BEFORE and AFTER the standalone load:
#
#   1. modal 1x xclbin   mode@512 (must PASS) + GEMM stream   -> the candidate, and bracket 1
#   2. standalone xclbin lnaffcast_512x1024 (must PASS)       -> the incumbent being priced
#   3. modal 1x xclbin   GEMM stream                          -> bracket 2
#
# Brackets 1 and 3 agreeing is what says the session did not drift across the load in between --
# the same evidence gemm-stream-controls-the-cross-xclbin-comparison provides, by a different route
# because the usual route is unavailable here. If they disagree by more than the ~2.6% seen across
# three loads on the 1x run, the standalone number sits on a moving session and the verdict waits.
#
# Both sides are gated against the SAME host reference (same seeds, same draw order), so the parity
# numbers are comparable rather than merely similar in kind.
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort. The free-device check is fuser + pgrep, never
# `systemctl is-active` -- it prints inactive for stopped, absent AND running-as-a-plain-process
# alike, and `npu-asr` was renamed to `xdna-engine`, so the old name returns a false all-clear.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/lnaffcast_standalone.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

# Which vehicle carries the mode. Default is the research one this was first written against; the
# merge needs the price on the ENCODER'S OWN resident, where the mode is a mode of the xclbin the
# FFN already loads:
#   BASE=512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024lnaff1024 bash scripts/...
# The standalone arm is the same shipped op either way -- it is the mode side that moves.
BASE="${BASE:-512x1024x1024_32x32x128_8c_modalidbf16outkrtpkrllnaff1024}"
X1=${BASE}scat21x
REPS="${REPS:-5}"
ROWS="${ROWS:-512}"
LN_DIR="${LN_DIR:-/mnt/data/models/xdna-artifacts/parakeet/ln}"

log "===== lnaffcast: mode vs the SHIPPED standalone op at ${ROWS} rows  $(date -Is) ====="
log "[svc] stopping xdna-engine + npu-vox"
systemctl --user stop xdna-engine.service npu-vox.service >/dev/null 2>&1
sleep 2
held=0
fuser /dev/accel/accel0 >/dev/null 2>&1 && held=1
if pgrep -af '(^|/)npu serve' >/dev/null 2>&1; then held=1; fi
if [ "$held" = 1 ]; then
  log "FATAL: device still held -- another session has the NPU. Aborting (single-tenant)."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"
  pgrep -af '(^|/)npu serve' | tee -a "$LOG"
  exit 75
fi
log "[svc] device clear"

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
rc=0

log "\n========== 1/3  mode @ ${ROWS} rows, + GEMM stream (bracket 1) =========="
.venv-iron/bin/python scripts/lnaffcast_burst_isolation.py \
    --host "${X1}rtp18r${ROWS}g4ctgc" \
    --arms "${X1}rtp18r${ROWS}g4ctgc" "${X1}" \
    --parity-must-pass "${X1}rtp18r${ROWS}g4ctgc" \
    --reps "$REPS" --rows "$ROWS" --one-x \
    --out "artifacts/lnaffcast_standalone_mode.json" 2>&1 | tee -a "$LOG"
r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r

log "\n========== 2/3  SHIPPED standalone lnaffcast_${ROWS}x1024 =========="
.venv-iron/bin/python scripts/lnaffcast_standalone_price.py \
    --ln-dir "$LN_DIR" --rows "$ROWS" --reps "$REPS" \
    --out "artifacts/lnaffcast_standalone.json" 2>&1 | tee -a "$LOG"
r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r

log "\n========== 3/3  GEMM stream again (bracket 2) =========="
.venv-iron/bin/python scripts/lnaffcast_burst_isolation.py \
    --host "${X1}rtp18r${ROWS}g4ctgc" \
    --arms "${X1}rtp18r${ROWS}g4ctgc" "${X1}" \
    --parity-must-pass "${X1}rtp18r${ROWS}g4ctgc" \
    --reps "$REPS" --rows "$ROWS" --one-x \
    --out "artifacts/lnaffcast_standalone_bracket2.json" 2>&1 | tee -a "$LOG"
r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r

log "\n========== verdict =========="
.venv-iron/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, statistics
def med(p, key=None):
    d = json.load(open(p))
    r = d["results"]
    if key is None:
        b = r["blocks"]
    else:
        b = [v for k, v in r.items() if k.endswith(key)][0]["blocks"]
    return statistics.mean(b[n]["median_us"] for n in ("forward", "reversed")), b

m1 = json.load(open("artifacts/lnaffcast_standalone_mode.json"))
m2 = json.load(open("artifacts/lnaffcast_standalone_bracket2.json"))
def arms(d):
    out = {}
    for k, v in d["results"].items():
        out[k] = statistics.mean(v["blocks"][n]["median_us"] for n in ("forward", "reversed"))
    return out
a1, a2 = arms(m1), arms(m2)
gemm1 = [v for k, v in a1.items() if k.endswith("scat21x")][0]
gemm2 = [v for k, v in a2.items() if k.endswith("scat21x")][0]
mode1 = [v for k, v in a1.items() if "ctgc" in k][0]
mode2 = [v for k, v in a2.items() if "ctgc" in k][0]
st = json.load(open("artifacts/lnaffcast_standalone.json"))["results"]
stand = st["mean_us"]

drift = abs(gemm2 - gemm1) / statistics.mean([gemm1, gemm2])
print(f"  GEMM bracket 1        {gemm1:8.1f} us")
print(f"  GEMM bracket 2        {gemm2:8.1f} us     drift {drift:+.1%}")
print(f"  mode @512 (b1 / b2)   {mode1:8.1f} / {mode2:.1f} us")
print(f"  standalone @512       {stand:8.1f} us     block spread {st['block_spread']:.1%}")
mode = statistics.mean([mode1, mode2])
d = mode - stand
print(f"\n  mode - standalone     {d:+8.1f} us per dispatch  ({d / stand:+.1%})")
for n in (48, 96):
    print(f"    x{n:3d} dispatches/clip -> {d * n / 1000.0:+8.1f} ms/clip against the merge's -240.0")
if drift > 0.05:
    print(f"\n  WARNING: brackets disagree by {drift:.1%} -- the session moved across the "
          f"standalone load, so this delta is not clean.")
PY

log ""
log "log: $LOG   rc=$rc"
exit $rc
