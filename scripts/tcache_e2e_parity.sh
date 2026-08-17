#!/usr/bin/env bash
# Token-parity gate for the M0.5 transposed self-V cache, driven through the ENGINE.
#
# verify_tcache_parity.py already proved the ARM is correct by driving decode.elf directly. This
# gates the other half: that the Rust host drives the 4-param contract, so the arm is REACHABLE.
# Both dirs come from one gen_decode differing in --coalesce-self-tr alone, so any token difference
# is the host's param handling.
#
# Greedy argmax per step -> the token sequence is a hard equality, not a tolerance. NOT a WER gate:
# the 17-clip WER is chaotic at ~1e-5 and is never the bar. Input is the SAVED REAL encoder output
# (refs/encoded.npy) -- whisper-small has no mel preprocessor.onnx here, and the decoder is what is
# tested. Fixed-seed synthetic states are NOT usable as the gate: they decode to a 2-token constant
# that a broadly wrong cache also reproduces (the bin keeps them behind --synthetic).
#
# ALSO REPORTS THE SPEED NUMBER, from the same run and so at no extra device cost: the bin times
# fd.step() alone per step. Arms are timed back-to-back in one quiesced window, and the drift control
# is that each arm runs TWICE in A,B,B,A order -- if an arm's two medians differ by more than the
# base-vs-tr delta, the delta is drift and not the arm.
#
#   bash scripts/tcache_e2e_parity.sh [BASE_DIR] [TR_DIR] [STEPS]
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
BASE="${1:-$WT/artifacts/dec12_base}"
TR="${2:-$WT/artifacts/dec12_tr}"
STEPS="${3:-256}"
BIN="$WT/rust/target/release/tcache_fused_parity"
LDLIB=~/.local/lib/npu-asr
TS="$(date +%Y%m%d_%H%M%S)"; OUT="$WT/artifacts/tcache_e2e_${TS}"
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "[ERR] missing $BIN (cargo build --release -p npu-probes --bin tcache_fused_parity)"; exit 1; }
for d in "$BASE" "$TR"; do
  [ -f "$d/decode.elf" ] || { echo "[ERR] missing $d/decode.elf"; exit 1; }
done
# Refuse a comparison whose arms differ in more than the self-V cache.
LADDER="${LADDER:-0}" python3 - "$BASE" "$TR" <<'PY' || exit 1
import json, os, sys
b, t = (json.load(open(f"{d}/meta.json")) for d in sys.argv[1:3])
# Two stackable arms live on this axis, so the DEFAULT pairing rule is "differ in exactly ONE of
# them": coalesce_self_tr (M0.5, transposed cache) and vstage_direct (M0.6, the stage writes that
# cache itself and op_scv is gone). Pairing M0.6 against M0.5 isolates the fused write; pairing
# either against the plain arm isolates the transposed cache. Pairing M0.6 against plain moves both,
# so a delta from it is not attributable to one flag -- refused unless the caller says it wants the
# LADDER, which is a different question: what the fully-stacked arm buys against today's default,
# the number the default-flip decision actually turns on. Summing the two one-flag pairings is NOT a
# substitute; that sum is only defensible while their shared arm times the same in both.
AXES = ("coalesce_self_tr", "vstage_direct")
ladder = os.environ.get("LADDER") == "1"
moved = [k for k in AXES if bool(b.get(k)) != bool(t.get(k))]
if not moved:
    sys.exit(f"[ERR] arms are identical on {AXES}; they differ in none")
if len(moved) > 1 and not ladder:
    sys.exit(f"[ERR] arms must differ in exactly one of {AXES}; they differ in {moved}. "
             f"Set LADDER=1 to compare plain against the fully-stacked arm on purpose -- "
             f"the delta is then cumulative, NOT attributable to one flag")
for k in moved:
    if not bool(t.get(k)):
        sys.exit(f"[ERR] {sys.argv[1]} is the {k} arm; pass it as TR and the other as BASE")
if ladder and len(moved) > 1 and any(bool(b.get(k)) for k in AXES):
    sys.exit(f"[ERR] LADDER wants plain (neither axis) as BASE; {sys.argv[1]} already has "
             f"{[k for k in AXES if bool(b.get(k))]}")
for k in ("coalesce_cross", "coalesce_self", "int8_cross_k", "int8_cross_v", "int8_ffn",
          "int8_attn_w", "npu_logits") + tuple(k for k in AXES if k not in moved):
    if bool(b.get(k)) != bool(t.get(k)):
        sys.exit(f"[ERR] arms differ in {k} ({b.get(k)} vs {t.get(k)}) as well as {moved}")
if b["dims"] != t["dims"]:
    sys.exit(f"[ERR] arms differ in dims: {b['dims']} vs {t['dims']}")
if len(moved) > 1:
    print(f"[pair] LADDER: arms differ in {moved} together -- delta is CUMULATIVE, not "
          f"attributable to one flag; tr params = {sorted(t['scratchpad']['params'])}")
else:
    print(f"[pair] arms differ in {moved[0]} alone; tr params = {sorted(t['scratchpad']['params'])}")
PY

echo "[gate] $STEPS steps/arm, base=$BASE tr=$TR"

# Single-tenant NPU. `systemctl is-active` is NOT the check -- it prints inactive for a service
# running as a plain process too; fuser + pgrep are.
# The voice daemon's unit is npu-vox.service. Scripts here inherited `voxd.service`, which does not
# exist on this box (`systemctl --user stop voxd.service` is a silent no-op, rc ignored), so the
# daemon stayed UP through timed runs and the single-tenant claim was nominal. Naming a unit that
# cannot be verified is the same silent failure as a stopped one.
SVC="xdna-engine.service npu-vox.service"
echo "[svc] stopping $SVC ..."
# shellcheck disable=SC2086
systemctl --user stop $SVC >/dev/null 2>&1
restart(){ # shellcheck disable=SC2086
           systemctl --user start $SVC >/dev/null 2>&1
           # POLL, do not sleep-once: the engine takes ~4 s to answer here, so a fixed `sleep 2`
           # reported it down (or, with the old check, silently "up") on a box that was merely slow.
           local code=""
           for _ in $(seq 15); do
             # -w always prints a code (000 = no connection), so a `|| echo 000` fallback concatenates
             # into "000000" and silently passes an equality test against "000" -- report it healthy
             # while it is down. Test what curl printed, and nothing else.
             code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:11434/ 2>/dev/null)
             # any HTTP response means it is serving; `/` is not a route, so 404 is healthy.
             case "$code" in ""|000) sleep 1 ;; *) break ;; esac
           done
           case "$code" in
             ""|000) echo "[svc] WARNING: :11434 not answering (code=${code:-none}) -- restart it by hand" ;;
             *)      echo "[svc] restarted; :11434 answering (code=$code)" ;;
           esac; }
trap restart EXIT
for _ in $(seq 10); do pgrep -f 'npu serve' >/dev/null 2>&1 || break; sleep 1; done
if fuser /dev/accel/accel0 >/dev/null 2>&1 || pgrep -f 'npu serve' >/dev/null 2>&1; then
  echo "[ERR] device still busy:"; fuser -v /dev/accel/accel0 2>&1; pgrep -af 'npu serve'
  exit 1
fi
echo "[svc] device clear"

# Stopping the NPU units is NOT the same as a quiesced box, and the timing half is what notices.
# MEASURED 2026-08-17: with a Wine/Proton app at ~392% CPU (load 8.4) the drift control widened
# 0.539 -> 5.690 ms, which is wide enough to hide any per-step effect smaller than the ops we are
# removing. Record the host load around the timed section so a delta can never be read as a result
# on a box that was busy; the verdict below downgrades itself rather than blocking the run.
LOAD_MAX="${LOAD_MAX:-2.0}"
cut -d' ' -f1 /proc/loadavg > "$OUT/loadavg.pre"
echo "[load] 1-min load average before timed runs: $(cat "$OUT/loadavg.pre") (quiet bar $LOAD_MAX)"

run(){ # $1 label  $2 dir
  LD_LIBRARY_PATH="$LDLIB" "$BIN" "$2" "$STEPS" 2>"$OUT/$1.err" > "$OUT/$1.ids"
  local rc=$?
  echo "[run] $1 rc=$rc steps=$(wc -l < "$OUT/$1.ids") $(grep -c . "$OUT/$1.err" >/dev/null && grep -m1 '^\[enc\]' "$OUT/$1.err")"
  return $rc
}
# A,B,B,A: the repeat of each arm is the drift control for the timing half. Parity is gated on the
# first pair; base-vs-base2 doubles as a determinism check on the harness itself.
run base "$BASE" || { echo "[ERR] baseline arm failed:"; tail -20 "$OUT/base.err"; exit 1; }
run tr   "$TR"   || { echo "[ERR] tr arm failed:";       tail -20 "$OUT/tr.err";   exit 1; }
run tr2  "$TR"   || { echo "[ERR] tr arm (repeat) failed:";       tail -20 "$OUT/tr2.err";  exit 1; }
run base2 "$BASE" || { echo "[ERR] baseline arm (repeat) failed:"; tail -20 "$OUT/base2.err"; exit 1; }
cmp -s "$OUT/base.ids" "$OUT/base2.ids" \
  && echo "[det] baseline reruns bit-identical" \
  || echo "[det] WARNING: baseline is NOT run-to-run deterministic -- treat the parity gate with suspicion"

echo "=============== TOKEN PARITY ==============="
# Report WHAT diverged, not just pass/fail: a hash-only difference means the numerics moved while
# the argmax survived, and an odd-step-only pattern is the staged-pair hazard's signature.
python3 - "$OUT/base.ids" "$OUT/tr.ids" "$STEPS" <<'PY'
import sys
b = [l.split(":") for l in open(sys.argv[1]).read().split()]
t = [l.split(":") for l in open(sys.argv[2]).read().split()]
want = int(sys.argv[3])
if len(b) != want or len(t) != want:
    sys.exit(f"GATE FAILED: short output (base={len(b)} tr={len(t)}, want {want})")
uniq = len({x[0] for x in b})
tok_bad = [i for i, (x, y) in enumerate(zip(b, t)) if x[0] != y[0]]
hash_bad = [i for i, (x, y) in enumerate(zip(b, t)) if x[1] != y[1]]
print(f"[signal] baseline emitted {uniq} distinct tokens over {len(b)} steps"
      + ("  <-- DEGENERATE, gate is weak" if uniq <= 2 else ""))
if not hash_bad:
    print(f"GATE MET: {len(b)}/{len(b)} steps bit-identical (argmax AND full logit vector); "
          f"{sum(1 for i in range(len(b)) if i % 2 == 0)} even + "
          f"{sum(1 for i in range(len(b)) if i % 2)} odd columns")
    sys.exit(0)
print(f"GATE FAILED: {len(tok_bad)}/{len(b)} argmax differ, {len(hash_bad)}/{len(b)} logit vectors "
      f"differ; first divergence at step {hash_bad[0]} (parity {hash_bad[0] % 2}); "
      f"even-step={sum(1 for i in hash_bad if i % 2 == 0)}, odd-step={sum(1 for i in hash_bad if i % 2)}")
for i in hash_bad[:10]:
    print(f"  step {i:3d} (parity {i % 2}): base={b[i][0]}/{b[i][1]} tr={t[i][0]}/{t[i][1]}")
sys.exit(1)
PY
rc=$?

cut -d' ' -f1 /proc/loadavg > "$OUT/loadavg.post"

echo "=============== PER-STEP SPEED ==============="
grep -h '^\[time\]' "$OUT"/base.err "$OUT"/tr.err "$OUT"/tr2.err "$OUT"/base2.err 2>/dev/null \
  | sed 's/^/  /' || true
# The verdict is deliberately conservative: a delta the drift control cannot separate from run-to-run
# noise is reported as NO MEASURABLE DIFFERENCE, not as a win.
python3 - "$OUT" "$LOAD_MAX" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
load_max = float(sys.argv[2])
def load(which):
    p = out / f"loadavg.{which}"
    return float(p.read_text().strip()) if p.exists() else None
lo_pre, lo_post = load("pre"), load("post")
busy = max(x for x in (lo_pre, lo_post) if x is not None) > load_max \
    if any(x is not None for x in (lo_pre, lo_post)) else False
def median(label):
    p = out / f"{label}.err"
    if not p.exists(): return None
    m = re.search(r'\[time\].*?median=([\d.]+) ms', p.read_text(), re.S)
    return float(m.group(1)) if m else None
b1, b2, t1, t2 = (median(x) for x in ("base", "base2", "tr", "tr2"))
if None in (b1, b2, t1, t2):
    print(f"[speed] incomplete timing (base={b1},{b2} tr={t1},{t2})"); sys.exit(0)
base, tr = (b1 + b2) / 2, (t1 + t2) / 2
delta, spread = tr - base, max(abs(b1 - b2), abs(t1 - t2))
print(f"[speed] baseline median {base:.3f} ms/step (runs {b1:.3f}, {b2:.3f})")
print(f"[speed] tr arm   median {tr:.3f} ms/step (runs {t1:.3f}, {t2:.3f})")
print(f"[speed] delta {delta:+.3f} ms/step ({100 * delta / base:+.2f}%); "
      f"drift control (worst same-arm spread) {spread:.3f} ms")
print(f"[speed] host load average {lo_pre} -> {lo_post} (quiet bar {load_max})")
if busy:
    print(f"[speed] VERDICT: INCONCLUSIVE -- the box was NOT quiesced (load > {load_max}), which is "
          f"what the {spread:.3f} ms drift control is measuring. Stopping the NPU units does not "
          f"stop host CPU load. Re-run on an idle box before reading the delta as anything.")
elif abs(delta) <= spread:
    print("[speed] VERDICT: NO MEASURABLE DIFFERENCE -- |delta| is within run-to-run drift, so this "
          "run does not support a speed claim in either direction.")
else:
    print(f"[speed] VERDICT: {'REGRESSION' if delta > 0 else 'WIN'} of {abs(delta):.3f} ms/step, "
          f"{abs(delta) / spread:.1f}x the drift control.")
PY
echo "artifacts: $OUT"
exit $rc
