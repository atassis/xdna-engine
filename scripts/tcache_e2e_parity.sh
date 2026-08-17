#!/usr/bin/env bash
# Token-parity gate for the M0.5 transposed self-V cache, driven through the ENGINE.
#
# verify_tcache_parity.py already proved the ARM is correct by driving decode.elf directly. This
# gates the other half: that the Rust host drives the 4-param contract, so the arm is REACHABLE.
# Both dirs come from one gen_decode differing in --coalesce-self-tr alone, so any token difference
# is the host's param handling.
#
# Greedy argmax per step -> the token sequence is a hard equality, not a tolerance. NOT a WER gate:
# the 17-clip WER is chaotic at ~1e-5 and is never the bar. Input is fixed-seed synthetic encoder
# hidden states (whisper-small has no mel preprocessor.onnx here, and the decoder is what is tested).
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
python3 - "$BASE" "$TR" <<'PY' || exit 1
import json, sys
b, t = (json.load(open(f"{d}/meta.json")) for d in sys.argv[1:3])
if not t.get("coalesce_self_tr"):
    sys.exit(f"[ERR] {sys.argv[2]} was not built with --coalesce-self-tr")
if b.get("coalesce_self_tr"):
    sys.exit(f"[ERR] {sys.argv[1]} IS a tr arm; it is the baseline")
for k in ("coalesce_cross", "coalesce_self", "int8_cross_k", "int8_cross_v", "int8_ffn",
          "int8_attn_w", "npu_logits"):
    if b.get(k) != t.get(k):
        sys.exit(f"[ERR] arms differ in {k} ({b.get(k)} vs {t.get(k)}) as well as the self-V cache")
if b["dims"] != t["dims"]:
    sys.exit(f"[ERR] arms differ in dims: {b['dims']} vs {t['dims']}")
print(f"[pair] arms differ in coalesce_self_tr alone; tr params = {sorted(t['scratchpad']['params'])}")
PY

echo "[gate] $STEPS steps/arm, base=$BASE tr=$TR"

# Single-tenant NPU. `systemctl is-active` is NOT the check -- it prints inactive for a service
# running as a plain process too; fuser + pgrep are.
echo "[svc] stopping xdna-engine + voxd ..."
systemctl --user stop xdna-engine.service voxd.service >/dev/null 2>&1
restart(){ systemctl --user start xdna-engine.service voxd.service >/dev/null 2>&1
           sleep 2
           local code; code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:11434/ 2>/dev/null || echo 000)
           # any HTTP response means it is serving; `/` is not a route, so 404 is healthy.
           if [ "$code" = "000" ]; then echo "[svc] WARNING: :11434 not answering (code=$code)"
           else echo "[svc] restarted; :11434 answering (code=$code)"; fi; }
trap restart EXIT
for _ in $(seq 10); do pgrep -f 'npu serve' >/dev/null 2>&1 || break; sleep 1; done
if fuser /dev/accel/accel0 >/dev/null 2>&1 || pgrep -f 'npu serve' >/dev/null 2>&1; then
  echo "[ERR] device still busy:"; fuser -v /dev/accel/accel0 2>&1; pgrep -af 'npu serve'
  exit 1
fi
echo "[svc] device clear"

run(){ # $1 label  $2 dir
  LD_LIBRARY_PATH="$LDLIB" "$BIN" "$2" "$STEPS" 2>"$OUT/$1.err" > "$OUT/$1.ids"
  local rc=$?
  echo "[run] $1 rc=$rc steps=$(wc -l < "$OUT/$1.ids") $(grep -c . "$OUT/$1.err" >/dev/null && grep -m1 '^\[enc\]' "$OUT/$1.err")"
  return $rc
}
run base "$BASE" || { echo "[ERR] baseline arm failed:"; tail -20 "$OUT/base.err"; exit 1; }
run tr   "$TR"   || { echo "[ERR] tr arm failed:";       tail -20 "$OUT/tr.err";   exit 1; }

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
echo "artifacts: $OUT"
exit $rc
