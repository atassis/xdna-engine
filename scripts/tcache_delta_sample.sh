#!/usr/bin/env bash
# Pooled per-step delta for the two stacked self-V-cache arms, N repeats each, INTERLEAVED.
#
# Why this exists. tcache_e2e_parity.sh estimates one delta per invocation and controls drift with
# the worst same-arm spread from n=2. MEASURED 2026-08-17: that spread is itself unstable across
# invocations on a box the load guard calls quiet -- 0.068, 0.089, 0.922, 2.677 ms in four
# consecutive runs -- so a single invocation lands on WIN or NO MEASURABLE DIFFERENCE partly by luck
# of the draw. The delta's SIGN was negative in every one of them. Repeating the whole invocation and
# pooling turns that into a mean with a standard error instead of one estimate with a volatile bar.
#
# Interleaved M0.5, M0.6, M0.5, ... so any slow trend (thermal, DVFS, a background app starting)
# lands on both pairings equally instead of entirely on whichever ran second.
#
#   bash scripts/tcache_delta_sample.sh [N]     # N repeats per pairing, default 5
#   PAIRS='direct|dec12_base|dec12_direct|plain -> stage writes cache' ... [N]
#
# PAIRS overrides which pairings run, one `tag|baseArtifact|armArtifact|label` per element,
# newline-separated. Default is the stacked ladder M0.5 then M0.6. Override it to measure a pairing
# DIRECTLY rather than by summing two: the stacked plain -> direct figure is a sum of the two default
# pairings, which is only defensible while their shared arm (`dec12_tr`) times the same in both.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
N="${1:-5}"
LOCK="${NPU_LOCK:-}"   # optional device serializer (was an absolute path into a private checkout)
TS="$(date +%Y%m%d_%H%M%S)"; OUT="$WT/artifacts/tcache_sample_${TS}"; mkdir -p "$OUT"

# M0.5 = the transposed cache (plain -> tr). M0.6 = the stage writing it (tr -> direct).
PAIRS="${PAIRS:-m05|dec12_base|dec12_tr|M0.5  plain -> transposed cache
m06|dec12_tr|dec12_direct|M0.6  tr -> stage writes cache}"
run_one(){ # $1 tag  $2 base dir  $3 tr dir
  # The parity harness restarts xdna-engine on EXIT, so the device is busy again by the next
  # iteration; npu_lock defers rather than stopping it. Clear it here, every time.
  systemctl --user stop xdna-engine.service npu-vox.service >/dev/null 2>&1
  sleep 2
  "$LOCK" run -- bash scripts/tcache_e2e_parity.sh "$WT/artifacts/$2" "$WT/artifacts/$3" 256 \
    > "$OUT/$1.log" 2>&1
  local delta spread load parity
  delta=$(grep -oP '^\[speed\] delta \K[-+0-9.]+' "$OUT/$1.log")
  spread=$(grep -oP 'drift control \(worst same-arm spread\) \K[0-9.]+' "$OUT/$1.log")
  load=$(grep -oP '^\[speed\] host load average \K[0-9.]+ -> [0-9.]+' "$OUT/$1.log")
  parity=$(grep -c '^GATE MET' "$OUT/$1.log")
  printf '%s\tdelta=%s\tdrift=%s\tload=%s\tparity_met=%s\n' "$1" "${delta:-ERR}" "${spread:-ERR}" "${load:-?}" "$parity" \
    | tee -a "$OUT/deltas.tsv"
}

# Read the spec into an array ONCE. Do not drive the run loop from a `read` on stdin -- the timed
# child would be competing for the same fd and could swallow the remaining pairings.
mapfile -t SPEC <<< "$PAIRS"

# The label sidecar is what the pooling step reads, so a PAIRS override needs no second edit.
for line in "${SPEC[@]}"; do
  [ -n "$line" ] || continue
  IFS='|' read -r tag base arm label <<< "$line"
  printf '%s\t%s\n' "$tag" "$label" >> "$OUT/pairings.tsv"
done

echo "[sample] $N repeats per pairing, interleaved; out=$OUT"
for i in $(seq 1 "$N"); do
  for line in "${SPEC[@]}"; do
    [ -n "$line" ] || continue
    IFS='|' read -r tag base arm label <<< "$line"
    run_one "${tag}_$i" "$base" "$arm" < /dev/null
  done
done

echo "=============== POOLED ==============="
python3 - "$OUT/deltas.tsv" "$OUT/pairings.tsv" <<'PY'
import re, sys, statistics as st
rows = [l.split('\t') for l in open(sys.argv[1]).read().splitlines() if l.strip()]
pairs = [l.split('\t') for l in open(sys.argv[2]).read().splitlines() if l.strip()]
for arm, label in pairs:
    d = [float(r[1].split('=')[1]) for r in rows if r[0].startswith(arm) and 'ERR' not in r[1]]
    p = [r[4].split('=')[1] for r in rows if r[0].startswith(arm)]
    if not d:
        print(f"{label}: no usable runs"); continue
    n, mean = len(d), st.mean(d)
    sd = st.stdev(d) if n > 1 else float('nan')
    se = sd / n**0.5 if n > 1 else float('nan')
    neg = sum(1 for x in d if x < 0)
    print(f"\n{label}")
    print(f"  n={n}  deltas ms/step: {', '.join(f'{x:+.3f}' for x in d)}")
    print(f"  mean {mean:+.3f}  sd {sd:.3f}  se {se:.3f}  95% CI [{mean-1.96*se:+.3f}, {mean+1.96*se:+.3f}]")
    print(f"  sign: {neg}/{n} negative (faster)   parity gate met: {sum(1 for x in p if x=='1')}/{n}")
    # The CI, not any single run's drift control, is the claim: a CI straddling 0 is not a win.
    print(f"  VERDICT: {'WIN' if mean + 1.96*se < 0 else 'REGRESSION' if mean - 1.96*se > 0 else 'NOT SEPARABLE FROM ZERO'}")
PY
echo "artifacts: $OUT"
