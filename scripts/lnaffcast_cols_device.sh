#!/usr/bin/env bash
# Measure the lnaffcast 1x route's row response at 8 AND 4 columns inside ONE quiesced window,
# with the 8c arm run twice to bracket the 4c one (task mode-switched-multi-program-xclbin).
#
# WHY BRACKETED RATHER THAN TWO RUNS. The point of the column axis is the DIFFERENCE between the
# two intercepts, and this box drifts 8.0% across sessions against 2.2% within one (measured, and
# persistent rather than thermal). A 4c number from one session minus an 8c
# number from another is differencing two quantities whose separation is smaller than the boundary
# between them. Running 8c -> 4c -> 8c in one window makes the drift measurable instead of assumed:
# the two 8c passes bound it, and any 4c-vs-8c effect has to clear that bound to be read.
#
# Columns change the array program, so each pass loads its own xclbin -- that is unavoidable here
# and is exactly what the repeated 8c pass controls for.
#
# Single-tenant NPU: stops xdna-engine + npu-vox ONCE for all three passes, verifies the device is
# free with fuser + pgrep (never `systemctl is-active`), and always restores on exit.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/lnaffcast_cols_device.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

log "===== lnaffcast row response, 8c bracketing 4c  $(date -Is) ====="
log "[svc] stopping xdna-engine + npu-vox"
systemctl --user stop xdna-engine.service npu-vox.service >/dev/null 2>&1
sleep 2
held=0
fuser /dev/accel/accel0 >/dev/null 2>&1 && held=1
if pgrep -af '(^|/)npu serve' >/dev/null 2>&1; then held=1; fi
if [ "$held" = 1 ]; then
  log "FATAL: device still held -- another session has the NPU. Aborting (single-tenant)."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"
  exit 75
fi
log "[svc] device clear"

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
B8=512x1024x1024_32x32x128_8c_modalidbf16outkrtpkrllnaff1024scat21x
B4=512x1024x1024_32x32x128_4c_modalidbf16outkrtpkrllnaff1024scat21x
DEPTHS="${DEPTHS:-1 8 32}"
REPS="${REPS:-9}"

pass() {   # <tag> <base>
  log "\n########## pass $1  ($2)  $(date -Is) ##########"
  .venv-iron/bin/python scripts/lnaffcast_dispatch_floor.py \
     --base "$2" --depths $DEPTHS --reps "$REPS" --depth-reps "$REPS" \
     --out "artifacts/lnaffcast_cols_$1.json" 2>&1 | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

pass 8c_pre  "$B8" || exit 1
pass 4c      "$B4" || exit 1
pass 8c_post "$B8" || exit 1

log "\n===== bracket ====="
python3 - "$WT" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, os
wt = sys.argv[1]
def load(tag):
    return json.load(open(os.path.join(wt, "artifacts", f"lnaffcast_cols_{tag}.json")))

def rows_us(d, depth="1"):
    """Mean of the forward and reversed blocks -- this vehicle has a known order effect, so a
    single block is not the arm's number."""
    out = {}
    for arm in d["results"].values():
        if arm["kind"] != "lnaff":
            continue
        vals = [b["depths"][depth]["per_cmd_us"] for b in arm["blocks"].values()
                if depth in b["depths"]]
        if vals:
            out[arm["rows"]] = sum(vals) / len(vals)
    return out

try:
    pre, post, four = (load("8c_pre"), load("8c_post"), load("4c"))
except Exception as e:
    print("bracket: could not read a pass (%s) -- read the per-pass output above" % e)
    raise SystemExit(0)

for depth in ("1", "32"):
    p, q, f = rows_us(pre, depth), rows_us(post, depth), rows_us(four, depth)
    print(f"\ndepth {depth}")
    print(f"{'rows':>6}{'8c pre':>10}{'8c post':>10}{'8c drift':>10}{'4c':>10}{'4c/8c':>9}")
    for r in sorted(set(p) & set(q) & set(f)):
        mid = 0.5 * (p[r] + q[r])
        print(f"{r:>6}{p[r]:>10.1f}{q[r]:>10.1f}{100*(q[r]-p[r])/p[r]:>9.1f}%{f[r]:>10.1f}{f[r]/mid:>9.2f}")

print("\nrow fit (slope us/row + intercept us, max |resid|):")
for tag, d in (("8c_pre", pre), ("4c", four), ("8c_post", post)):
    fit = d["row_fits"]["all_points"]
    print(f"  {tag:<8} {fit['slope_us_per_row']:>8.4f} {fit['intercept_us']:>8.1f}"
          f"   resid {fit['max_abs_resid_us']:>6.1f}")
print("\nThe 8c drift column bounds what this window can resolve, and a residual far above the\n"
      "other passes' says that pass did not hold still -- read no intercept off it.")
PY

log "\nlog: $LOG"
