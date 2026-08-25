#!/usr/bin/env bash
# Device wrapper: re-score the GELU ladder on a REAL ENCODER's fc1 band (task k768-gelu-rail, (a)).
#
# THE QUESTION. The 2x2 ladder was measured on a random-normal accumulator -- A ~ N(0,1),
# B ~ N(0,1)/sqrt(K) -- and returned 1757x for the combined arm. That is the number a default flip
# would be argued on, and it is scored on a distribution no encoder produces. The one place the
# host model and the device already disagreed (sw-tanh alone: ~10x modelled on encoder bands,
# 2.18x measured) is exactly band-dependent, because the hardware tanh LUT is worst near +/-0.5.
#
# bge-base's own fc1 bands, measured host-side (artifacts/bge_fc1_bands/bands.json): median |x| is
# 2.3-3.7 and only 1-9% of the mass falls under 0.5, against 0.67 and 38% for the random-normal
# accumulator. So the ladder was scored where the LUT is worst, on a distribution the encoder
# spends 1-9% of its mass in. Whether that helps or hurts the prize is not predictable from the
# band alone -- rel-L2 is norm-relative, and GELU saturates to identity on the tail that dominates
# that norm -- so it is measured here rather than argued.
#
# THREE LAYERS, not one: L0/L6/L11 bracket the encoder, and L10/L11 are visibly a different regime
# from the rest (frac|x|<0.5 drops to 0.011-0.017). A single layer would not show whether the
# verdict is a property of the model or of one block.
#
# The random-normal arm is re-run LAST in the same session as the within-run control, because
# absolute ms moved 1.45x across the previous two passes on unpinned power mode; only ratios
# measured inside one session are comparable.
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort. The free-device check is fuser + pgrep, never
# `systemctl is-active`.
#
# QUIESCE THEN LOCK, in that order, and NOT under `npu_lock.sh run --`. npu_lock is deliberately
# non-destructive -- it defers when anything holds the device and never stops a service -- so
# wrapping this script in it deadlocks: it waits out xdna-engine, which only this script stops.
# (xdna-engine does not merely idle-hold. One /v1/embeddings request loads a resident hw_context
# and keeps it for idle_unload_s = 900, so the device reads free before a validation call and busy
# after one.) So quiesce first, then take npu_lock's OWN lock file directly, and the autonomous
# sessions still serialise against each other on the same flock.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/gelu_encoder_bands.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

. "$WT/scripts/_npu_services.sh"
trap 'npu_svc_start' EXIT
npu_svc_assert_units
npu_svc_stop

exec 9>"${NPU_LOCK:-/tmp/xdna2-npu-autonomous.lock}"
flock -w "${NPU_WAIT_S:-120}" 9 || { log "[lock] another session holds the NPU lock; DEFER"; exit 75; }
export CUDA_VISIBLE_DEVICES=""   # never touch the discrete GPU / eGPU

npu_svc_require_device_free

PY="$WT/.venv-iron/bin/python"
BANDS="$WT/artifacts/bge_fc1_bands"

# Operands first (device-free): a real forward pass, validated at cos 0.9998 against the served engine.
if [ ! -f "$BANDS/bands.json" ]; then
  log "[host] capturing bge-base fc1 operands"
  "$PY" route_b_kernels/probes/bge_ffn1_bands.py 2>&1 | tee -a "$LOG"
fi

for L in 0 6 11; do
  log "\n=== bge-base L$L ==========================================================="
  ENC_NPZ="$BANDS/bge_fc1_L$L.npz" "$PY" scripts/probe_gelu_f32_arms.py 2>&1 | tee -a "$LOG"
done

log "\n=== random-normal (within-session control, the arm the 1757x came from) ====="
"$PY" scripts/probe_gelu_f32_arms.py 2>&1 | tee -a "$LOG"

log "\nlog: $LOG"
