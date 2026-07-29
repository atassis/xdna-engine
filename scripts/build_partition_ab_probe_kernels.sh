#!/usr/bin/env bash
# Build the narrow-partition xclbins that `partition_ab_probe` A/Bs (CPU-only; no NPU required).
#
# WHY A SEPARATE SCRIPT. These four are DIAGNOSTIC inputs, not encoder inputs -- nothing in
# npu-parakeet loads them, only rust/npu-probes/src/bin/partition_ab_probe.rs does. Keeping them out
# of build_parakeet_modal_kernels.sh means the encoder rebuild (which every device gate pays for)
# does not carry probe weight, while these stay reproducible instead of being hand-built one-offs.
#
# WHAT THE PROBE ASKS. `ctxln` is the one encoder brick whose shim-DMA footprint fits in 4 columns.
# Built at width 4 it should be able to co-reside with another 4-column design (4+4 = 8), so an
# alternating dispatch would cost no array reprogram. It cannot, because aiecc emits ONE allowed
# start column and both designs demand it. The `W` variants widen start_columns to every legal
# offset so the driver's column solver can place them disjointly -- that pair is the co-residency arm.
#
#   _p4c   column_width=4, start_columns=["1"]                  (as aiecc emits it)
#   _p4cW  column_width=4, start_columns=["0".."4"]             (widened, post-processed)
#
# The AIE program is identical across the pair -- the instruction streams are byte-identical, only
# the partition claim differs -- so the W step is a pure metadata rewrite (scripts/widen_start_columns.py).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
source scripts/kernel_sandbox.sh

LNML=mlir-aie/programming_examples/ml/layernorm
LNDIR=artifacts/parakeet/ln
mkdir -p "$LNDIR"

bash scripts/sync_kernels.sh

echo "== PARTITION A/B: ctxln + cast at column_width=4 =="
make -C $LNML -f Makefile.ctxln NPU2=1 rows=512 cols=1024 partcols=4 build/final_ctxln_512x1024_p4c.xclbin
make -C $LNML -f Makefile.cast  NPU2=1 rows=512 cols=1024 partcols=4 build/final_cast_512x1024_p4c.xclbin

for tag in ctxln_512x1024_p4c cast_512x1024_p4c; do
  cp "$LNML/build/final_${tag}.xclbin" "$LNML/build/insts_${tag}.txt" "$LNDIR/"
done

# The widened (_p4cW) arm is BEST-EFFORT: it needs an xclbinutil that can re-encode AIE_PARTITION
# from JSON, which XRT 2.21.75 (the system one here) cannot -- see widen_start_columns.py for the
# measurement. The narrow (_p4c) pair above always builds; if the widen step is unavailable the
# shipped _p4cW copies are left in place and the probe's last arm uses those.
echo "== PARTITION A/B: widened start_columns (the co-residency arm; best-effort) =="
widen_ok=1
for tag in ctxln_512x1024 cast_512x1024; do
  if python3 scripts/widen_start_columns.py \
       --in  "$LNDIR/final_${tag}_p4c.xclbin" \
       --out "$LNDIR/final_${tag}_p4cW.xclbin.new" \
       --device-columns 8; then
    mv "$LNDIR/final_${tag}_p4cW.xclbin.new" "$LNDIR/final_${tag}_p4cW.xclbin"
    # Same program, so the narrow build's instruction stream is the W build's too.
    cp "$LNDIR/insts_${tag}_p4c.txt" "$LNDIR/insts_${tag}_p4cW.txt"
  else
    widen_ok=0
    rm -f "$LNDIR/final_${tag}_p4cW.xclbin.new"
  fi
done
if [ "$widen_ok" = 0 ]; then
  echo "NOTE: _p4cW not rebuilt (no AIE_PARTITION JSON writer). Shipped copies retained;"
  echo "      the _p4c pair IS reproducible. This is the one artifact class still hand-made."
fi

echo "Built + staged partition-A/B probe xclbins -> $LNDIR"
ls -la "$LNDIR"/*_p4c*.xclbin "$LNDIR"/*_p4c*.txt
