#!/usr/bin/env bash
# Build the thin-M GEMV xclbin the on-NPU decoder needs (CPU-only; no NPU required).
#
# Decode is M=1 (single token). The resident encoder GEMM is tuned for M=512. This builds the
# SMALLEST legal M the whole_array design supports and pads the single query row up to it on the host.
#
# Smallest-M derivation (whole_array_iron.py + aie_kernels/aie2p/mm.cc, npu2 native bf16, r=4):
#   microkernel  static_assert(m % (2*r) == 0)  with r=4  =>  smallest tile m = 8
#   iron tiler   M % (m * n_aie_rows) == 0       with n_aie_rows = 4
#   iron tiler   C-output ping-pong needs >=2 C row-tile-groups:
#                M // m // n_aie_rows >= tb_n_rows (=2)  =>  M >= 2 * m * n_aie_rows
#   => with m=8: smallest legal M = 2 * 8 * 4 = 64  (M=16/32 fail; m<8 fails the microkernel assert)
#   K=768 % k(=32) == 0 ok ;  N % (n * n_aie_cols) == 0  =>  768 % (32*8=256) == 0 ok
#
# So: M=64, m=8, k=32, n=32, n_aie_cols=8, dtype_in=bf16, dtype_out=f32 (native bf16, NO bfp16 emul).
# The single decode query (M=1) is zero-padded up to 64 rows on the host; row 0 holds the result.
# Output xclbin + insts land in the whole_array build dir with shape-encoded names:
#   final_64x${K}x${N}_8x32x32_8c.xclbin   insts_64x${K}x${N}_8x32x32_8c.txt
#
# Usage:  scripts/build_decode_kernels.sh [K] [N]   (defaults K=768 N=768)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
bash scripts/sync_kernels.sh >/dev/null   # copy whole_array_iron.py + Makefile.resident into the sandbox

K="${1:-768}"
N="${2:-768}"
M=64          # smallest legal M for the whole_array 8-col native-bf16 design (see header)
m=8; k=32; n=32; COLS=8

MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
SUFFIX="${M}x${K}x${N}_${m}x${k}x${n}_${COLS}c"

echo "== decode GEMV: M=${M} K=${K} N=${N} (tile ${m}x${k}x${n}, ${COLS} cols, native bf16->f32) =="
# Stale-object trap: the per-tile kernel object is named by tile size only (not by M/K/N/dtype),
# so drop it before building to avoid picking up an object from a different dtype/define set.
rm -f "$MMW/build/mm_${m}x${k}x${n}.o"
rm -f "$MMW/build/aie_${SUFFIX}.mlir"

# Drive the MLIR-emitting whole_array_iron.py via Makefile.resident: the rewritten upstream
# makefile-common names insts .bin (engine reads insts_*.txt) and lacks the WA_C_DEPTH knob.
make -f Makefile.resident -C "$MMW" NPU2=1 M="$M" K="$K" N="$N" m="$m" k="$k" n="$n" \
  dtype_in=bf16 dtype_out=f32 n_aie_cols="$COLS" use_iron=1 \
  "build/final_${SUFFIX}.xclbin"

echo "Built: $MMW/build/final_${SUFFIX}.xclbin"
echo "       $MMW/build/insts_${SUFFIX}.txt"
