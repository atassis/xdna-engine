#!/usr/bin/env bash
# Sourceable env recipe for the vendored mlir-air toolchain (the dedicated airenv 3.12 venv).
# Reconstructs the env the 2026-06 decode-attention spike used (the original /tmp/air-env.sh was lost).
# Gives: aircc (AIR compiler driver), aie-opt, the `air`+`aie` python dialects, Peano clang++ (PEANO).
#
#   source scripts/air_env.sh
#   python3 $AIR/programming_examples/attention_decode/attn_decode_npu2.py -p --nkv 12 --n 64 --seq-len 448 ...
#
# Proven (2026-06-19): attn_decode_npu2.py LOWERS to AIR MLIR at Whisper self-attn shapes
# (--nkv 12 --n 64 --seq-len 448 --k 768 = 12 heads x 448 x 64). See log/2026-06/p0b-resident-attn-spike.md.
AIR=~/mlir-air
export PEANO_INSTALL_DIR="$AIR/airenv/lib/python3.12/site-packages/llvm-aie"
export PYTHONPATH="$AIR/install/python:$AIR/airenv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$AIR/install/bin:$AIR/airenv/bin:$PATH"
export AIR
# Use $AIR/airenv/bin/python3 (3.12) for air/aie imports.
echo "[air_env] airenv ready: aircc=$(command -v aircc), aie-opt=$(command -v aie-opt), PEANO=$PEANO_INSTALL_DIR"
