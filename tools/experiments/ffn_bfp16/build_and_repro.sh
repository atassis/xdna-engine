#!/usr/bin/env bash
# A2: Parakeet FFN/proj GEMM bfp16 variant -- build-check + llvm-aie #847 repro.
#
# (a) Confirms the mmul whole_array path (IRON aie2p/mm.cc) compiles at Parakeet
#     FFN tile dims, baseline bf16 (4x8x8) AND bfp16-emulated (8x8x8).
# (b) Reproduces the OPEN Peano miscompile llvm-aie #847:
#       -O0  -> ICE   (selectG_AIE_LOAD_STORE / getReg() isReg() assert)
#       -O1/-O2 -> compiles, but #847 reports element-0-of-8 +offset MISCOMPILE
#                  (runtime-only; NOT verifiable on CPU -- needs NPU).
#
# CPU/build only. Env: source scripts/air_env.sh  (Peano clang, aie_api headers).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
source "$REPO/scripts/air_env.sh" 2>/dev/null
INC="$(dirname "$PEANO_INSTALL_DIR")/mlir_aie/include"
[ -d "$INC/aie_api" ] || INC=~/mlir-air/airenv/lib/python3.12/site-packages/mlir_aie/include
CXX="$PEANO_INSTALL_DIR/bin/clang++"
MM="$INC/aie_kernels/aie2p/mm.cc"
OUT="$(mktemp -d)"
BASE=(-std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes
  -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw
  -I "$INC")
# Parakeet FFN/proj tile dims: 1024 and 4096 both divisible by the 8x8x8 2x2-expanded
# kernel reqs (m%16==0, k%8==0, n%16==0). Per-core L1 tile = 64x64x64.
TILE=(-DDIM_M=64 -DDIM_K=64 -DDIM_N=64)

echo "### (a) mmul whole_array path -- baseline bf16 (4x8x8), -O2"
"$CXX" -O2 "${BASE[@]}" -DNDEBUG -Dbf16_bf16_ONLY "${TILE[@]}" -c "$MM" -o "$OUT/base.o" \
  && echo "   PASS: mm.cc bf16 mmul compiles for d=1024/ff=4096 tiles" || echo "   FAIL"

echo "### (a') same path, bfp16 emul (8x8x8), -O2  (production opt level)"
"$CXX" -O2 "${BASE[@]}" -DNDEBUG -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DROUND_CONV_EVEN \
  -Dbf16_bf16_ONLY "${TILE[@]}" -c "$MM" -o "$OUT/bfp16.o" \
  && echo "   COMPILES (but #847 says runtime element-0 +offset miscompile)" || echo "   FAIL"

echo "### (b) #847 ICE at -O0 -- mm.cc bfp16 8x8x8"
"$CXX" -O0 "${BASE[@]}" -DNDEBUG -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DROUND_CONV_EVEN \
  -Dbf16_bf16_ONLY "${TILE[@]}" -c "$MM" -o "$OUT/o0.o" 2>"$OUT/mm_o0.log"
echo "   rc=$? ; $(grep -oE "selectG_AIE_LOAD_STORE|isReg\(\)" "$OUT/mm_o0.log" | tr '\n' ' ')"

echo "### (b') MINIMAL repro -- a single aie::mmul<8,8,8> bf16, one .mul()"
for OPT in -O0 -O1 -O2; do
  "$CXX" $OPT "${BASE[@]}" -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
    -c "$HERE/repro_847_mmul888_bfp16.cc" -o "$OUT/min_$OPT.o" 2>"$OUT/min_$OPT.log"
  echo "   $OPT rc=$? $(grep -oE 'selectG_AIE_LOAD_STORE' "$OUT/min_$OPT.log" | head -1)"
done
echo "### control: minimal repro WITHOUT bfp16 flag, -O0  (isolates the flag as trigger)"
"$CXX" -O0 "${BASE[@]}" -c "$HERE/repro_847_mmul888_bfp16.cc" -o "$OUT/min_noemul.o" 2>/dev/null
echo "   rc=$?  (expect 0 -> bfp16 emul flag is the sole trigger)"
rm -rf "$OUT"
