#!/usr/bin/env bash
# Build ALL NPU xclbins the npu_asr encoder needs (CPU-only; no NPU required).
# Idempotent. Run after scripts/setup_route_b.sh. The kernel-object stale-trap
# (objects named by tile size, not dtype) is handled with explicit `rm` before
# bf16 builds. See docs/08.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
bash scripts/sync_kernels.sh   # copy canonical custom kernels/designs into the build sandbox

PE=mlir-aie/programming_examples
MM=$PE/basic/matrix_multiplication/single_core
MMW=$PE/basic/matrix_multiplication/whole_array

echo "== dwconv1d k=5 =="
make -C $PE/ml/dwconv1d NPU2=1

echo "== layernorm [400x768] =="
make -C $PE/ml/layernorm NPU2=1 rows=400 cols=768

echo "== silu (two sizes; length/cols/chans must be a multiple of 1024) =="
rm -f $PE/ml/silu/build/*.o $PE/ml/silu/build/kernels.a
make -C $PE/ml/silu NPU2=1 length=307200 cols=4 chans=1
cp $PE/ml/silu/build/final.xclbin $PE/ml/silu/build/final_307200.xclbin
cp $PE/ml/silu/build/insts.bin     $PE/ml/silu/build/insts_307200.bin
make -C $PE/ml/silu NPU2=1 length=1228800 cols=4 chans=2
cp $PE/ml/silu/build/final.xclbin $PE/ml/silu/build/final_1228800.xclbin
cp $PE/ml/silu/build/insts.bin     $PE/ml/silu/build/insts_1228800.bin

echo "== matmul bf16->f32 (rm stale dtype-agnostic objects first) =="
rm -f $MM/build/mm_*.o $MMW/build/mm_*.o
for KN in 768x768 3072x768 768x1536; do K=${KN%x*}; N=${KN#*x}
  make -C $MM NPU2=1 M=512 K=$K N=$N dtype_in=bf16 dtype_out=f32           # single_core (1 col)
done
# --- V2 encoder whole_array kernels (default = FAST BFP16_IREE, tile 64x32x96) ---
# The shipped V2 encoder (two_ctx) runs the WHOLE encoder on ONE resident 768x3072 xclbin via per-N
# instruction streams (768/1536/3072). The fast kernel/dataflow-2x BFP16_IREE microkernel gives ~2x
# (n=96 chosen so the resident-stream reuse holds across all served widths). Needs the mlir-aie patch
# applied (setup_route_b.sh does this: BFP16_IREE microkernel + bfp16_iree flag + WA_C_DEPTH).
rm -f $MMW/build/mm_64x32x96.o
for N in 3072 1536 768; do
  rm -f $MMW/build/aie_512x768x${N}_64x32x96_8c.mlir
  WA_C_DEPTH=1 make -C $MMW NPU2=1 M=512 K=768 N=$N m=64 k=32 n=96 \
     dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 \
     emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 \
     build/final_512x768x${N}_64x32x96_8c.xclbin
done
# native bf16 32x32x32 (precise, rel<0.08) — selectable via NPU_PRECISION=native.
rm -f $MMW/build/mm_32x32x32.o
for KN in 768x768 3072x768 768x1536 768x3072; do K=${KN%x*}; N=${KN#*x}
  make -C $MMW NPU2=1 M=512 K=$K N=$N dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1  # whole_array (8 col)
done
# int8 64x64x96 (integer-exact kernel, ~3.6x, half weight bytes) — selectable via NPU_PRECISION=int8
# (RU GigaAM WER-validated: G4 = 9.2%, == precise). Native 8x8x8 path: NO bfp16 flags.
rm -f $MMW/build/mm_64x64x96.o
for N in 3072 1536 768; do
  rm -f $MMW/build/aie_512x768x${N}_64x64x96_8c.mlir
  WA_C_DEPTH=1 make -C $MMW NPU2=1 M=512 K=768 N=$N m=64 k=64 n=96 \
     dtype_in=i8 dtype_out=i32 n_aie_cols=8 use_iron=1 \
     build/final_512x768x${N}_64x64x96_8c.xclbin
done

echo "== FUSION xclbins (docs/10): whole_array matmul+epilogue + softmax-400 =="
WAF=$MMW   # whole_array build dir
rm -f $MMW/build/mm_*.o $MMW/build/mm_silu_epilogue_*.o
# linear1 silu(A@B+bias): Kaug=800,N=3072 ; linear2/proj/pw bias: Kaug per K+32
# NOTE: -C changes dir BEFORE -f is resolved, so the makefile path must be relative to $MMW
# (i.e. `-C $MMW -f Makefile.silu`, NOT `-f $MMW/Makefile.silu -C $MMW` which doubles the path).
make -C $MMW -f Makefile.silu NPU2=1 M=512 K=800  N=3072 n_aie_cols=8          build/final_512x800x3072_32x32x32_8c_silu.xclbin
make -C $MMW -f Makefile.silu NPU2=1 M=512 K=3104 N=768  n_aie_cols=8 no_silu=1 build/final_512x3104x768_32x32x32_8c_bias.xclbin
make -C $MMW -f Makefile.silu NPU2=1 M=512 K=800  N=1536 n_aie_cols=8 no_silu=1 build/final_512x800x1536_32x32x32_8c_bias.xclbin
make -C $MMW -f Makefile.silu NPU2=1 M=512 K=800  N=768  n_aie_cols=8 no_silu=1 build/final_512x800x768_32x32x32_8c_bias.xclbin
make -C $PE/ml/softmax400 NPU2=1 build/final.xclbin   # softmax-400 (pad->416)

echo "All encoder + fusion xclbins built."
echo "Verify host-orchestrated: .venv-iron/bin/python scripts/verify_encoder.py --backend npu --accurate"
echo "Verify FUSED encoder:     .venv-iron/bin/python scripts/verify_fused_encoder.py --blocks 16"
