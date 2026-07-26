#!/usr/bin/env bash
#
# npu_xrt_dynseq_harness.sh -- build + run the dyn-seq dynamic whole-array i16
# GEMM example (mlir-aie PR #3368, "runtime M/K/N, multi-column") on an XDNA2 /
# Strix NPU against SYSTEM XRT.
#
# Why this exists: our normal toolchain build disables XRT (we drive the device
# through pyxrt), so the upstream C++ npu-xrt test host cannot be built by the
# in-tree lit flow here. This harness compiles that host directly against the
# system XRT headers/libs and wires up the dyn-seq lowering by hand, so we can
#   (A) independently confirm the example on our own silicon, and
#   (B) build the artifacts a throughput benchmark times.
#
# It needs a mlir-aie toolchain INSTANCE that carries the dynamic-BD-pool pass
# (--aie-lower-dynamic-bd-pool, from the #3358 stack). A pure-upstream pin from
# before that merge does NOT have it; point INSTANCE at an instance built from a
# branch/commit that does.
#
# Recipe (mirrors test/npu-xrt/matmul_whole_array_dynamic/run.lit):
#   mm.cc --Peano--> mm.o                  (compute kernel, built once, col-agnostic)
#   example.py --emit MLIR--> normalize kernel symbol to mm.o
#   aiecc --aie-generate-xclbin --aie-generate-input-with-addresses
#   aie-opt <dyn-seq pass list> --> lowered.mlir
#   aie-translate --aie-npu-to-cpp --> gen.h   (the runtime C++ TXN builder)
#   host compiler: test.cpp + test_utils.cpp + gen.h + system XRT --> exe
#   exe M K N  (on device) --> "PASS!"
#
# Gotchas baked in (each cost real time to find):
#   - aie-translate MUST be the source-built one (has --aie-npu-to-cpp); the
#     vendored wheel binary lacks the EmitC TXN target.
#   - aie_api headers come from the instance's build/include, not the source
#     tree's (often-empty) third_party/aie_api.
#   - gen.h #includes aie/Runtime/TxnEncoding.h, so the host compile needs
#     -I <mlir-aie>/include.
#   - kernel tile dims are compile-time -D defines (DIM_M/DIM_K/DIM_N) + the
#     dtype-combo flag (i16_i16_ONLY).
#
# Usage:
#   npu_xrt_dynseq_harness.sh kernel                 # build mm.o once
#   npu_xrt_dynseq_harness.sh build <cols>           # build xclbin + gen.h + exe for a column count
#   npu_xrt_dynseq_harness.sh run   <cols> <M> <K> <N>   # run one shape on device
#   npu_xrt_dynseq_harness.sh sweep                  # leg A: build + run cols 1/2/4 over the PR's shapes
#
# Config (env-overridable; defaults are for this workspace):
#   MLIR_AIE_SRC  mlir-aie tree carrying the #3368 example+test
#   INSTANCE      prebuilt toolchain instance dir with the dyn-seq passes
#   PEANO         llvm-aie (Peano) install dir
#   VENV_PY       python interpreter with IRON deps
#   CXX           host C++ compiler (default: clang++)
#   WORK          scratch build dir
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"

MLIR_AIE_SRC="${MLIR_AIE_SRC:-$WORKSPACE/wt-dynseq}"
INSTANCE="${INSTANCE:-$WORKSPACE/.cache/instances/303c90318dd6}"
PEANO="${PEANO:-$WORKSPACE/xdna-engine/.venv-iron/lib/python3.14/site-packages/llvm-aie}"
VENV_PY="${VENV_PY:-$WORKSPACE/xdna-engine/.venv-iron/bin/python}"
CXX="${CXX:-clang++}"
WORK="${WORK:-/tmp/dynseq-harness}"

TESTDIR="$MLIR_AIE_SRC/test/npu-xrt/matmul_whole_array_dynamic"
EXAMPLE="$MLIR_AIE_SRC/programming_examples/basic/matrix_multiplication/whole_array/whole_array_dynamic.py"
export PYTHONPATH="$INSTANCE/python${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$WORK"

# Kernel tile dims -- mirror kernels.mm() for m=64 k=64 n=32, i16->i16.
KDIM=( -DDIM_M=64 -DDIM_K=64 -DDIM_N=32 -Di16_i16_ONLY )

log() { printf '[harness] %s\n' "$*" >&2; }
die() { printf '[harness] ERROR: %s\n' "$*" >&2; exit 1; }

preflight() {
  [ -f "$EXAMPLE" ]                 || die "example not found: $EXAMPLE"
  [ -f "$TESTDIR/test.cpp" ]        || die "host not found: $TESTDIR/test.cpp"
  [ -x "$INSTANCE/bin/aie-opt" ]    || die "aie-opt not in instance: $INSTANCE/bin"
  "$INSTANCE/bin/aie-translate" --help 2>&1 | grep -q 'aie-npu-to-cpp' \
      || die "instance aie-translate lacks --aie-npu-to-cpp (rebuild from source)"
  [ -x "$PEANO/bin/clang++" ]       || die "Peano clang++ not found: $PEANO/bin"
}

build_kernel() {
  log "building compute kernel mm.o (col-agnostic, once)"
  "$PEANO/bin/clang++" --target=aie2p-none-unknown-elf -O2 \
    -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    "${KDIM[@]}" \
    -I"$MLIR_AIE_SRC/include" \
    -I"$INSTANCE/build/include" \
    -I"$MLIR_AIE_SRC/aie_kernels/aie2p" \
    -c "$MLIR_AIE_SRC/aie_kernels/aie2p/mm.cc" -o "$WORK/mm.o"
}

build_config() { # <cols>
  local cols="$1"
  local prj="$WORK/$cols.prj"
  [ -f "$WORK/mm.o" ] || build_kernel
  mkdir -p "$prj"

  log "cols=$cols: emit MLIR"
  "$VENV_PY" "$EXAMPLE" --dev npu2 -M 512 -K 512 -N 512 -m 64 -k 64 -n 32 \
    --dtype_in i16 --dtype_out i16 --n-aie-cols "$cols" -o "$WORK/$cols.mlir"

  # Normalize the content-hash kernel symbol to the stable mm.o built above.
  sed -E 's/matmul_i16_i16_[0-9a-f]+\.o/mm.o/g; s/@"[0-9a-f]+_matmul_i16_i16"/@matmul_i16_i16/g; s/@[0-9a-f]+_matmul_i16_i16/@matmul_i16_i16/g' \
    "$WORK/$cols.mlir" > "$WORK/$cols.norm.mlir"

  log "cols=$cols: aiecc -> xclbin + input_with_addresses"
  ( cd "$WORK" && PATH="$PEANO/bin:$INSTANCE/bin:$PATH" PEANO_INSTALL_DIR="$PEANO" \
    "$VENV_PY" "$INSTANCE/bin/aiecc.py" --no-xchesscc --no-xbridge --no-aiesim \
      --aie-generate-xclbin --aie-generate-input-with-addresses \
      --output-dir="$prj" --no-compile-host \
      --peano="$PEANO" \
      --xclbin-name="$WORK/$cols.xclbin" --tmpdir="$prj" "$WORK/$cols.norm.mlir" )

  log "cols=$cols: aie-opt (dyn-seq lowering) -> lowered.mlir"
  "$INSTANCE/bin/aie-opt" \
    --aie-materialize-bd-chains --aie-substitute-shim-dma-allocations \
    --aie-unroll-runtime-sequence-loops --canonicalize \
    --aie-lower-dynamic-bd-pool --canonicalize \
    --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
    --aie-dma-to-npu --aie-lower-set-lock \
    "$prj/input_with_addresses.mlir" -o "$WORK/$cols.lowered.mlir"

  log "cols=$cols: aie-translate -> gen.h (runtime C++ TXN builder)"
  "$INSTANCE/bin/aie-translate" --aie-npu-to-cpp "$WORK/$cols.lowered.mlir" > "$WORK/$cols.gen.h"

  log "cols=$cols: host compile test.cpp -> exe"
  "$CXX" "$TESTDIR/test.cpp" "$MLIR_AIE_SRC/runtime_lib/test_lib/test_utils.cpp" \
    -o "$WORK/$cols.exe" -std=c++23 -Wall \
    -DGEN_HDR="\"$WORK/$cols.gen.h\"" -DXCLBIN="std::string(\"$WORK/$cols.xclbin\")" \
    -I"$MLIR_AIE_SRC/include" \
    -I"$MLIR_AIE_SRC/programming_examples/basic/matrix_multiplication" \
    -I"$MLIR_AIE_SRC/runtime_lib/test_lib" \
    -I/usr/include/xrt -I/usr/include \
    -lxrt_coreutil -luuid
  log "cols=$cols: built $WORK/$cols.exe"
}

build_bench() { # <cols>  -- compile the timing host against a built cols config
  local cols="$1"
  [ -f "$WORK/$cols.xclbin" ] || build_config "$cols"
  log "cols=$cols: host compile bench.cpp -> bench exe"
  "$CXX" "$SCRIPT_DIR/npu_xrt_dynseq_bench.cpp" \
    "$MLIR_AIE_SRC/runtime_lib/test_lib/test_utils.cpp" \
    -o "$WORK/$cols.bench" -O2 -std=c++23 -Wall \
    -DGEN_HDR="\"$WORK/$cols.gen.h\"" -DXCLBIN="std::string(\"$WORK/$cols.xclbin\")" \
    -I"$MLIR_AIE_SRC/include" \
    -I"$MLIR_AIE_SRC/programming_examples/basic/matrix_multiplication" \
    -I"$MLIR_AIE_SRC/runtime_lib/test_lib" \
    -I/usr/include/xrt -I/usr/include \
    -lxrt_coreutil -luuid
}

bench() { # <cols> <M> <K> <N> [iters] [warmup]
  local cols="$1"; shift
  [ -x "$WORK/$cols.bench" ] || build_bench "$cols"
  log "cols=$cols bench $*"
  "$WORK/$cols.bench" "$@"
}

run_config() { # <cols> <M> <K> <N>
  local cols="$1" M="$2" K="$3" N="$4"
  [ -x "$WORK/$cols.exe" ] || build_config "$cols"
  log "cols=$cols run M=$M K=$K N=$N"
  "$WORK/$cols.exe" "$M" "$K" "$N"
}

sweep() { # leg A: the PR's run.lit matrix
  preflight
  build_kernel
  local fail=0
  for cols in 1 2 4; do
    build_config "$cols"
    run_config "$cols" 512 512 512 || fail=1
    if [ "$cols" = 4 ]; then run_config "$cols" 768 512 768 || fail=1
    else                     run_config "$cols" 256 512 1024 || fail=1; fi
  done
  [ "$fail" = 0 ] && log "SWEEP PASS (cols 1/2/4)" || die "SWEEP had failures"
}

cmd="${1:-sweep}"; shift || true
case "$cmd" in
  kernel) preflight; build_kernel ;;
  build)  preflight; build_config "$@" ;;
  run)    preflight; run_config "$@" ;;
  bench)  preflight; bench "$@" ;;
  sweep)  sweep ;;
  *)      die "unknown command: $cmd (kernel|build|run|sweep)" ;;
esac
