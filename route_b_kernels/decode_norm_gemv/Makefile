##===- Makefile: fused norm+GEMV decode primitive -------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Builds norm_gemv_iron.py, which emits the same f32-out whole_array GEMV MLIR as the plain
# decode GEMV (the norm fold is separable and runs host-side; the kind is carried in the
# filename for CtxDecode-loader symmetry -- see the generator header).
#
# WHY THIS EXISTS: there was no Makefile and no build/ here, so the tile-shape question the
# traffic model raises -- does dispatch time track STREAMED bytes at the decode shape? -- could
# not be answered without hand-driving aiecc. Streamed traffic is
# (M/m)(N/n)(K/k)*(m*k + k*n)*2, so growing m is the only software lever on it at fixed M,K,N.
#
# TILE-SHAPE RANGE, and it is narrower than the frontmatter arithmetic suggests. Three
# constraints stack at M=64:
#   mm.cc microkernel   m % (2*r) == 0, r=4          => m >= 8
#   iron tiler          M % (m * n_aie_rows) == 0     => m in {1,2,4,8,16} at n_aie_rows=4
#   C ping-pong         M // m // n_aie_rows >= 2     => m <= 8
# so m=8 is the ONLY legal tile at M=64, and m=16 needs M=128. `make sweep` builds the range
# and reports which shapes actually produce an xclbin rather than assuming the middle one does.
#
# Invoke:
#   source scripts/iron_env.sh
#   make -f route_b_kernels/decode_norm_gemv/Makefile NPU2=1            # default m=8
#   make -f ... NPU2=1 M=128 m=16                                       # the m=16 comparison
#   make -f ... sweep                                                   # legality sweep
#
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
ifneq ($(devicename),npu2)
$(error this design is npu2/AIE2P only -- pass NPU2=1)
endif

# Fail loud rather than resolve kernels from a drifted submodule sandbox: the shipped resident
# matmul once compiled a pre-#3442 mm.cc while toolchain.lock said otherwise, for 5 days.
ifeq ($(MLIR_AIE_INSTANCE),)
$(error MLIR_AIE_INSTANCE is unset -- source scripts/iron_env.sh before building)
endif
kernels_dir = ${MLIR_AIE_INSTANCE}/src/aie_kernels/aie2p
MLIR_AIE_DIR := $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")

PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses \
  -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG \
  -I ${MLIR_AIE_DIR}/include -D__AIE_API_AIE_ADF_HPP__
KERNEL_CC = ${PEANO_INSTALL_DIR}/bin/clang++
AIECC := $(if $(AIECC_PATH),$(AIECC_PATH),aiecc)

# GEMM dims. M=64 is the smallest legal whole_array M at m=8 (see the range note above).
M ?= 64
K ?= 768
N ?= 768
m ?= 8
k ?= 32
n ?= 32
n_aie_cols ?= 8
norm ?= ln

suffix   := ${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c_${norm}
mlir     := build/aie_${suffix}.mlir
xclbin   := build/final_${suffix}.xclbin
insts    := build/insts_${suffix}.txt

# Single-buffer the L1 C tile, as the whole_array family does; the generator reads it from env.
WA_C_DEPTH ?= 1
export WA_C_DEPTH

# mm.cc names its symbols by the combo and defaults DIM_* to 64, so the tile must be passed.
MM_DEFINES = -Dbf16_f32_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -DVECTORIZED_ONLY

.PHONY: all clean sweep help
all: ${xclbin}

build/mm_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${PEANOWRAP2P_FLAGS} ${MM_DEFINES} -c $< -o ${@F}

build/kernels_${m}x${k}x${n}.a: build/mm_${m}x${k}x${n}.o
	ar rvs $@ $+

${mlir}: ${srcdir}/norm_gemv_iron.py
	mkdir -p ${@D}
	python3 $< --dev ${devicename} -M ${M} -K ${K} -N ${N} -m ${m} -k ${k} -n ${n} \
	  --n-aie-cols ${n_aie_cols} --norm ${norm} > $@

# Grouped co-target: one aiecc call yields BOTH the xclbin and the .txt insts, so a direct
# `make .../insts_*.txt` resolves instead of treating it as an untracked side effect.
${xclbin} ${insts} &: ${mlir} build/kernels_${m}x${k}x${n}.a
	mkdir -p ${@D}
	cd ${@D} && ${AIECC} --alloc-scheme=basic-sequential --get-xclbin \
	  --xclbin-name=$(notdir ${xclbin}) --no-xchesscc --no-xbridge \
	  --peano ${PEANO_INSTALL_DIR} \
	  --get-npu-insts --npu-insts-name=$(notdir ${insts}) $(notdir ${mlir})

# Which tile shapes actually build. Reports per-shape legality instead of asserting a range.
sweep:
	@for mm in 1 2 4 8 16; do \
	  printf '  m=%-3s ' $$mm; \
	  $(MAKE) -s -f $(firstword $(MAKEFILE_LIST)) NPU2=1 M=${M} K=${K} N=${N} m=$$mm k=${k} n=${n} \
	    >/dev/null 2>&1 && echo "BUILDS" || echo "rejected"; \
	done

clean:
	rm -rf build

help:
	@echo "norm+GEMV decode primitive (whole_array, bf16 in / f32 out)"
	@echo "  make NPU2=1 [M=64 K=768 N=768 m=8 k=32 n=32 n_aie_cols=8 norm=ln]"
	@echo "  make sweep      which tile shapes are legal at this M"
	@echo "  make clean"
