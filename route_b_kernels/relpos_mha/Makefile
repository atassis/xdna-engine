##===- Makefile -----------------------------------------------------------===##
#
# Parakeet rel-pos MHA scores->softmax STANDALONE step-1 kernel. Build:
# `source scripts/iron_env.sh` then `make NPU2=1 T=32` (xclbin + insts.bin).
# Driven on the NPU via pyxrt (scripts/run_npu_relpos_scores.py); the C++ host
# harness is unbuildable on Arch. T MUST match both the generator (-T) and the
# kernel (-DRELPOS_T); the recipes below thread the single T= var into both.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
T ?= 32

# STEP selects which entry point / generator this xclbin drives:
#   STEP=1 (default) -- relpos_scores_softmax : host-fed AC + BD -> probs (no matmul).
#   STEP=2           -- relpos_ac_scores_softmax : on-chip AC=qu@k^T matmul feeding
#                       the resident-in-L1 f32 score tile -> scores->softmax -> probs.
# Both build the SAME build/kernels.a from relpos_mha.cc (which holds both entries);
# only the generated .mlir / xclbin differ. `make clean` when switching STEP (the
# xclbin/insts names are fixed, so a stale build/ from the other step would linger).
STEP ?= 1
ifeq ($(STEP),2)
targetname = relpos_ac_scores_softmax
aie_py_src = relpos_ac_scores_softmax_iron.py
else
targetname = relpos_scores_softmax
aie_py_src = relpos_scores_softmax_iron.py
endif

ifeq ($(devicename),npu2)
VPATH :=${srcdir}/../../../aie_kernels/aie2p
else
VPATH :=${srcdir}/../../../aie_kernels/aie2
endif

.PHONY all: build/${targetname}.mlir build/final.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DRELPOS_T=${T} -c $< -o ${@F}
else
	echo "Only npu2 supported for this example"
endif

build/kernels.a: build/relpos_mha.cc.o
	ar rvs $@ $+

build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -T ${T} > $@

build/final.xclbin: build/${targetname}.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		    --xclbin-name=${@F} \
		    --no-xchesscc --no-xbridge \
		    --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
