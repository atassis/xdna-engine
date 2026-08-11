##===- Makefile -----------------------------------------------------------===##
#
# exp2_ab A/B device probe: hw aie::exp2<bfloat16> vs the relpos_mha sw poly.
# Build: `source scripts/iron_env.sh` then `make NPU2=1` (xclbin + insts.bin) from
# the SYNCED sandbox copy (mlir-aie/programming_examples/ml/exp2_ab), same as
# relpos_mha/Makefile. Not built via sync_kernels.sh (this is a one-off probe, not
# a tracked route_b module); copy exp2_ab.cc/-> aie_kernels/aie2p/, this Makefile
# and exp2_ab_iron.py -> programming_examples/ml/exp2_ab/ by hand.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
# Input/output vector length, baked as -DEXP2AB_N. Must match exp2_ab_iron.py's N.
# Capped by the aie2p BD length limit: 4*N f32 words <= 16383 -> N <= 4095.
N ?= 1024

targetname = exp2_ab
aie_py_src = exp2_ab_iron.py

ifeq ($(devicename),npu2)
VPATH :=${srcdir}/../../../aie_kernels/aie2p
else
VPATH :=${srcdir}/../../../aie_kernels/aie2
endif

.PHONY all: build/${targetname}.mlir build/final.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DEXP2AB_N=${N} -c $< -o ${@F}
else
	echo "Only npu2 supported for this example"
endif

build/kernels.a: build/exp2_ab.cc.o
	ar rvs $@ $+

build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@

build/final.xclbin: build/${targetname}.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc --get-xclbin --get-npu-insts \
		    --xclbin-name=${@F} \
		    \
		    --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
