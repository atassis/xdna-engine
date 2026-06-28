##===- Makefile -----------------------------------------------------------===##
#
# Depthwise conv1d (k=9, Parakeet ConvModule) example. Build: `source scripts/iron_env.sh` then
# `make NPU2=1` (xclbin + insts.bin). Driven on the NPU via pyxrt
# (scripts/run_npu_dwconv1d.py); the C++ host harness is unbuildable on Arch.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = dwconv1d
cols ?= 4

ifeq ($(devicename),npu2)
VPATH :=${srcdir}/../../../aie_kernels/aie2p
else
VPATH :=${srcdir}/../../../aie_kernels/aie2
endif

aie_py_src=dwconv1d.py

.PHONY all: build/${targetname}.mlir build/final.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Only npu2 supported for this example"
endif

build/kernels.a: build/${targetname}.cc.o
	ar rvs $@ $+

build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -co ${cols} > $@

build/final.xclbin: build/${targetname}.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		    --xclbin-name=${@F} \
		    --no-xchesscc --no-xbridge \
		    --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
