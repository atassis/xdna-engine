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
include ${srcdir}/toolchain_stamp.mk

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = dwconv1d
cols ?= 4

ifeq ($(devicename),npu2)
VPATH :=${srcdir}/../../../aie_kernels/aie2p
else
VPATH :=${srcdir}/../../../aie_kernels/aie2
endif

aie_py_src=dwconv1d.py

# `.PHONY all: <prereqs>` declares the target `.PHONY` WITH these as PREREQUISITES, and .PHONY's
# prerequisites are marked phony -- so build/final.xclbin was phony and rebuilt unconditionally on
# every invocation, forever. One missing colon; found by census on 2026-09-03 in 5 Makefiles here.
all: build/${targetname}.mlir build/final.xclbin
.PHONY: all

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
	cd ${@D} && aiecc --get-xclbin --get-npu-insts \
		    --xclbin-name=${@F} \
		    \
		    --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
