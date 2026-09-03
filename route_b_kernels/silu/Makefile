##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common
include ${srcdir}/toolchain_stamp.mk

aie2_runtime_dir = ${MLIR_AIE_DIR}/aie_runtime_lib/AIE2

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = silu
length ?= 16384
cols ?= 4
chans ?= 2

ifeq ($(devicename),npu2)
VPATH :=${srcdir}/../../../aie_kernels/aie2p
else
VPATH :=${srcdir}/../../../aie_kernels/aie2
endif

aie_py_src=silu.py

# `.PHONY all: <prereqs>` declares the target `.PHONY` WITH these as PREREQUISITES, and .PHONY's
# prerequisites are marked phony -- so build/final.xclbin was phony and rebuilt unconditionally on
# every invocation, forever. One missing colon; found by census on 2026-09-03 in 5 Makefiles here.
all: build/${targetname}.mlir build/final.xclbin
.PHONY: all

ifeq ($(devicename),npu)
build/lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -c $< -o ${@F}
endif

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -I${aie2_runtime_dir} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

ifeq ($(devicename),npu)
build/kernels.a: build/${targetname}.cc.o build/lut_based_ops.o
	ar rvs $@ $+
else ifeq ($(devicename),npu2)
build/kernels.a: build/${targetname}.cc.o
	ar rvs $@ $+
else
	echo "Device type not supported"
endif

build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -l ${length} -co ${cols} -ch ${chans} > $@

build/final.xclbin: build/${targetname}.mlir build/kernels.a
	mkdir -p ${@D}
	# aiecc flag set updated 2026-09-01: upstream renamed --aie-generate-xclbin/--aie-generate-npu-insts
	# to --get-xclbin/--get-npu-insts and INVERTED the toolchain selection -- Peano/lld is now the
	# default, and --xchesscc/--xbridge opt IN to Chess, so the old --no-xchesscc/--no-xbridge are
	# rejected outright ("Unknown command line argument '--no-xbridge'"). Matches ml/dwconv1d.
	cd ${@D} && aiecc --get-xclbin --get-npu-insts \
	            --xclbin-name=${@F} \
	            --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -l ${length}

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
