##===- Makefile: trivial 2-stage conveyor prototype ----------------------===##
# Build:  source scripts/iron_env.sh ; make NPU2=1   ->  build/final.xclbin + insts.bin
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
# VARIANT=stage (trivial +1/*2 mechanism proof) | attn (real scores->softmax)
VARIANT ?= stage
ifeq ($(VARIANT),qt)
targetname = conveyor_qt
aie_py_src = conveyor_qt_iron.py
kernel_obj = build/conveyor_stage.cc.o
else ifeq ($(VARIANT),attn)
targetname = conveyor_attn
aie_py_src = conveyor_attn_iron.py
kernel_obj = build/conveyor_attn.cc.o
else
targetname = conveyor
aie_py_src = conveyor_iron.py
kernel_obj = build/conveyor_stage.cc.o
endif

.PHONY: all
all: build/final.xclbin

# The two kernels are tracked in aie_kernels/<kebab>/<snake>.cc and sync_kernels.sh lays them
# into the sandbox's shared aie2p kernel dir, like every other design's kernels. They used to sit
# beside this Makefile, which is why the rule read `%.cc` from the current directory.
# (NOT ${lib_kernels_dir}: that is defined by whole_array_fused/design_override.mk for its own
# tree, not by makefile-common, so it expands empty here.)
kernels_aie2p := ${srcdir}/../../../aie_kernels/aie2p
build/conveyor_attn.cc.o:  ${kernels_aie2p}/conveyor_attn.cc
build/conveyor_stage.cc.o: ${kernels_aie2p}/conveyor_stage.cc

build/%.cc.o:
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${KFLAGS} -c $< -o $@
else
	echo "Only npu2 supported for this prototype"
endif

build/kernels.a: ${kernel_obj}
	ar rvs $@ $+

GEN_EXTRA = $(if $(filter 1,$(MONO)),--mono,) $(if $(TRIVIAL),--trivial $(TRIVIAL),) $(if $(filter 1,$(RELPOS)),--relpos,) $(if $(filter 1,$(BDON)),--relpos-bd-onchip,) $(if $(filter 1,$(MASK)),--tactive-mask,) $(if $(filter 1,$(PRES)),--p-resident,) $(if $(filter 1,$(SIO)),--stream-io,) $(if $(filter 1,$(MMUL)),--mmul,) $(if $(TRACE),-t $(TRACE) --trace-worker $(TW),)

build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} ${GEN_EXTRA} > $@

build/final.xclbin: build/${targetname}.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc --get-xclbin --get-npu-insts \
	    --xclbin-name=final.xclbin \
	    --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
