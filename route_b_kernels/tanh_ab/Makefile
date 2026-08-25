##===- Makefile -----------------------------------------------------------===##
#
# tanh_ab A/B device probe: hw aie::exp2<bfloat16> vs the relpos_mha sw poly.
# Build: `source scripts/iron_env.sh` then `make NPU2=1` (xclbin + insts.bin) from
# the SYNCED sandbox copy (mlir-aie/programming_examples/ml/tanh_ab), same as
# relpos_mha/Makefile. Not built via sync_kernels.sh (this is a one-off probe, not
# a tracked route_b module); copy tanh_ab.cc/-> aie_kernels/aie2p/, this Makefile
# and tanh_ab_iron.py -> programming_examples/ml/tanh_ab/ by hand.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
# Input/output vector length, baked as -DTANHAB_N. Must match tanh_ab_iron.py's N.
# Capped by the aie2p BD length limit: 5*N f32 words <= 16383 -> N <= 3276.
N ?= 512
NOSW ?= 0
# Core stack reservation, bytes. aie.core's stack_size defaults to 0x400 and the
# generated ld.script puts tout_buff_0 at the very next address, so a frame over
# 1024 bytes silently overwrites the output buffer -- measured here as `raw`
# passthrough corruption proportional to the overflow. Empty = leave the default.
STACK ?=

# Extra clang flags, appended verbatim. Exists for the aie2p subregister-liveness
# wrong-code defect that silently drops the second aie::tanh's result -- see
# route_b_kernels/probes/tanh_subreg_clobber.ll. Either of these makes it correct:
#   EXTRA_CFLAGS="-mllvm -enable-subreg-liveness=false"
#   EXTRA_CFLAGS="-mllvm -join-liveintervals=false"
EXTRA_CFLAGS ?=

targetname = tanh_ab
aie_py_src = tanh_ab_iron.py

ifeq ($(devicename),npu2)
VPATH :=${srcdir}/../../../aie_kernels/aie2p
else
VPATH :=${srcdir}/../../../aie_kernels/aie2
endif

.PHONY all: build/${targetname}.mlir build/final.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${EXTRA_CFLAGS} -DTANHAB_N=${N} -DTANHAB_NOSW=${NOSW} -c $< -o ${@F}
else
	echo "Only npu2 supported for this example"
endif

build/kernels.a: build/tanh_ab.cc.o
	ar rvs $@ $+

build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@
ifneq (${STACK},)
	python3 -c "import sys; p=sys.argv[1]; s=open(p).read(); k='      aie.end\n    }\n    aie.runtime_sequence'; assert s.count(k)==1, 'core-op close not found uniquely'; open(p,'w').write(s.replace(k, '      aie.end\n    } {stack_size = ${STACK} : i32}\n    aie.runtime_sequence'))" $@
	@grep -q 'stack_size = ${STACK}' $@ && echo "[stack] reserved ${STACK} bytes"
endif

build/final.xclbin: build/${targetname}.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc --get-xclbin --get-npu-insts \
		    --xclbin-name=${@F} \
		    \
		    --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
