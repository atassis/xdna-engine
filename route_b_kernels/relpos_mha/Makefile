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
# Query-tile row count for STEP>=6 (row-tiled block); baked as -DRELPOS_TQ. Need
# not divide T (kernel handles a ragged final tile). Ignored by STEP<6.
TQ ?= 8
# Key-block row count for STEP>=7 (block-streamed k/p/V); baked as -DRELPOS_KB.
# T=172=4*43 -> KB=43 blocks k/V with no pad. Ignored by STEP<7.
KB ?= 43

# STEP selects which entry point / generator this xclbin drives:
#   STEP=1 (default) -- relpos_scores_softmax : host-fed AC + BD -> probs (no matmul).
#   STEP=2           -- relpos_ac_scores_softmax : on-chip AC=qu@k^T matmul feeding
#                       the resident-in-L1 f32 score tile -> scores->softmax -> probs.
#   STEP=3/4/5       -- both score matmuls resident / ctx AV matmul / full composition.
#   STEP=6           -- relpos_rowtiled : ROW-TILED, MemTile-staged block. quv[2T,DK]
#                       + kpv[(2T+P),DK] packed bf16 -> ctx[T,DK]; T query rows tiled
#                       by TQ (=RELPOS_TQ), GLOBAL-index rel_shift. Target T up to 172.
# All build the SAME build/kernels.a from relpos_mha.cc (which holds every entry);
# only the generated .mlir / xclbin differ. `make clean` when switching STEP (the
# xclbin/insts names are fixed, so a stale build/ from the other step would linger).
#   STEP=7           -- relpos_kpvstream : MONOLITHIC reference for the block-
#                       decomposed arithmetic (column-slice AC/BD + resident-f32
#                       ctx accumulate over V-blocks). Same packed (quv,kpv,ctx)
#                       ABI as STEP=6 -> gated by the SAME runner at the T where
#                       kpv fits L1; de-risks the streaming core's compute half.
#   STEP=8           -- relpos_rowtiled_stream : FULL-T (up to 172) MemTile-
#                       STREAMED block (k/p/V staged in L2, streamed to L1 in
#                       KB-row blocks, replayed per query tile). Removes the L1
#                       wall. Needs the runner's --stream packing (tile-interleaved
#                       QUV + padded KPV); see BUILD-AND-BENCH.md.
STEP ?= 1
ifeq ($(STEP),8)
targetname = relpos_rowtiled_stream
aie_py_src = relpos_rowtiled_stream_iron.py
else ifeq ($(STEP),7)
targetname = relpos_kpvstream
aie_py_src = relpos_kpvstream_iron.py
else ifeq ($(STEP),6)
targetname = relpos_rowtiled
aie_py_src = relpos_rowtiled_iron.py
else ifeq ($(STEP),5)
targetname = relpos_full
aie_py_src = relpos_full_iron.py
else ifeq ($(STEP),4)
targetname = relpos_ctx
aie_py_src = relpos_ctx_iron.py
else ifeq ($(STEP),3)
targetname = relpos_qkp_scores_softmax
aie_py_src = relpos_qkp_scores_softmax_iron.py
else ifeq ($(STEP),2)
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
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DRELPOS_T=${T} -DRELPOS_TQ=${TQ} -DRELPOS_KB=${KB} -c $< -o ${@F}
else
	echo "Only npu2 supported for this example"
endif

build/kernels.a: build/relpos_mha.cc.o
	ar rvs $@ $+

# STEP=8 (streaming) generator also needs TQ/KB to compute block counts + unroll;
# STEP<=7 bake TQ/KB in the .cc only, so pass just -T there.
ifeq ($(STEP),8)
GEN_EXTRA = --tq ${TQ} --kb ${KB}
else
GEN_EXTRA =
endif

build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -T ${T} ${GEN_EXTRA} > $@

build/final.xclbin: build/${targetname}.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		    --xclbin-name=${@F} \
		    --no-xchesscc --no-xbridge \
		    --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
