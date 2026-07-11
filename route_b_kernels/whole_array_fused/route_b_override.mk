##===- route_b_override.mk ------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Shared route_b override for the whole_array matmul FAMILY (mm.cc + emulate/bfp16
# fast tile: Makefile.modal, Makefile.modal.int8, and Phase-3 siblings).
#
# WHY: the blessed toolchain instance's NEW matrix_multiplication/makefile-common is a
# rewrite that dropped the aiecc-from-MLIR flow. Its xclbin rule now drives @iron.jit with
# compile-only args (--use-chess / --xclbin-path / --insts-path) that our MLIR-only
# generators reject, and it no longer defines KERNEL_CC / KERNEL_CFLAGS / kernels_dir / the
# mlir_target rule / the emulate+bfp16 KERNEL_DEFINES. This snippet re-supplies the OLD
# generator -> .mlir -> aiecc flow so the modal generators keep working.
#
# HOW TO USE: `include` this AFTER `include ${srcdir}/../makefile-common`, once the
# per-Makefile has set target_suffix, xclbin_target, aie_py_src, kernels, buffer_aloc_flag
# and the MM/EPI defines. Every variable used below is supplied by makefile-common or the
# per-Makefile, so this block is generic for the whole_array family.
# Do NOT reuse for ctxln/mstat -- those are different kernels.
#
# NOTE: the Makefiles that include this are COPIED into the mlir-aie example dir before
# they run (srcdir resolves there), so callers reference this file by its source-tree
# relative path, e.g. `include ${srcdir}/../../../../../route_b_kernels/whole_array_fused/route_b_override.mk`.
#
# Idempotent; ASCII-clean; safe on the shared checkout.

# KERNEL_CC is identical in both device branches; only kernels_dir / CFLAGS differ.
KERNEL_CC=${PEANO_INSTALL_DIR}/bin/clang++
ifeq ($(devicename),npu2)
kernels_dir=${srcdir}/../../../../aie_kernels/aie2p
KERNEL_CFLAGS=${PEANOWRAP2P_FLAGS}
else
kernels_dir=${srcdir}/../../../../aie_kernels/aie2
KERNEL_CFLAGS=${PEANOWRAP2_FLAGS}
endif

# Fast-tile defines the new common no longer sets (mm.cc names symbols by these).
ifeq ($(emulate_bfloat16_mmul_with_bfp16),1)
KERNEL_DEFINES += -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
endif
ifeq ($(bfp16_iree),1)
KERNEL_DEFINES += -DBFP16_IREE
endif

aiecc_peano_flags=--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR}
# Prefer the wired fork-instance aiecc (AIECC_PATH from iron_env.sh); fall back to PATH.
AIECC := $(if $(AIECC_PATH),$(AIECC_PATH),aiecc)

# Reuse makefile-common's aieargs but drop the compile-only --use-chess the MLIR-only
# generator does not accept (we never pass --xclbin-path). COUPLING: this subst matches the
# space-separated spelling `--use-chess <val>`; if upstream ever emits `--use-chess=0`
# instead, update the pattern below.
gen_args = $(subst --use-chess $(use_chess),,$(aieargs))

# Pin insts to .txt (engine + Phase-3 kernel_registry expect insts_*.txt, not the new .bin).
mlir_target := build/aie_${target_suffix}.mlir
insts_target := build/insts_${target_suffix}.txt

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${gen_args} --trace_size 0 > $@

# Grouped co-target (&:): ONE aiecc invocation produces BOTH the xclbin AND the .txt insts,
# so make treats insts_${target_suffix}.txt as a real target (a direct `make .../insts_*.txt`
# resolves) rather than an untracked side-effect. This rule deliberately WINS over the new
# makefile-common's xclbin rule; make prints two EXPECTED warnings ("overriding recipe for
# target" / "ignoring old recipe for target") -- do NOT try to "fix" them.
${xclbin_target} ${insts_target} &: ${mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && ${AIECC} --alloc-scheme=${buffer_aloc_flag} --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(notdir ${xclbin_target}) ${aiecc_peano_flags} \
	    --aie-generate-npu-insts --npu-insts-name=$(notdir ${insts_target}) $(notdir ${mlir_target})
