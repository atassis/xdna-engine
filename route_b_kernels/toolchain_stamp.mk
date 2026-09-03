##===- toolchain_stamp.mk -------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Make every target in this directory depend on the IDENTITY of the toolchain that
# builds it.
#
# WHY: an artifact's content depends on the aiecc/Peano that produced it, and nothing
# declared that. make compares mtimes of declared prerequisites, so an object or xclbin
# built by an older pin looks up to date forever. MEASURED 2026-09-03:
# final_512x800x3072_64x32x96_8c_modalsilu was 149344 B from 06-29 while every sibling
# was 170895 B from 09-03, and it silently entered a same-day A/B as the fc1 kernel --
# the A/B then attributed a hang to the kernel TILE when the arms also differed in
# toolchain vintage. Upstream #3559 changes the emitted transaction blob, so an old
# artifact can be speaking an older host<->device wire format.
#
# Freshness is an IDENTITY question -- which toolchain built this -- and an mtime cannot
# answer it. MLIR_AIE_INSTANCE is content-addressed by the toolchain.lock hash, so the
# path IS the identity; PEANO_INSTALL_DIR is included because it is what the .o recipes
# actually invoke.
#
# HOW TO USE: `include ${srcdir}/toolchain_stamp.mk` anywhere after srcdir is set. It is
# copied next to the Makefiles by sync_kernels.sh, so the path is same-directory and does
# not depend on how deep the toolchain instance nests the example dir.
#
# WHY .EXTRA_PREREQS rather than editing each rule: these directories carry 23 Makefiles
# whose object and xclbin targets are all named differently (build/acc_add${dtsuf}.o,
# build/final_${tag}.xclbin, ...). .EXTRA_PREREQS (GNU make >= 4.3) adds the prerequisite
# to EVERY target, so the guard cannot be defeated by a rule someone adds later and
# forgets to annotate -- which is the failure mode that produced the stale kernel.
#
# The stamp is refreshed at PARSE time and only when the identity actually changed, so a
# warm rebuild with an unchanged toolchain is a no-op (verified: no spurious relinks).
# It never DELETES anything -- a purge on a shared build dir is its own hazard.
#
# Idempotent; ASCII-clean; safe on the shared checkout.

toolchain_id := $(MLIR_AIE_INSTANCE) $(PEANO_INSTALL_DIR)

# `cmp -s` against the existing stamp, so the file's mtime moves only on a real change.
_toolchain_stamp_sync := $(shell mkdir -p build; printf '%s\n' '$(toolchain_id)' \
    | cmp -s - build/.toolchain.stamp || printf '%s\n' '$(toolchain_id)' > build/.toolchain.stamp)

.EXTRA_PREREQS += build/.toolchain.stamp
# ...except for the stamp itself, which would otherwise depend on itself.
build/.toolchain.stamp: .EXTRA_PREREQS :=
