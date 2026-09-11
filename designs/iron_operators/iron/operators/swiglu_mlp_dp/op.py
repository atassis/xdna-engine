# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
from typing import ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils
from iron.common.device_utils import get_kernel_dir


@dataclass
class SwiGLUMLPDataParallel(MLIROperator):
    """Decode SwiGLU MLP block as ONE `aie.device`, data-parallel across `num_aie_columns` cores
    (one core per column, n_aie_rows fixed at 1 -- see design.py's module docstring). Every core
    runs every stage on its own 1/N slice instead of fuse/mlp-block's 5-core spatial pipeline.

    Runtime interface: cur, a, n_pf, Wg, Wu, Wd -> nxt, plus one internal `gh_scratch` (FF
    elements) DRAM round-trip buffer for the cross-core all-gather -- there is no internal-DRAM-
    scratch primitive in plain Runtime/Program (every DMA endpoint must be a formal Runtime
    argument), so it is exposed as a genuine 7th argument here. Callers that want the operator's
    logical 6-in/1-out surface should allocate it once and never touch its contents (matches how
    OperatorSequence's own `buffer_sizes=` auto-allocates unnamed intermediates for the unfused
    arm in the A/B harness this operator was built for).
    """

    D: int
    FF: int
    num_aie_columns: int = 8
    num_aie_rows: int = 1
    epsilon: float = 1e-5
    QD: int = None
    fuse_o: bool = False
    # Group-quantized weight stream (Wg/Wu/Wd, and Wo under fuse_o). bf16 is the
    # byte-for-byte pre-existing path; see design.py's WEIGHT WIRE UNITS block.
    weight_dtype: str = field(default="bf16", repr=False)
    group_size: int = field(default=0, repr=False)
    # Weight ObjectFifo depth. 2 is plain double-buffering; deeper hides more of the
    # shim->L1 latency at the cost of L1 (the budget check above follows it).
    weight_depth: int = field(default=2, repr=False)
    tile_rows_gu: int = field(default=0, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "epsilon": "eps",
        "num_aie_columns": "cols",
        "num_aie_rows": "rows",
        "fuse_o": "fo",
        "weight_dtype": "wdt",
        "group_size": "g",
        "weight_depth": "wd",
        "tile_rows_gu": "tr",
    }

    def __post_init__(self):
        if self.D <= 0 or self.FF <= 0:
            raise ValueError(f"D ({self.D}) and FF ({self.FF}) must be positive")
        if self.fuse_o:
            if not self.QD or self.QD <= 0:
                raise ValueError("fuse_o requires QD (the attention context width)")
            if self.num_aie_rows != 1:
                raise ValueError("fuse_o is only derived for num_aie_rows=1 (see design.py)")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _wo_rows_padded(self):
        """Wo's own arg-spec row count once padded for the TSI_O overlap window -- see
        design.py's FUSE_O section. Mirrors design.py's arithmetic exactly (kept in sync by hand;
        design.py itself asserts the same divisibility this depends on)."""
        N = self.num_aie_columns * self.num_aie_rows
        D_PER_CORE = self.D // N
        WTILE_ELEMS = 6 * self.D  # TSI_GU * D, design.py's shared-tile constant
        TSI_O = WTILE_ELEMS // self.QD
        N_O_TILES = -(-D_PER_CORE // TSI_O)
        O_WINDOW = N_O_TILES * TSI_O
        return self.D + (O_WINDOW - D_PER_CORE)

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_swiglu_mlp_dp",
                (aie_utils.get_current_device(), self.D, self.FF, self.epsilon),
                {
                    "stack_size": 0x800,
                    "n_aie_cols": self.num_aie_columns,
                    "n_aie_rows": self.num_aie_rows,
                    "QD": self.QD,
                    "fuse_o": self.fuse_o,
                    "weight_dtype": self.weight_dtype,
                    "group_size": self.group_size,
                    "weight_depth": self.weight_depth,
                    "tile_rows_gu": self.tile_rows_gu or None,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"

        add_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        mul_obj = KernelObjectArtifact(
            "mul.o", dependencies=[SourceArtifact(kdir / "generic" / "mul.cc")]
        )
        rms_norm_obj = KernelObjectArtifact(
            f"rms_norm_{self.D}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.D}"],
        )
        silu_obj = KernelObjectArtifact(
            "silu.o", dependencies=[SourceArtifact(kdir / arch_dir / "silu.cc")]
        )
        # One source and one flag set for both weight formats: mv_quant.cc exports
        # matvec_vectorized_{int4,int8}_bf16 with mv.cc's exact signature except that `a_in` is
        # int8, which is the only thing the WTILE_ty change alters. GROUP_SIZE is the extra flag.
        _qsrc = kdir / "generic" / ("mv.cc" if self.weight_dtype == "bf16" else "mv_quant.cc")
        _qtag = "" if self.weight_dtype == "bf16" else f"_{self.weight_dtype}g{self.group_size}"
        # A vector chunk must not straddle a quant group, so the kernel's VEC_SIZE is capped by
        # the group width. 64 is the native width and the only value the symmetric builds ever
        # used (their smallest shipped group is 128); a 32-wide group -- FastFlowLM's operating
        # point -- needs 32. It is in the object name because two objects compiled at different
        # VEC_SIZE export the SAME symbol, which is the artifact-key collision this file already
        # documents for the dtype axis.
        _vec = 64 if self.weight_dtype == "bf16" else min(64, self.group_size)
        _qflags = ([] if self.weight_dtype == "bf16"
                   else [f"-DGROUP_SIZE={self.group_size}"])
        mv_gu_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_{_vec}vs{_qtag}.o",
            dependencies=[SourceArtifact(_qsrc)],
            extra_flags=[f"-DDIM_K={self.D}", f"-DVEC_SIZE={_vec}"] + _qflags,
        )
        # Same exported symbol as mv_gu_obj (DIM_K is baked in, not part of the name); this
        # object's own device-wide symbol table entry must be distinct, so it is compiled with a
        # prefix -- see design.py's mv_d_kernel comment and fuse/mlp-block's identical mechanism.
        mv_d_obj = KernelObjectArtifact(
            f"down_gemv_{self.FF}k_{_vec}vs{_qtag}.o",
            dependencies=[SourceArtifact(_qsrc)],
            extra_flags=[f"-DDIM_K={self.FF}", f"-DVEC_SIZE={_vec}"] + _qflags,
            prefix_symbols="down_",
        )
        deps = [add_obj, mul_obj, rms_norm_obj, silu_obj, mv_gu_obj, mv_d_obj]
        if self.fuse_o:
            mv_o_obj = KernelObjectArtifact(
                f"o_gemv_{self.QD}k_{_vec}vs{_qtag}.o",
                dependencies=[SourceArtifact(_qsrc)],
                extra_flags=[f"-DDIM_K={self.QD}", f"-DVEC_SIZE={_vec}"] + _qflags,
                prefix_symbols="o_",
            )
            # copy_offset_bf16_vector (in add.cc) is generic (pointer + runtime size/offset, no
            # compile-time shape), but a func.func declaration's symbol table entry is keyed by
            # NAME only, not by memref type -- two Kernel() Python bindings to the SAME symbol
            # with different declared shapes (QD_ty vs DPC_ty/OWIN_ty here) is a device-wide
            # "redefinition of symbol" verifier error, not a harmless overload (confirmed
            # device-free by aiecc's own MLIR verifier). Same fix as mv_d_kernel's "down_" prefix:
            # compile add.cc again per new call-site shape, under a distinct renamed symbol.
            cx_copy_obj = KernelObjectArtifact(
                "add_cxcopy.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                prefix_symbols="cx_",
            )
            oa_copy_obj = KernelObjectArtifact(
                "add_oacopy.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                prefix_symbols="oa_",
            )
            deps += [mv_o_obj, cx_copy_obj, oa_copy_obj]
        core_archive = KernelArchiveArtifact(
            f"swiglu_mlp_dp_core{_qtag}.a", dependencies=deps)
        return [core_archive]

    def _wrow(self, K):
        """Wire units per weight ROW of width K: bf16 elements, or packed bytes when quantized."""
        if self.weight_dtype == "bf16":
            return K
        from iron.common.quant import row_stride_bytes
        return row_stride_bytes(K, self.group_size, self.weight_dtype)

    def _wspec(self, n_units, comment_unused=None):
        """One weight argument, sized in wire units. dtype is OMITTED for bf16 rather than passed
        as None -- AIERuntimeArgSpec's default is not None, and passing it explicitly sized the
        arena at 8 bytes/element instead of 2 (caught by the sequence's own layout assert)."""
        if self.weight_dtype == "bf16":
            return AIERuntimeArgSpec("in", (n_units,))
        return AIERuntimeArgSpec("in", (n_units,), dtype=np.int8)

    def get_arg_spec(self):
        if self.fuse_o:
            return [
                AIERuntimeArgSpec("in", (self.D,)),                       # cur
                AIERuntimeArgSpec("in", (self.QD,)),                      # cx
                AIERuntimeArgSpec("in", (self.D,)),                       # n_pf
                self._wspec(self._wo_rows_padded * self._wrow(self.QD)),   # Wo, flat [D+pad,QD]
                self._wspec(self.FF * self._wrow(self.D)),        # Wg, flat [FF,D]
                self._wspec(self.FF * self._wrow(self.D)),        # Wu, flat [FF,D]
                self._wspec(self.D * self._wrow(self.FF)),        # Wd, flat [D,FF]
                AIERuntimeArgSpec("inout", (self.FF,)),                   # gh_scratch
                AIERuntimeArgSpec("inout", (self.D,)),                    # a_scratch
                AIERuntimeArgSpec("out", (self.D,)),                      # nxt
            ]
        return [
            AIERuntimeArgSpec("in", (self.D,)),                 # cur
            AIERuntimeArgSpec("in", (self.D,)),                 # a
            AIERuntimeArgSpec("in", (self.D,)),                 # n_pf
            self._wspec(self.FF * self._wrow(self.D)),  # Wg, flat [FF,D]
            self._wspec(self.FF * self._wrow(self.D)),  # Wu, flat [FF,D]
            self._wspec(self.D * self._wrow(self.FF)),  # Wd, flat [D,FF]
            AIERuntimeArgSpec("inout", (self.FF,)),             # gh_scratch (internal round-trip)
            AIERuntimeArgSpec("out", (self.D,)),                # nxt
        ]

    def reference(self, cur, cx_or_a, n_pf, *rest):
        from iron.operators.swiglu_mlp_dp.reference import reference, reference_fused_o

        if self.fuse_o:
            Wo, Wg, Wu, Wd = rest[:4]
            return reference_fused_o(
                cur, cx_or_a, n_pf, Wo, Wg, Wu, Wd, self.D, self.FF, self.QD, self.epsilon
            )
        Wg, Wu, Wd = rest[:3]
        return reference(cur, cx_or_a, n_pf, Wg, Wu, Wd, self.D, self.FF, self.epsilon)
