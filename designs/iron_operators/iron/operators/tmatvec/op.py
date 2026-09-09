# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

from ml_dtypes import bfloat16

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
    KernelObjectArtifact,
    SourceArtifact,
)
import aie.utils as aie_utils


@dataclass
class TMatVec(MLIROperator):
    """Transposed-A matvec: C[b][j] = sum_p W[b][p] * A[b // batch_group][p][j].

    gemv reduces ALONG a row and gives one output per row. This reduces DOWN the rows and gives one
    output per COLUMN, which is what attention's context step needs against a V cache stored
    [S][head_dim]. Doing it as a gemv is what forces a physical transpose of the whole cache first.
    """

    M: int  # output width == the matrix's row width (head_dim)
    K: int  # reduction extent == the matrix's row COUNT (sequence length)
    num_aie_columns: int = 1
    num_batches: int = 1
    batch_group: int = 1
    rows_per_chunk: int = 64
    # Rows ALLOCATED per matrix when that differs from the rows REDUCED -- gemv's alloc_M one axis
    # over. The window is a row PREFIX here, so only the per-matrix stride moves.
    alloc_K: int | None = field(default=None, repr=False)
    kwargs: dict = field(default_factory=dict, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "num_batches": "batch",
        "batch_group": "bgrp",
        "rows_per_chunk": "rpc",
    }

    def __post_init__(self):
        if self.num_batches % self.batch_group != 0:
            raise ValueError(
                f"num_batches ({self.num_batches}) must be a multiple of batch_group "
                f"({self.batch_group})"
            )
        n_matrices = self.num_batches // self.batch_group
        if n_matrices != self.num_aie_columns:
            raise ValueError(
                f"this design places one matrix per column: num_batches//batch_group "
                f"({n_matrices}) must equal num_aie_columns ({self.num_aie_columns})"
            )
        if self.K % self.rows_per_chunk != 0:
            raise ValueError(
                f"rows_per_chunk ({self.rows_per_chunk}) must divide K ({self.K})"
            )
        if self.alloc_K is not None and self.alloc_K < self.K:
            raise ValueError(
                f"alloc_K ({self.alloc_K}) must be >= K ({self.K}): it is the ALLOCATED row "
                f"count per matrix, not a second window"
            )
        # K008 -- the tiling must FIT, not merely divide. Checked HERE, at construction, because
        # the only other thing that notices is aiecc, which reports it as a placement failure
        # naming a tile and not a size. Arithmetic lives once, in design.py.
        from iron.operators.tmatvec.design import check_l1_fits

        msg = check_l1_fits(self.M, self.K, self.batch_group, self.rows_per_chunk)
        if msg is not None:
            raise ValueError(msg)
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # A windowed read is a different design from the plain one at the same K: same reduction
        # extent, different buffer size and per-matrix stride. Without this they collide in the
        # build dir and a cached plain build silently satisfies the windowed op.
        base = super().name
        if self.alloc_K is not None and self.alloc_K != self.K:
            base = f"{base}_ak{self.alloc_K}"
        return base

    @property
    def _kernel_object(self) -> str:
        return f"tmv_{self.M}n.o"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "transposed_matvec",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.M,
                    self.K,
                    self.num_batches,
                    self.batch_group,
                    self.rows_per_chunk,
                ),
                {
                    **self.kwargs,
                    "kernel_object": self._kernel_object,
                    "alloc_K": self.alloc_K,
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv_taccum.cc"
                    )
                ],
                # DIM_N is the output width each core owns. It is the WHOLE row here, which is what
                # keeps the A read contiguous -- a per-core column slice would be M/cols wide, 32 B
                # at head_dim 128 over 8 columns, under the measured ~128 B contiguity knee.
                extra_flags=[f"-DDIM_N={self.M}"],
            )
        ]

    def get_arg_spec(self):
        n_matrices = self.num_batches // self.batch_group
        a_rows = self.K if self.alloc_K is None else self.alloc_K
        return [
            AIERuntimeArgSpec("in", (n_matrices, a_rows, self.M)),  # matrix (A)
            AIERuntimeArgSpec("in", (self.num_batches, self.K)),  # vector (W)
            AIERuntimeArgSpec("out", (self.num_batches, self.M)),  # output (C)
        ]
