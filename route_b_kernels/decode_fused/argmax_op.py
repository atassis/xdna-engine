# SPDX-License-Identifier: Apache-2.0
# Local Argmax MLIROperator for the e2e/NPU lm-head (step-2). Defined HERE (route_b_kernels) — NOT in the
# shared IRON tree — so `operator_dir` resolves to this directory and the op points at our argmax_design.py
# + argmax_slice.cc (no shared-IRON edit). Per-column partial argmax: input [N] bf16 → out_idx [cols] i32 +
# out_val [cols] f32 (each column's LOCAL max index + value); the host does the trivial cols-way reduce.
from dataclasses import dataclass, field

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils


@dataclass
class Argmax(MLIROperator):
    """Per-column partial argmax over a [N] bf16 vector → [cols] local indices (i32) + [cols] values (f32)."""

    N: int
    cols: int = 8
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        assert self.N % self.cols == 0, "N must split evenly across cols"
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "argmax_design.py",
                "my_argmax",
                (aie_utils.get_current_device(), self.N, self.cols),
                {"kernel_object": "argmax_slice.o"},
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                "argmax_slice.o",
                dependencies=[SourceArtifact(self.operator_dir / "argmax_slice.cc")],
                extra_flags=[],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.N,)),            # logits (bf16)
            AIERuntimeArgSpec("out", (self.cols * 4,)),    # packed [val:f32 | idx:i32] per column (bf16-typed)
        ]
