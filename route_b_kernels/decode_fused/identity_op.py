# SPDX-License-Identifier: Apache-2.0
# Identity MLIROperator -- a PLUMBING probe for the FusedMLIROperator path, defined here (not in
# the shared IRON tree) so `operator_dir` resolves to this directory, exactly like argmax_op.py.
#
# Why it exists: probe_argmax_vocab.py returns all-zero reads through FusedMLIROperator, and that
# survived both a residency/sync fix and a bisect down to the stock un-chunked ASR-size shape. The
# same kernel is index-exact on the brick verify rail. So the suspect is the fusion path itself.
# An identity op removes every other variable: if a straight copy cannot round-trip, the path is
# broken and no compute kernel on it can be trusted.
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
class Identity(MLIROperator):
    """Copy [N] bf16 in -> out, streamed as tiles of `tile`."""

    N: int
    tile: int = None
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.tile is None:
            self.tile = self.N
        assert self.N % self.tile == 0, "N must split evenly into tiles"
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "identity_design.py",
                "my_identity",
                (aie_utils.get_current_device(), self.N),
                {"kernel_object": "identity_copy.o", "tile": self.tile},
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                "identity_copy.o",
                dependencies=[SourceArtifact(self.operator_dir / "identity_copy.cc")],
                extra_flags=[],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.N,)),
            AIERuntimeArgSpec("out", (self.N,)),
        ]
