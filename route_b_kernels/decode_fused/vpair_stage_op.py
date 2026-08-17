# SPDX-License-Identifier: Apache-2.0
# Parity-staging MLIROperator, defined here (not in the shared IRON tree) so `operator_dir`
# resolves to this directory, like identity_op.py and argmax_op.py.
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
class VPairStage(MLIROperator):
    """Merge [N] bf16 into slot `parity` of [N,2], carrying the other slot through.

    `parity` is a core-kind scratchpad parameter, written per dispatch by the host. Pass the same
    buffer name for `pair_in` and `pair_out` in the runlist: the staged pair is read and rewritten
    in place, which is what carries the other slot's token across dispatches.
    """

    N: int
    parity_parameter: str
    context: object = field(default=None, repr=False)

    _name_aliases = {**MLIROperator._name_aliases, "parity_parameter": "par"}

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "vpair_stage_design.py",
                "vpair_stage",
                (aie_utils.get_current_device(), self.N, self.parity_parameter),
                {"kernel_object": "vpair_stage.o"},
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                "vpair_stage.o",
                dependencies=[SourceArtifact(self.operator_dir / "vpair_stage.cc")],
                extra_flags=[],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.N,)),
            AIERuntimeArgSpec("in", (2 * self.N,)),
            AIERuntimeArgSpec("out", (2 * self.N,)),
        ]
