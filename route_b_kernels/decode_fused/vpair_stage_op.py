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

    DIRECT mode (`pair_sizes`/`pair_strides`/`pair_buffer_size` given): the pair lives IN the
    transposed cache rather than in a separate arena buffer, so the stage's own fill/drain address
    `cache[h, d, base:base+2]` with `pair_offset_parameter` supplying `base` per dispatch. That
    deletes the follow-on StridedCopy -- the -1 op the arm was scoped for. `fill`/`drain` take a tap
    and an `offset_parameter` on any ObjectFifoHandle, so this needs no new toolchain mechanism; it
    is what StridedCopy itself is built out of.
    """

    N: int
    parity_parameter: str
    pair_sizes: tuple | None = None
    pair_strides: tuple | None = None
    pair_buffer_size: int | None = None
    pair_offset_parameter: str | None = None
    context: object = field(default=None, repr=False)

    _name_aliases = {
        **MLIROperator._name_aliases,
        "parity_parameter": "par",
        "pair_offset_parameter": "poff",
        "pair_buffer_size": "pbuf",
    }

    def __post_init__(self):
        direct = [self.pair_sizes, self.pair_strides, self.pair_buffer_size]
        if any(x is not None for x in direct) and any(x is None for x in direct):
            raise ValueError("direct mode needs pair_sizes, pair_strides AND pair_buffer_size")
        if self.pair_sizes is not None:
            if len(self.pair_sizes) != len(self.pair_strides):
                raise ValueError("pair_sizes and pair_strides must have equal rank")
            import numpy as np

            if int(np.prod(self.pair_sizes)) != 2 * self.N:
                raise ValueError(f"pair_sizes must cover 2*N={2 * self.N} elements")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _direct(self):
        return self.pair_sizes is not None

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "vpair_stage_design.py",
                "vpair_stage",
                (aie_utils.get_current_device(), self.N, self.parity_parameter),
                {
                    "kernel_object": "vpair_stage.o",
                    "pair_sizes": self.pair_sizes,
                    "pair_strides": self.pair_strides,
                    "pair_buffer_size": self.pair_buffer_size,
                    "pair_offset_parameter": self.pair_offset_parameter,
                },
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
        pair = (self.pair_buffer_size,) if self._direct else (2 * self.N,)
        return [
            AIERuntimeArgSpec("in", (self.N,)),
            AIERuntimeArgSpec("in", pair),
            AIERuntimeArgSpec("out", pair),
        ]
