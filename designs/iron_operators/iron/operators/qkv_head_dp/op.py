# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
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
class QKVHeadDataParallel(MLIROperator):
    """Decode QKV head as ONE `aie.device`, data-parallel across `num_aie_columns` cores.

    Fuses RMSNorm(in) -> concatenated [Hq*HD + 2*Hkv*HD, D] QKV GEMV -> per-head qk-RMSNorm ->
    RoPE(q, k). Every core owns a contiguous row slice of the concatenated weight and runs every
    stage on it; see design.py for why that shape and not fuse/qkv-head's spatial one.

    Runtime interface: cur, n_in, Wqkv, n_qn, n_kn, ang -> qkv, where `qkv` is the concatenated
    [q | k | v] the caller then slices. The weight and the output are concatenated because the
    caller's graph already concatenates them (gen_llm_decode.py's FUSE_QKV_GEMV), not to save an
    argument.
    """

    D: int
    HD: int
    Hq: int
    Hkv: int
    max_seq: int
    num_aie_columns: int = 8
    epsilon: float = 1e-6
    tile_size_input: int = 4
    stack_size: int = 0xD00
    kv_offset_parameter: str | None = "kv_off"
    # Weight ObjectFifo depth; the L1 budget check in design.py follows it.
    weight_depth: int = field(default=2, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
        "tile_size_input": "tsi",
        "stack_size": "ss",
        "max_seq": "S",
        "kv_offset_parameter": "kvpar",
        "weight_depth": "wd",
    }

    def __post_init__(self):
        heads = self.Hq + 2 * self.Hkv
        if heads % self.num_aie_columns:
            raise ValueError(
                f"Hq + 2*Hkv ({heads}) must be divisible by num_aie_columns "
                f"({self.num_aie_columns}) -- every core owns a whole number of heads"
            )
        if self.HD % self.tile_size_input:
            raise ValueError(
                f"head_dim ({self.HD}) must be divisible by tile_size_input "
                f"({self.tile_size_input})"
            )
        if self.D % self.HD:
            raise ValueError(
                f"d_model ({self.D}) must be a whole number of head_dim ({self.HD}) chunks -- "
                "`cur` and `n_in` ride the HD-wide misc channel"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qkv_head_dp",
                (aie_utils.get_current_device(), self.D, self.HD, self.Hq, self.Hkv,
                 self.max_seq),
                {
                    "epsilon": self.epsilon,
                    "kv_offset_parameter": self.kv_offset_parameter,
                    "weight_depth": self.weight_depth,
                    "tile_size_input": self.tile_size_input,
                    "stack_size": self.stack_size,
                    "n_aie_cols": self.num_aie_columns,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        copy_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        rms_obj = KernelObjectArtifact(
            f"rms_norm_{self.D}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.D}"],
        )
        # Same source, second symbol: this core calls weighted_rms_norm at D and at HD, and one
        # Kernel() binding fixes one signature per symbol. Prefixing a second object is what
        # swiglu_mlp_dp does for its two matvec DIM_Ks; the alternative -- a local copy of the
        # vendored kernel under two names -- is the duplication one-kernel-three-repos warns about.
        rms_hd_obj = KernelObjectArtifact(
            f"hd_rms_norm_{self.HD}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.HD}"],
            prefix_symbols="hd_",
        )
        mv_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_64vs.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.D}", "-DVEC_SIZE=64"],
        )
        rope_obj = KernelObjectArtifact(
            "rope_0.o",
            dependencies=[SourceArtifact(kdir / "generic" / "rope.cc")],
            extra_flags=["-DTWO_HALVES"],
        )
        return [
            KernelArchiveArtifact(
                "qkv_head_dp_core.a",
                dependencies=[copy_obj, rms_obj, rms_hd_obj, mv_obj, rope_obj],
            )
        ]

    def get_arg_spec(self):
        QD, KVD = self.Hq * self.HD, self.Hkv * self.HD
        cache = self.Hkv * self.max_seq * self.HD
        return [
            AIERuntimeArgSpec("in", (self.D,)),                    # cur
            AIERuntimeArgSpec("in", (self.D,)),                    # n_in
            AIERuntimeArgSpec("in", ((QD + 2 * KVD) * self.D,)),   # Wqkv, flat [QD+2KVD, D]
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_qn
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_kn
            AIERuntimeArgSpec("in", (self.HD,)),                   # ang
            AIERuntimeArgSpec("out", (QD,)),                       # q
            AIERuntimeArgSpec("inout", (cache,)),                  # kc, appended at kv_off
            AIERuntimeArgSpec("inout", (cache,)),                  # vc, appended at kv_off
        ]

    def reference(self, cur, n_in, wqkv, n_qn, n_kn, ang):
        """Returns the concatenated [q | k | v]; the caller places k and v itself. The device
        appends them to the caches directly, so there is no single output to compare against."""
        from iron.operators.qkv_head_dp.reference import reference

        return reference(cur, n_in, wqkv, n_qn, n_kn, ang,
                         self.D, self.HD, self.Hq, self.Hkv, self.epsilon)
