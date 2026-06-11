"""bf16/f32 helpers. The engine runs a bf16 dataflow: every op takes bf16 inputs,
accumulates in fp32, and rounds to bf16 — matching the AIE matmul convention so the
NPU and host paths agree to ~1 bf16 ULP."""
import numpy as np
from ml_dtypes import bfloat16

__all__ = ["bf16", "f32", "bfloat16"]


def bf16(x):
    return np.asarray(x, np.float32).astype(bfloat16)


def f32(x):
    return np.asarray(x, np.float32)
