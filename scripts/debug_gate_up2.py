import sys
import numpy as np
from ml_dtypes import bfloat16

import os
MLIR_AIE_PIN = os.environ.get("MLIR_AIE_PIN", os.path.join(os.path.dirname(__file__), "..", "..", "wt-mlir-aie-pin"))
sys.path.insert(0, f"{MLIR_AIE_PIN}/test/npu-xrt/matmul_whole_array_dynamic")
import aie.iron as iron
from whole_array_dynamic import whole_array_dynamic

K, N = 3840, 15360
M_TILE, K_TILE, N_TILE = 32, 128, 64
N_AIE_COLS = 4
M_MAX = 2048
m_real = 128
m_pad = 128

rng = np.random.default_rng(1)
a_real = rng.uniform(-1, 1, size=(m_real * K,)).astype(bfloat16)
b_real = rng.uniform(-1, 1, size=(K * N,)).astype(bfloat16)

a = np.zeros((M_MAX * K,), dtype=bfloat16)
a[: m_real * K] = a_real
b = b_real.copy()

A = iron.tensor(a, dtype=bfloat16, device="npu")
B = iron.tensor(b, dtype=bfloat16, device="npu")
C = iron.zeros((M_MAX * N,), dtype=bfloat16, device="npu")

# SAME as main script: oversized A_elements/C_elements (M_MAX), dispatched M=m_pad
design = whole_array_dynamic.specialize(
    A_elements=M_MAX * K, B_elements=K * N, C_elements=M_MAX * N,
    K=K, m=M_TILE, k=K_TILE, n=N_TILE, n_aie_cols=N_AIE_COLS,
    dtype_in_str="bf16", dtype_out_str="bf16",
)
design(A, B, C, M=m_pad, N=N)
c = C.numpy()[: m_pad * N].reshape(m_pad, N).copy()
expected = (a_real.reshape(m_real, K).astype(np.float32) @ b_real.reshape(K, N).astype(np.float32))

print("actual[0,:8] =", c[0, :8].astype(np.float32))
print("expected[0,:8] =", expected[0, :8])
rel = np.abs(c.astype(np.float32) - expected) / np.maximum(np.abs(expected), 1e-3)
print("max rel err:", rel.max(), "at", np.unravel_index(rel.argmax(), rel.shape))
print("actual overall max abs:", np.abs(c.astype(np.float32)).max())
