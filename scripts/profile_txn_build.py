import cProfile
import pstats
import sys
import io
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
m_pad = 256

rng = np.random.default_rng(9)
A = iron.tensor(rng.uniform(-1, 1, size=(M_MAX * K,)).astype(bfloat16), dtype=bfloat16, device="npu")
B = iron.tensor(rng.uniform(-1, 1, size=(K * N,)).astype(bfloat16), dtype=bfloat16, device="npu")
C = iron.zeros((M_MAX * N,), dtype=bfloat16, device="npu")

design = whole_array_dynamic.specialize(
    A_elements=M_MAX * K, B_elements=K * N, C_elements=M_MAX * N,
    K=K, m=M_TILE, k=K_TILE, n=N_TILE, n_aie_cols=N_AIE_COLS,
    dtype_in_str="bf16", dtype_out_str="bf16",
)
design(A, B, C, M=m_pad, N=N)  # warmup/compile

pr = cProfile.Profile()
pr.enable()
for _ in range(30):
    design(A, B, C, M=m_pad, N=N)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())
