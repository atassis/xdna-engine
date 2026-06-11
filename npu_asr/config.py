"""Model dimensions + build/artifact paths. Single source of truth."""
import os

# repo root = parent of this package dir
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- GigaAM-v3 encoder dims ---
D_MODEL = 768
D_FF = 3072
N_BLOCKS = 16
N_HEADS = 16
HEAD_DIM = 48           # 768 / 16
T_OUT = 400             # frames after ÷4 subsampling (from 1600)
T_SUB0 = 800            # after first stride-2 conv
N_MEL = 64
DW_K = 5                # depthwise conv kernel
SUB_K = 5               # subsampling conv kernel
SUB_STRIDE = 2
SUB_PAD = 2
LN_EPS = 1e-5
PAD_M = 512             # matmul M padded to a tile-aligned 512 (from 400)

# --- artifacts (produced by scripts/extract_encoder.py, run in .venv) ---
ARTIFACTS = os.path.join(ROOT, "artifacts", "encoder")

# --- xclbin build dirs (Route B; built by scripts/setup_route_b.sh + make) ---
_PE = os.path.join(ROOT, "mlir-aie", "programming_examples")
MM_DIR = os.path.join(_PE, "basic", "matrix_multiplication", "single_core", "build")
MM_WHOLE_DIR = os.path.join(_PE, "basic", "matrix_multiplication", "whole_array", "build")
DW_DIR = os.path.join(_PE, "ml", "dwconv1d", "build")
LN_DIR = os.path.join(_PE, "ml", "layernorm", "build")
SILU_DIR = os.path.join(_PE, "ml", "silu", "build")

# matmul shapes (K, N) at M=PAD_M.
# single_core (1 col): N=3072 overflows -> tiled as 2x1536.
MM_SHAPES = {(768, 768), (3072, 768), (768, 1536)}
# whole_array (8 cols, ~20-38x faster, plain row-major, same ABI): N=3072 in one shot.
MM_WHOLE_SHAPES = {(768, 768), (3072, 768), (768, 1536), (768, 3072)}

# Whisper-NPU floor to beat (~3.3 s on the 11.9 s clip); GigaAM-v3 CPU ~0.89 s.
TARGET_WHISPER_S = 3.3
TARGET_CPU_S = 0.89
