"""npu_asr — an open GigaAM-v3 Conformer encoder on the AMD XDNA2 NPU (Route B).

Layering (each op is host/NPU-swappable; fused ObjectFifo is a future backend):
  config   — model dims + xclbin/artifact paths
  dtypes   — bf16/f32 helpers (bf16 dataflow: bf16 in -> fp32 accumulate -> bf16 out)
  device   — single shared pyxrt device + kernel/context cache (NPU is single-tenant)
  ops      — Ops facade: matmul/dwconv/layernorm/silu/softmax, NPU engine or host fallback
  weights  — load + bf16-quantize encoder weights from artifacts/encoder
  block    — the verified Conformer block recipe (RoPE-before-proj, GLU, macaron, ...)
  encoder  — subsampling front-end + N stacked blocks
  verify   — relative-error comparison vs ONNX reference tensors

Correctness is proven op-by-op vs the ONNX reference; performance (whole-array
matmul, ObjectFifo fusion) is layered in behind the same Ops facade.
"""
from .ops import Ops
from .encoder import Encoder
from .weights import WeightStore

__all__ = ["Ops", "Encoder", "WeightStore"]
