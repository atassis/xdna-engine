"""Full GigaAM-v3 encoder: subsampling front-end + N stacked Conformer blocks.

  audio_signal [N_MEL, 1600]  -- log-mel features
    -> pre_encode: 2x (conv1d k5 s2 pad2 + ReLU)   (÷4 in time)  -> [D, 400]
    -> transpose -> [400, D]
    -> 16x conformer_block
    -> transpose -> encoded [D, 400]

Subsampling is dense conv1d done as im2col->matmul; it runs on host today (small,
and its shapes aren't in the shipped matmul xclbin set) — a clear offload TODO. The
16 blocks' heavy ops run through `ops` (NPU or host).
"""
import numpy as np

from . import config as C
from .dtypes import bf16, f32
from .block import conformer_block


def im2col_conv1d(x, w, b, k=C.SUB_K, stride=C.SUB_STRIDE, pad=C.SUB_PAD):
    """x[Cin, L], w[Cout, Cin, k], b[Cout] -> [Cout, Lout] (no activation)."""
    Cin, L = x.shape
    Cout = w.shape[0]
    xp = np.pad(f32(x), ((0, 0), (pad, pad)))
    Lout = (L + 2 * pad - k) // stride + 1
    cols = np.empty((Lout, Cin * k), np.float32)
    for t in range(Lout):
        s = t * stride
        cols[t] = xp[:, s:s + k].reshape(-1)        # [Cin,k] -> flat (Cin,k order)
    W2 = f32(w).reshape(Cout, Cin * k)              # same (Cin,k) order
    return (cols @ W2.T + f32(b)).T                 # [Cout, Lout]


def subsample(audio, pw):
    """audio [N_MEL, 1600] -> block input [400, D] (host im2col->matmul + ReLU)."""
    h = im2col_conv1d(audio, pw["pre_encode.conv.0.weight"], pw["pre_encode.conv.0.bias"])
    h = np.maximum(h, 0.0)                           # ReLU
    h = im2col_conv1d(h, pw["pre_encode.conv.2.weight"], pw["pre_encode.conv.2.bias"])
    h = np.maximum(h, 0.0)
    return h.T                                       # [400, D] == /pre_encode/Transpose


class Encoder:
    def __init__(self, weights, ops):
        self.w = weights
        self.ops = ops

    def subsample(self, audio):
        return subsample(audio, self.w.pre_encode)

    def encode(self, audio, n_blocks=C.N_BLOCKS, x0=None):
        """audio [N_MEL,1600] -> encoded [D,400]. x0 overrides the block input
        (e.g. the ONNX subsampling output, to isolate the block stack)."""
        x = bf16(self.subsample(audio) if x0 is None else x0)   # [400, D]
        for i in range(n_blocks):
            x = conformer_block(x, self.w.block(i), self.ops, self.w.cos, self.w.sin)
        return f32(x).T                                          # [D, 400] == encoded
