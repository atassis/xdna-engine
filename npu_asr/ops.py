"""Op facade + NPU engines.

`Ops` is the single interface the block/encoder call: matmul / dwconv / layernorm /
silu / softmax. Each op runs on the NPU if its engine is attached, else on a host
(numpy) reference. This keeps the block code backend-agnostic — host for correctness
bring-up, NPU engines for offload, and (future) a fused ObjectFifo backend swap in the
same place. Build `Ops.on_npu(...)` to attach engines, or `Ops.host()` for pure numpy.
"""
import os
import numpy as np

from . import config as C
from .dtypes import bf16, f32, bfloat16
from .device import NpuDevice


# ----------------------------- host reference ops -----------------------------
def host_matmul(x, w, b=None):
    y = f32(x) @ f32(w)
    return y if b is None else y + f32(b)


def host_layernorm_normalize(x):
    xf = f32(x)
    return (xf - xf.mean(-1, keepdims=True)) / np.sqrt(xf.var(-1, keepdims=True) + C.LN_EPS)


def host_silu(x):
    xf = f32(x)
    return xf / (1.0 + np.exp(-xf))


def host_softmax(x, axis=-1):
    xf = f32(x)
    e = np.exp(xf - xf.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def host_dwconv_k5(x, taps):
    """x[C,T] bf16, taps[C,5] -> [C,T] fp32, 'same' pad=2."""
    xf, w = f32(x), f32(taps)
    T = xf.shape[1]
    pad = np.pad(xf, ((0, 0), (2, 2)))
    out = np.zeros_like(xf)
    for i in range(5):
        out += w[:, i:i + 1] * pad[:, i:i + T]
    return out


# ----------------------------- NPU engines -----------------------------
class MatmulEngine:
    """bf16->f32 matmuls. Pads M->PAD_M; picks xclbin by (K,N). Default `whole`=8-col
    whole_array (~20-38x faster, N=3072 in one shot); `whole=False` = single_core
    (1 col, N=3072 tiled as 2x1536). Both: plain row-major A/B/C, identical ABI."""
    def __init__(self, dev, whole=True):
        self.dev = dev
        self.shapes = C.MM_WHOLE_SHAPES if whole else C.MM_SHAPES
        d = C.MM_WHOLE_DIR if whole else C.MM_DIR
        suf = (lambda K, N: f"{C.PAD_M}x{K}x{N}_32x32x32_8c") if whole \
            else (lambda K, N: f"{C.PAD_M}x{K}x{N}_32x32x32")
        self.k = {}; self.instr = {}
        for (K, N) in self.shapes:
            s = suf(K, N)
            self.k[(K, N)] = dev.kernel(os.path.join(d, f"final_{s}.xclbin"))
            self.instr[(K, N)] = np.fromfile(os.path.join(d, f"insts_{s}.txt"), np.uint32)

    def _one(self, A_bf16_512, B_bf16):
        K, N = B_bf16.shape
        k, instr = self.k[(K, N)], self.instr[(K, N)]
        bi = self.dev.bo_instr(k, 1, instr)
        ba = self.dev.bo_in(k, 3, np.ascontiguousarray(A_bf16_512).view(np.uint16))
        bb = self.dev.bo_in(k, 4, np.ascontiguousarray(B_bf16).view(np.uint16))
        bc = self.dev.bo_out(k, 5, C.PAD_M * N * 4)
        bt = self.dev.bo_dummy(k, 6, 1)
        btr = self.dev.bo_dummy(k, 7, 4)
        k(3, bi, instr.size, ba, bb, bc, bt, btr).wait()
        bc.sync(self.dev.FROM)
        return np.frombuffer(bc.read(C.PAD_M * N * 4, 0), np.float32).reshape(C.PAD_M, N)

    def matmul(self, x, w, b=None):
        x = f32(x); w = f32(w)
        M, K = x.shape; _, N = w.shape
        Ap = np.zeros((C.PAD_M, K), np.float32); Ap[:M] = x
        Ap, Bb = bf16(Ap), bf16(w)
        if (K, N) in self.shapes:
            out = self._one(Ap, Bb)
        elif K == 768 and N == 3072:                    # single_core: tile N as 2x1536
            out = np.concatenate([self._one(Ap, Bb[:, :1536]), self._one(Ap, Bb[:, 1536:])], 1)
        else:
            raise ValueError(f"no matmul xclbin for (K,N)=({K},{N})")
        out = out[:M]
        return out if b is None else out + f32(b)


class DwconvEngine:
    """depthwise conv1d k=5 'same' on [768,400], one channel per ObjectFifo tile."""
    def __init__(self, dev):
        self.dev = dev
        self.k = dev.kernel(os.path.join(C.DW_DIR, "final.xclbin"))
        self.instr = np.fromfile(os.path.join(C.DW_DIR, "insts.bin"), np.uint32)

    def dwconv(self, x, taps):                # x[768,400] bf16, taps[768,5] -> [768,400] f32
        ch = x.shape[0]
        w = np.zeros((ch, 16), np.float32); w[:, :5] = f32(taps)
        k = self.k
        bi = self.dev.bo_instr(k, 1, self.instr)
        bx = self.dev.bo_in(k, 3, np.ascontiguousarray(bf16(x)).reshape(-1).view(np.uint16))
        bw = self.dev.bo_in(k, 4, np.ascontiguousarray(bf16(w)).reshape(-1).view(np.uint16))
        nb = ch * x.shape[1] * 2
        by = self.dev.bo_out(k, 5, nb)
        k(3, bi, self.instr.size, bx, bw, by).wait()
        by.sync(self.dev.FROM)
        return np.frombuffer(by.read(nb, 0), np.uint16).view(bfloat16).reshape(x.shape).astype(np.float32)


class LayerNormEngine:
    """normalize-only LayerNorm [400,768] (kernel gamma=1,beta=0); affine on host."""
    def __init__(self, dev):
        self.dev = dev
        self.k = dev.kernel(os.path.join(C.LN_DIR, "final.xclbin"))
        self.instr = np.fromfile(os.path.join(C.LN_DIR, "insts.bin"), np.uint32)

    def normalize(self, x):                   # [400,768] -> [400,768] f32 normalized
        k = self.k
        X = np.ascontiguousarray(bf16(x)).reshape(-1).view(np.uint16)
        bi = self.dev.bo_instr(k, 1, self.instr)
        bx = self.dev.bo_in(k, 3, X)
        by = self.dev.bo_out(k, 4, X.nbytes)
        bt = self.dev.bo_dummy(k, 5, 1); bcp = self.dev.bo_dummy(k, 6, 8); btr = self.dev.bo_dummy(k, 7, 1)
        k(3, bi, self.instr.size, bx, by, bt, bcp, btr).wait()
        by.sync(self.dev.FROM)
        return np.frombuffer(by.read(X.nbytes, 0), np.uint16).view(bfloat16).reshape(x.shape).astype(np.float32)


class SiluEngine:
    """SiLU (tanh-approx sigmoid). One xclbin per element count: 400*768 and 400*3072."""
    def __init__(self, dev):
        self.dev = dev
        self.k = {}; self.instr = {}
        for L in (C.T_OUT * C.D_MODEL, C.T_OUT * C.D_FF):
            self.k[L] = dev.kernel(os.path.join(C.SILU_DIR, f"final_{L}.xclbin"))
            self.instr[L] = np.fromfile(os.path.join(C.SILU_DIR, f"insts_{L}.bin"), np.uint32)

    def silu(self, x):
        shape = x.shape
        X = np.ascontiguousarray(bf16(x)).reshape(-1).view(np.uint16)
        L = X.size
        k, instr = self.k[L], self.instr[L]
        bi = self.dev.bo_instr(k, 1, instr)
        bx = self.dev.bo_in(k, 3, X)
        by = self.dev.bo_out(k, 4, X.nbytes)
        k(3, bi, instr.size, bx, by).wait()
        by.sync(self.dev.FROM)
        return np.frombuffer(by.read(X.nbytes, 0), np.uint16).view(bfloat16).reshape(shape).astype(np.float32)


# ----------------------------- the facade -----------------------------
class Ops:
    """Backend-agnostic op facade. Attach NPU engines or fall back to host numpy.
    All methods take/return bf16 (rounding at op boundaries = the bf16 dataflow)."""
    def __init__(self, matmul=None, dwconv=None, layernorm=None, silu=None, softmax=None):
        self._mm, self._dw, self._ln, self._si = matmul, dwconv, layernorm, silu

    @classmethod
    def host(cls):
        return cls()

    @classmethod
    def on_npu(cls, matmul=True, dwconv=True, layernorm=True, silu=True, whole_matmul=True):
        """Attach NPU engines. layernorm/silu use approximation kernels (invsqrt, tanh)
        that compound over depth — set them False to keep those on host for accuracy."""
        dev = NpuDevice.get()
        return cls(
            matmul=MatmulEngine(dev, whole=whole_matmul) if matmul else None,
            dwconv=DwconvEngine(dev) if dwconv else None,
            layernorm=LayerNormEngine(dev) if layernorm else None,
            silu=SiluEngine(dev) if silu else None,
        )

    def matmul(self, x, w, b=None):
        y = self._mm.matmul(x, w, b) if self._mm else host_matmul(x, w, b)
        return bf16(y)

    def dwconv(self, x, taps, bias=None):
        y = self._dw.dwconv(x, taps) if self._dw else host_dwconv_k5(x, taps)
        if bias is not None:
            y = f32(y) + f32(bias)[:, None]
        return bf16(y)

    def layernorm(self, x, w, b):
        norm = self._ln.normalize(x) if self._ln else host_layernorm_normalize(x)
        return bf16(f32(norm) * f32(w) + f32(b))

    def silu(self, x):
        return bf16(self._si.silu(x) if self._si else host_silu(x))

    def softmax(self, x, axis=-1):            # host today; NPU shaping is a TODO
        return bf16(host_softmax(x, axis))

    def placement(self):
        return {"matmul": bool(self._mm), "dwconv": bool(self._dw),
                "layernorm": bool(self._ln), "silu": bool(self._si), "softmax": False}
