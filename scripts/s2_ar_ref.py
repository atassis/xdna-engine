#!/usr/bin/env python3
"""Host numpy reference for the Fish-Audio S2 Dual-AR half: text tokens -> [10, T] codec codes.

WHY this exists: `xdna-engine`'s S2 port has so far only touched the codec DECODER (codes ->
audio; see `scripts/codec_decoder_ref.py` on the `feat/s2-codec-bricks` branch and
`route_b_kernels/codec_block/`). The autoregressive half that PRODUCES those codes -- a "Slow"
transformer emitting one semantic token per audio frame plus a "Fast" transformer emitting the
9 residual RVQ codebook tokens for that frame -- has no reference, no shape inventory, and no NPU
kernel plan. This script is that reference, built by reading `s2.cpp/src/s2_model.cpp` (the ggml
graph, `SlowARModel::eval_cached` / `::fast_decode`) op-for-op and `s2.cpp/models/s2-pro-q6_k.gguf`
for real shapes and hyperparameters. The companion doc is `docs/s2-ar-graph-map.md` -- read that
first for the full op-by-op map, the brick-vocabulary gap analysis, the parity/dump-point spec,
and the L1 sizing note; this module docstring only covers what's specific to running the code.

THE HEADLINE FINDING: every big AR weight (embeddings, codebook_embeddings, wqkv/wo/w1/w2/w3,
fast_*) is GGUF type q6_k. `scripts/gguf_extract.py` (borrowed from `feat/s2-codec-bricks`, also
copied into this worktree) only decodes f32/f16/bf16 -- so by itself it reads NOTHING useful here;
only the tiny RMSNorm gamma vectors (attention_norm/ffn_norm/q_norm/k_norm/fast_norm, all f16) are
readable through it. Rather than stub the graph out with random weights (a previous framing this
task explicitly allows, "if quantization blocks you"), this module PORTS ggml's q6_k dequant
(`ggml/src/ggml-quants.c:dequantize_row_q6_K`, block layout `ggml/src/ggml-common.h:352-358`,
QK_K=256) to vectorized numpy -- see `q6k_dequant_blocks()` below. That was verified byte-exact
against a scalar transliteration of the same C function on real bytes from the shipped GGUF (see
this file's own `__main__` self-check, function `_selftest_q6k_matches_scalar_ref`) before being
trusted for anything downstream. Once that lands, EVERY weight this graph needs is readable, and
this reference runs on REAL trained weights, not placeholders -- `__main__` prints dequantized
weight statistics (rms/min/max) as evidence, and runs small real forward passes end to end.

WHAT'S VERIFIED vs WHAT'S INFERRED (be honest about the boundary):
  VERIFIED   -- op graph and shapes: read directly from `s2_model.cpp` + the GGUF's own tensor
                table and KV metadata (this file's `read_ar_hparams` reads the SAME keys with the
                SAME defaults as `SlowARModel::load_shared`, so a drifted GGUF fails loud, not
                silently). q6_k dequant: verified byte-exact vs a scalar port of the ggml C source
                on real GGUF bytes (see self-test). RoPE convention: `ggml_rope_ext(..., mode=0,
                ...)` is `GGML_ROPE_TYPE_NORMAL` (`ggml.h:250`, `1809`) -- ADJACENT-PAIR rotation
                ("[cscs0000]" per that header's own diagram), NOT the split-half NEOX convention.
  INFERRED   -- that a from-scratch (non-KV-cached) recompute of the whole sequence at every AR
                step is mathematically equivalent to `eval_cached`'s incremental KV-cache path.
                This follows from attention being a pure function of the full causal history (the
                cache is a speed optimization, not a different computation), but was not diffed
                token-by-token against a live s2.cpp run -- that requires the new dump point this
                script is designed to consume once it exists (see docs/s2-ar-graph-map.md #3).
  NOT BUILT  -- the tokenizer / prompt template (interleaving text tokens with prior-frame
                codebook tokens into the `[n_tokens, num_codebooks+1]` flat_tokens contract) --
                out of scope per the task framing (this mirrors the codec side already drawing the
                line at "codes -> audio", not "text -> codes -> audio"). `generate_greedy` takes a
                pre-built flat_tokens prompt as input; the self-check below only exercises the
                graph with a tiny synthetic one.

USAGE:
    python3 scripts/s2_ar_ref.py                    # self-check: hparams, dequant proof, small
                                                      # real-weight forward passes (few seconds)
    python3 scripts/s2_ar_ref.py --report-types      # which AR tensors are readable vs blocked
    python3 scripts/s2_ar_ref.py --full-forward      # one full 36-layer slow forward + full
                                                      # 4-layer fast forward on real weights
                                                      # (dequantizes ~4.5B q6_k elements, ~1 min)
    S2_GGUF=/path/to/model.gguf python3 scripts/s2_ar_ref.py   # override model path

Path resolution follows `scripts/codec_paths.py` (also borrowed): $S2_GGUF env var first, then the
sibling `s2.cpp/models/` checkout layout.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codec_paths  # noqa: E402  (sibling script, path-resolution only)

QK_K = 256
Q6K_BLOCK_BYTES = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2  # ql + qh + scales + d(f16) = 210
GGUF_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1",
    10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k", 15: "q8_k",
    24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64", 30: "bf16",
}
# Types this module can actually turn into a numpy float32 array.
SUPPORTED_GGUF_TYPES = {0, 1, 14, 30}  # f32, f16, q6_k, bf16


# --------------------------------------------------------------------------------------------
# Low-level GGUF reading: tensor directory, KV metadata, byte-range fetch.
# Self-contained (doesn't depend on gguf_extract.py/gguf_shapes.py) because it needs the q6_k
# byte layout, which those two tools deliberately don't implement.
# --------------------------------------------------------------------------------------------

class UnsupportedGGUFType(RuntimeError):
    """Raised when a tensor this module was asked to read is a quant type nobody's ported yet
    (i.e. NOT in SUPPORTED_GGUF_TYPES). This is meant to fail LOUD -- see module docstring's
    "fail loud, not silently" stance: a future model version quantized to, say, q4_k should not
    silently produce garbage floats."""


@dataclass
class GGUFFile:
    path: str
    alignment: int
    data_start: int
    infos: dict  # name -> (dims: tuple[int,...] ne-order, type: int, rel_offset: int)
    kv: dict


def _rd_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _rd_kv_val(f, t):
    scalar = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
              5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
    if t in scalar:
        fmt, sz = scalar[t]
        (v,) = struct.unpack(fmt, f.read(sz))
        return bool(v) if t == 7 else v
    if t == 8:
        return _rd_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_rd_kv_val(f, et) for _ in range(n)]
    raise ValueError(f"unknown GGUF kv type {t}")


def open_gguf(path: str) -> GGUFFile:
    """Parse the GGUF header, KV metadata, and tensor directory. Does not read tensor data."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", f"{path} is not a GGUF file"
    struct.unpack("<I", f.read(4))  # version
    (n_tensors,) = struct.unpack("<Q", f.read(8))
    (n_kv,) = struct.unpack("<Q", f.read(8))

    alignment = 32
    kv = {}
    for _ in range(n_kv):
        key = _rd_str(f)
        (t,) = struct.unpack("<I", f.read(4))
        v = _rd_kv_val(f, t)
        kv[key] = v
        if key == "general.alignment":
            alignment = int(v)

    infos = {}
    for _ in range(n_tensors):
        name = _rd_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        (ty,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        infos[name] = (dims, ty, off)

    pos = f.tell()
    data_start = pos + (-pos) % alignment
    f.close()
    return GGUFFile(path=path, alignment=alignment, data_start=data_start, infos=infos, kv=kv)


def tensor_numel(dims: tuple[int, ...]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def tensor_numpy_shape(dims: tuple[int, ...]) -> tuple[int, ...]:
    """ggml ne is fastest-first; numpy wants slowest-first (matches gguf_extract.py's convention,
    so a linear weight's numpy shape is (out_features, in_features), PyTorch nn.Linear style)."""
    return tuple(reversed(dims))


# --------------------------------------------------------------------------------------------
# q6_k dequantization (the unblock). Ported from ggml/src/ggml-quants.c:dequantize_row_q6_K
# and the block_q6_K layout in ggml/src/ggml-common.h:352-358 (in s2.cpp's vendored ggml copy).
#
#   struct block_q6_K {
#       uint8_t ql[QK_K/2];      // 128 bytes: quants, lower 4 bits
#       uint8_t qh[QK_K/4];      //  64 bytes: quants, upper 2 bits
#       int8_t  scales[QK_K/16]; //  16 bytes: per-16-element scale, signed 8-bit
#       ggml_half d;             //   2 bytes: super-block f16 scale
#   };                           // = 210 bytes for 256 output elements (~6.56 bits/weight)
#
# Dequant per 256-element block, 2 sub-chunks of 128 output elements each:
#   for l in [0,32): is = l//16
#     q1 = ((ql[l]    & 0xF) | ((qh[l]>>0 & 3)<<4)) - 32   -> y[l   ] = d*scales[is+0]*q1
#     q2 = ((ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4)) - 32   -> y[l+32] = d*scales[is+2]*q2
#     q3 = ((ql[l]    >> 4)  | ((qh[l]>>4 & 3)<<4)) - 32   -> y[l+64] = d*scales[is+4]*q3
#     q4 = ((ql[l+32] >> 4)  | ((qh[l]>>6 & 3)<<4)) - 32   -> y[l+96] = d*scales[is+6]*q4
# --------------------------------------------------------------------------------------------

def q6k_dequant_blocks(raw: np.ndarray) -> np.ndarray:
    """raw: (nblocks, 210) uint8, one block_q6_K per row. Returns (nblocks*256,) float32,
    element order matching the source tensor's flat (ggml ne[0]-fastest) byte order."""
    assert raw.ndim == 2 and raw.shape[1] == Q6K_BLOCK_BYTES, raw.shape
    nb = raw.shape[0]
    ql = raw[:, 0:128].reshape(nb, 2, 64).astype(np.int16)
    qh = raw[:, 128:192].reshape(nb, 2, 32).astype(np.int16)
    sc = raw[:, 192:208].copy().view(np.int8).reshape(nb, 2, 8).astype(np.int16)
    d = raw[:, 208:210].copy().view(np.float16).reshape(nb, 1, 1).astype(np.float32)

    l = np.arange(32)
    isv = l // 16  # (32,) in {0,1}

    ql_l = ql[:, :, l]        # (nb,2,32)
    ql_l32 = ql[:, :, l + 32]
    qh_l = qh[:, :, l]

    q1 = ((ql_l & 0xF) | (((qh_l >> 0) & 3) << 4)) - 32
    q2 = ((ql_l32 & 0xF) | (((qh_l >> 2) & 3) << 4)) - 32
    q3 = ((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) - 32
    q4 = ((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) - 32

    y1 = d * sc[:, :, isv + 0] * q1
    y2 = d * sc[:, :, isv + 2] * q2
    y3 = d * sc[:, :, isv + 4] * q3
    y4 = d * sc[:, :, isv + 6] * q4

    out = np.empty((nb, 2, 128), dtype=np.float32)
    out[:, :, 0:32] = y1
    out[:, :, 32:64] = y2
    out[:, :, 64:96] = y3
    out[:, :, 96:128] = y4
    return out.reshape(-1)


def read_tensor(gg: GGUFFile, name: str, row_range: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Read one tensor as float32, in numpy C-order (see tensor_numpy_shape). Supports
    f32/f16/bf16 (like gguf_extract.py) plus q6_k (this module's addition).

    row_range=(lo,hi): only read numpy-shape rows [lo,hi) of a 2D tensor. Each GGUF row is
    ne[0] contiguous elements; for q6_k that's only exact (whole number of 256-blocks per row)
    when ne[0] % 256 == 0, which holds for every 2D weight in this model (dim=2560=10*256,
    ff=9728=38*256, q_size=4096=16*256, kv-concat=6144=24*256) -- asserted below rather than
    assumed. This turns an embedding gather (a handful of rows out of 155776) into reading a
    few KB instead of dequantizing the whole ~327 MB table.
    """
    if name not in gg.infos:
        raise KeyError(f"tensor not found in {gg.path}: {name}")
    dims, ty, off = gg.infos[name]
    n = tensor_numel(dims)

    if ty not in SUPPORTED_GGUF_TYPES:
        raise UnsupportedGGUFType(
            f"{name}: GGUF type {GGUF_TYPE_NAMES.get(ty, ty)} not supported by read_tensor "
            f"(supported: {sorted(GGUF_TYPE_NAMES[t] for t in SUPPORTED_GGUF_TYPES)})")

    f = open(gg.path, "rb")
    try:
        if row_range is not None:
            assert len(dims) == 2, "row_range only defined for 2D tensors"
            row_elems = dims[0]  # ne[0] = fastest axis = one numpy row's element count
            lo, hi = row_range
            assert 0 <= lo < hi <= dims[1]
            if ty == 14:
                assert row_elems % QK_K == 0, f"{name}: ne[0]={row_elems} not a multiple of {QK_K}"
                blocks_per_row = row_elems // QK_K
                f.seek(gg.data_start + off + lo * blocks_per_row * Q6K_BLOCK_BYTES)
                nbytes = (hi - lo) * blocks_per_row * Q6K_BLOCK_BYTES
                raw = np.frombuffer(f.read(nbytes), dtype=np.uint8).reshape(-1, Q6K_BLOCK_BYTES)
                return q6k_dequant_blocks(raw).reshape(hi - lo, row_elems)
            itemsize = {0: 4, 1: 2, 30: 2}[ty]
            f.seek(gg.data_start + off + lo * row_elems * itemsize)
            raw = f.read((hi - lo) * row_elems * itemsize)
            return _decode_flat(raw, ty).reshape(hi - lo, row_elems)

        f.seek(gg.data_start + off)
        if ty == 14:
            assert n % QK_K == 0, f"{name}: numel {n} not a multiple of {QK_K}"
            nbytes = (n // QK_K) * Q6K_BLOCK_BYTES
            raw = np.frombuffer(f.read(nbytes), dtype=np.uint8).reshape(-1, Q6K_BLOCK_BYTES)
            flat = q6k_dequant_blocks(raw)
        else:
            itemsize = {0: 4, 1: 2, 30: 2}[ty]
            flat = _decode_flat(f.read(n * itemsize), ty)
        return flat.reshape(tensor_numpy_shape(dims))
    finally:
        f.close()


def _decode_flat(raw: bytes, ty: int) -> np.ndarray:
    if ty == 0:
        return np.frombuffer(raw, dtype=np.float32).copy()
    if ty == 1:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if ty == 30:
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).copy()
    raise UnsupportedGGUFType(f"GGUF type {ty}")


def report_tensor_types(gg: GGUFFile, names: list[str]) -> list[tuple[str, str, bool]]:
    """[(name, type_str, is_readable)] for the given tensor names -- the 'testable blocker
    report' the task asks for. Currently everything in this model IS readable (q6_k + f16), but
    the mechanism stays generic: a future model with e.g. q4_k tensors reports them as blocked
    instead of read_tensor() silently doing the wrong thing."""
    out = []
    for name in names:
        if name not in gg.infos:
            out.append((name, "MISSING", False))
            continue
        _, ty, _ = gg.infos[name]
        type_str = GGUF_TYPE_NAMES.get(ty, f"unknown({ty})")
        out.append((name, type_str, ty in SUPPORTED_GGUF_TYPES))
    return out


# --------------------------------------------------------------------------------------------
# Hyperparameters -- same GGUF keys and defaults as SlowARModel::load_shared in s2_model.cpp
# (lines 245-288 as read for this task). Kept as a flat dataclass rather than reusing ggml's
# struct so this file has zero non-numpy dependencies.
# --------------------------------------------------------------------------------------------

@dataclass
class ARHParams:
    context_length: int = 32768
    vocab_size: int = 155776
    embedding_length: int = 2560
    feed_forward_length: int = 9728
    block_count: int = 36
    head_count: int = 32
    head_count_kv: int = 8
    rope_freq_base: float = 1e6
    rms_norm_eps: float = 1e-6
    codebook_size: int = 4096
    num_codebooks: int = 10
    semantic_begin_id: int = 151678
    semantic_end_id: int = 155773
    tie_word_embeddings: bool = True
    attention_qk_norm: bool = False
    scale_codebook_embeddings: bool = False

    fast_context_length: int = 11
    fast_embedding_length: int = 2560
    fast_feed_forward_length: int = 9728
    fast_block_count: int = 4
    fast_head_count: int = 32
    fast_head_count_kv: int = 8
    fast_head_dim: int = 128
    fast_rope_freq_base: float = 1e6
    fast_rms_norm_eps: float = 1e-6
    fast_attention_qk_norm: bool = False
    fast_has_project_in: bool = False
    has_fast_decoder: bool = False

    im_end_id: int = -1  # tokenizer special token, not in this GGUF's hparams; caller-supplied

    @property
    def head_dim(self) -> int:
        return self.embedding_length // self.head_count if not self.attention_qk_norm else 128

    @property
    def codebook_dim(self) -> int:
        """flat_tokens row width: 1 semantic/text id + num_codebooks RVQ ids."""
        return self.num_codebooks + 1


def read_ar_hparams(gg: GGUFFile) -> ARHParams:
    kv = gg.kv
    arch = kv.get("general.architecture", "fish-speech")
    arch_prefix = arch + "."
    has_fast = (arch == "fish-speech")

    def g(key, default):
        return kv.get(key, default)

    hp = ARHParams(
        context_length=int(g(arch_prefix + "context_length", 32768)),
        vocab_size=int(g(arch_prefix + "vocab_size", 155776)),
        embedding_length=int(g(arch_prefix + "embedding_length", 2560)),
        feed_forward_length=int(g(arch_prefix + "feed_forward_length", 9728)),
        block_count=int(g(arch_prefix + "block_count", 36)),
        head_count=int(g(arch_prefix + "attention.head_count", 32)),
        head_count_kv=int(g(arch_prefix + "attention.head_count_kv", 8)),
        rope_freq_base=float(g(arch_prefix + "rope.freq_base", 1e6)),
        rms_norm_eps=float(g(arch_prefix + "attention.layer_norm_rms_epsilon", 1e-6)),
        codebook_size=int(g("fish_speech.codebook_size", 4096)),
        num_codebooks=int(g("fish_speech.num_codebooks", 10)),
        semantic_begin_id=int(g("fish_speech.semantic_begin_id", 151678)),
        semantic_end_id=int(g("fish_speech.semantic_end_id", 155773)),
        tie_word_embeddings=bool(g("fish_speech.tie_word_embeddings", True)),
        attention_qk_norm=bool(g("fish_speech.attention_qk_norm", False)),
        scale_codebook_embeddings=bool(g("fish_speech.scale_codebook_embeddings", False)),
        has_fast_decoder=has_fast,
    )
    if has_fast:
        hp.fast_context_length = int(g("fish_speech.fast_context_length", 11))
        hp.fast_embedding_length = int(g("fish_speech.fast_embedding_length", 2560))
        hp.fast_feed_forward_length = int(g("fish_speech.fast_feed_forward_length", 9728))
        hp.fast_block_count = int(g("fish_speech.fast_block_count", 4))
        hp.fast_head_count = int(g("fish_speech.fast_head_count", 32))
        hp.fast_head_count_kv = int(g("fish_speech.fast_head_count_kv", 8))
        hp.fast_head_dim = int(g("fish_speech.fast_head_dim", 128))
        hp.fast_rope_freq_base = float(g("fish_speech.fast_rope_freq_base", 1e6))
        hp.fast_rms_norm_eps = float(g("fish_speech.fast_layer_norm_rms_eps", 1e-6))
        hp.fast_attention_qk_norm = bool(g("fish_speech.fast_attention_qk_norm", False))
        hp.fast_has_project_in = bool(g("fish_speech.fast_project_in", False))
    hp.im_end_id = int(g("fish_speech.audio_pad_token_id", -1))
    return hp


# All (needed_tensor_name_template, count) the AR half reads -- used by report_tensor_types and
# by the loaders below.
SLOW_TOP_LEVEL = ["embeddings.weight", "codebook_embeddings.weight", "norm.weight"]
FAST_TOP_LEVEL = ["fast_embeddings.weight", "fast_norm.weight", "fast_output.weight"]


def slow_layer_names(il: int, qk_norm: bool) -> list[str]:
    s = f"layers.{il}."
    names = [s + "attention_norm.weight", s + "ffn_norm.weight", s + "attention.wqkv.weight",
             s + "attention.wo.weight", s + "feed_forward.w1.weight", s + "feed_forward.w2.weight",
             s + "feed_forward.w3.weight"]
    if qk_norm:
        names += [s + "attention.q_norm.weight", s + "attention.k_norm.weight"]
    return names


def fast_layer_names(il: int, qk_norm: bool) -> list[str]:
    s = f"fast_layers.{il}."
    names = [s + "attention_norm.weight", s + "ffn_norm.weight", s + "attention.wqkv.weight",
             s + "attention.wo.weight", s + "feed_forward.w1.weight", s + "feed_forward.w2.weight",
             s + "feed_forward.w3.weight"]
    if qk_norm:
        names += [s + "attention.q_norm.weight", s + "attention.k_norm.weight"]
    return names


# --------------------------------------------------------------------------------------------
# Weight containers -- dequantize lazily and cache, so a small self-check (n_layers=2) doesn't
# pay for the other 34.
# --------------------------------------------------------------------------------------------

@dataclass
class LayerWeights:
    attention_norm: np.ndarray
    ffn_norm: np.ndarray
    wqkv: np.ndarray
    wo: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    q_norm: Optional[np.ndarray] = None
    k_norm: Optional[np.ndarray] = None


class ARWeights:
    """Lazy weight loader over the GGUF. `.slow_layer(i)` / `.fast_layer(i)` dequantize and
    cache on first access. Embedding tables are read ROW-WISE (see read_tensor's row_range) so
    a forward pass over a short prompt never materializes the full 155776x2560 / 40960x2560
    tables -- only `norm.weight`/logits-projection paths touch the full embeddings table."""

    def __init__(self, gg: GGUFFile, hp: ARHParams):
        self.gg = gg
        self.hp = hp
        self._slow_cache: dict[int, LayerWeights] = {}
        self._fast_cache: dict[int, LayerWeights] = {}
        self.norm = read_tensor(gg, "norm.weight")
        self.fast_norm = read_tensor(gg, "fast_norm.weight") if hp.has_fast_decoder else None
        self.fast_output = read_tensor(gg, "fast_output.weight") if hp.has_fast_decoder else None

    def embedding_rows(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        lo, hi = int(ids.min()), int(ids.max()) + 1
        block = read_tensor(self.gg, "embeddings.weight", row_range=(lo, hi))
        return block[ids - lo]

    def codebook_embedding_rows(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        lo, hi = int(ids.min()), int(ids.max()) + 1
        block = read_tensor(self.gg, "codebook_embeddings.weight", row_range=(lo, hi))
        return block[ids - lo]

    def fast_embedding_rows(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        lo, hi = int(ids.min()), int(ids.max()) + 1
        block = read_tensor(self.gg, "fast_embeddings.weight", row_range=(lo, hi))
        return block[ids - lo]

    def logits_rows(self, row_ids: np.ndarray) -> np.ndarray:
        """Rows of the (tied) output projection == same table as embedding_rows, ne[1]=vocab."""
        return self.embedding_rows(row_ids)

    def logits_full(self) -> np.ndarray:
        """Full (vocab, dim) tied projection matrix -- ~400M q6_k elements, ~5s. Only for
        --full-forward / real bit-parity runs; the generation driver uses logits_rows instead."""
        return read_tensor(self.gg, "embeddings.weight")

    def slow_layer(self, il: int) -> LayerWeights:
        if il not in self._slow_cache:
            names = slow_layer_names(il, self.hp.attention_qk_norm)
            vals = {n.split(".")[-2]: read_tensor(self.gg, n) for n in names}
            self._slow_cache[il] = LayerWeights(
                attention_norm=vals["attention_norm"], ffn_norm=vals["ffn_norm"],
                wqkv=vals["wqkv"], wo=vals["wo"], w1=vals["w1"], w2=vals["w2"], w3=vals["w3"],
                q_norm=vals.get("q_norm"), k_norm=vals.get("k_norm"))
        return self._slow_cache[il]

    def fast_layer(self, il: int) -> LayerWeights:
        if il not in self._fast_cache:
            names = fast_layer_names(il, self.hp.fast_attention_qk_norm)
            vals = {n.split(".")[-2]: read_tensor(self.gg, n) for n in names}
            self._fast_cache[il] = LayerWeights(
                attention_norm=vals["attention_norm"], ffn_norm=vals["ffn_norm"],
                wqkv=vals["wqkv"], wo=vals["wo"], w1=vals["w1"], w2=vals["w2"], w3=vals["w3"],
                q_norm=vals.get("q_norm"), k_norm=vals.get("k_norm"))
        return self._fast_cache[il]


# --------------------------------------------------------------------------------------------
# Op primitives. Each mirrors one ggml call in s2_model.cpp exactly (function docstring cites
# the C++ call it matches). All internal math in float32 (ggml's compute dtype for this graph).
# --------------------------------------------------------------------------------------------

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Matches rms_norm_weighted() (s2_model.cpp:46-55): ggml_rms_norm then elementwise mul by
    a (possibly-repeated) weight over the last axis."""
    x = x.astype(np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + eps)
    return (x * inv_rms) * weight.astype(np.float32)


def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """y = x @ w.T. w is (out,in) numpy-shape (see tensor_numpy_shape), matching ggml_mul_mat(w,
    x): mul_mat's ne[0] is the shared contraction axis on both operands."""
    return x.astype(np.float32) @ w.astype(np.float32).T


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def swiglu_ffn(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    """gate=w1(x), up=w3(x), h=silu(gate)*up, out=w2(h). Matches ggml_swiglu_split(gate,up) ==
    silu(src0)*src1 with src0=gate,src1=up (ggml-cpu/ops.cpp ggml_vec_swiglu_f32: y=silu(x)*g,
    called with x=src0=gate,g=src1=up)."""
    gate = linear(x, w1)
    up = linear(x, w3)
    return linear(silu(gate) * up, w2)


def rope_interleaved(x: np.ndarray, positions: np.ndarray, n_dims: int, base: float) -> np.ndarray:
    """x: (n_tokens, n_heads, head_dim). Matches ggml_rope_ext(..., mode=0, ...) ==
    GGML_ROPE_TYPE_NORMAL: ADJACENT-PAIR rotation (dims (0,1),(2,3),... each get one (cos,sin)),
    NOT the NEOX split-half convention (ggml.h:250 GGML_ROPE_TYPE_NORMAL=0 vs NEOX=2; ggml.h:1809
    diagrams NORMAL n_dims=4 as "[cscs0000]"). ext_factor=0 in every call site here, i.e. plain
    RoPE (no YaRN scaling) -- freq_scale=1, attn_factor=1 collapse rope_yarn to the textbook form.
    n_dims == head_dim in every call site in this model (full rotary, no partial-rotary tail)."""
    n_tokens, n_heads, head_dim = x.shape
    assert n_dims == head_dim, "partial rotary not exercised by this model; extend if needed"
    half = n_dims // 2
    theta_scale = base ** (-2.0 / n_dims)
    inv_freq = theta_scale ** np.arange(half, dtype=np.float64)  # base^(-2i/n_dims)
    theta = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None, :]  # (n_tokens,half)
    cos = np.cos(theta).astype(np.float32)[:, None, :]  # (n_tokens,1,half)
    sin = np.sin(theta).astype(np.float32)[:, None, :]

    xr = x.reshape(n_tokens, n_heads, half, 2)
    x0 = xr[..., 0]
    x1 = xr[..., 1]
    out = np.empty_like(xr)
    out[..., 0] = x0 * cos - x1 * sin
    out[..., 1] = x0 * sin + x1 * cos
    return out.reshape(n_tokens, n_heads, n_dims)


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """x: (n_tokens, n_head_kv, head_dim) -> (n_tokens, n_head_kv*n_rep, head_dim), REPEAT-
    INTERLEAVE (each kv head's block of n_rep consecutive output heads maps to it), matching
    repeat_interleave_heads() (s2_model.cpp:57-68): the ggml_repeat broadcast inserts the repeat
    axis BETWEEN head_dim and n_head_kv in ne-order, so after flattening, repeat-index is faster-
    varying than kv-head-index -- i.e. np.repeat(x, n_rep, axis=1), not np.tile."""
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=1)


def causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    """q: (n_tokens,n_head,head_dim); k,v: (n_tokens,n_head,head_dim) (already GQA-expanded to
    n_head via repeat_kv). n_past=0 causal mask (query i attends keys 0..i) -- matches
    ggml_diag_mask_inf(..., n_past_) specialized to a from-scratch (non-cached) recompute of the
    whole sequence, which is what this reference always does (see module docstring, INFERRED).
    Returns (n_tokens,n_head,head_dim)."""
    n_tokens, n_head, head_dim = q.shape
    scores = np.einsum("qhd,khd->hqk", q, k) * scale  # (n_head,n_tokens,n_tokens)
    mask = np.triu(np.full((n_tokens, n_tokens), -np.inf, dtype=np.float32), k=1)
    scores = scores + mask[None, :, :]
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = np.einsum("hqk,khd->qhd", probs, v)  # (n_tokens,n_head,head_dim)
    return out


# --------------------------------------------------------------------------------------------
# Slow (main) transformer forward. Mirrors SlowARModel::eval_cached (s2_model.cpp:860-1097),
# minus KV-cache bookkeeping: this recomputes attention over the FULL given sequence every call
# (see module docstring, INFERRED-equivalence note).
# --------------------------------------------------------------------------------------------

def slow_transformer_forward(hp: ARHParams, w: ARWeights, flat_tokens: np.ndarray,
                              n_layers: Optional[int] = None,
                              logits_row_ids: Optional[np.ndarray] = None):
    """flat_tokens: (n_tokens, codebook_dim) int, row t = [semantic_or_text_id, cb0_id, ..,
    cb{K-1}_id] (K=num_codebooks). Returns (hidden_last[dim], logits) where logits covers
    `logits_row_ids` if given (else the full vocab -- ~5s extra, see ARWeights.logits_full).

    Op order per layer (exact, from eval_cached): rmsnorm(attn_in) -> wqkv -> split q/k/v ->
    [qk_norm] -> RoPE(q,k) -> GQA-repeat(k,v) -> causal attn -> wo -> residual -> rmsnorm(ffn_in)
    -> swiglu(w1,w2,w3) -> residual. Final: rmsnorm(norm) -> last-token hidden -> tied logits.
    """
    n_tokens, cbdim = flat_tokens.shape
    assert cbdim == hp.codebook_dim, (cbdim, hp.codebook_dim)
    n_layers = hp.block_count if n_layers is None else n_layers

    semantic_ids = flat_tokens[:, 0]
    is_semantic = (semantic_ids >= hp.semantic_begin_id) & (semantic_ids <= hp.semantic_end_id)

    x = w.embedding_rows(semantic_ids).astype(np.float32)  # (n_tokens, dim)

    codebook_sum = np.zeros_like(x)
    for cb in range(hp.num_codebooks):
        raw_ids = flat_tokens[:, cb + 1]
        ids = np.where(is_semantic, raw_ids + cb * hp.codebook_size, cb * hp.codebook_size)
        codebook_sum += w.codebook_embedding_rows(ids)
    x = x + codebook_sum * is_semantic[:, None].astype(np.float32)

    if hp.scale_codebook_embeddings:
        sem_scale = 1.0 / np.sqrt(hp.codebook_dim)
        token_scale = np.where(is_semantic, sem_scale, 1.0).astype(np.float32)
        x = x * token_scale[:, None]

    n_head, n_head_kv = hp.head_count, hp.head_count_kv
    head_dim = hp.head_dim
    q_size, kv_size = n_head * head_dim, n_head_kv * head_dim
    n_rep = n_head // n_head_kv
    scale = 1.0 / np.sqrt(head_dim)
    positions = np.arange(n_tokens)

    for il in range(n_layers):
        lw = w.slow_layer(il)
        attn_in = rms_norm(x, lw.attention_norm, hp.rms_norm_eps)
        qkv = linear(attn_in, lw.wqkv)
        q = qkv[:, 0:q_size].reshape(n_tokens, n_head, head_dim)
        k = qkv[:, q_size:q_size + kv_size].reshape(n_tokens, n_head_kv, head_dim)
        v = qkv[:, q_size + kv_size:q_size + 2 * kv_size].reshape(n_tokens, n_head_kv, head_dim)

        if hp.attention_qk_norm:
            q = rms_norm(q, lw.q_norm, hp.rms_norm_eps)
            k = rms_norm(k, lw.k_norm, hp.rms_norm_eps)

        q = rope_interleaved(q, positions, head_dim, hp.rope_freq_base)
        k = rope_interleaved(k, positions, head_dim, hp.rope_freq_base)

        k_rep = repeat_kv(k, n_rep)
        v_rep = repeat_kv(v, n_rep)
        attn = causal_attention(q, k_rep, v_rep, scale).reshape(n_tokens, q_size)
        attn_out = linear(attn, lw.wo)

        h = x + attn_out
        ff_in = rms_norm(h, lw.ffn_norm, hp.rms_norm_eps)
        ff_out = swiglu_ffn(ff_in, lw.w1, lw.w2, lw.w3)
        x = h + ff_out

    slow_out = rms_norm(x, w.norm, hp.rms_norm_eps)
    hidden_last = slow_out[-1]  # (dim,)

    if logits_row_ids is not None:
        rows = w.logits_rows(logits_row_ids)
        logits = rows @ hidden_last
    else:
        rows = w.logits_full()
        logits = rows @ hidden_last
    return hidden_last, logits


# --------------------------------------------------------------------------------------------
# Fast (residual codebook) transformer forward. Mirrors SlowARModel::fast_decode
# (s2_model.cpp:1099-1255).
# --------------------------------------------------------------------------------------------

def fast_transformer_forward(hp: ARHParams, w: ARWeights, hidden: np.ndarray,
                              prefix_codes: list[int]) -> np.ndarray:
    """hidden: (embedding_length,) -- the slow model's hidden_last for this frame. prefix_codes:
    already-decided codebook token ids for this frame (codebook 0's sem_code is NOT included
    here -- fast_decode's loop starts at cb_idx=1, see s2_generate.cpp:131-143). Returns
    (codebook_size,) logits for the NEXT residual codebook.

    fast_project_in is None in this GGUF (fast_has_project_in=False, fast_embedding_length ==
    embedding_length == 2560) so hidden feeds the Fast stack directly; the projection path is
    still implemented for completeness (untested against this checkpoint)."""
    assert hp.has_fast_decoder
    assert len(prefix_codes) < hp.num_codebooks

    x0 = hidden.astype(np.float32)[None, :]  # (1, fast_dim)
    if len(prefix_codes) > 0:
        prefix_emb = w.fast_embedding_rows(np.asarray(prefix_codes, dtype=np.int64))
        x = np.concatenate([x0, prefix_emb], axis=0)
    else:
        x = x0
    n_tokens = x.shape[0]
    positions = np.arange(n_tokens)

    n_head, n_head_kv = hp.fast_head_count, hp.fast_head_count_kv
    head_dim = hp.fast_head_dim
    q_size, kv_size = n_head * head_dim, n_head_kv * head_dim
    n_rep = n_head // n_head_kv
    scale = 1.0 / np.sqrt(head_dim)

    for il in range(hp.fast_block_count):
        lw = w.fast_layer(il)
        attn_in = rms_norm(x, lw.attention_norm, hp.fast_rms_norm_eps)
        qkv = linear(attn_in, lw.wqkv)
        q = qkv[:, 0:q_size].reshape(n_tokens, n_head, head_dim)
        k = qkv[:, q_size:q_size + kv_size].reshape(n_tokens, n_head_kv, head_dim)
        v = qkv[:, q_size + kv_size:q_size + 2 * kv_size].reshape(n_tokens, n_head_kv, head_dim)

        if hp.fast_attention_qk_norm:
            q = rms_norm(q, lw.q_norm, hp.fast_rms_norm_eps)
            k = rms_norm(k, lw.k_norm, hp.fast_rms_norm_eps)

        q = rope_interleaved(q, positions, head_dim, hp.fast_rope_freq_base)
        k = rope_interleaved(k, positions, head_dim, hp.fast_rope_freq_base)

        k_rep = repeat_kv(k, n_rep)
        v_rep = repeat_kv(v, n_rep)
        attn = causal_attention(q, k_rep, v_rep, scale).reshape(n_tokens, q_size)
        attn_out = linear(attn, lw.wo)

        h = x + attn_out
        ff_in = rms_norm(h, lw.ffn_norm, hp.fast_rms_norm_eps)
        ff_out = swiglu_ffn(ff_in, lw.w1, lw.w2, lw.w3)
        x = h + ff_out

    fast_out = rms_norm(x, w.fast_norm, hp.fast_rms_norm_eps)
    last = fast_out[-1]
    return last @ w.fast_output.T


# --------------------------------------------------------------------------------------------
# End-to-end greedy driver. Mirrors s2::generate() (s2_generate.cpp:11-216) with temperature=0.
#
# WHY temperature=0 gives exact parity: sample_token (s2_sampler.cpp:42-99) sorts logits
# descending, applies top_k/top_p FILTERING (which only REMOVES lower-ranked entries, never
# reorders), then: `if (params.temperature <= 0.0f) return filtered[0].second;` -- filtered[0]
# is always the same as the pre-filter argmax regardless of top_k/top_p, because filtering never
# promotes anything above rank 0. So temperature=0 greedy == masked argmax, independent of
# top_p/top_k, and needs no RNG.
#
# CAVEAT (real finding, not hedging): generate()'s RAS window (s2_generate.cpp:99-116) can
# override temperature=0 and re-sample at temperature=1.0/top_p=0.9 -- WITH the process's
# thread_local std::mt19937(random_device{}()) -- whenever the just-picked semantic token repeats
# within the last 10 semantic tokens. That reintroduces real, unseeded randomness even in a
# "temperature=0" s2.cpp run. This driver does NOT implement the RAS escape hatch (there is no
# way to make its RNG bit-match libstdc++'s mt19937/discrete_distribution from numpy even if it
# did), so exact end-to-end parity against a live s2.cpp run holds only for however many frames
# elapse before RAS would first trigger. Keep parity runs short, or gate on the pre-sampling
# logits cut instead (see docs/s2-ar-graph-map.md #3) which sidesteps this entirely.
# --------------------------------------------------------------------------------------------

def generate_greedy(hp: ARHParams, w: ARWeights, prompt: np.ndarray, max_new_tokens: int,
                     n_slow_layers: Optional[int] = None):
    """prompt: (n_prompt_tokens, codebook_dim) int, the pre-built flat_tokens prompt (tokenizer/
    prompt-template construction is out of scope, see module docstring). Returns
    codes: (num_codebooks, n_frames) int array, matching s2::GenerateResult.codes' layout."""
    sem_lo, sem_hi = hp.semantic_begin_id, hp.semantic_end_id
    mask_row_ids = np.arange(sem_lo, sem_hi + 1)
    if hp.im_end_id >= 0:
        mask_row_ids = np.concatenate([mask_row_ids, [hp.im_end_id]])

    history = prompt.copy()
    frames = []

    def masked_argmax(logits, row_ids, block_im_end):
        # logits correspond 1:1 to row_ids (restricted-vocab fast path).
        biased = logits.copy()
        if block_im_end and hp.im_end_id >= 0:
            biased[row_ids == hp.im_end_id] = -np.inf
        return int(row_ids[np.argmax(biased)])

    hidden, logits = slow_transformer_forward(hp, w, history, n_layers=n_slow_layers,
                                               logits_row_ids=mask_row_ids)
    main_token = masked_argmax(logits, mask_row_ids, block_im_end=True)

    step = 0
    while main_token != hp.im_end_id and step < max_new_tokens:
        sem_code = int(np.clip(main_token - sem_lo, 0, hp.codebook_size - 1))
        codebooks_cb = [sem_code]
        for cb_idx in range(1, hp.num_codebooks):
            fast_logits = fast_transformer_forward(hp, w, hidden, codebooks_cb)
            codebooks_cb.append(int(np.argmax(fast_logits)))
        frames.append(codebooks_cb)

        new_row = np.array([main_token] + codebooks_cb, dtype=history.dtype)[None, :]
        history = np.concatenate([history, new_row], axis=0)

        step += 1
        hidden, logits = slow_transformer_forward(hp, w, history, n_layers=n_slow_layers,
                                                   logits_row_ids=mask_row_ids)
        main_token = masked_argmax(logits, mask_row_ids, block_im_end=(step < max_new_tokens))

    codes = np.array(frames, dtype=np.int32).T if frames else np.zeros((hp.num_codebooks, 0), np.int32)
    return codes


# --------------------------------------------------------------------------------------------
# Self-check.
# --------------------------------------------------------------------------------------------

def _scalar_q6k_dequant_block(block_bytes: bytes) -> np.ndarray:
    """Direct, unoptimized transliteration of dequantize_row_q6_K's inner loop (one block), used
    ONLY to cross-check q6k_dequant_blocks() -- not used by the actual reference path."""
    ql = np.frombuffer(block_bytes[0:128], dtype=np.uint8).astype(np.int32)
    qh = np.frombuffer(block_bytes[128:192], dtype=np.uint8).astype(np.int32)
    sc = np.frombuffer(block_bytes[192:208], dtype=np.int8).astype(np.int32)
    d = np.frombuffer(block_bytes[208:210], dtype=np.float16).astype(np.float64)[0]
    y = np.zeros(256, dtype=np.float64)
    ql_o = qh_o = sc_o = y_o = 0
    for _ in range(0, QK_K, 128):
        for l in range(32):
            is_ = l // 16
            q1 = ((ql[ql_o + l] & 0xF) | (((qh[qh_o + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[ql_o + l + 32] & 0xF) | (((qh[qh_o + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[ql_o + l] >> 4) | (((qh[qh_o + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[ql_o + l + 32] >> 4) | (((qh[qh_o + l] >> 6) & 3) << 4)) - 32
            y[y_o + l] = d * sc[sc_o + is_ + 0] * q1
            y[y_o + l + 32] = d * sc[sc_o + is_ + 2] * q2
            y[y_o + l + 64] = d * sc[sc_o + is_ + 4] * q3
            y[y_o + l + 96] = d * sc[sc_o + is_ + 6] * q4
        y_o += 128
        ql_o += 64
        qh_o += 32
        sc_o += 8
    return y


def _selftest_q6k_matches_scalar_ref(gg: GGUFFile):
    dims, ty, off = gg.infos["embeddings.weight"]
    assert ty == 14, "expected embeddings.weight to be q6_k in this checkpoint"
    f = open(gg.path, "rb")
    f.seek(gg.data_start + off)
    raw = np.frombuffer(f.read(5 * Q6K_BLOCK_BYTES), dtype=np.uint8).reshape(5, Q6K_BLOCK_BYTES)
    f.close()
    for bi in range(5):
        scalar = _scalar_q6k_dequant_block(raw[bi].tobytes())
        vec = q6k_dequant_blocks(raw[bi:bi + 1])
        assert np.array_equal(scalar, vec.astype(np.float64)), f"q6_k dequant mismatch at block {bi}"
    print(f"  [ok] q6_k dequant: vectorized == scalar ggml-port, byte-exact, on 5 real blocks")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-types", action="store_true",
                     help="report GGUF type + readability of every AR tensor this graph needs, then exit")
    ap.add_argument("--full-forward", action="store_true",
                     help="run a full 36-layer slow forward + full fast forward on real weights (~1 min)")
    args = ap.parse_args()

    gguf_path = codec_paths.gguf()
    print(f"GGUF: {gguf_path}")
    gg = open_gguf(gguf_path)
    hp = read_ar_hparams(gg)
    print(f"hparams: arch={gg.kv.get('general.architecture')} block_count={hp.block_count} "
          f"embedding_length={hp.embedding_length} head_count={hp.head_count}/{hp.head_count_kv} "
          f"vocab_size={hp.vocab_size} codebook_size={hp.codebook_size} "
          f"num_codebooks={hp.num_codebooks} attention_qk_norm={hp.attention_qk_norm} "
          f"scale_codebook_embeddings={hp.scale_codebook_embeddings}")
    print(f"fast: block_count={hp.fast_block_count} head_count={hp.fast_head_count}/"
          f"{hp.fast_head_count_kv} head_dim={hp.fast_head_dim} qk_norm={hp.fast_attention_qk_norm} "
          f"project_in={hp.fast_has_project_in}")

    needed = list(SLOW_TOP_LEVEL) + list(FAST_TOP_LEVEL)
    for il in range(hp.block_count):
        needed += slow_layer_names(il, hp.attention_qk_norm)
    for il in range(hp.fast_block_count):
        needed += fast_layer_names(il, hp.fast_attention_qk_norm)
    report = report_tensor_types(gg, needed)
    blocked = [r for r in report if not r[2]]

    if args.report_types:
        by_type: dict[str, int] = {}
        for _, t, ok in report:
            by_type[t] = by_type.get(t, 0) + 1
        print(f"\n{len(needed)} AR tensors needed, by GGUF type: {by_type}")
        if blocked:
            print(f"BLOCKED ({len(blocked)}, not decodable by this module):")
            for n, t, _ in blocked:
                print(f"  {n}: {t}")
        else:
            print("BLOCKED: none -- every needed tensor type is supported (f16 norms + q6_k weights).")
        return

    print(f"\n{len(needed)} AR tensors needed; {len(blocked)} blocked "
          f"(supported types: {sorted(GGUF_TYPE_NAMES[t] for t in SUPPORTED_GGUF_TYPES)})")
    assert not blocked, f"cannot proceed, blocked tensors: {blocked[:5]}..."

    print("\n--- q6_k dequant correctness ---")
    _selftest_q6k_matches_scalar_ref(gg)

    print("\n--- weight sanity (real dequantized values, not placeholders) ---")
    w = ARWeights(gg, hp)
    for label, arr in [("norm.weight (f16)", w.norm), ("layers.0.attention.wqkv.weight (q6_k)",
                        read_tensor(gg, "layers.0.attention.wqkv.weight")),
                       ("layers.0.feed_forward.w1.weight (q6_k)",
                        read_tensor(gg, "layers.0.feed_forward.w1.weight"))]:
        arr64 = arr.astype(np.float64)
        rms = float(np.sqrt((arr64 ** 2).mean()))
        print(f"  {label}: shape={arr.shape} finite={bool(np.isfinite(arr).all())} "
              f"rms={rms:.5f} min={arr.min():+.4f} max={arr.max():+.4f}")
        assert np.isfinite(arr).all()
        assert 1e-4 < rms < 5.0, f"implausible weight magnitude for {label}: rms={rms}"

    print("\n--- small real-weight forward pass (n_layers=2, synthetic tiny prompt) ---")
    rng = np.random.default_rng(0)
    n_prompt = 4
    flat = np.zeros((n_prompt, hp.codebook_dim), dtype=np.int64)
    flat[:, 0] = hp.semantic_begin_id + rng.integers(0, hp.codebook_size, size=n_prompt)
    for cb in range(hp.num_codebooks):
        flat[:, cb + 1] = rng.integers(0, hp.codebook_size, size=n_prompt)

    t0 = time.time()
    hidden, logits = slow_transformer_forward(hp, w, flat, n_layers=2,
                                               logits_row_ids=np.arange(hp.semantic_begin_id,
                                                                         hp.semantic_end_id + 1))
    t1 = time.time()
    assert hidden.shape == (hp.embedding_length,)
    assert np.isfinite(hidden).all()
    assert logits.shape == (hp.codebook_size,)
    assert np.isfinite(logits).all()
    print(f"  [ok] slow_transformer_forward(n_layers=2): hidden finite, logits finite, "
          f"{t1-t0:.2f}s, argmax_semantic_code={int(np.argmax(logits))}")

    t0 = time.time()
    fast_logits = fast_transformer_forward(hp, w, hidden, prefix_codes=[])
    t1 = time.time()
    assert fast_logits.shape == (hp.codebook_size,)
    assert np.isfinite(fast_logits).all()
    print(f"  [ok] fast_transformer_forward (full {hp.fast_block_count} layers, empty prefix): "
          f"finite, {t1-t0:.2f}s, argmax={int(np.argmax(fast_logits))}")

    t0 = time.time()
    fast_logits2 = fast_transformer_forward(hp, w, hidden, prefix_codes=[fast_logits.argmax()])
    t1 = time.time()
    assert np.isfinite(fast_logits2).all()
    print(f"  [ok] fast_transformer_forward (prefix len=1): finite, {t1-t0:.2f}s")

    if args.full_forward:
        print("\n--- FULL forward: 36 slow layers + full-vocab logits (~1 min) ---")
        t0 = time.time()
        hidden_f, logits_f = slow_transformer_forward(hp, w, flat, n_layers=None,
                                                        logits_row_ids=None)
        t1 = time.time()
        assert logits_f.shape == (hp.vocab_size,)
        assert np.isfinite(logits_f).all()
        print(f"  [ok] full slow forward: {t1-t0:.1f}s, hidden rms={float(np.sqrt((hidden_f.astype(np.float64)**2).mean())):.4f}, "
              f"argmax_token={int(np.argmax(logits_f))}")

        print("\n--- greedy generation smoke test (temperature=0 semantics, 2 frames, n_layers=2) ---")
        t0 = time.time()
        codes = generate_greedy(hp, w, flat, max_new_tokens=2, n_slow_layers=2)
        t1 = time.time()
        assert codes.shape[0] == hp.num_codebooks
        print(f"  [ok] generate_greedy: codes.shape={codes.shape}, {t1-t0:.1f}s")
        if codes.size:
            print(f"  codes[:, 0] = {codes[:, 0]}")

    print("\nAll self-checks passed.")


if __name__ == "__main__":
    main()
