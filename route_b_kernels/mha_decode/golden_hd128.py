#!/usr/bin/env python3
"""Host numpy golden for the S2 fast-decoder instance of mha_decode (HD=128, 8->32 GQA, n_rep=4).

WHAT THIS PROVES, device-free, before verify_mha_decode_hd128.py ever touches the NPU:

  1. INDEX == EXPAND-THEN-SLICE (the GQA claim in mha_decode.cc's header). Per
     `scripts/s2_ar_ref.py::repeat_kv`'s docstring, GQA is REPEAT-INTERLEAVE
     (`np.repeat(x, n_rep, axis=1)`, NOT `np.tile`): output head `oh` reads KV head `oh // n_rep`.
     `reference_full()` computes attention the reference's way -- expand K/V to n_head via
     `repeat_kv` THEN slice head `oh`. `reference_direct()` never expands: it indexes head
     `oh // n_rep` directly out of the unexpanded K/V. `_selftest_index_equals_expand()` asserts
     these are identical. This is the numeric evidence that mha_decode.cc's kernel (which only
     ever sees ONE head's data per call, GQA or not) can be fed the unexpanded KV head directly --
     the doctrine "index the map, don't materialize the copy" -- without changing the result.

  2. THE KERNEL'S TILE ALGORITHM == THE REFERENCE. `flash_tiled_reference()` is a numpy port of
     `mha_tile_impl`'s actual per-tile online-softmax loop (mha_decode.cc): same reset-on-tile-0,
     same running (m, l, acc), same finalize-on-last-tile. `_selftest_flash_matches_direct()`
     asserts it reproduces `reference_direct()`. "Flash attention is mathematically equivalent to
     softmax" is a standard fact, but a tiling-boundary bug (off-by-one on the last partial tile,
     wrong reset condition) would not show up in that fact alone -- this check is specific to the
     actual loop structure the kernel implements.

  3. `pack_kv_tiles()` builds the exact on-wire bf16 tile layout (K-tile | V-tile | bit-exact
     int32 runtime-S header via 2 bf16 lanes) that `verify_mha_decode_hd128.py` feeds the real
     device kernel -- the same convention as `mha_decode_iron.py` / `mha_decode_probe.rs`.

CONFIG: S2 fast decoder hyperparameters (`scripts/s2_ar_ref.py::ARHParams` defaults --
fast_head_count=32, fast_head_count_kv=8, fast_head_dim=128, fast_rope_freq_base=1e6). n_rep =
32/8 = 4. HD=128 forces TKV=32 to hold the same L1 double-buffer footprint as the shipped HD=64,
TKV=64 kernel -- see mha_decode.cc's "L1 FOOTPRINT" comment for the arithmetic.

Gate the SINGLE-QUERY step (M=1 decode): the reference always recomputes the whole S-token
sequence (see s2_ar_ref.py's module docstring on why that is equivalent to the real KV-cached
incremental path), so "this decode step" = the LAST row of that recompute's output, which is what
`reference_full`/`reference_direct` return.

Usage: python3 golden_hd128.py   (host-only self-checks, no device)
"""
from pathlib import Path

import numpy as np
import ml_dtypes

HD = 128
N_HEAD = 32
N_HEAD_KV = 8
N_REP = N_HEAD // N_HEAD_KV  # 4
ROPE_BASE = 1.0e6  # ARHParams.fast_rope_freq_base default
TKV = 32  # see mha_decode.cc's L1 FOOTPRINT comment: required pairing for MHA_HD=128
SCALE = 1.0 / np.sqrt(HD)

_bf16 = ml_dtypes.bfloat16


def _load_s2_ar_ref():
    """importlib-load scripts/s2_ar_ref.py by path, registering it in sys.modules first --
    required because it uses @dataclass, whose machinery looks the defining class up via
    sys.modules[cls.__module__] (see route_b_kernels/bricks/rope-interleaved/golden.py, which
    uses the identical pattern for the same reason)."""
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[2]  # mha_decode -> route_b_kernels -> repo root
    path = repo / "scripts" / "s2_ar_ref.py"
    spec = importlib.util.spec_from_file_location("s2_ar_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s2_ar_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def bf16_round(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 -> f32, so host reference and device kernel see identical (already-quantized)
    inputs -- same convention as rust/npu-probes/src/bin/mha_decode_probe.rs's bf16_round."""
    return x.astype(_bf16).astype(np.float32)


def build_qkv(seed: int, s_tokens: int):
    """Random q_full (s_tokens, N_HEAD, HD), k_full/v_full (s_tokens, N_HEAD_KV, HD), bf16-rounded,
    RoPE'd (q,k only, per the real op order RoPE(q,k) -> GQA-repeat(k,v) -> causal attn), then
    bf16-rounded again (RoPE's own output re-quantizes going into the next op on a real bf16
    pipeline). positions = arange(s_tokens), matching every call site in s2_ar_ref.py."""
    ar_ref = _load_s2_ar_ref()
    rng = np.random.default_rng(seed)

    # ~N(0,1)-ish via sum of uniforms (same shape of reasoning as the Rust probe): scores with
    # std~1 exercise the softmax properly instead of being near-uniform.
    def randn_like(shape):
        u = rng.random(shape + (3,), dtype=np.float32)
        return (u.sum(axis=-1) - 1.5) * 1.1547

    q_full = bf16_round(randn_like((s_tokens, N_HEAD, HD)).astype(np.float32))
    k_full = bf16_round(randn_like((s_tokens, N_HEAD_KV, HD)).astype(np.float32))
    v_full = bf16_round(randn_like((s_tokens, N_HEAD_KV, HD)).astype(np.float32))

    positions = np.arange(s_tokens)
    q_full = bf16_round(ar_ref.rope_interleaved(q_full, positions, HD, ROPE_BASE))
    k_full = bf16_round(ar_ref.rope_interleaved(k_full, positions, HD, ROPE_BASE))

    return ar_ref, q_full, k_full, v_full


def reference_full(ar_ref, q_full, k_full, v_full, out_head: int):
    """Reference path: repeat_kv-EXPAND k,v to N_HEAD via GQA repeat-interleave, then run the
    full causal_attention and slice `out_head`'s LAST row (the M=1 decode step)."""
    k_rep = ar_ref.repeat_kv(k_full, N_REP)
    v_rep = ar_ref.repeat_kv(v_full, N_REP)
    out = ar_ref.causal_attention(q_full, k_rep, v_rep, SCALE)  # (s_tokens, N_HEAD, HD)
    return out[-1, out_head, :]


def reference_direct(q_full, k_full, v_full, out_head: int):
    """Index-map path: NEVER expand. out_head's KV head is out_head // N_REP (repeat-interleave:
    each KV head's block of N_REP consecutive Q heads maps to it). The query is the LAST token
    (M=1 decode); causal masking is a no-op for the last position (it attends every key 0..S-1),
    so this is plain (non-causal) softmax attention over the unexpanded KV head's full history."""
    h_kv = out_head // N_REP
    q_last = q_full[-1, out_head, :].astype(np.float64)          # (HD,)
    k = k_full[:, h_kv, :].astype(np.float64)                    # (S, HD)
    v = v_full[:, h_kv, :].astype(np.float64)                    # (S, HD)
    scores = (k @ q_last) * SCALE                                # (S,)
    scores = scores - scores.max()
    probs = np.exp(scores)
    probs = probs / probs.sum()
    return (probs @ v).astype(np.float32)                        # (HD,)


def flash_tiled_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float, tkv: int):
    """Numpy port of mha_tile_impl's ACTUAL per-tile online-softmax loop (mha_decode.cc): reset
    (m,l,acc) at tile 0, stream TKV-key tiles updating the running max/denom/accumulator, finalize
    (ctx = acc/l) on the last tile. q: (HD,) f32. k,v: (S,HD) f32. Self-checked against
    `reference_direct` below -- catches tiling-boundary bugs a generic "flash==softmax" argument
    would not (off-by-one on the partial last tile, wrong reset condition, etc)."""
    s_tokens, hd = k.shape
    n_tiles = -(-s_tokens // tkv)  # ceil
    q64 = q.astype(np.float64)
    m = l = 0.0
    acc = np.zeros(hd, dtype=np.float64)
    ctx = None
    for t in range(n_tiles):
        start, end = t * tkv, min(t * tkv + tkv, s_tokens)
        s_real = end - start
        last = t == n_tiles - 1
        if t == 0:
            m, l, acc = -3.0e38, 0.0, np.zeros(hd, dtype=np.float64)
        if s_real == 0:
            continue
        for s in range(start, end):
            score = float(q64 @ k[s].astype(np.float64)) * scale
            m_new = max(m, score)
            corr = np.exp(m - m_new)
            p = np.exp(score - m_new)
            l = l * corr + p
            acc = acc * corr + p * v[s].astype(np.float64)
            m = m_new
        if last:
            ctx = acc / l
    assert ctx is not None, "n_tiles must be >= 1 (s_tokens >= 1)"
    return ctx.astype(np.float32)


def int32_pair_to_bf16(val: int) -> np.ndarray:
    """Bit-exact int32 -> 2x bfloat16-shaped lanes (reinterpret, NOT a numeric conversion) -- the
    RUNTIME-S header transport mha_decode.cc reads via
    `*reinterpret_cast<const int32_t*>(kv + 2*Tkv*Hd)`. Same convention as
    rust/npu-probes/src/bin/mha_decode_probe.rs's `s_in_tile.to_le_bytes()` packing."""
    u16 = np.array([np.int32(val)]).view(np.uint16)  # 2 elements, little-endian lanes
    return u16.view(_bf16)


def pack_kv_tiles(k: np.ndarray, v: np.ndarray, tkv: int) -> np.ndarray:
    """k,v: (S,HD) f32 (already bf16-representable). Returns (n_tiles, 2*tkv*HD+2) bf16 array:
    per tile, K-tile (tkv*HD) | V-tile (tkv*HD) | int32 s_in_tile header (2 bf16 lanes).
    >0 normal tile, <0 last non-empty tile (finalize), matching mha_decode.cc's RUNTIME S."""
    s_tokens, hd = k.shape
    n_tiles = -(-s_tokens // tkv)
    kv_tile_bf16 = 2 * tkv * hd + 2
    out = np.zeros((n_tiles, kv_tile_bf16), dtype=_bf16)
    k_bf = k.astype(_bf16)
    v_bf = v.astype(_bf16)
    for t in range(n_tiles):
        start, end = t * tkv, min(t * tkv + tkv, s_tokens)
        s_real = end - start
        out[t, 0:s_real * hd] = k_bf[start:end].reshape(-1)
        out[t, tkv * hd:tkv * hd + s_real * hd] = v_bf[start:end].reshape(-1)
        last = t == n_tiles - 1
        header = -s_real if last else s_real
        out[t, 2 * tkv * hd:2 * tkv * hd + 2] = int32_pair_to_bf16(header)
    return out


def rel_l2(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(a - b))


def build_case(seed: int = 0, s_tokens: int = 100, out_head: int = 5):
    """One full device-test case: bf16-rounded, RoPE'd q/k/v; expected ctx for `out_head` (via
    both reference paths, asserted equal); the packed device-ready q/kv-tile buffers."""
    ar_ref, q_full, k_full, v_full = build_qkv(seed, s_tokens)
    h_kv = out_head // N_REP

    exp_full = reference_full(ar_ref, q_full, k_full, v_full, out_head)
    exp_direct = reference_direct(q_full, k_full, v_full, out_head)
    exp_flash = flash_tiled_reference(
        q_full[-1, out_head, :], k_full[:, h_kv, :], v_full[:, h_kv, :], SCALE, TKV)

    q_bf = q_full[-1, out_head, :].astype(_bf16)
    kv_bf = pack_kv_tiles(k_full[:, h_kv, :], v_full[:, h_kv, :], TKV)
    n_tiles = kv_bf.shape[0]

    return dict(exp_full=exp_full, exp_direct=exp_direct, exp_flash=exp_flash,
                q_bf=q_bf, kv_bf=kv_bf, n_tiles=n_tiles, out_head=out_head, h_kv=h_kv,
                s_tokens=s_tokens)


def _selftest_index_equals_expand():
    ar_ref, q_full, k_full, v_full = build_qkv(seed=0, s_tokens=100)
    worst = 0.0
    for oh in range(N_HEAD):
        full = reference_full(ar_ref, q_full, k_full, v_full, oh)
        direct = reference_direct(q_full, k_full, v_full, oh)
        worst = max(worst, rel_l2(direct, full))
    print(f"[selftest] index (oh -> oh//n_rep) vs repeat_kv-expand-then-slice, "
          f"all {N_HEAD} heads: worst rel_l2={worst:.3e}")
    assert worst < 1e-5, "index map does NOT match repeat_kv's expand -- np.repeat vs np.tile bug?"


def _selftest_flash_matches_direct():
    case = build_case(seed=0, s_tokens=100, out_head=5)
    r = rel_l2(case["exp_flash"], case["exp_direct"])
    print(f"[selftest] flash-tiled kernel algorithm vs direct-index reference: rel_l2={r:.3e}  "
          f"(n_tiles={case['n_tiles']}, TKV={TKV}, S={case['s_tokens']})")
    assert r < 1e-5, "flash-tiled numpy port disagrees with the reference -- tiling bug"


if __name__ == "__main__":
    _selftest_index_equals_expand()
    _selftest_flash_matches_direct()
    print("All host self-checks passed.")
