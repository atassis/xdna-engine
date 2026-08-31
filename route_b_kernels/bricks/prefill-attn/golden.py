#!/usr/bin/env python3
"""Host numpy golden for the prefill-attn brick: causal, GQA, no-relative-position attention over
the S2 AR fast decoder's short prefill sequence (M = fast_context_length <= 11).

ORACLE (scripts/s2_ar_ref.py, loaded by path below so its @dataclass machinery resolves via
sys.modules -- same pattern route_b_kernels/mha_decode/golden_hd128.py uses, for the same reason):
  * `causal_attention(q, k, v, scale)` (~line 582): q (n_tokens,n_head,head_dim); k,v
    (n_tokens,n_head,head_dim) ALREADY GQA-expanded to n_head. n_past=0 causal mask (query i
    attends keys 0..i, `np.triu(-inf, k=1)` additive mask). Returns (n_tokens,n_head,head_dim).
  * `repeat_kv(x, n_rep)` (~line 571) -- docstring: REPEAT-INTERLEAVE, `np.repeat(x, n_rep,
    axis=1)`, explicitly NOT `np.tile`. Output head `oh` therefore reads KV head `oh // n_rep`
    (each KV head's block of n_rep CONSECUTIVE output heads maps to it). This is the exact
    convention `_selftest_gqa_mapping_has_teeth` below proves this brick's test data cannot pass
    if it were backwards (oh % n_head_kv, the np.tile convention).
  Fast-decoder shapes (s2_ar_ref.py:708-711, `fast_transformer_forward`): head_dim=128
  (fast_head_dim default), n_head=32 (fast_head_count), n_head_kv=8 (fast_head_count_kv), n_rep=4,
  scale=1/sqrt(head_dim). fast_context_length<=11 (s2_ar_ref.py:337) is M.

WHAT THIS PROVES, device-free, before verify_prefill_attn.py ever touches the NPU:

  1. INDEX == EXPAND-THEN-SLICE, for the FULL PREFILL (all M rows), not just mha_decode's M=1
     decode-step slice. `reference_full()` expands K/V to n_head via `repeat_kv` then runs the
     real `causal_attention` and slices head `oh`. `reference_direct()` never expands: it indexes
     head `oh // n_rep` directly and reimplements the SAME causal-softmax math the kernel itself
     does (row-by-row, `np.triu` mask). `_selftest_index_equals_expand()` asserts these agree
     across all 32 heads -- the numeric evidence that prefill_attn.cc (which only ever sees ONE
     head's data per call) can be fed the unexpanded KV head directly.

  2. THE CAUSAL MASK HAS TEETH. `_selftest_causal_mask_has_teeth()` proves two things about the
     TEST DATA itself (so a broken, unmasked kernel could not pass this gate by accident): row 0
     of a correct causal reference must equal V[0] EXACTLY (position 0 can only ever see key 0);
     and the causal and a deliberately-bidirectional (unmasked) reference on the SAME q/k/v
     diverge well above the 3e-2 device gate.

  3. THE GQA INDEX MAP HAS TEETH. `_selftest_gqa_mapping_has_teeth()` proves the repeat-interleave
     (`oh // n_rep`) and tile (`oh % n_head_kv`) conventions actually disagree on this test data
     for heads outside the first KV group -- so a backwards driver could not pass silently either.

  4. THE MASK-AS-DATA DESIGN HAS TEETH. prefill_attn.cc's kernel takes its causal mask as an
     additive DATA row (packed alongside q_row in the streamed operand) rather than a `row_idx`
     scalar + branch -- see the kernel's own "MASK IS DATA" header for why. Data is easier to get
     silently wrong than a branch (a packing-offset bug could deliver an all-zero mask and nothing
     would complain at the type level). `_selftest_mask_data_has_teeth()` proves an all-zero mask
     reduces to the bidirectional (unmasked) reference and the real causal mask reduces to
     reference_direct, with the two well-separated -- so a broken (e.g. all-zero) mask cannot pass
     this brick's gate by accident.

Usage: python3 golden.py   (host-only self-checks, no device)
"""
from pathlib import Path

import numpy as np
import ml_dtypes

HD = 128
N_HEAD = 32
N_HEAD_KV = 8
N_REP = N_HEAD // N_HEAD_KV  # 4
ROPE_BASE = 1.0e6  # ARHParams.fast_rope_freq_base default
M = 11  # fast_context_length default (s2_ar_ref.py:337) -- matches PREFILL_M's default in the .cc
SCALE = 1.0 / np.sqrt(HD)
SPAD = 16  # prefill_attn.cc's softmax_core<VL> chunk width; M<=SPAD always for this brick.

_bf16 = ml_dtypes.bfloat16


def _load_s2_ar_ref():
    """importlib-load scripts/s2_ar_ref.py by path, registering it in sys.modules first --
    required because it uses @dataclass, whose machinery looks the defining class up via
    sys.modules[cls.__module__] (same pattern as route_b_kernels/mha_decode/golden_hd128.py and
    route_b_kernels/bricks/rope-interleaved/golden.py)."""
    import importlib.util
    import sys

    # this file -> prefill-attn -> bricks -> route_b_kernels -> repo root: 3 parents up.
    repo = Path(__file__).resolve().parents[3]
    path = repo / "scripts" / "s2_ar_ref.py"
    spec = importlib.util.spec_from_file_location("s2_ar_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s2_ar_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def bf16_round(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 -> f32, so host reference and device kernel see identical (already-quantized)
    inputs -- same convention as golden_hd128.py's bf16_round."""
    return x.astype(_bf16).astype(np.float32)


def build_qkv(seed: int, m_tokens: int):
    """Random q_full (m_tokens, N_HEAD, HD), k_full/v_full (m_tokens, N_HEAD_KV, HD), bf16-rounded,
    RoPE'd (q,k only, per the real op order RoPE(q,k) -> GQA-repeat(k,v) -> causal attn -- this
    brick does NOT do RoPE itself, same division of labor as mha_decode.cc; RoPE'd data here is
    just realistic test input, not something prefill_attn.cc depends on), bf16-rounded again."""
    ar_ref = _load_s2_ar_ref()
    rng = np.random.default_rng(seed)

    # ~N(0,1)-ish via sum of uniforms (same shape of reasoning as golden_hd128.py's randn_like):
    # scores with std~1 exercise the softmax properly instead of being near-uniform.
    def randn_like(shape):
        u = rng.random(shape + (3,), dtype=np.float32)
        return (u.sum(axis=-1) - 1.5) * 1.1547

    q_full = bf16_round(randn_like((m_tokens, N_HEAD, HD)).astype(np.float32))
    k_full = bf16_round(randn_like((m_tokens, N_HEAD_KV, HD)).astype(np.float32))
    v_full = bf16_round(randn_like((m_tokens, N_HEAD_KV, HD)).astype(np.float32))

    positions = np.arange(m_tokens)
    q_full = bf16_round(ar_ref.rope_interleaved(q_full, positions, HD, ROPE_BASE))
    k_full = bf16_round(ar_ref.rope_interleaved(k_full, positions, HD, ROPE_BASE))

    return ar_ref, q_full, k_full, v_full


def reference_full(ar_ref, q_full, k_full, v_full, out_head: int) -> np.ndarray:
    """Reference path: repeat_kv-EXPAND k,v to N_HEAD via GQA repeat-interleave, run the REAL
    `causal_attention` oracle, slice `out_head` -> (m_tokens, HD) -- EVERY prefill row's context,
    not just the last row (unlike mha_decode's M=1 decode-step golden)."""
    k_rep = ar_ref.repeat_kv(k_full, N_REP)
    v_rep = ar_ref.repeat_kv(v_full, N_REP)
    out = ar_ref.causal_attention(q_full, k_rep, v_rep, SCALE)  # (m_tokens, N_HEAD, HD)
    return out[:, out_head, :]


def _causal_softmax_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Plain numpy causal-softmax attention for ONE (already-selected) head's q,k,v -- the same
    math `causal_attention` does, transcribed directly (not calling it) so `reference_direct` can
    feed it the unexpanded (M,HD) KV-head slice instead of an (M,N_HEAD,HD) tensor. q,k,v: (M,HD)
    f64. Returns (M,HD) f64."""
    m_tokens = q.shape[0]
    scores = (q @ k.T) * SCALE  # (M, M)
    mask = np.triu(np.full((m_tokens, m_tokens), -np.inf, dtype=np.float64), k=1)
    scores = scores + mask
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs @ v


def reference_direct(q_full, k_full, v_full, out_head: int) -> np.ndarray:
    """Index-map path: NEVER expand. out_head's KV head is out_head // N_REP (repeat-interleave).
    Full causal mask over ALL m_tokens query rows -- matches prefill_attn.cc's per-row
    causal-softmax algorithm exactly (additive mask, see `build_causal_mask` /
    `reference_from_mask` below and the kernel's own "MASK IS DATA" header section). The kernel
    computes one row per call (prefill_attn_row); this reference still returns the full
    (m_tokens, HD) stack, which is what the verify script assembles M single-row device calls
    into, so it stays the right shape to gate against without any change here."""
    h_kv = out_head // N_REP
    q = q_full[:, out_head, :].astype(np.float64)
    k = k_full[:, h_kv, :].astype(np.float64)
    v = v_full[:, h_kv, :].astype(np.float64)
    return _causal_softmax_attention(q, k, v).astype(np.float32)


def reference_direct_wrong_tile(q_full, k_full, v_full, out_head: int) -> np.ndarray:
    """Same as reference_direct but with the WRONG (np.tile) GQA convention: h_kv = oh %
    n_head_kv. Exists only so `_selftest_gqa_mapping_has_teeth` can prove this brick's test data
    would catch a backwards driver -- never used as an expected value."""
    h_kv = out_head % N_HEAD_KV
    q = q_full[:, out_head, :].astype(np.float64)
    k = k_full[:, h_kv, :].astype(np.float64)
    v = v_full[:, h_kv, :].astype(np.float64)
    return _causal_softmax_attention(q, k, v).astype(np.float32)


def reference_bidirectional(q_full, k_full, v_full, out_head: int) -> np.ndarray:
    """Same head selection as reference_direct (correct GQA index), but WITHOUT the causal mask --
    exists only so the verify script can prove the causal mask is actually doing something on the
    real test data (device output must be far from THIS, close to reference_direct)."""
    h_kv = out_head // N_REP
    q = q_full[:, out_head, :].astype(np.float64)
    k = k_full[:, h_kv, :].astype(np.float64)
    v = v_full[:, h_kv, :].astype(np.float64)
    scores = (q @ k.T) * SCALE
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return (probs @ v).astype(np.float32)


def pack_head(q_full, k_full, v_full, out_head: int):
    """Device-ready buffers for ONE (Q head, its GQA-mapped KV head): q_bf [M,HD] bf16 (all M rows
    -- the verify script slices out ONE row per prefill_attn_row() call, since the kernel takes
    one query row per call, see prefill_attn.cc's header "ONE ROW PER CALL"), kv_bf [2*M,HD] bf16
    (K-block then V-block, M rows each) for h_kv = out_head // N_REP -- this is the FULL per-head
    K/V block, held RESIDENT across the M single-row calls that share it, unchanged in layout from
    before the one-row-per-call fix (only the Q/ctx side is now sliced per row, not KV)."""
    h_kv = out_head // N_REP
    m_tokens = q_full.shape[0]
    q_bf = q_full[:, out_head, :].astype(_bf16)
    kv_bf = np.zeros((2 * m_tokens, HD), dtype=_bf16)
    kv_bf[0:m_tokens] = k_full[:, h_kv, :].astype(_bf16)
    kv_bf[m_tokens:2 * m_tokens] = v_full[:, h_kv, :].astype(_bf16)
    return q_bf, kv_bf, h_kv


def build_causal_mask(m_tokens: int) -> np.ndarray:
    """Additive causal mask (m_tokens, SPAD): mask[i,j] = 0.0 for j<=i (visible), -1e9 for j>i
    (a real future key, if j<m_tokens, OR padding beyond the resident KV block, if j>=m_tokens --
    prefill_attn.cc's kernel scores padding slots 0 internally and relies on this mask ALWAYS
    carrying -1e9 there, see its header PADDING section). Same 0/-1e9 additive convention as
    `scripts/codec_quantizer_ref.py::_causal_window_mask` (~line 350) and the `np.triu(-inf,k=1)`
    term `scripts/s2_ar_ref.py::causal_attention` (~line 590) adds to its raw scores -- this is
    the SAME masking those two oracles already use, now supplied to the kernel as DATA instead of
    a `j > row_idx` branch (see prefill_attn.cc's "MASK IS DATA, NOT A SCALAR" header)."""
    mask = np.zeros((m_tokens, SPAD), dtype=np.float32)
    for i in range(m_tokens):
        mask[i, i + 1:] = -1.0e9
    return mask


def reference_from_mask(q_full, k_full, v_full, out_head: int, mask: np.ndarray) -> np.ndarray:
    """Reference attention using an EXPLICIT additive mask array (m_tokens, SPAD), matching
    prefill_attn.cc's data-driven masking exactly: score[j] = dot(q,k[j])*scale for j<m_tokens
    (0 for j>=m_tokens, no real key there), THEN mask is added, THEN softmax over all SPAD slots,
    THEN the first m_tokens softmax weights combine V. Used to prove the mask-as-data design still
    has teeth (see `_selftest_mask_data_has_teeth`): an all-zero mask must reduce to the
    bidirectional reference; `build_causal_mask`'s mask must reduce to `reference_direct`; the two
    must differ well above the device gate."""
    h_kv = out_head // N_REP
    q = q_full[:, out_head, :].astype(np.float64)
    k = k_full[:, h_kv, :].astype(np.float64)
    v = v_full[:, h_kv, :].astype(np.float64)
    m_tokens = q.shape[0]
    scores = np.zeros((m_tokens, SPAD), dtype=np.float64)
    scores[:, :m_tokens] = (q @ k.T) * SCALE
    scores = scores + mask.astype(np.float64)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return (probs[:, :m_tokens] @ v).astype(np.float32)


def pack_head_rows(q_full, k_full, v_full, out_head: int, mask: np.ndarray = None):
    """Device-ready buffers for bricklib.verify_streamed, ONE (Q head, its GQA-mapped KV head):
      in_tiles  : (m_tokens, HD+SPAD) bf16 -- row i is [q_row (HD) | mask_row (SPAD)] packed,
                  exactly prefill_attn.cc's `qm_row` streamed-operand layout. `mask` defaults to
                  `build_causal_mask(m_tokens)` (the real causal mask); passing an explicit
                  all-zero (or otherwise) mask here is how `verify_prefill_attn.py`'s negative
                  control run exercises the "mask is data" claim on real device output.
      kv_resident : (2*m_tokens*HD,) bf16 flattened K-block then V-block for h_kv = out_head //
                  N_REP -- the RESIDENT operand, unchanged in layout from `pack_head`'s kv_bf.
    Returns (in_tiles, kv_resident, h_kv)."""
    h_kv = out_head // N_REP
    m_tokens = q_full.shape[0]
    if mask is None:
        mask = build_causal_mask(m_tokens)
    q_bf = q_full[:, out_head, :].astype(_bf16)            # (m_tokens, HD)
    mask_bf = mask.astype(_bf16)                            # (m_tokens, SPAD)
    in_tiles = np.concatenate([q_bf, mask_bf], axis=1)      # (m_tokens, HD+SPAD)

    kv_resident = np.zeros((2 * m_tokens, HD), dtype=_bf16)
    kv_resident[0:m_tokens] = k_full[:, h_kv, :].astype(_bf16)
    kv_resident[m_tokens:2 * m_tokens] = v_full[:, h_kv, :].astype(_bf16)

    return in_tiles, kv_resident.reshape(-1), h_kv


def rel_l2(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(a - b))


def build_case(seed: int = 0, m_tokens: int = M, out_head: int = 5):
    """One full device-test case for a single head: bf16-rounded, RoPE'd q/k/v; expected (M,HD)
    context for `out_head` via both reference paths (asserted equal by the caller); device-ready
    packed buffers."""
    ar_ref, q_full, k_full, v_full = build_qkv(seed, m_tokens)
    exp_full = reference_full(ar_ref, q_full, k_full, v_full, out_head)
    exp_direct = reference_direct(q_full, k_full, v_full, out_head)
    exp_bidir = reference_bidirectional(q_full, k_full, v_full, out_head)
    q_bf, kv_bf, h_kv = pack_head(q_full, k_full, v_full, out_head)
    return dict(exp_full=exp_full, exp_direct=exp_direct, exp_bidir=exp_bidir,
                q_bf=q_bf, kv_bf=kv_bf, out_head=out_head, h_kv=h_kv, m_tokens=m_tokens,
                q_full=q_full, k_full=k_full, v_full=v_full)


def _selftest_index_equals_expand():
    ar_ref, q_full, k_full, v_full = build_qkv(seed=0, m_tokens=M)
    worst = 0.0
    for oh in range(N_HEAD):
        full = reference_full(ar_ref, q_full, k_full, v_full, oh)
        direct = reference_direct(q_full, k_full, v_full, oh)
        worst = max(worst, rel_l2(direct, full))
    print(f"[selftest] index (oh -> oh//n_rep) + row-by-row causal softmax vs "
          f"repeat_kv-expand-then-causal_attention, all {N_HEAD} heads, all {M} rows: "
          f"worst rel_l2={worst:.3e}")
    assert worst < 1e-5, "index map / row-by-row softmax does NOT match the oracle -- bug"


def _selftest_causal_mask_has_teeth():
    case = build_case(seed=0, m_tokens=M, out_head=5)
    row0 = case["exp_direct"][0]
    v0 = case["v_full"][0, case["h_kv"], :]
    d_row0 = rel_l2(row0, v0)
    print(f"[selftest] causal row 0 (can only see key 0) vs V[0] directly: rel_l2={d_row0:.3e}")
    assert d_row0 < 1e-5, "row 0 of a causal reference must equal V[0] exactly -- masking is broken"

    d_bidir = rel_l2(case["exp_direct"], case["exp_bidir"])
    print(f"[selftest] causal vs bidirectional (unmasked) reference, same q/k/v: "
          f"rel_l2={d_bidir:.3e} (must be >> 3e-2 device gate, or this test data has no teeth)")
    assert d_bidir > 0.1, "causal and bidirectional references are suspiciously close -- test " \
        "data would not catch an unmasked (non-causal) kernel"


def _selftest_gqa_mapping_has_teeth():
    _, q_full, k_full, v_full = build_qkv(seed=0, m_tokens=M)
    # oh=8: correct h_kv = 8//4 = 2; wrong (tile) h_kv = 8%8 = 0 -- different KV heads, so the two
    # references must diverge (unless the RNG produced near-identical heads by sheer coincidence,
    # vanishingly unlikely with independent random K/V per head).
    oh = 8
    correct = reference_direct(q_full, k_full, v_full, oh)
    wrong = reference_direct_wrong_tile(q_full, k_full, v_full, oh)
    d = rel_l2(correct, wrong)
    print(f"[selftest] GQA repeat-interleave (h_kv={oh // N_REP}) vs np.tile convention "
          f"(h_kv={oh % N_HEAD_KV}) for out_head={oh}: rel_l2={d:.3e} "
          f"(must be >> 3e-2, or a backwards driver could pass silently)")
    assert d > 0.1, "repeat-interleave and tile GQA conventions agree too closely on this data " \
        "-- this test could not catch a backwards (oh % n_head_kv) driver"


def _selftest_mask_data_has_teeth():
    """Prove the MASK-AS-DATA design (prefill_attn.cc's kernel now takes an additive mask ROW as
    part of its streamed operand instead of branching on a `row_idx` scalar) still catches a
    broken mask, exactly as thoroughly as the retired branch-based design did:
      (a) `reference_from_mask` with the REAL causal mask must reproduce `reference_direct`.
      (b) `reference_from_mask` with the CAUSAL PART zeroed but PADDING (j>=m_tokens) still
          correctly masked at -1e9 must reproduce `reference_bidirectional` (the unmasked
          reference) -- this is the realistic failure mode of a broken mask-construction (e.g.
          the host forgets `mask[i, i+1:] = -1e9` for real keys but the SPAD-vs-m_tokens sizing,
          a separate piece of code, still correctly zero-pads/masks beyond m_tokens). NOTE: an
          mask that is ALSO zero in the padding columns is NOT equivalent to bidirectional -- the
          padding slots have no real K data (the kernel scores them 0, see prefill_attn.cc's
          PADDING section) and an unmasked score-0 padding slot would compete in the softmax
          against the m_tokens real keys, contaminating the result with SPAD-m_tokens phantom
          "keys". Verified concretely: a literal all-zero (M,SPAD) mask measured rel_l2=3.028e-01
          against reference_bidirectional here, NOT the near-zero this docstring's claim (b)
          requires -- that failure is itself evidence the padding-vs-causal distinction matters
          and this test needed the FIX below, not a loosened gate.
      (c) The two must differ well above the device gate, or this test has no teeth."""
    ar_ref, q_full, k_full, v_full = build_qkv(seed=0, m_tokens=M)
    oh = 5
    causal_mask = build_causal_mask(M)
    no_causal_mask = np.zeros((M, SPAD), dtype=np.float32)
    no_causal_mask[:, M:] = -1.0e9  # padding beyond m_tokens still correctly masked; only the
                                     # causal (j<m_tokens) part is broken (all-zero -> bidirectional)

    via_causal_mask = reference_from_mask(q_full, k_full, v_full, oh, causal_mask)
    via_broken_mask = reference_from_mask(q_full, k_full, v_full, oh, no_causal_mask)

    d_causal_vs_direct = rel_l2(via_causal_mask, reference_direct(q_full, k_full, v_full, oh))
    d_broken_vs_bidir = rel_l2(via_broken_mask, reference_bidirectional(q_full, k_full, v_full, oh))
    d_causal_vs_broken = rel_l2(via_causal_mask, via_broken_mask)

    print(f"[selftest] mask-as-data: causal-mask formulation vs reference_direct: "
          f"rel_l2={d_causal_vs_direct:.3e}")
    print(f"[selftest] mask-as-data: causal-part-zeroed formulation vs bidirectional reference: "
          f"rel_l2={d_broken_vs_bidir:.3e}")
    print(f"[selftest] mask-as-data: causal-mask vs causal-part-zeroed (must be >> 3e-2 device "
          f"gate): rel_l2={d_causal_vs_broken:.3e}")
    assert d_causal_vs_direct < 1e-5, \
        "additive-mask formulation does not match reference_direct -- reference_from_mask bug"
    assert d_broken_vs_bidir < 1e-5, \
        "causal-part-zeroed mask does not reduce to the bidirectional reference -- " \
        "reference_from_mask bug"
    assert d_causal_vs_broken > 0.1, "causal and causal-part-zeroed masks produce suspiciously " \
        "similar results on this data -- this test would not catch a broken causal mask on device"


def _selftest_m1_degenerate():
    """M=1: a single query, single key -- softmax over one (unmasked) element is exactly 1.0, so
    ctx must equal V[0] exactly. Exercises the SPAD=16 padding path at its most extreme (15 of 16
    softmax slots masked)."""
    ar_ref, q_full, k_full, v_full = build_qkv(seed=1, m_tokens=1)
    out_head = 3
    direct = reference_direct(q_full, k_full, v_full, out_head)
    h_kv = out_head // N_REP
    d = rel_l2(direct[0], v_full[0, h_kv, :])
    print(f"[selftest] M=1 degenerate: ctx vs V[0]: rel_l2={d:.3e}")
    assert d < 1e-5, "M=1 case must reduce to exactly V[0]"


if __name__ == "__main__":
    _selftest_index_equals_expand()
    _selftest_causal_mask_has_teeth()
    _selftest_gqa_mapping_has_teeth()
    _selftest_mask_data_has_teeth()
    _selftest_m1_degenerate()
    print("All host self-checks passed.")
