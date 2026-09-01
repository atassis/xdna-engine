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


#===============================================================================================#
# RVQ / prefill_attn_chunk golden -- S2 quantizer post_module transformer (rvq_transformer),
# the capacity-extension consumer (see prefill_attn.cc's "prefill_attn_chunk" section header).
# NO GQA here (rvq_transformer asserts kv_size==dim: n_local_heads=-1 means "use n_head" for K/V
# too), RVQ_HEAD_DIM=64 (vs this file's HD=128 default above), and a SLIDING-WINDOW causal mask
# (RVQ_WINDOW_SIZE=128) instead of plain causal -- degenerates to plain causal whenever
# t_tokens <= window_size, which is every real captured clip (T=66) but not asserted in general.
#===============================================================================================#

VL = 16  # prefill_attn_chunk's fixed per-call key-chunk width == softmax_core<VL>'s N.


def _load_codec_quantizer_ref():
    """importlib-load scripts/codec_quantizer_ref.py by path (same pattern as
    `_load_s2_ar_ref` above, for the same @dataclass/sys.modules reason)."""
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[3]
    path = repo / "scripts" / "codec_quantizer_ref.py"
    spec = importlib.util.spec_from_file_location("codec_quantizer_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["codec_quantizer_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_qkv_rvq(seed: int, t_tokens: int, n_head: int = 4):
    """Random q/k/v [n_head, RVQ_HEAD_DIM, t_tokens] (codec_quantizer_ref's own channel-major
    layout, NOT this file's token-major [m_tokens,N_HEAD,HD] above), bf16-rounded, RoPE'd via
    codec_quantizer_ref._rope_normal (GGML NORMAL/interleaved -- NOT s2_ar_ref.rope_interleaved
    used above; the two conventions are a known hazard in this codebase, see rope-lut's history,
    so this brick's flash config imports its OWN consumer's RoPE rather than reusing the other
    oracle's). RMSNorm and the wqkv projection are deliberately NOT modelled: prefill_attn_chunk,
    like prefill_attn_row above, only ever consumes already-projected Q/K/V (see prefill_attn_row's
    own build_qkv: "this brick does NOT do RoPE itself... RoPE'd data here is just realistic test
    input" -- the same scope boundary applies one stage further upstream to RMSNorm+wqkv here)."""
    cq = _load_codec_quantizer_ref()
    rng = np.random.default_rng(seed)

    def randn_like(shape):
        u = rng.random(shape + (3,), dtype=np.float32)
        return (u.sum(axis=-1) - 1.5) * 1.1547

    hd = cq.RVQ_HEAD_DIM
    q = bf16_round(randn_like((n_head, hd, t_tokens)).astype(np.float32))
    k = bf16_round(randn_like((n_head, hd, t_tokens)).astype(np.float32))
    v = bf16_round(randn_like((n_head, hd, t_tokens)).astype(np.float32))

    positions = np.arange(t_tokens, dtype=np.float64)
    q = bf16_round(cq._rope_normal(q, positions, cq.RVQ_ROPE_BASE))
    k = bf16_round(cq._rope_normal(k, positions, cq.RVQ_ROPE_BASE))
    return cq, q, k, v


def reference_rvq_head(cq, q_full, k_full, v_full, head: int, mask: np.ndarray) -> np.ndarray:
    """One head's windowed-causal attention, transcribed from rvq_transformer's attention block
    (codec_quantizer_ref.py ~437-448: scores=k.q, /sqrt(RVQ_HEAD_DIM), +mask,
    `_softmax_over_keys`, V-weighted sum) -- same functions, same call order, CALLING
    `cq._softmax_over_keys` rather than reimplementing it, so this golden cannot silently drift
    from the real oracle's normalization. q_full/k_full/v_full: [n_head, HD, T]. mask: [T_k, T_q]
    additive (codec_quantizer_ref._causal_window_mask's own convention). Returns [T_q, HD] f64
    (query-major, matching prefill_attn_chunk's per-row ctx output)."""
    q = q_full[head].astype(np.float64)  # [HD, T]
    k = k_full[head].astype(np.float64)
    v = v_full[head].astype(np.float64)
    scores = np.einsum("dk,dq->kq", k, q) / np.sqrt(cq.RVQ_HEAD_DIM)  # [T_k, T_q]
    scores = scores + mask.astype(np.float64)
    probs = cq._softmax_over_keys(scores[None, :, :])[0]  # add/drop the n_head axis _softmax_over_keys expects
    attn = np.einsum("dk,kq->dq", v, probs)  # [HD, T_q]
    return attn.T.astype(np.float32)  # [T_q, HD]


def pack_head_chunks(cq, q_full, k_full, v_full, head: int, t_tokens: int, n_chunks: int,
                      mask: np.ndarray = None):
    """Device-ready buffers for prefill_attn_chunk: qm_tiles (t_tokens*n_chunks, HD+VL) bf16 and
    kv_tiles (t_tokens*n_chunks, 2*VL*HD) bf16, in (row, chunk) order matching the verify script's
    nested `range_(t_tokens)` (row) / Python-unrolled `range(n_chunks)` (chunk) call order -- see
    that script's `_build_flash_design`. `mask` defaults to
    `cq._causal_window_mask(t_tokens, cq.RVQ_WINDOW_SIZE)`, padded on the KEY axis to
    `n_chunks*VL` with -1e9 (padding beyond the real key count is masked unconditionally,
    regardless of row -- the SAME convention `build_causal_mask` above uses for M<SPAD)."""
    hd = cq.RVQ_HEAD_DIM
    if mask is None:
        mask = cq._causal_window_mask(t_tokens, cq.RVQ_WINDOW_SIZE)  # [t_tokens(k), t_tokens(q)]
    n_keys_padded = n_chunks * VL
    mask_padded = np.full((n_keys_padded, t_tokens), -1.0e9, dtype=np.float32)
    mask_padded[:t_tokens, :] = mask.astype(np.float32)

    q_h = q_full[head]  # [HD, T]
    k_h = np.zeros((hd, n_keys_padded), dtype=np.float32)
    v_h = np.zeros((hd, n_keys_padded), dtype=np.float32)
    k_h[:, :t_tokens] = k_full[head]
    v_h[:, :t_tokens] = v_full[head]

    qm_tiles = np.zeros((t_tokens * n_chunks, hd + VL), dtype=_bf16)
    kv_tiles = np.zeros((t_tokens * n_chunks, 2 * VL * hd), dtype=_bf16)
    idx = 0
    for q_idx in range(t_tokens):
        q_row_bf = q_h[:, q_idx].astype(_bf16)  # [HD]
        for c in range(n_chunks):
            mask_chunk_bf = mask_padded[c * VL:(c + 1) * VL, q_idx].astype(_bf16)  # [VL]
            qm_tiles[idx] = np.concatenate([q_row_bf, mask_chunk_bf])
            k_chunk = k_h[:, c * VL:(c + 1) * VL].T.astype(_bf16)  # [VL, HD]
            v_chunk = v_h[:, c * VL:(c + 1) * VL].T.astype(_bf16)  # [VL, HD]
            kv_tiles[idx] = np.concatenate([k_chunk.reshape(-1), v_chunk.reshape(-1)])
            idx += 1
    return qm_tiles, kv_tiles


def _selftest_rvq_chunked_matches_windowed():
    """Prove chunking is bookkeeping, not a different algorithm: `reference_rvq_head`'s windowed-
    causal attention (computed in one shot over the whole padded key range) must match what the
    flash/chunked KERNEL ALGORITHM computes when the SAME masked scores are split into VL-wide
    blocks and merged via the online-softmax recurrence -- transcribed here in plain numpy (not the
    device kernel) as the closest device-free check on the merge algebra itself."""
    cq = _load_codec_quantizer_ref()
    t_tokens, n_chunks, head = 20, 2, 1  # ceil(20/16)=2 chunks; deliberately NOT a multiple of VL
    _, q, k, v = build_qkv_rvq(seed=2, t_tokens=t_tokens, n_head=4)
    mask = cq._causal_window_mask(t_tokens, cq.RVQ_WINDOW_SIZE)
    direct = reference_rvq_head(cq, q, k, v, head, mask)  # [T,HD], one-shot softmax

    n_keys_padded = n_chunks * VL
    mask_padded = np.full((n_keys_padded, t_tokens), -1.0e9, dtype=np.float64)
    mask_padded[:t_tokens, :] = mask
    hd = cq.RVQ_HEAD_DIM
    k_pad = np.zeros((hd, n_keys_padded)); k_pad[:, :t_tokens] = k[head]
    v_pad = np.zeros((hd, n_keys_padded)); v_pad[:, :t_tokens] = v[head]
    q_h = q[head].astype(np.float64)
    scale = 1.0 / np.sqrt(cq.RVQ_HEAD_DIM)

    chunked = np.zeros((t_tokens, hd), dtype=np.float64)
    for qi in range(t_tokens):
        m, l, acc = -np.inf, 0.0, np.zeros(hd)
        for c in range(n_chunks):
            k_c = k_pad[:, c * VL:(c + 1) * VL].astype(np.float64)  # [HD,VL]
            v_c = v_pad[:, c * VL:(c + 1) * VL].astype(np.float64)
            score = (k_c.T @ q_h[:, qi]) * scale + mask_padded[c * VL:(c + 1) * VL, qi]
            m_new = max(m, score.max())
            corr = np.exp(m - m_new) if np.isfinite(m) else 0.0
            p = np.exp(score - m_new)
            l = l * corr + p.sum()
            acc = acc * corr + v_c @ p
            m = m_new
        chunked[qi] = acc / l

    d = rel_l2(chunked, direct)
    print(f"[selftest] flash-chunked merge vs one-shot windowed softmax (T={t_tokens}, "
          f"n_chunks={n_chunks}): rel_l2={d:.3e}")
    assert d < 1e-6, "online-softmax chunk merge does not reproduce the one-shot reference"


if __name__ == "__main__":
    _selftest_index_equals_expand()
    _selftest_causal_mask_has_teeth()
    _selftest_gqa_mapping_has_teeth()
    _selftest_mask_data_has_teeth()
    _selftest_m1_degenerate()
    _selftest_rvq_chunked_matches_windowed()
    print("All host self-checks passed.")
