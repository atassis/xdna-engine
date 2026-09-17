# SPDX-License-Identifier: Apache-2.0
"""Batched prefill past a sliding window's circular KV ring, checked against stepwise ring decode.

Numpy and pytest only, no IRON/aie import: mechanism only, no weights, no bf16.

Reference (what the served decode does, npu_decode.rs mask_writes / kv_layout::kv_off_circular):
per position p, write k,v to slot p % W, then attend slots [0, min(p+1, W)). Global layers: slot p,
width p+1.

Device form under test (sliding layers only): read-first. Scores over the concat [ring W | batch M],
mask per row (hole_lo, hole_hi, width) = mask [hole_lo, hole_hi) and [width, W+M), ctx as two
products summed, then commit the batch rows at slot base % W. Global layers keep today's write-first
graph with widths min(base+i+1, S).

Run:  .venv-iron/bin/python -m pytest designs/decode_fused/test_prefill_ring.py -v
(no IRON needed -- plain numpy/pytest venv is enough)
"""
import numpy as np
import pytest

W, M, S = 1024, 64, 4096
D, HQ, HKV, HD, VOCAB = 32, 4, 2, 8, 97
GRP = HQ // HKV
KINDS = ("sliding", "global", "sliding")
G = 8            # decode steps compared after the prompt
TOL = 1e-9


def ring_mask_rows(base, m, w):
    """Device form: int32 [m, 3] of (hole_lo, hole_hi, width) for a chunk at `base`.

    Legal only for base % m == 0 and w % m == 0, which is what keeps the hole and the commit from
    wrapping the ring end.
    """
    if w % m:
        raise ValueError(f"W={w} % M={m} != 0: a chunk would straddle the ring end")
    if base % m:
        raise ValueError(f"base={base} % M={m} != 0: chunk not aligned")
    b0 = base % w
    i = np.arange(m)
    hole_lo = np.full(m, b0)
    hole_hi = np.full(m, w) if base < w else b0 + i + 1
    return np.stack([hole_lo, hole_hi, w + i + 1], 1).astype(np.int32)


def expand_mask(rows, cols):
    c = np.arange(cols)[None, :]
    lo, hi, wid = (rows[:, j:j + 1] for j in range(3))
    return ~(((c >= lo) & (c < hi)) | (c >= wid))


def global_widths(base, m, s, heads):
    """Today's per-row causal widths (gen_llm_prefill.py causal_widths), unchanged."""
    return np.tile(np.clip(np.arange(m) + base + 1, 1, s), heads).astype(np.int32)


def softmax(s):
    mx = np.max(s, -1, keepdims=True)
    e = np.exp(s - mx)
    return e / e.sum(-1, keepdims=True)


def rope(x, pos):
    half = HD // 2
    inv = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


class Model:
    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        self.embed = r.standard_normal((VOCAB, D))
        self.U = r.standard_normal((VOCAB, D)) / np.sqrt(D)
        self.L = []
        for _ in KINDS:
            self.L.append(dict(
                q=r.standard_normal((HQ * HD, D)) / np.sqrt(D),
                k=r.standard_normal((HKV * HD, D)) / np.sqrt(D),
                v=r.standard_normal((HKV * HD, D)) / np.sqrt(D),
                o=r.standard_normal((D, HQ * HD)) / np.sqrt(HQ * HD),
                f=r.standard_normal((D, D)) / np.sqrt(D)))

    def qkv(self, l, x, pos):
        P = self.L[l]
        n = x.shape[0]
        q = rope((x @ P["q"].T).reshape(n, HQ, HD), pos)
        k = rope((x @ P["k"].T).reshape(n, HKV, HD), pos)
        v = (x @ P["v"].T).reshape(n, HKV, HD)
        return q, k, v

    def post(self, l, x, ctx):
        P = self.L[l]
        x = x + ctx.reshape(x.shape[0], HQ * HD) @ P["o"].T
        return x + np.tanh(x @ P["f"].T)


def fresh_cache(stale_seed=None):
    caps = [W if k == "sliding" else S for k in KINDS]
    if stale_seed is None:
        mk = lambda c: np.zeros((HKV, c, HD))
    else:
        r = np.random.default_rng(stale_seed)
        mk = lambda c: r.standard_normal((HKV, c, HD)) * 5.0
    return [dict(kc=mk(c), vc=mk(c), slot_pos=np.full(c, -1)) for c in caps]


def attend(q, keys, vals, valid):
    out = np.empty((q.shape[0], HQ, HD))
    for h in range(HQ):
        g = h // GRP
        s = np.where(valid, q[:, h] @ keys[g].T / np.sqrt(HD), -np.inf)
        out[:, h] = softmax(s) @ vals[g]
    return out


def step(model, cache, tok, pos):
    """The reference: one stepwise decode position, as the served decode addresses the ring."""
    x = model.embed[np.array([tok])]
    for l, kind in enumerate(KINDS):
        c = cache[l]
        q, k, v = model.qkv(l, x, [pos])
        cap = W if kind == "sliding" else S
        slot = pos % cap
        c["kc"][:, slot], c["vc"][:, slot], c["slot_pos"][slot] = k[0], v[0], pos
        width = min(pos + 1, cap)
        valid = np.arange(cap)[None, :] < width
        x = model.post(l, x, attend(q, c["kc"], c["vc"], valid))
    return (model.U @ x[0])


def general_valid(slot_pos, base, m):
    """Mask from the ring's slot->position ledger, alignment-free: the definition, not the encoding."""
    p = base + np.arange(m)[:, None]
    lo = np.maximum(0, p - W + 1)
    ring = (slot_pos[None, :] >= 0) & (slot_pos[None, :] >= lo)
    batch = np.arange(m)[None, :] <= np.arange(m)[:, None]
    return np.concatenate([ring, batch], 1)


def check_attended(slot_pos, base, m, valid):
    labels = np.concatenate([slot_pos, base + np.arange(m)])
    for i in range(m):
        p = base + i
        got = np.sort(labels[valid[i]])
        want = np.arange(max(0, p - W + 1), p + 1)
        if not np.array_equal(got, want):
            raise AssertionError(f"row p={p}: attends {got[:3]}..{got[-3:]} ({got.size}), "
                                 f"window is {want[0]}..{want[-1]} ({want.size})")


def sliding_chunk_device(model, l, c, x, base, n_rows):
    """Read-first concat with the (hole_lo, hole_hi, width) mask. Returns the attention context."""
    q, k, v = model.qkv(l, x, base + np.arange(n_rows))
    rows = ring_mask_rows(base, n_rows, W)
    valid = expand_mask(rows, W + n_rows)
    general = general_valid(c["slot_pos"], base, n_rows)
    if not np.array_equal(valid, general):
        raise AssertionError(f"device mask != general mask at base={base}")
    check_attended(c["slot_pos"], base, n_rows, valid)
    ctx = np.empty((n_rows, HQ, HD))
    for h in range(HQ):
        g = h // GRP
        keys = np.concatenate([c["kc"][g], k[:, g]], 0)
        p = softmax(np.where(valid, q[:, h] @ keys.T / np.sqrt(HD), -np.inf))
        ctx[:, h] = p[:, :W] @ c["vc"][g] + p[:, W:] @ v[:, g]
    b0 = base % W
    c["kc"][:, b0:b0 + n_rows] = k.transpose(1, 0, 2)
    c["vc"][:, b0:b0 + n_rows] = v.transpose(1, 0, 2)
    c["slot_pos"][b0:b0 + n_rows] = base + np.arange(n_rows)
    return ctx


def sliding_chunk_general(model, l, c, x, base, n_rows):
    """Same algebra with a circular commit and the ledger mask -- for chunks the device refuses."""
    q, k, v = model.qkv(l, x, base + np.arange(n_rows))
    valid = general_valid(c["slot_pos"], base, n_rows)
    check_attended(c["slot_pos"], base, n_rows, valid)
    ctx = np.empty((n_rows, HQ, HD))
    for h in range(HQ):
        g = h // GRP
        keys = np.concatenate([c["kc"][g], k[:, g]], 0)
        p = softmax(np.where(valid, q[:, h] @ keys.T / np.sqrt(HD), -np.inf))
        ctx[:, h] = p[:, :W] @ c["vc"][g] + p[:, W:] @ v[:, g]
    slots = (base + np.arange(n_rows)) % W
    c["kc"][:, slots] = k.transpose(1, 0, 2)
    c["vc"][:, slots] = v.transpose(1, 0, 2)
    c["slot_pos"][slots] = base + np.arange(n_rows)
    return ctx


def sliding_chunk_write_first(model, l, c, x, base, n_rows):
    """NEGATIVE CONTROL: today's graph order (append, then read) with widths min(base+i+1, W)."""
    q, k, v = model.qkv(l, x, base + np.arange(n_rows))
    slots = (base + np.arange(n_rows)) % W
    c["kc"][:, slots] = k.transpose(1, 0, 2)
    c["vc"][:, slots] = v.transpose(1, 0, 2)
    c["slot_pos"][slots] = base + np.arange(n_rows)
    valid = np.arange(W)[None, :] < np.minimum(base + np.arange(n_rows) + 1, W)[:, None]
    return attend(q, c["kc"], c["vc"], valid)


def global_chunk(model, l, c, x, base, n_rows):
    q, k, v = model.qkv(l, x, base + np.arange(n_rows))
    c["kc"][:, base:base + n_rows] = k.transpose(1, 0, 2)
    c["vc"][:, base:base + n_rows] = v.transpose(1, 0, 2)
    c["slot_pos"][base:base + n_rows] = base + np.arange(n_rows)
    widths = global_widths(base, n_rows, S, HQ)[:n_rows]
    valid = np.arange(S)[None, :] < widths[:, None]
    return attend(q, c["kc"], c["vc"], valid)


def run_chunk(model, cache, toks, base, n_rows, sliding_fn):
    x = model.embed[toks]
    for l, kind in enumerate(KINDS):
        fn = sliding_fn if kind == "sliding" else global_chunk
        x = model.post(l, x, fn(model, l, cache[l], x, base, n_rows))


def prime(model, cache, tokens, n, m, policy, sliding_fn=sliding_chunk_device):
    """Prime positions [0, n) in chunks of m, host-side. Returns positions primed.

    policy: 'pad'      pad the last chunk whatever it overwrites (NEGATIVE CONTROL past the wrap)
            'tail'     stage 1: skip a padded last chunk that would overwrite ring content the
                       following decode steps read; the caller finishes it stepwise
            'restore'  stage 2: pad, but save and restore the ring slots of pad rows p >= n+1
    """
    start = 0
    while start < n:
        real = min(m, n - start)
        pads = m - real
        harmful = pads >= 2 and start + m > W
        if policy == "tail" and harmful:
            return start
        toks = [tokens[start + i] if i < real else tokens[start + real - 1] for i in range(m)]
        saved = None
        if policy == "restore" and harmful:
            slots = np.arange(n + 1, start + m) % W
            saved = [(l, slots, c["kc"][:, slots].copy(), c["vc"][:, slots].copy(),
                      c["slot_pos"][slots].copy())
                     for l, c in enumerate(cache) if KINDS[l] == "sliding"]
        run_chunk(model, cache, toks, start, m, sliding_fn)
        for l, slots, kc, vc, sp in saved or ():
            cache[l]["kc"][:, slots], cache[l]["vc"][:, slots] = kc, vc
            cache[l]["slot_pos"][slots] = sp
        start += m
    return n


def reference_logits(model, tokens, n_pos, stale_seed=None):
    cache = fresh_cache(stale_seed)
    return np.stack([step(model, cache, tokens[p], p) for p in range(n_pos)])


def batched_logits(model, tokens, prompt_len, m, policy, sliding_fn, stale_seed=None):
    cache = fresh_cache(stale_seed)
    n = prompt_len - 1
    primed = prime(model, cache, tokens, n, m, policy, sliding_fn)
    for p in range(primed, n):
        step(model, cache, tokens[p], p)
    logits = np.stack([step(model, cache, tokens[p], p)
                       for p in range(n, n + G)])
    return primed, logits


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(1e-30, float(np.max(np.abs(b)))))


# --------------------------------------------------------------------------------------------
# Fixtures: one model/token stream/reference pass shared by every test (module-scoped -- the
# stepwise reference over 2200+ positions is the expensive part, ~seconds, and is read-only).
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return Model(0)


@pytest.fixture(scope="module")
def tokens():
    return np.random.default_rng(1).integers(0, VOCAB, 2300 + G)


@pytest.fixture(scope="module")
def ref(model, tokens):
    return reference_logits(model, tokens, 2200 + G)


@pytest.fixture(scope="module")
def ref_stale(model, tokens):
    return reference_logits(model, tokens, 1100 + G, stale_seed=7)


# --------------------------------------------------------------------------------------------
# 1. The closed form, row by row, for aligned chunks with a contiguous history from 0.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("base", [0, 960, 1024, 1088, 1984, 2048, 2112])
def test_closed_form_mask_matches_ledger_and_window(base):
    sp = np.full(W, -1)
    hist = np.arange(max(0, base - W), base)
    sp[hist % W] = hist
    rows = ring_mask_rows(base, M, W)
    v = expand_mask(rows, W + M)
    assert np.array_equal(v, general_valid(sp, base, M)), f"base={base}: device mask != ledger mask"
    check_attended(sp, base, M, v)  # raises on mismatch


# --------------------------------------------------------------------------------------------
# 2. End to end against stepwise ring decode, both tail policies that are meant to be safe.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("prompt_len", [1025, 1026, 1088, 1089, 1100, 2113, 2200])
@pytest.mark.parametrize("policy", ["tail", "restore"])
def test_batched_matches_stepwise_past_the_wrap(model, tokens, ref, prompt_len, policy):
    primed, lg = batched_logits(model, tokens, prompt_len, M, policy, sliding_chunk_device)
    r = rel(lg, ref[prompt_len - 1:prompt_len - 1 + G])
    assert r < TOL, f"P={prompt_len} policy={policy} primed_batched={primed} rel={r:.2e}"


# --------------------------------------------------------------------------------------------
# 3. Stale ring from a previous request must be masked, not read.
# --------------------------------------------------------------------------------------------

def test_stale_ring_is_masked_not_read(model, tokens, ref, ref_stale):
    primed, lg = batched_logits(model, tokens, 1100, M, "restore", sliding_chunk_device, stale_seed=7)
    r = rel(lg, ref_stale[1099:1099 + G])
    r_clean = rel(ref_stale[:1100], ref[:1100])
    assert r < TOL and r_clean < TOL, f"stale ring P=1100 rel={r:.2e} (stale vs clean ref {r_clean:.2e})"


# --------------------------------------------------------------------------------------------
# 4. A chunk starting at 1000 that straddles the wrap: the device form refuses it, the
#    alignment-free algebra is still exact (M=40, 1000 = 25*40, W % 40 != 0).
# --------------------------------------------------------------------------------------------

def test_device_form_refuses_a_straddling_chunk():
    with pytest.raises(ValueError):
        ring_mask_rows(1000, 40, W)


def test_general_form_is_exact_across_a_straddling_chunk(model, tokens, ref):
    cache = fresh_cache()
    n = 1099
    start = 0
    while start + 40 <= n:
        run_chunk(model, cache, list(tokens[start:start + 40]), start, 40, sliding_chunk_general)
        start += 40
    for p in range(start, n):
        step(model, cache, tokens[p], p)
    lg = np.stack([step(model, cache, tokens[p], p) for p in range(n, n + G)])
    r = rel(lg, ref[n:n + G])
    assert r < TOL, f"general form, chunks of 40 incl. [1000,1040) across the wrap, rel={r:.2e}"


# --------------------------------------------------------------------------------------------
# 5. Global layers: widths identical to today's generator formula.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("base", [0, 1024, 2048, 4032])
def test_global_widths_match_the_generators_causal_widths(base):
    want = np.tile(np.clip(np.arange(M) + base + 1, 1, S), HQ).astype(np.int32)
    assert np.array_equal(global_widths(base, M, S, HQ), want)


# --------------------------------------------------------------------------------------------
# Negative controls and one control: each must FAIL (or pass, for the control) the same
# tolerance the positive checks above use.
# --------------------------------------------------------------------------------------------

def test_negative_write_first_past_the_wrap_is_caught(model, tokens, ref):
    _, lg = batched_logits(model, tokens, 1100, M, "tail", sliding_chunk_write_first)
    r = rel(lg, ref[1099:1099 + G])
    assert r > 1e-6, f"NEGATIVE write-first + width clamp past the wrap should be caught, rel={r:.2e}"


def test_negative_padded_last_chunk_past_the_wrap_is_caught(model, tokens, ref):
    _, lg = batched_logits(model, tokens, 1026, M, "pad", sliding_chunk_device)
    r = rel(lg, ref[1025:1025 + G])
    assert r > 1e-6, (f"NEGATIVE padded last chunk past the wrap (P=1026, 62 harmful pads) should "
                     f"be caught, rel={r:.2e}")


def test_control_write_first_is_still_exact_below_the_wrap(model, tokens, ref):
    _, lg = batched_logits(model, tokens, 1025, M, "pad", sliding_chunk_write_first)
    r = rel(lg, ref[1024:1024 + G])
    assert r < TOL, f"CONTROL write-first below the wrap should be exact, rel={r:.2e}"


# --------------------------------------------------------------------------------------------
# 6. Pre-existing, stepwise only (sec 1.5): the prefix ledger resuming on a ring the previous
#    request advanced past the resume point. Expected to FAIL -- this is a known, filed defect
#    (prefix-ledger-reuse-over-a-wrapped-sliding-ring), out of THIS task's scope; the test
#    documents that it still reproduces, not that it is fixed.
# --------------------------------------------------------------------------------------------

def test_ledger_resume_over_a_wrapped_ring_is_wrong_today(model, tokens):
    alt = tokens.copy()
    alt[1090:] = (alt[1090:] + 1) % VOCAB
    ref_alt = reference_logits(model, alt, 1100 + G)
    cache = fresh_cache()
    for p in range(1100):
        step(model, cache, tokens[p], p)
    lg = np.stack([step(model, cache, alt[p], p) for p in range(1090, 1100 + G)])
    r = rel(lg, ref_alt[1090:1100 + G])
    assert r > 1e-6, f"LEDGER resume at 1090 over a ring advanced to 1100 rel={r:.2e} (expected wrong)"
