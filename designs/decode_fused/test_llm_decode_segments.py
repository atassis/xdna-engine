# SPDX-License-Identifier: Apache-2.0
"""Device-free gates for DECODE_SEGMENTS -- cutting the layer stack across dispatches.

The cut exists because a runtime buffer's arena offset is patched into its BD by
`aiex.npu.address_patch`, whose `arg_plus` mlir-aie declares as I32: past 2^32 the offset wraps and
the buffer is never written, silently, on a design that builds and runs. Gemma-4-12B at 48 layers
needs 9.13 GiB in one arena, so it cannot be addressed at all; three arenas of ~2.4 GiB can.

What is asserted here is that a cut CHANGES NOTHING but where the seams are. The runlist a split
build hands to IRON must concatenate back to the unsplit one entry for entry -- a segmentation bug
that dropped or reordered a layer would otherwise present as a numerical fault on device, which is
the most expensive place this project has to find things.

OperatorSequence is stubbed, so this runs in seconds with no aiecc and no device.

Run inside the IRON env:
  PYTHONPATH=designs/decode_fused:$IRON .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_llm_decode_segments.py -v
"""
import dataclasses
import os

import numpy as np
import pytest

gen = pytest.importorskip("gen_llm_decode")

SPEC = "gemma3-270m"
WEIGHTS = "/mnt/data/xdna/artifacts/gemma3-270m/weights"
WEIGHTS_G4 = "/mnt/data/xdna/artifacts/gemma4-12b/weights_int4g32qat_hf"
LAYERS = 6

pytestmark = pytest.mark.skipif(
    not os.path.isdir(WEIGHTS), reason=f"no dumped weights at {WEIGHTS}")


class _StubSeq:
    """Records what build_graph asked for, without compiling it."""

    def __init__(self, name, runlist, input_args=None, output_args=None,
                 buffer_sizes=None, context=None, extra_flags=None, share_designs=None):
        self.name = name
        self.runlist = list(runlist)
        self.input_args = list(input_args or [])
        self.output_args = list(output_args or [])
        self.declared_sizes = dict(buffer_sizes or {})
        # The real OperatorSequence exposes this as (input, output, scratch), not as the dict its
        # constructor takes -- build_graph reads [2] to report the arena.
        self.buffer_sizes = (0, 0, sum(self.declared_sizes.values()))

    def compile(self):
        pass

    def get_layout_for_buffer(self, name):
        raise KeyError(name)


@pytest.fixture
def build(monkeypatch):
    def _build(nseg):
        monkeypatch.setattr(gen, "OperatorSequence", _StubSeq)
        monkeypatch.setattr(gen, "DECODE_SEGMENTS", nseg)
        return gen.build_graph(SPEC, WEIGHTS, LAYERS, 2048)
    return _build


def test_unsplit_is_one_segment(build):
    _sp, fused, _w, md = build(1)
    assert len(md["segments"]) == 1
    assert md["segments"][0]["seq"] is fused
    assert md["segments"][0]["inlet"] == "x"


@pytest.mark.parametrize("nseg", [2, 3, 4])
def test_split_preserves_the_runlist_exactly(build, nseg):
    """The whole safety property: a cut moves no work and reorders none."""
    _sp, _f, _w, md1 = build(1)
    base = list(md1["segments"][0]["seq"].runlist)
    _sp, _f, _w, mdn = build(nseg)
    joined = [e for s in mdn["segments"] for e in s["seq"].runlist]
    assert joined == base


@pytest.mark.parametrize("nseg", [2, 3, 4])
def test_residual_chains_end_to_end(build, nseg):
    _sp, _f, _w, md = build(nseg)
    segs = md["segments"]
    assert len(segs) == nseg
    assert segs[0]["inlet"] == "x"
    assert segs[-1]["outlet"] in ("logits", "xf")
    for prev, nxt in zip(segs, segs[1:]):
        # The seam. A break here is a residual read from a buffer nothing wrote -- zeros, which
        # look exactly like the 4 GiB wrap this whole mechanism exists to avoid.
        assert nxt["inlet"] == prev["outlet"]


@pytest.mark.parametrize("nseg", [2, 3, 4])
def test_every_weight_lands_in_exactly_one_segment(build, nseg):
    _sp, _f, weights, md = build(nseg)
    claimed = [w for s in md["segments"] for w in s["weights"]]
    assert len(claimed) == len(set(claimed)), "a weight is claimed by two segments"
    assert set(claimed) == set(weights), "a weight is claimed by no segment"


@pytest.mark.parametrize("nseg", [2, 3, 4])
def test_seam_buffers_are_declared_and_sized(build, nseg):
    """A cut turns an implicit intermediate into a declared arg on both sides."""
    for i, s in enumerate(build(nseg)[3]["segments"]):
        assert s["inputs"][0] == s["inlet"]
        assert s["seq"].output_args == [s["outlet"]]
        if i:
            assert s["seq"].declared_sizes.get(s["inlet"]), \
                f"segment {i} inlet {s['inlet']} declared without a size"


@pytest.mark.parametrize("nseg", [2, 3])
def test_a_segment_declares_only_the_angle_tables_its_layers_read(build, nseg):
    """Same rule the unsplit build already follows, now per segment.

    IRON refuses a design declaring an input no op consumes ("Input argument rope_global not found
    in runlist buffers"), so a segment of sliding-only layers must not declare rope_global.
    """
    for s in build(nseg)[3]["segments"]:
        refs = gen.runlist_buffer_names(s["seq"].runlist)
        for ang in ("rope_global", "rope_local"):
            if ang in s["inputs"]:
                assert ang in refs, f"{s['seq'].name} declares {ang} but no op reads it"


def test_more_segments_than_layers_is_refused(build):
    with pytest.raises(SystemExit, match="exceeds the"):
        build(LAYERS + 1)


@pytest.mark.parametrize("nseg", [2, 3])
def test_each_segment_declares_only_its_own_kv_slots(build, nseg):
    """kv_off/kv_off1/... are named per GEOMETRY and baked into that geometry's StridedCopy.

    A segment's scratchpad therefore declares only the slots its own layers' head_dims use, and
    writing one it does not have raises "ParameterScratchpad: unknown parameter". Gemma-4-12B is
    where this bites -- sliding layers at head_dim 256, global at 512 -- and it cost a device run
    on 2026-09-09. gemma3-270m is uniform so this passes trivially here; the assertion exists to
    fail if the derivation regresses on a multi-geometry spec.
    """
    sp, _f, _w, md = build(nseg)
    segs = md["segments"]
    every = {n for s in segs for n, _, _ in s["kv_slots"]}
    for s in segs:
        la, lb = s["layers"]
        want = {sp.head_dim_for(l) for l in range(la, lb)}
        assert {hd for _, hd, _ in s["kv_slots"]} == want, \
            f"segment {s['seq'].name} slots do not match its layers' head_dims"
    # Union over segments must be the whole model's slot set -- a slot owned by nobody is a KV
    # cache the host never advances, which reads as a model that stops attending to its history.
    assert every == {n for n, _, _ in md["kv_slots"]}


def test_kv_skip_v_drops_vnorm_for_has_v_false_layer(monkeypatch):
    """Task 1.2b: under KV_SKIP_V=1 the has_v=False layer must not build a vnorm op.

    gemma3-270m's own sw_pattern=6 already makes layer 5 (of LAYERS=6) global, so
    v_from_k_on_global=True/v_norm=True on a copy of its spec gives has_v_proj(5)==False with no
    other dims touched -- geometry stays uniform, no extra weights needed (v_norm is gainless).

    vnorm entries are the only runlist tuples whose gain arg is the literal `ones_h{hd}` (qk-norm
    entries pass `n_qn`/`n_kn` instead) -- v_norm applies on EVERY layer (gainless RMSNorm on the
    value path), so the has_v=True layers still build it in place; only the has_v=False layer's
    cross (k-src, v-dst) form is dead. Buffer names are prefixed `L{layer}_`, so that literal's
    absence among a layer's own args is exactly "this layer's vnorm was not built".
    """
    monkeypatch.setattr(gen, "OperatorSequence", _StubSeq)
    monkeypatch.setattr(gen, "DECODE_SEGMENTS", 1)
    monkeypatch.setattr(gen, "KV_SKIP_V", True)
    vskip_spec = dataclasses.replace(
        gen.SPECS[SPEC], name="gemma3-270m-vskip", v_from_k_on_global=True, v_norm=True)
    monkeypatch.setitem(gen.SPECS, vskip_spec.name, vskip_spec)

    sp, fused, _w, _md = gen.build_graph(vskip_spec.name, WEIGHTS, LAYERS, 2048)
    assert not sp.has_v_proj(LAYERS - 1), "fixture no longer produces a has_v=False layer"

    def layer_has_vnorm(l):
        prefix = f"L{l}_"
        return any(
            isinstance(arg, str) and arg.startswith("ones_h")
            for entry in fused.runlist
            if any(isinstance(a, str) and a.startswith(prefix) for a in entry[1:])
            for arg in entry[1:])

    assert not layer_has_vnorm(LAYERS - 1), "vnorm construction still present under KV_SKIP_V=1"
    assert layer_has_vnorm(0), "has_v=True layer lost its (expected, in-place) vnorm too"


def test_attn_global_flash_gets_a_resident_rope_f_constant_and_kv_skip_v_flag(monkeypatch):
    """op.py's `AttnGlobalFlash.get_arg_spec` declares `rope_f` at HD (the on-chip inverse-RoPE
    constant, not the old capacity-sized angle table), and `kv_skip_v` must thread through to the
    operator.

    Setting global_head_dim/global_n_kv_heads to gemma3-270m's own uniform (head_dim, n_kv_heads)
    makes EVERY layer's geometry "the global one" (attn_global_flash_why's is_global_geom check),
    so the has_v=True layers and the has_v=False layer (same v_from_k_on_global fixture as the
    kv_skip_v vnorm test above) land in two distinct (hd, hkv, has_v) geometries and each gets its
    own AttnGlobalFlash -- the has_v=False one is attention_k_eq_v, the arm kv_skip_v targets.
    """
    monkeypatch.setattr(gen, "OperatorSequence", _StubSeq)
    monkeypatch.setattr(gen, "DECODE_SEGMENTS", 1)
    monkeypatch.setattr(gen, "KV_SKIP_V", True)
    monkeypatch.setattr(gen, "FUSE_ATTN_GLOBAL_FLASH", True)
    base = gen.SPECS[SPEC]
    flash_spec = dataclasses.replace(
        base, name="gemma3-270m-flash", v_from_k_on_global=True, v_norm=True,
        global_head_dim=base.head_dim, global_n_kv_heads=base.n_kv_heads)
    monkeypatch.setitem(gen.SPECS, flash_spec.name, flash_spec)

    sp, fused, w, md = gen.build_graph(flash_spec.name, WEIGHTS, LAYERS, 2048)
    assert not sp.has_v_proj(LAYERS - 1), "fixture no longer produces a has_v=False layer"

    all_flash_entries = [e for e in fused.runlist if type(e[0]).__name__ == "AttnGlobalFlash"]
    # attn_ops() memoizes per (hd, hkv, has_v), so every layer sharing a geometry reuses the
    # SAME op object -- dedupe by identity to get one entry per DISTINCT geometry.
    seen, flash_entries = set(), []
    for e in all_flash_entries:
        if id(e[0]) not in seen:
            seen.add(id(e[0]))
            flash_entries.append(e)
    assert len(flash_entries) == 2, "expected one AttnGlobalFlash per (has_v) geometry"

    skip_flags = sorted(e[0].kv_skip_v for e in flash_entries)
    assert skip_flags == [False, True], "only the has_v=False geometry should set kv_skip_v"

    partial = sp.rope_partial_rotary if sp.rope_type_global == "proportional" else None
    for op, ref_q, n_kn, rope_f, kc, vc, cx in flash_entries:
        assert rope_f not in ("rope_global", "rope_local"), \
            "AttnGlobalFlash must not read the single-row forward-RoPE angle buffer directly"
        assert rope_f not in md["cache_names"], "rope_f is a resident constant, not a cache buffer"
        assert fused.declared_sizes.get(rope_f) is None, \
            "rope_f needs no explicit bufsz: get_arg_spec's own HD entry sizes it"
        _assert_rope_f_undoes_forward_rotation(w[rope_f], op.HD, sp.rope_theta_global, partial)


def _assert_rope_f_undoes_forward_rotation(rope_f_bf16, hd, theta, partial):
    """rope_f (HD/2 uint32 turns, byte-reinterpreted into an HD-element bf16 buffer) must undo the
    SAME rotation verify_llm_decode.py's `rope_row` applies going forward (gated byte-for-byte
    against rust/npu-engine/src/llm/npu_decode.rs::rope_row) -- checked against an INDEPENDENT
    ground truth, measure_rope_poly.rope_inv_freq (already verified verbatim against
    gate_llm_reference.py's rope()), not against gen.rope_f_for_geometry a second time. A prior
    version of this check called gen.rope_f_for_geometry again as its own oracle, so a bug inside
    it (e.g. dropping the partial-rotary zeroing line) passed silently.

    BOUND is rope_int_phase's own measured accuracy (see
    scripts/test_gate_llm.py::test_int_phase_matches_exact_cos_sin_at_gemma4_geometries_and_large_
    positions, which gates the same scheme at the same bound); the cross-check against `rope_row`
    itself uses measure_rope_poly.GATE_ABS, the wider bf16-rounding bound, because rope_row's own
    output is bf16-quantized.
    """
    import measure_rope_poly as mrp
    import rope_int_phase as rp
    import verify_llm_decode as vld

    BOUND = 3e-4
    inv = mrp.rope_inv_freq(hd, theta, partial)
    half = hd // 2
    rotated = half if partial is None else int(partial * hd // 2)
    F = rope_f_bf16.view(np.uint32).astype(np.uint64)
    for pos in (0, 1, 4097, 262143):
        turns = ((F * np.uint64(pos)) & rp.U32_MASK).astype(np.uint32)
        s, c = rp.phase_cs(turns)
        ref_c, ref_s = np.cos(pos * inv), np.sin(pos * inv)
        assert np.max(np.abs(c - ref_c)) <= BOUND, \
            f"hd={hd} pos={pos}: max|dcos|={np.max(np.abs(c - ref_c)):.4e} exceeds {BOUND:.1e}"
        assert np.max(np.abs(s - ref_s)) <= BOUND, \
            f"hd={hd} pos={pos}: max|dsin|={np.max(np.abs(s - ref_s)):.4e} exceeds {BOUND:.1e}"
        assert np.all(c[rotated:] == 1.0) and np.all(s[rotated:] == 0.0), \
            f"hd={hd} pos={pos}: lanes past the rotated width must be the exact identity rotation"
        row = vld.rope_row(pos, hd, theta, partial).astype(np.float32)
        assert np.max(np.abs(c - row[0::2])) <= mrp.GATE_ABS, \
            f"hd={hd} pos={pos}: disagrees with the forward rope_row LUT past its bf16 bound"
        assert np.max(np.abs(s - row[1::2])) <= mrp.GATE_ABS, \
            f"hd={hd} pos={pos}: disagrees with the forward rope_row LUT past its bf16 bound"


def test_rope_f_satisfies_the_forward_inverse_rope_contract_with_partial_rotary():
    """Same contract as `_assert_rope_f_undoes_forward_rotation`, at a Gemma-4-12B-shaped
    geometry (hd=512, theta=1e6, proportional, partial=0.25 -- GEMMA4_12B's own spec fields).
    gemma3-270m's fixture above never sets rope_type_global, so it never exercises the
    partial-rotary zeroing branch of rope_f_for_geometry; this is the test that does.
    """
    g4 = gen.SPECS["gemma4-12b"]
    partial = g4.rope_partial_rotary if g4.rope_type_global == "proportional" else None
    assert partial is not None, "fixture drift: gemma4-12b no longer sets proportional partial rotary"
    rope_f = gen.rope_f_for_geometry(g4.global_head_dim, g4.rope_theta_global, partial)
    _assert_rope_f_undoes_forward_rotation(rope_f, g4.global_head_dim, g4.rope_theta_global, partial)


@pytest.mark.skipif(not os.path.isdir(WEIGHTS_G4), reason=f"no dumped weights at {WEIGHTS_G4}")
def test_gemma4_12b_flash_geometry_builds_against_the_real_operator_sequence(monkeypatch):
    """Every other test in this file stubs `OperatorSequence`, so a caller/operator mismatch that
    only `calculate_buffer_layout()` (the REAL class) would catch -- a conflicting-size collision,
    a missing runlist buffer, an arg-spec/runlist arity mismatch -- passes silently. Builds
    gemma4-12b for real (KV_SKIP_V/FUSE_ATTN_GLOBAL_FLASH on, FUSE_QKV_DP off so the unfused head
    carries n_kn/rope_f into AttnGlobalFlash), with the real `iron.common.sequence.OperatorSequence`
    and only `.compile()` stubbed out (no aiecc/mlir-aie, no device).

    Needs the on-chip-rope operator worktree (wt-kv-skip-v-onchip-rope) on PYTHONPATH: an IRON
    checkout whose `AttnGlobalFlash.get_arg_spec()` still declares the old capacity-sized `ang`
    (e.g. the default $IRON_DIR) has no `rope_f` slot at all, so skip rather than fail for a
    reason unrelated to this repo's own wiring.
    """
    from iron.operators.attn_global_dp.op import AttnGlobalFlash

    probe = AttnGlobalFlash(HD=8, Hq=8, capacity=64)
    probe_spec = probe.get_arg_spec()
    if len(probe_spec) != 6 or probe_spec[2].shape != (8,):
        pytest.skip("this IRON checkout's AttnGlobalFlash has no rope_f arg slot "
                    "(pre-onchip-rope get_arg_spec) -- point PYTHONPATH at "
                    "wt-kv-skip-v-onchip-rope")

    monkeypatch.setattr(gen, "KV_SKIP_V", True)
    monkeypatch.setattr(gen, "FUSE_ATTN_GLOBAL_FLASH", True)
    monkeypatch.setattr(gen, "FUSE_QKV_DP", False)
    monkeypatch.setattr(gen.OperatorSequence, "compile", lambda self, dry_run=False: self)

    sp, fused, w, md = gen.build_graph("gemma4-12b", WEIGHTS_G4, LAYERS, 2048)

    all_flash = [e for e in fused.runlist if type(e[0]).__name__ == "AttnGlobalFlash"]
    seen, flash_entries = set(), []
    for e in all_flash:
        if id(e[0]) not in seen:
            seen.add(id(e[0]))
            flash_entries.append(e)
    assert any(e[0].kv_skip_v for e in flash_entries), \
        "gemma4-12b's global/attention_k_eq_v geometry must set kv_skip_v"

    subbuffer_layout, _buffer_sizes, _slice_info = fused.calculate_buffer_layout()

    partial = sp.rope_partial_rotary if sp.rope_type_global == "proportional" else None
    for op, ref_q, n_kn, rope_f, kc, vc, cx in flash_entries:
        assert len(op.get_arg_spec()) == 6 == len(("q", n_kn, rope_f, kc, vc, cx))

        buf_type, _offset, length = subbuffer_layout[rope_f]
        assert buf_type == "scratch"
        assert length == op.HD * 2, f"rope_f layout length {length} != HD*2 ({op.HD * 2})"
        assert w[rope_f].shape == (op.HD,), f"rope_f weight shape {w[rope_f].shape} != (HD,)"

        # Content, not just shape: a swap of n_kn/rope_f in the caller runlist tuple would leave
        # this position holding n_kn's trained qk-norm gain (or an all-zero rope_f in its place),
        # neither of which equals the deterministic rope_f formula -- shape alone cannot tell them
        # apart, since qk-norm gain is head_dim-wide too.
        expected = gen.rope_f_for_geometry(op.HD, sp.rope_theta_global, partial)
        np.testing.assert_array_equal(w[rope_f].view(np.uint32), expected.view(np.uint32))


def test_rope_f_refuses_ordinary_partial_rotary_under_kv_skip_v(monkeypatch):
    """rope_f_for_geometry only undoes Gemma-4's PROPORTIONAL partial rotary (full head_dim
    width, exponent over head_dim). A spec that also sets the ordinary `rope_rotary_dim` axis
    (llm_decode_spec.py:533-536 -- first N dims rotated, split at their own half-point) under
    kv_skip_v would silently build a rope_f that does not undo it; must refuse by name instead.
    """
    monkeypatch.setattr(gen, "OperatorSequence", _StubSeq)
    monkeypatch.setattr(gen, "DECODE_SEGMENTS", 1)
    monkeypatch.setattr(gen, "KV_SKIP_V", True)
    monkeypatch.setattr(gen, "FUSE_ATTN_GLOBAL_FLASH", True)
    base = gen.SPECS[SPEC]
    bad_spec = dataclasses.replace(
        base, name="gemma3-270m-flash-bad-rope", v_from_k_on_global=True, v_norm=True,
        global_head_dim=base.head_dim, global_n_kv_heads=base.n_kv_heads,
        rope_rotary_dim=base.head_dim // 2)
    monkeypatch.setitem(gen.SPECS, bad_spec.name, bad_spec)

    with pytest.raises(NotImplementedError, match="rope_rotary_dim"):
        gen.build_graph(bad_spec.name, WEIGHTS, LAYERS, 2048)
