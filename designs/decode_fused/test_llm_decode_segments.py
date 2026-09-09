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
import os

import pytest

gen = pytest.importorskip("gen_llm_decode")

SPEC = "gemma3-270m"
WEIGHTS = "/mnt/data/models/xdna-artifacts/gemma3-270m/weights"
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
    every = {n for s in segs for n, _ in s["kv_slots"]}
    for s in segs:
        la, lb = s["layers"]
        want = {sp.head_dim_for(l) for l in range(la, lb)}
        assert {hd for _, hd in s["kv_slots"]} == want, \
            f"segment {s['seq'].name} slots do not match its layers' head_dims"
    # Union over segments must be the whole model's slot set -- a slot owned by nobody is a KV
    # cache the host never advances, which reads as a model that stops attending to its history.
    assert every == {n for n, _ in md["kv_slots"]}
