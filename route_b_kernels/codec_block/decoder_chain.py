#!/usr/bin/env python3
"""The decoder's device chain -- head -> stage1..4 -> tail -> tanh -- via window_driver.

Extracted from verify_whole_decoder.py, which originally defined `run_stage` inline and inlined the
head/tail calls, entangled with its own iso-vs-chain gating. That gate still needs BOTH an iso call
(fed the true stream) and a chain call (fed the previous stage's device output) per stage, so it
keeps calling these pieces individually rather than through `run_chain`; generate_wav.py wants only
the chain, over the WHOLE stream rather than one V_FINAL-sized window, so it calls `run_chain`
directly. Same four functions, one definition, two callers -- a fix here reaches both.

`g` is every caller's own weight cache (`gx.load(GGUF, name)`, memoized) -- passed in rather than
imported, since verify_whole_decoder.py's cache is also warmed by its host-truth computation and
must stay the same dict.

`chain_offset` is the position/length arithmetic every caller needs to know WHERE a chain's output
lands in the true (unwindowed) audio stream and how long it is, walked FORWARD from the per-stage
formula verify_whole_decoder.py's own docstring already derives for its backward V_FINAL solve:
`out_start = (S + UP_CTX) * STRIDE + RES_CTX`. Backward (there) picks the minimum input window for a
target output length; forward (here) reports what an ALREADY-CHOSEN input window's output actually
covers -- same formula, opposite direction, so both must agree on the same CTX_HEAD/CTX_TAIL/
UP_CTX/RES_CTX constants read from stage_shapes.py.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stage_shapes as ss  # noqa: E402
import window_driver as wd  # noqa: E402

PREFIX = ss.PREFIX

CTX_HEAD = (ss.head_shape()[2] - 1) * 1     # 6
CTX_TAIL = (ss.tail_shape()[2] - 1) * 1     # 6
STRIDES = {stage: ss.stage_shape(stage)[3] for stage in (1, 2, 3, 4)}

STAGE_PLAN = {1: ss.plan(1), 2: ss.plan(2), 3: ss.plan(3),
              4: dict(up_ci_chunk=None, res_ci_chunk=None, resident_depth=2)}


def run_head(z, g, tag="head"):
    """[1024, L] -> [1536, L - CTX_HEAD]. model.0.conv, k=7 dilation=1."""
    plan = ss.head_plan()
    return wd.conv(z, g(f"{PREFIX}.model.0.conv.weight"), g(f"{PREFIX}.model.0.conv.bias").reshape(-1),
                   7, 1, tag, ci_chunk=plan["ci_chunk"], resident_depth=plan["resident_depth"])


def run_stage(x, stage, g, suffix):
    """One full decoder stage (upsample + 3 residual units) via window_driver -- the exact sequence
    verify_stage123.py/verify_stage4_block.py run, parameterized over `x` so it can be called on
    either the true-fed or device-chained input."""
    p = f"{PREFIX}.model.{stage}"

    def gg(name):
        return g(f"{p}.{name}")

    _, _, k, stride = ss.stage_shape(stage)
    plan = STAGE_PLAN[stage]
    up = wd.upsample_unfused(x, gg("block.0.alpha").reshape(-1), gg("block.1.conv.weight"),
                             gg("block.1.conv.bias").reshape(-1), stride, f"s{stage}up_{suffix}",
                             ci_chunk=plan["up_ci_chunk"], resident_depth=plan["resident_depth"])
    cur = up
    for i, (sub, dil) in enumerate((("block.2", 1), ("block.3", 3), ("block.4", 9))):
        wts = (gg(f"{sub}.block.0.alpha").reshape(-1), gg(f"{sub}.block.1.conv.weight"),
               gg(f"{sub}.block.1.conv.bias").reshape(-1), gg(f"{sub}.block.2.alpha").reshape(-1),
               gg(f"{sub}.block.3.conv.weight"), gg(f"{sub}.block.3.conv.bias").reshape(-1))
        cur = wd.residual_unit(cur, wts, dil, f"s{stage}u{i}_{suffix}",
                               ci_chunk=plan["res_ci_chunk"], resident_depth=plan["resident_depth"])
    return cur


def run_tail(x, g, suffix=""):
    """[96, L] -> tanh'd audio [1, L - CTX_TAIL]. model.5.alpha snake, model.6.conv k=7, then tanh --
    a pointwise nonlinearity applied here in numpy, same as codec_decoder_ref.decode(), since there
    is nothing device-side to gain from running it as a kernel."""
    last = len(ss.DECODER_RATES) + 1  # 5: model.5.alpha, model.6.conv
    alpha_tail = g(f"{PREFIX}.model.{last}.alpha").reshape(-1)
    w_tail = g(f"{PREFIX}.model.{last + 1}.conv.weight")
    if w_tail.ndim == 2:
        w_tail = w_tail.reshape(1, *w_tail.shape)
    b_tail = g(f"{PREFIX}.model.{last + 1}.conv.bias").reshape(-1)
    snaked = wd.snake(x, alpha_tail, f"tailsnake{suffix}")
    raw = wd.conv(snaked, w_tail, b_tail, 7, 1, f"tailconv{suffix}")
    return np.tanh(raw.astype(np.float64)).astype(np.float32)


def run_chain(z, g):
    """[1024, L] latent -> tanh'd audio [1, L']. head -> stage1..4 -> tail, run ONCE end to end over
    the WHOLE given window -- no iso comparison (verify_whole_decoder.py's gate does that; this is
    the plain generation path). `L'` is whatever `chain_offset(0, L)` predicts; window_driver's own
    per-op windowing (window_driver.py's T=64/UPSAMPLE_T=16 tiles) is internal to each call below and
    needs nothing from the caller beyond `z` itself, however long it is."""
    chain = run_head(z, g)
    for stage in (1, 2, 3, 4):
        chain = run_stage(chain, stage, g, f"s{stage}")
    return run_tail(chain, g)


def chain_offset(latent_start, latent_len):
    """Forward per-stage position tracking for a `run_chain` call fed `latent_len` samples starting
    at `latent_start` in some true (unwindowed) latent stream. Returns `(audio_start, audio_len)`:
    `audio_start` is where the first sample `run_chain` returns lands in the TRUE (unwindowed) audio
    stream -- e.g. codec_audio.bin, whose tanh'd samples are never context-starved because
    codec_decoder_ref.decode()'s causal convs zero-pad instead of dropping; window_driver's ops
    refuse to fabricate that missing context and instead just don't emit samples that would need it,
    so a windowed chain's output is always offset from the true stream by however much context it
    consumed. `audio_len` is `run_chain`'s actual output length, computed the same way `run_chain`
    itself arrives at it (conv: L-ctx; upsample+3 residual units: (L-UP_CTX)*stride-RES_CTX) so
    asserting `run_chain(...).shape[1] == audio_len` after the fact catches a desync here, not a
    device numerics issue.
    """
    p = latent_start + CTX_HEAD
    length = latent_len - CTX_HEAD
    assert length > 0, f"head: {latent_len} latent samples <= CTX_HEAD={CTX_HEAD}"
    for stage in (1, 2, 3, 4):
        p = (p + ss.UP_CTX) * STRIDES[stage] + ss.RES_CTX
        length = (length - ss.UP_CTX) * STRIDES[stage] - ss.RES_CTX
        assert length > 0, (
            f"stage {stage}: window exhausted (output length {length} <= 0) -- widen the input "
            "or raise --limit-frames")
    audio_start = p + CTX_TAIL
    audio_len = length - CTX_TAIL
    assert audio_len > 0, f"tail: output length {audio_len} <= 0"
    return audio_start, audio_len
