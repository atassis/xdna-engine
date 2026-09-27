"""SPAN x2's conv chain as the device runs it: stage order, row layouts, skip depths, weight
groups, and the host side of the data (input rows, weights, output rows). No IRON, so it is
checked on the CPU (test_net_layout.py); net_design.py builds the design from it.
"""
from collections import namedtuple

import numpy as np

from span_int import c3

C = 48
PAD = c3.PAD  # zero pixels on each side of every row

# kind per stage; row layouts in `layout`
STAGES = ([("conv_1", "conv1")]
          + [(f"b{i}c{j}", kind) for i in range(1, 7) for j, kind in
             ((1, "silu16" if i == 1 else "silu_x"), (2, "silu_i16" if i == 1 else "silu"),
              (3, "gate"))]
          + [("conv_2", "plain"), ("conv_cat", "cat"), ("up", "up")])

# conv_cat's sources in its weight order, with each source row's offset (in rows of 48 int8
# channels) in the joined element; b6c1's row carries x after its SiLU output
CAT_SOURCES = [("conv_1", 0), ("conv_2", 1), ("b1c3", 2), ("b6c1", 3)]
CAT_HALVES = 5

# policy: rows of headroom past the minimum a skip must hold, so a source is not throttled by
# its skip before the main path catches up. Was 2; raised to 8 2026-09-27 (device dose-response,
# TRACE_RESULTS.md skip-ring latency-throttle section): whole-net wall-clock fell from ~2624 to
# ~659 cyc/px between slack=2 and slack=8, with slack=16/25 buying no further win in the same
# run (noise-bound plateau at 660-750). A same-process trace at the two ends (slack=2 vs 25)
# corroborates the mechanism directly: b3c3's gap fell 919.72 -> 224.92 cyc/px with compute
# pinned at 406.2 throughout. 8 sits well under the 512 KB MemTile ring ceiling (max fitting
# slack=25 at W=32), leaving headroom for later MAIN_DEPTH/PROD_DEPTH work.
SKIP_SLACK = 8
# policy: a MemTile has 6 MM2S channels, so one weight split feeds at most 6 cores
WEIGHT_GROUP = 6

# stages the device gate checks, in bring-up order: span_int tensor, channels, dtype
GOLDEN = {"conv_1": ("conv_1", 48, np.int8), "b1c1": ("b1.c1.silu", 48, np.int16),
          "b1c3": ("b1.out", 48, np.int8), "b3c3": ("b3.out", 48, np.int8),
          "b6c3": ("b6.out", 48, np.int8), "conv_2": ("conv_2", 48, np.int8),
          "conv_cat": ("conv_cat", 48, np.int8), "up": ("rgba", 16, np.uint8)}

Layout = namedtuple("Layout", "in_bytes out_bytes x_in x_out")


def half(w):
    """Bytes of one int8 48-channel row at strip width w."""
    return (w + 2 * PAD) * C


def layout(kind, w):
    h = half(w)
    return {
        "conv1": Layout((w + 2 * PAD) * 8, h, None, None),
        "silu16": Layout(h, 3 * h, 0, 2 * h),
        "silu_i16": Layout(3 * h, 2 * h, 2 * h, h),
        "silu_x": Layout(h, 2 * h, 0, h),
        "silu": Layout(2 * h, 2 * h, h, h),
        "gate": Layout(2 * h, h, h, None),
        "plain": Layout(h, h, None, None),
        "cat": Layout(CAT_HALVES * h, h, None, None),
        "up": Layout(h, (w + 2 * PAD) * 16, None, None),
    }[kind]


def stage_names(upto="up"):
    names = [n for n, _ in STAGES]
    return names[:names.index(upto) + 1]


def rows_ahead(src):
    """Rows `src` must have emitted before conv_cat can emit row 0: one, plus one per 3x3 stage
    strictly between them on the main path (each needs row y+1 to emit row y)."""
    names = [n for n, _ in STAGES]
    between = STAGES[names.index(src) + 1:names.index("conv_cat")]
    return 1 + sum(1 for _, kind in between if kind != "cat")


def skip_depth(src):
    return rows_ahead(src) + SKIP_SLACK


def weight_groups(names):
    return [names[i:i + WEIGHT_GROUP] for i in range(0, len(names), WEIGHT_GROUP)]


def weights_blob(NP, names):
    return np.concatenate([NP[n]["blob"] for n in names]).astype(np.int8)


def conv1_rows(rgb01, mean255):
    """conv_1's host input [H+2][1][W+16][8] uint8: the frame padded by one row and column of the
    rounded mean colour, as span_int.int_tensors pads it. The padding column sits in the zero
    margin, where the kernel reads it as pixel -1 and W; channels 3..7 are zero."""
    x8 = np.round(rgb01 * 255).astype(np.int64)
    _, h, w = x8.shape
    pad = np.round(mean255).astype(np.int64)[:, None]
    xp = np.zeros((8, h + 2, w + 2 * PAD), np.int64)
    xp[:3, :, PAD - 1:PAD + w + 1] = pad[:, :, None]
    xp[:3, 1:-1, PAD:PAD + w] = x8
    return np.ascontiguousarray(
        xp.reshape(1, 8, h + 2, w + 2 * PAD).transpose(2, 0, 3, 1)).astype(np.uint8)


def split_gate_params(p, cin=C, cout=C, lo_channels=32):
    """Split a gate stage's net_params() dict into two output-channel halves: golden.pack_params
    concatenates [weights][bias][mult], each ordered by ob (8-channel block), so each section
    slices at the same ob boundary. conv3x3_core processes ob in PAIRS (ob += 2), so each half's
    channel count must be a multiple of 16 -- cout=48 has no even 24/24 split point, so the
    default lo_channels=32 gives 32/16, the closest the kernel's block-pairing allows.
    ga/gb/gs1/gc/gs2/tables/pre/shift are per-net scalars, shared by both halves as is."""
    assert lo_channels % 16 == 0 and 0 < lo_channels < cout
    ocb, lo_ocb = cout // 8, lo_channels // 8
    wblk = 3 * (cin // 8) * 3 * 64
    w_end, b_end = ocb * wblk, ocb * wblk + ocb * 256

    def part(lo):
        a, z = (0, lo_ocb) if lo else (lo_ocb, ocb)
        blob = p["blob"]
        return np.concatenate([blob[a * wblk:z * wblk], blob[w_end + a * 256:w_end + z * 256],
                               blob[b_end + a * 128:b_end + z * 128]])

    return {**p, "blob": part(True)}, {**p, "blob": part(False)}


def unpack_out(y, upto, h, w):
    """Device output rows of stage `upto` -> [ch, h, w], dropping any in-band x."""
    _, ch, dt = GOLDEN[upto]
    own = ch * (w + 2 * PAD) * np.dtype(dt).itemsize
    rows = np.ascontiguousarray(np.asarray(y).view(np.int8).reshape(h, -1)[:, :own])
    return c3.unpack_rows(rows.view(dt).reshape(-1), ch, h, w)
