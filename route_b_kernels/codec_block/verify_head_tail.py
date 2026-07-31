#!/usr/bin/env python3
"""The head conv and the tail conv, gated INDIVIDUALLY on device. Gate 3e-2 each.

    head: causal_conv_1d(model.0.conv, z, stride=1, dilation=1)          1024 -> 1536, k 7
    tail: x = snake(x, model.5.alpha); causal_conv_1d(model.6.conv, x)    96 -> 1,   k 7

Same shape as verify_stage123.py/verify_stage4_block.py: one segment, well inside the stream, valid
outputs only (window_driver's LENGTH CONVENTION -- output j corresponds to input position ctx + j,
ctx = (k-1)*dilation = 6 for both here). Head needs stage_shapes.head_plan()'s ci_chunk=128,
resident_depth=1 (c_in=1024 blows a 64 KiB tile unchunked -- see stage_shapes.py). Tail is
shape-identical to stage 4 (c_in=96, k=7) and needs no chunking, same as the residual convs stage 4
already runs unchunked.

Unlike stages 1-4, neither head nor tail has a captured true-in/true-out pair sitting in
stage_io.npz: dump_stage_io.py starts capturing only AFTER the head conv (its `stage1_in` key IS
head's true output, but head's true INPUT -- the raw latent -- was never saved), and stops at
`stage4_out`, before the final snake/tail/tanh. So this script's primary input is a DUMP DIR (same
`codec_latent.bin` / `codec_audio.bin` + `.shape` pair codec_decoder_ref.py and dump_stage_io.py
already read), which alone is sufficient. An optional stage_io.npz, in the same trailing-positional
spot verify_stage123.py puts it, is used only as a cross-check against a freshly host-derived
`stage1_in` (should be rel-L2 0.0 -- both are the same op on the same weights) -- it is never wired
into the actual gates, so a stale or differently-clipped npz cannot silently change the result.

Tail's truth is `codec_audio.bin` itself (the real s2.cpp oracle, post-tanh) -- tanh is applied here
in numpy after the device conv, exactly where codec_decoder_ref.py applies it, since it is a
pointwise nonlinearity with nothing device-side to gain from running it as a kernel.

    python3 verify_head_tail.py <dump_dir> [stage_io.npz]

dump_dir needs codec_latent.{bin,shape} and codec_audio.{bin,shape} (e.g. an s2dump_cpu-style
capture; the codes-only s2dump lacks codec_latent and will fail loud on the missing file).

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import window_driver as wd  # noqa: E402
import stage_shapes as ss  # noqa: E402
import gguf_extract as gx  # noqa: E402
import codec_decoder_ref as R  # noqa: E402

GATE = 3e-2
V_TARGET = 128
GGUF = ss.GGUF
PREFIX = R.PREFIX

if len(sys.argv) < 2:
    raise SystemExit(
        f"usage: {sys.argv[0]} <dump_dir> [stage_io.npz]\n"
        "  dump_dir must contain codec_latent.{bin,shape} (head's true input) and\n"
        "  codec_audio.{bin,shape} (tail's true output, the real s2.cpp oracle, post-tanh).\n"
        "  stage_io.npz is optional and used only as a cross-check, not wired into the gates.")
DUMP_DIR = Path(sys.argv[1])
NPZ_PATH = sys.argv[2] if len(sys.argv) > 2 else None

_cache = {}


def g(name):
    if name not in _cache:
        _cache[name] = gx.load(GGUF, name).astype(np.float32)
    return _cache[name]


def alignment(cur, truth, V):
    """Same explicit shift-search every stage gate uses: a one-sample misalignment reads as a small
    rel-L2 on smooth audio, so this has to be checked rather than trusted from the rel-L2 alone."""
    return min(range(-2, 3),
               key=lambda s_: np.linalg.norm(cur[:, 2:V - 2] - truth[:, 2 + s_:V - 2 + s_]))


def report(label, cur, truth):
    assert cur.shape == truth.shape, f"{label}: {cur.shape} vs {truth.shape}"
    num = np.linalg.norm((cur - truth).astype(np.float64))
    den = np.linalg.norm(truth.astype(np.float64))
    rl2 = float(num / den)
    err = np.abs(cur - truth)
    V = cur.shape[1]
    best = alignment(cur, truth, V)
    print(f"  {label} rel-L2 vs truth      : {rl2:.3e}   (gate {GATE:.1e})")
    print(f"  {label} max abs err          : {float(err.max()):.3e}")
    print(f"  {label} best alignment shift : {best} (0 means the index arithmetic is right)")
    assert best == 0, f"{label} output is misaligned by {best} samples"
    assert rl2 <= GATE, f"{label} gate failed: rel_l2={rl2}"
    return rl2


# ---- host truth -----------------------------------------------------------------------------
# latent -> head conv (causal_conv_1d, zero-padded like codec_decoder_ref's own reference) gives
# head's true output == stage1_in. Getting to stage4_out needs the full 4-stage reference chain --
# there is no shortcut, tail's true input is genuinely downstream of every earlier stage.
zf, zshape = R.load_dump(DUMP_DIR, "codec_latent")
z_full = zf.reshape(zshape[1], zshape[0]).T.copy()
af, ashape = R.load_dump(DUMP_DIR, "codec_audio")
oracle_audio = af.reshape(-1)
print(f"latent {z_full.shape}  oracle audio {oracle_audio.shape}")

head_out_full = R.conv_1d_causal(
    z_full, g(f"{PREFIX}.model.0.conv.weight"), g(f"{PREFIX}.model.0.conv.bias").reshape(-1), 1)
stage1_in_full = head_out_full

if NPZ_PATH:
    npz = np.load(NPZ_PATH)
    npz_stage1_in = npz["stage1_in"].astype(np.float32)
    if npz_stage1_in.shape == head_out_full.shape:
        chk = (np.linalg.norm((npz_stage1_in - head_out_full).astype(np.float64)) /
               np.linalg.norm(head_out_full.astype(np.float64)))
        print(f"cross-check: npz stage1_in vs freshly-derived head_out rel-L2 = {chk:.3e} "
              f"(informational only, not wired into the gate)")
    else:
        print(f"cross-check skipped: npz stage1_in shape {npz_stage1_in.shape} != "
              f"derived head_out shape {head_out_full.shape} (different clip?)")

x = head_out_full
for i, stride in enumerate(R.DECODER_RATES):
    p = f"{PREFIX}.model.{i + 1}"
    x = R.conv_transpose_1d(R.snake(x, g(f"{p}.block.0.alpha").reshape(-1)),
                            g(f"{p}.block.1.conv.weight"),
                            g(f"{p}.block.1.conv.bias").reshape(-1), stride, crop_right=stride)
    for unit, dil in ((2, 1), (3, 3), (4, 9)):
        # R.residual_unit calls its W(name) argument with the FULL "prefix.block.N..." key already,
        # which is exactly what g() takes -- pass it straight through, no wrapping needed.
        x = R.residual_unit(x, g, f"{p}.block.{unit}", dil)
stage4_out_full = x

last = len(R.DECODER_RATES) + 1  # 5: model.5.alpha, model.6.conv
alpha_tail = g(f"{PREFIX}.model.{last}.alpha").reshape(-1)
w_tail = g(f"{PREFIX}.model.{last + 1}.conv.weight")
if w_tail.ndim == 2:
    w_tail = w_tail.reshape(1, *w_tail.shape)
b_tail = g(f"{PREFIX}.model.{last + 1}.conv.bias").reshape(-1)

wd.reset_stats()

# ---- HEAD -------------------------------------------------------------------------------------
c_in, c_out, k = ss.head_shape()
CTX_HEAD = (k - 1) * 1
hp = ss.head_plan()
latent_len = z_full.shape[1]
L_IN = V_TARGET + CTX_HEAD
assert latent_len >= L_IN, (
    f"latent only {latent_len} samples, need >= {L_IN} for a head gate at V_TARGET={V_TARGET} "
    "-- shrink V_TARGET or use a longer dump (e.g. s2dump_long)")
S_HEAD = max(0, min(40, latent_len - L_IN))

x_head = np.ascontiguousarray(z_full[:, S_HEAD:S_HEAD + L_IN])
print(f"\nHEAD: c_in={c_in} c_out={c_out} k={k}  plan={hp}")
print(f"  input {x_head.shape} from latent index {S_HEAD}")
head_dev = wd.conv(x_head, g(f"{PREFIX}.model.0.conv.weight"),
                   g(f"{PREFIX}.model.0.conv.bias").reshape(-1), k, 1, "head",
                   ci_chunk=hp["ci_chunk"], resident_depth=hp["resident_depth"])
out_start = S_HEAD + CTX_HEAD
head_truth = stage1_in_full[:, out_start:out_start + V_TARGET].astype(np.float32)
print(f"  expect {V_TARGET} outputs at [{out_start}, {out_start + V_TARGET})")
report("HEAD", head_dev, head_truth)

# ---- TAIL ---------------------------------------------------------------------------------
tc_in, tc_out, tk = ss.tail_shape()
CTX_TAIL = (tk - 1) * 1
s4_len = stage4_out_full.shape[1]
L_IN_T = V_TARGET + CTX_TAIL
S_TAIL = 20000
assert s4_len >= S_TAIL + L_IN_T, (
    f"stage4_out only {s4_len} samples, need >= {S_TAIL + L_IN_T} for the tail gate")

x_tail = np.ascontiguousarray(stage4_out_full[:, S_TAIL:S_TAIL + L_IN_T])
print(f"\nTAIL: c_in={tc_in} c_out={tc_out} k={tk}  (unchunked, resident_depth=2, same shape "
      "as stage 4)")
print(f"  input {x_tail.shape} from stage4_out index {S_TAIL}")
tail_snake_dev = wd.snake(x_tail, alpha_tail, "tailsnake")
tail_raw_dev = wd.conv(tail_snake_dev, w_tail, b_tail, tk, 1, "tailconv")
tail_audio_dev = np.tanh(tail_raw_dev.astype(np.float64)).astype(np.float32)
out_start_t = S_TAIL + CTX_TAIL
tail_truth = oracle_audio[out_start_t:out_start_t + V_TARGET].reshape(1, -1).astype(np.float32)
print(f"  expect {V_TARGET} outputs at [{out_start_t}, {out_start_t + V_TARGET}) of codec_audio.bin")
report("TAIL", tail_audio_dev, tail_truth)

st = wd.stats()
print(f"\ndispatches {st['dispatches']}, useful {st['useful']}/{st['computed']} "
      f"-> {st['overhead'] * 100:.1f}% recompute")
print("PASS")
