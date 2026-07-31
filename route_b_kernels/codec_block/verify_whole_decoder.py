#!/usr/bin/env python3
"""THE decoder, head to tail, chained on device. Gate: final audio rel-L2 vs codec_audio.bin <= 3e-2.

    head -> stage1 -> stage2 -> stage3 -> stage4 -> tail -> tanh

Every gate before this one (verify_stage123.py, verify_stage4_block.py, verify_head_tail.py) tests
ONE op or ONE stage in ISOLATION: fed the TRUE unblocked stream, sliced straight out of a captured
truth array. That proves each op's own kernel/windowing arithmetic is right, but it never proves the
INTERFACE between two device ops is right, because op N+1 has only ever seen op N's TRUE output, not
its DEVICE output. This script is the first one where op N+1 consumes op N's actual device result --
so an interface or alignment mistake between stages shows up here for the first time, if anywhere.

To make that visible rather than just producing one final number, every stage is run TWICE:
  - "iso"   : fed the TRUE stream, sliced at the position this call's output should land at -- the
              same thing verify_stage123.py already does, repeated here as a same-script control.
  - "chain" : fed the PREVIOUS stage's own DEVICE output -- the real end-to-end path.
Both calls get identical-shape input (same absolute stream position, same length), so their outputs
are directly comparable against the same truth slice, and the DIFFERENCE between the two rel-L2s is
the error this stage's own device output introduces into the next stage -- the compounding number
nobody has measured before. Doubling every stage roughly doubles dispatch count; see the total
printed at the end.

SIZING, worked BACKWARD from the desired number of final audio samples V_FINAL (default 128, the
same V_TARGET every other gate here uses -- overridable via $V_FINAL). Each decoder stage consumes
UP_CTX=2 (at its own input rate) then RES_CTX=78 (at its own output rate) before producing anything
valid; head and tail each consume ctx=6 (k=7, dilation=1, same as any residual dilated conv at
dilation 1). Solving each of those backward (ceil division, same as verify_stage123.py's own per-
stage formula) from tail up to head gives the minimum latent WINDOW length that survives the whole
chain and still lands >= V_FINAL valid audio samples. The forward pass then never re-derives a length
from a formula -- every length used past the first window is read off the actual array `.shape`
returned by a device call, so ceiling slack anywhere upstream cannot desync the two sides.

POSITION TRACKING is the same per-stage formula verify_stage123.py and verify_stage4_block.py already
use (`out_start = (S + UP_CTX) * STRIDE + RES_CTX`), just composed six times running forward from a
chosen latent start S: P[1] = S + 6 (head's ctx), P[k+1] = (P[k] + UP_CTX) * STRIDE[k] + RES_CTX for
k=1..4 (stage k's own stride, read from stage_shapes.py, not hardcoded), P_audio = P[5] + 6 (tail's
ctx). P[k] is "the position, in the TRUE full stream this stage reads from, where this call's INPUT
starts" -- it is what lets a truth array sliced at [P[k+1], P[k+1]+V_actual) be the right comparison
for either the chain or the iso output, since both have the same shape by construction.

    python3 verify_whole_decoder.py <dump_dir> [stage_io.npz]

dump_dir needs codec_latent.{bin,shape} (decoder input) and codec_audio.{bin,shape} (the oracle),
same as codec_decoder_ref.py and dump_stage_io.py already read (e.g. an s2dump_cpu-style capture).
stage_io.npz is optional: if given, its stage{1..4}_in/out arrays are cross-checked against a fresh
host-derived reference (informational only) but are NOT wired into the gates -- this script derives
its own truth via codec_decoder_ref.py's primitives from the SAME dump_dir passed in, so it is
self-sufficient and cannot silently disagree with itself about which clip is being gated.

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import os
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
V_FINAL = int(os.environ.get("V_FINAL", "128"))
GGUF = ss.GGUF
PREFIX = R.PREFIX
UP_CTX = ss.UP_CTX          # 2
RES_CTX = ss.RES_CTX        # 78

if len(sys.argv) < 2:
    raise SystemExit(
        f"usage: {sys.argv[0]} <dump_dir> [stage_io.npz]\n"
        "  dump_dir must contain codec_latent.{bin,shape} and codec_audio.{bin,shape}.\n"
        "  stage_io.npz is optional and used only as a cross-check, not wired into the gates.\n"
        "  V_FINAL env var overrides the default 128 target valid final-audio samples.")
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


def rel_l2(cur, truth):
    num = np.linalg.norm((cur - truth).astype(np.float64))
    den = np.linalg.norm(truth.astype(np.float64))
    return float(num / den)


def check(label, cur, truth, gate=None):
    """Report rel-L2 + alignment. Alignment is ALWAYS asserted -- a nonzero shift means the index
    arithmetic in this script is wrong, not a numerical compounding effect, and must not be papered
    over (see the journal AGENTS.md HAZARDS note this task was briefed with). rel-L2 is only asserted
    against `gate` when the caller passes one -- per-stage numbers are reported, not gated; only the
    final audio is gated, per spec."""
    assert cur.shape == truth.shape, f"{label}: {cur.shape} vs {truth.shape}"
    V = cur.shape[1]
    rl2 = rel_l2(cur, truth)
    best = alignment(cur, truth, V)
    print(f"    {label:28s} rel-L2 {rl2:.3e}  align {best:+d}  shape {cur.shape}")
    assert best == 0, f"{label}: misaligned by {best} samples -- index arithmetic is wrong"
    if gate is not None:
        assert rl2 <= gate, f"{label}: gate failed rel_l2={rl2} > {gate}"
    return rl2


# =====  host truth: latent -> head -> stage1..4 -> tail -> tanh, via codec_decoder_ref primitives  =
zf, zshape = R.load_dump(DUMP_DIR, "codec_latent")
z_full = zf.reshape(zshape[1], zshape[0]).T.copy()
af, ashape = R.load_dump(DUMP_DIR, "codec_audio")
oracle_audio = af.reshape(-1)
print(f"latent {z_full.shape}  oracle audio {oracle_audio.shape}")

# TRUE_STREAM[0] = head's true output (== stage1_in). TRUE_STREAM[k] = stage k's true output
# (== stage (k+1)_in for k<4), k=1..4. One array per stage boundary, all at that boundary's own rate.
TRUE_STREAM = {}
TRUE_STREAM[0] = R.conv_1d_causal(
    z_full, g(f"{PREFIX}.model.0.conv.weight"), g(f"{PREFIX}.model.0.conv.bias").reshape(-1), 1)

x = TRUE_STREAM[0]
for i, stride in enumerate(R.DECODER_RATES):
    stage = i + 1
    p = f"{PREFIX}.model.{stage}"
    x = R.conv_transpose_1d(R.snake(x, g(f"{p}.block.0.alpha").reshape(-1)),
                            g(f"{p}.block.1.conv.weight"),
                            g(f"{p}.block.1.conv.bias").reshape(-1), stride, crop_right=stride)
    for unit, dil in ((2, 1), (3, 3), (4, 9)):
        x = R.residual_unit(x, g, f"{p}.block.{unit}", dil)
    TRUE_STREAM[stage] = x

last = len(R.DECODER_RATES) + 1  # 5: model.5.alpha, model.6.conv
alpha_tail = g(f"{PREFIX}.model.{last}.alpha").reshape(-1)
w_tail = g(f"{PREFIX}.model.{last + 1}.conv.weight")
if w_tail.ndim == 2:
    w_tail = w_tail.reshape(1, *w_tail.shape)
b_tail = g(f"{PREFIX}.model.{last + 1}.conv.bias").reshape(-1)
tail_raw_host = R.conv_1d_causal(R.snake(TRUE_STREAM[4], alpha_tail), w_tail, b_tail, 1)
host_audio = np.tanh(tail_raw_host.astype(np.float64)).astype(np.float32).reshape(-1)

host_vs_oracle = rel_l2(host_audio.reshape(1, -1), oracle_audio.reshape(1, -1))
print(f"host reference (codec_decoder_ref primitives) vs codec_audio.bin oracle: "
      f"{host_vs_oracle:.6e}  (this is REFERENCE error, not device error)")

if NPZ_PATH:
    npz = np.load(NPZ_PATH)
    worst = 0.0
    for stage in (1, 2, 3, 4):
        for key, truth_arr in ((f"stage{stage}_in", TRUE_STREAM[stage - 1]),
                                (f"stage{stage}_out", TRUE_STREAM[stage])):
            if key in npz.files and npz[key].shape == truth_arr.shape:
                c = rel_l2(npz[key].astype(np.float32), truth_arr)
                worst = max(worst, c)
    print(f"cross-check: npz stage arrays vs freshly-derived truth, worst rel-L2 = {worst:.3e} "
          f"(informational only, not wired into the gates)")


# =====  sizing: solve backward from V_FINAL for the latent window length  ==========================
CTX_HEAD = (ss.head_shape()[2] - 1) * 1     # 6
CTX_TAIL = (ss.tail_shape()[2] - 1) * 1     # 6

STRIDES = {stage: ss.stage_shape(stage)[3] for stage in (1, 2, 3, 4)}

v_target = V_FINAL + CTX_TAIL               # stage4's own output length needed
L = {}
for stage in (4, 3, 2, 1):
    L[stage] = UP_CTX + -(-(v_target + RES_CTX) // STRIDES[stage])   # ceil division
    v_target = L[stage]                     # stage (stage-1)'s output length required
L_head = L[1] + CTX_HEAD

latent_len = z_full.shape[1]
assert latent_len >= L_head, (
    f"latent only {latent_len} samples, need >= {L_head} for V_FINAL={V_FINAL} "
    "-- lower $V_FINAL or use a longer dump (e.g. s2dump_long)")
S = max(0, min(40, latent_len - L_head))

print(f"\nV_FINAL={V_FINAL}  backward-solved required lengths: head_in(latent)={L_head} "
      f"stage1_in={L[1]} stage2_in={L[2]} stage3_in={L[3]} stage4_in={L[4]}")
print(f"latent window: [{S}, {S + L_head}) of {latent_len}")


STAGE_PLAN = {1: ss.plan(1), 2: ss.plan(2), 3: ss.plan(3),
              4: dict(up_ci_chunk=None, res_ci_chunk=None, resident_depth=2)}


def run_stage(x, stage, suffix):
    """One full decoder stage (upsample + 3 residual units) via window_driver -- the exact same
    sequence verify_stage123.py/verify_stage4_block.py run, just parameterized over `x` so it can be
    called on EITHER the true-fed or device-chained input."""
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


# =====  forward pass: head, then stages 1-4 chained + iso, then tail  ==============================
wd.reset_stats()
disp = {"prev": 0}


def phase(name):
    now = wd.stats()["dispatches"]
    print(f"    [{name}: {now - disp['prev']} dispatches]")
    disp["prev"] = now


results = {}     # label -> rel-L2, for the final summary table

z_window = np.ascontiguousarray(z_full[:, S:S + L_head])
head_dev = wd.conv(z_window, g(f"{PREFIX}.model.0.conv.weight"),
                   g(f"{PREFIX}.model.0.conv.bias").reshape(-1), 7, 1, "head",
                   ci_chunk=ss.head_plan()["ci_chunk"], resident_depth=ss.head_plan()["resident_depth"])
phase("head")

P = [None] * 6
P[1] = S + CTX_HEAD
head_truth = TRUE_STREAM[0][:, P[1]:P[1] + head_dev.shape[1]].astype(np.float32)
print("\n  HEAD (feeds stage1's device-chain input; no iso/chain split -- it is the chain's start)")
results["head"] = check("head", head_dev, head_truth)

chain = head_dev            # the real end-to-end path, carried forward stage to stage
for stage in (1, 2, 3, 4):
    iso_in = TRUE_STREAM[stage - 1][:, P[stage]:P[stage] + chain.shape[1]].astype(np.float32)
    assert iso_in.shape == chain.shape, (
        f"stage {stage}: iso input {iso_in.shape} vs chain input {chain.shape} -- "
        f"P[{stage}]={P[stage]} runs past TRUE_STREAM[{stage - 1}] (len "
        f"{TRUE_STREAM[stage - 1].shape[1]}); widen the dump or lower V_FINAL")

    chain_out = run_stage(chain, stage, f"s{stage}chain")
    phase(f"stage{stage} chain")
    iso_out = run_stage(iso_in, stage, f"s{stage}iso")
    phase(f"stage{stage} iso")

    assert chain_out.shape == iso_out.shape, (
        f"stage {stage}: chain output {chain_out.shape} != iso output {iso_out.shape} -- "
        "same-shape input should always give same-shape output; a real bug, not compounding")

    P[stage + 1] = (P[stage] + UP_CTX) * STRIDES[stage] + RES_CTX
    truth = TRUE_STREAM[stage][:, P[stage + 1]:P[stage + 1] + chain_out.shape[1]].astype(np.float32)

    print(f"\n  STAGE {stage} (c_in={ss.stage_shape(stage)[0]} c_out={ss.stage_shape(stage)[1]} "
          f"stride={STRIDES[stage]})  input len {chain.shape[1]} -> output len {chain_out.shape[1]}")
    iso_rl2 = check(f"stage{stage} fed TRUE (iso)", iso_out, truth)
    chain_rl2 = check(f"stage{stage} fed DEVICE (chain)", chain_out, truth)
    delta = chain_rl2 - iso_rl2
    ratio = (chain_rl2 / iso_rl2) if iso_rl2 > 0 else float("inf")
    print(f"    stage{stage} COMPOUNDING: chain - iso = {delta:+.3e}   "
          f"(chain/iso = {ratio:.2f}x)")
    results[f"stage{stage}_iso"] = iso_rl2
    results[f"stage{stage}_chain"] = chain_rl2

    chain = chain_out

# ---- tail: snake -> conv -> tanh, chain-fed and iso-fed, gated against the real oracle ----------
iso_tail_in = TRUE_STREAM[4][:, P[5]:P[5] + chain.shape[1]].astype(np.float32)
assert iso_tail_in.shape == chain.shape

chain_tail_snake = wd.snake(chain, alpha_tail, "tailsnake_chain")
chain_tail_raw = wd.conv(chain_tail_snake, w_tail, b_tail, 7, 1, "tailconv_chain")
chain_audio = np.tanh(chain_tail_raw.astype(np.float64)).astype(np.float32)
phase("tail chain")

iso_tail_snake = wd.snake(iso_tail_in, alpha_tail, "tailsnake_iso")
iso_tail_raw = wd.conv(iso_tail_snake, w_tail, b_tail, 7, 1, "tailconv_iso")
iso_audio = np.tanh(iso_tail_raw.astype(np.float64)).astype(np.float32)
phase("tail iso")

P_audio = P[5] + CTX_TAIL
oracle_slice = oracle_audio[P_audio:P_audio + chain_audio.shape[1]].reshape(1, -1).astype(np.float32)

print(f"\n  TAIL  input len {chain.shape[1]} -> output len {chain_audio.shape[1]}  "
      f"(audio index [{P_audio}, {P_audio + chain_audio.shape[1]}))")
iso_tail_rl2 = check("tail fed TRUE (iso) vs oracle", iso_audio, oracle_slice)
chain_tail_rl2 = check("tail fed DEVICE (chain) vs oracle -- THE GATE", chain_audio, oracle_slice,
                       gate=GATE)
print(f"    tail COMPOUNDING: chain - iso = {chain_tail_rl2 - iso_tail_rl2:+.3e}")


# =====  summary  ====================================================================================
st = wd.stats()
print(f"\n{'=' * 78}")
print("SUMMARY")
print(f"  final audio rel-L2 vs codec_audio.bin (device, whole chain) : {chain_tail_rl2:.6e}  "
      f"(gate {GATE:.1e}) {'PASS' if chain_tail_rl2 <= GATE else 'FAIL'}")
print(f"  host reference vs codec_audio.bin (codec_decoder_ref.py)    : {host_vs_oracle:.6e}  "
      f"-- reference error, not device error, do not conflate")
print(f"  device (chain) vs host reference, same segment              : "
      f"{rel_l2(chain_audio, host_audio[P_audio:P_audio + chain_audio.shape[1]].reshape(1, -1)):.6e}"
      f"  -- isolates pure device error from the reference's own {host_vs_oracle:.3e}")
print(f"\n  per-stage compounding (chain rel-L2 minus iso rel-L2, both vs the true stream):")
for stage in (1, 2, 3, 4):
    d = results[f"stage{stage}_chain"] - results[f"stage{stage}_iso"]
    print(f"    stage{stage}: iso={results[f'stage{stage}_iso']:.3e}  "
          f"chain={results[f'stage{stage}_chain']:.3e}  delta={d:+.3e}")
print(f"    tail  : iso={iso_tail_rl2:.3e}  chain={chain_tail_rl2:.3e}  "
      f"delta={chain_tail_rl2 - iso_tail_rl2:+.3e}")
print(f"\n  total dispatches {st['dispatches']}, useful {st['useful']}/{st['computed']} "
      f"-> {st['overhead'] * 100:.1f}% recompute")

assert chain_tail_rl2 <= GATE, f"whole-decoder gate failed: rel_l2={chain_tail_rl2} > {GATE}"
print("\nPASS")
