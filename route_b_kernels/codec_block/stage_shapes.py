#!/usr/bin/env python3
"""Per-stage decoder shapes, read from the real GGUF, and the L1-fit arithmetic that follows.

Stage 4 (the smallest: k=4, 192 -> 96, stride 2) is already device-green through window_driver.py,
gated at rel-L2 8.869e-07 (verify_stage4_block.py). The other three stages are the SAME graph shape
(snake -> conv_transpose_1d -> 3x residual_unit) at successively larger channel counts:

    stage 1: k=16, 1536 -> 768, stride 8      c_in*k = 24576
    stage 2: k=16,  768 -> 384, stride 8      c_in*k = 12288
    stage 3: k= 8,  384 -> 192, stride 4      c_in*k =  3072
    stage 4: k= 4,  192 ->  96, stride 2      c_in*k =   768   <- the one that works today

`c_in * k` is the width of ONE streamed weight row (one output channel's slice of the
conv_transpose weight), and it alone -- before the resident activation window or the output tile
are even counted -- already exceeds a 64 KiB core tile at stages 1 and 2 (see UPSAMPLE_CI_CHUNK
below). The residual units' own dilated conv (k=7, square c x c) has the same problem at stage 1
and 2's channel counts, and is TIGHT even at stage 3's. None of this is a kernel gap -- conv_1d.cc,
conv_transpose_channel.cc and snake.cc are unchanged and device-green; it is purely an L1-fit
question, answered here with arithmetic instead of guessed.

THE TWO LEVERS, and why only one of them helps here:
  - `resident_depth=1` on the activation window. Every window_driver op acquires its resident
    operand ONCE before the streamed-tile loop and releases it once after (bricklib._build_streamed),
    so depth 1 is always numerically safe here -- it is not a stage-specific hack, it is a property
    of how every op in this driver is already structured. It only ~halves the RESIDENT term though,
    and at stage 1/2 the STREAMED weight row is the dominant term, so depth 1 alone is not enough.
  - `ci_chunk`, splitting the channel dimension and accumulating on the host (window_driver.conv()
    already does this for the residual convs; conv_transpose() gains it here). This shrinks BOTH the
    resident window AND the streamed row, because both scale with the chunked channel count -- it is
    the only lever that fixes the streamed-row-alone-exceeds-64KiB case.
  Shrinking the window length t does NOT help conv_transpose: the streamed row is c_in*k+1 floats
  REGARDLESS of t (t only sets the resident window and the output tile), so a stage whose c_in*k
  alone exceeds 64 KiB stays over budget at any t.

    python3 stage_shapes.py                 # print the shape + L1-fit table for every stage + head/tail
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402

GGUF = codec_paths.gguf()
PREFIX = "c.decoder"
DECODER_RATES = [8, 8, 4, 2]            # model.1..model.4, pinned (gguf_shapes.py worked example)

# Mirror window_driver.py's module constants rather than importing it -- this file stays a pure
# shape/arithmetic utility with no aie.iron dependency, so it runs even if the toolchain doesn't.
# If window_driver.T / .UPSAMPLE_T ever change, change these too.
RES_T = 64                              # window_driver.T
UP_T = 16                               # window_driver.UPSAMPLE_T

L1_BUDGET = 0x00010000                  # 64 KiB: AIE2p core-tile local data memory, the hard aiecc
                                         # allocation ceiling (AIETargetModel.h getLocalMemorySize).
F32 = 4
RES_CTX = 6 * 1 + 6 * 3 + 6 * 9          # 78, at the output rate: 3 residual units, k=7, dil 1/3/9,
                                         # each unit's context is 6*dilation. Fixed, stage-independent.
UP_CTX = 2                               # ceil((k-1)/stride); k == 2*stride at every codec rate, so
                                         # this is 2 everywhere and pinned rather than recomputed.


# ---- shapes, from the real GGUF (not hardcoded) -------------------------------------------------

def stage_shape(stage):
    """(c_in, c_out, k, stride) for decoder stage 1..4, from the conv_transpose weight's own shape.
    ggml_conv_transpose_1d layout: ne=[k, c_out, c_in] == numpy [c_in, c_out, k] (gguf_extract.py)."""
    c_in, c_out, k = gx.load(GGUF, f"{PREFIX}.model.{stage}.block.1.conv.weight").shape
    stride = DECODER_RATES[stage - 1]
    assert k == 2 * stride, f"stage {stage}: k={k} != 2*stride={2 * stride} (codec invariant broken)"
    return int(c_in), int(c_out), int(k), int(stride)


def residual_channels(stage):
    """The residual units' own channel count. Must equal the stage's c_out (a residual unit is
    square, c_in==c_out) -- cross-checked against the unit's own dilated-conv weight shape rather
    than assumed, so a wrong prefix shows up as an assertion, not a silent shape mismatch later."""
    _, c_out, _, _ = stage_shape(stage)
    w1 = gx.load(GGUF, f"{PREFIX}.model.{stage}.block.2.block.1.conv.weight")
    assert w1.shape[:2] == (c_out, c_out), (
        f"stage {stage}: residual dilated-conv shape {w1.shape} is not square {c_out}")
    return c_out


def head_shape():
    """(c_in, c_out, k) of model.0.conv, the causal input conv 1024 -> 1536, k=7. ggml_conv_1d
    layout: ne=[k, c_in, c_out] == numpy [c_out, c_in, k]."""
    c_out, c_in, k = gx.load(GGUF, f"{PREFIX}.model.0.conv.weight").shape
    return int(c_in), int(c_out), int(k)


def tail_shape():
    """(c_in, c_out, k) of model.6.conv, 96 -> 1, k=7. Verified against the real file: ne=[7, 96],
    i.e. numpy (96, 7) -- a single output channel folds the c_out axis out entirely, so this is
    (c_in, k) on disk, NOT (k, c_in); codec_decoder_ref.py's own `w_tail.reshape(1, *w_tail.shape)`
    confirms that reading."""
    w = gx.load(GGUF, f"{PREFIX}.model.6.conv.weight")
    if w.ndim == 2:
        c_in, k = w.shape
        return int(c_in), 1, int(k)
    c_out, c_in, k = w.shape
    return int(c_in), int(c_out), int(k)


# ---- L1 arithmetic --------------------------------------------------------------------------
# These totals are exactly what bricklib._build_streamed allocates for one window_driver `_run`
# call: ONE resident operand (depth `resident_depth`) + ONE streamed input tile (always depth 2,
# genuinely wants double buffering) + ONE streamed output tile (always depth 2). f32 throughout.

def conv_l1(c, k, t=RES_T, resident_depth=1, has_add=False):
    """Bytes one window_driver.conv()/residual_unit() call needs, for `c` INPUT channels resident
    at once (i.e. post-chunking -- pass ci_chunk here, not the stage's full channel count).
    has_add=True is the residual unit's 1x1 conv, whose streamed tile also carries the T-wide
    residual add (see window_driver._conv_chunk)."""
    resident = c * t * F32 * resident_depth
    stream = (c * k + 1 + (t if has_add else 0)) * F32 * 2
    out = t * F32 * 2
    return resident, stream, out, resident + stream + out


def conv_transpose_l1(c_in, k, stride, t=UP_T, resident_depth=1):
    """Bytes one window_driver.conv_transpose() call needs, for `c_in` channels resident at once."""
    resident = c_in * t * F32 * resident_depth
    stream = (c_in * k + 1) * F32 * 2
    out = t * stride * F32 * 2
    return resident, stream, out, resident + stream + out


# ---- TIME arithmetic ------------------------------------------------------------------------
# A dispatch has a TIME budget as well as an L1 one, and until 2026-09-01 only L1 was encoded here.
# The head conv (c_in=1024, c_out=1536, k=7, ci_chunk=128) was device-green at 4.119e-07 on
# 2026-07-31 and returned ERT_CMD_STATE_TIMEOUT on every run afterwards, with the device provably
# healthy (snake re-gated at its recorded 5.143e-06 either side of the failure).
#
# MEASURED on device (probe_device_ms.py, hooking CachedXRTRuntime.run so the figure is the
# dispatch and not the aiecc build): device time is linear in the STREAMED operand across three
# decades -- 229632 B -> 223.9 ms, 1837056 B -> 1786.1 ms, 2066688 B -> 2009.3 ms -- and every
# failing shape returns at 2010-2060 ms regardless of how far over it is (3674112 B, which
# extrapolates to ~3570 ms, comes back at 2057.6 ms). That is amdxdna's `tdr_timeout_ms`, default
# 2000, killing a job it judges hung (drivers/accel/amdxdna/aie2_ctx.c:31).
#
# It is NOT a ~2 MiB size limit, which is what the byte figures alone look like: at this op's
# measured rate 2000 ms simply lands near 2 MiB. The two constants are numerically adjacent for
# unrelated reasons, so size and time have to be told apart by measuring time, not inferred.
# Rates are PER KERNEL and differ by 3.2x -- conv_transpose moves a streamed byte far cheaper than
# conv-1d does. Applying conv-1d's slope to an upsample would over-chunk it ~3x and multiply the
# dispatch count for nothing, so each kernel carries its own measured number (probe_device_ms.py,
# probe_ct_device_ms.py). Both are f32 at resident_depth=1; a different dtype or depth is unmeasured.
STREAM_MS_PER_KIB = 0.97                # conv-1d: 229632 B -> 223.9 ms .. 2066688 B -> 2009.3 ms
CT_STREAM_MS_PER_KIB = 0.30             # conv_transpose: 0.304/0.301/0.299/0.298 over 128 KiB..1 MiB
TDR_TIMEOUT_MS = 2000                   # amdxdna aie2_ctx.c:31 default; root-only to change
TDR_MARGIN = 0.7                        # aim well under the watchdog, not at its edge


def _snap_pow2(c):
    """Round DOWN to a power of two, keeping this file's existing "round numbers, not the tightest
    chunk that technically fits" policy. The raw cap is an awkward 34 at the head; 32 divides 1024
    exactly, so no call gets a ragged remainder chunk."""
    p = 1
    while p * 2 <= c:
        p *= 2
    return p


def max_stream_bytes(margin=TDR_MARGIN, ms_per_kib=STREAM_MS_PER_KIB):
    """Largest streamed operand one dispatch may carry and still finish inside the TDR window."""
    return int(TDR_TIMEOUT_MS * margin / ms_per_kib * 1024)


def stream_bytes(c_out, ci_chunk, k, has_add=False, t=RES_T):
    """Bytes window_driver streams for ONE dispatch: one weight row per OUTPUT channel."""
    return c_out * (ci_chunk * k + 1 + (t if has_add else 0)) * F32


def max_ci_chunk_time(c_out, k, has_add=False, t=RES_T, margin=TDR_MARGIN,
                      ms_per_kib=STREAM_MS_PER_KIB):
    """Largest ci_chunk whose dispatch fits the TDR budget. None if the widest sensible chunk
    already fits, mirroring max_ci_chunk's convention."""
    cap = max_stream_bytes(margin, ms_per_kib)
    for c in range(1, 4097):
        if stream_bytes(c_out, c, k, has_add, t) > cap:
            return c - 1 if c > 1 else 0
    return None


def max_ci_chunk(total_c, l1_fn, budget=L1_BUDGET, margin=1.0):
    """Largest chunk size <= total_c whose l1_fn(chunk)[-1] fits budget*margin, found by direct
    search over l1_fn (not a closed-form solve) so this stays correct if l1_fn's shape changes.
    Returns None if the WHOLE width already fits (no chunking needed)."""
    cap = budget * margin
    if l1_fn(total_c)[-1] <= cap:
        return None
    for c in range(total_c, 0, -1):
        if l1_fn(c)[-1] <= cap:
            return c
    raise ValueError(f"no chunk size fits even c=1 against budget {cap} -- check l1_fn")


# ---- the chosen policy, self-checked rather than merely asserted in a comment -------------------
# ROUND numbers, not the tightest chunk that technically fits (which is a less regular ~187-301
# depending on component -- see the printed table). Chosen so every stage lands with clear margin
# below the hard ceiling and so remainder chunks are at most one per call. resident_depth=1 for
# every stage 1-3 (+head) call: it is REQUIRED, not just a nice-to-have, once c_in is this large --
# e.g. stage 3's conv_transpose is 113.3% of budget unchunked at the driver's current default
# (resident_depth=2) and only fits (75.8%) once dropped to 1.
RESIDUAL_CI_CHUNK = {1: 128, 2: 128, 3: 128, 4: None}    # None == unchunked (driver default c=96)
UPSAMPLE_CI_CHUNK = {1: 256, 2: 256, 3: None, 4: None}
HEAD_CI_CHUNK = 128
NEW_RESIDENT_DEPTH = 1                                    # stage 1-3 and head; stage 4/tail keep 2


def plan(stage):
    """(up_ci_chunk, res_ci_chunk, resident_depth) for stage 1-3, self-checked against L1_BUDGET
    (the hard ceiling, not the softer margin the chosen numbers actually land at) before being
    handed back -- so a drifted RES_T/UP_T or a typo'd table entry fails loud here, not on device."""
    assert stage in (1, 2, 3), "stage 4 already fits the driver's defaults; see verify_stage4_block.py"
    c_in, c_out, k, stride = stage_shape(stage)
    c_res = residual_channels(stage)
    rd = NEW_RESIDENT_DEPTH

    # Both budgets bind and they do not move together: stage 1's upsample fits L1 at ci_chunk=256
    # and streams 12.59 MB there, ~3687 ms against a 2000 ms watchdog.
    up_t_cap = max_ci_chunk_time(c_out, k, ms_per_kib=CT_STREAM_MS_PER_KIB)
    up_chunk = UPSAMPLE_CI_CHUNK[stage]
    if up_t_cap is not None and up_t_cap < (up_chunk or c_in):
        up_chunk = _snap_pow2(up_t_cap)   # snap only when TIME binds; else leave the chosen policy
    up_c = up_chunk or c_in
    up_total = conv_transpose_l1(up_c, k, stride, resident_depth=rd)[-1]
    assert up_total <= L1_BUDGET, f"stage {stage} upsample @ ci_chunk={up_chunk}: {up_total} > {L1_BUDGET}"

    res_t_cap = min(max_ci_chunk_time(c_res, 7),
                    max_ci_chunk_time(c_res, 1, has_add=True))
    res_chunk = RESIDUAL_CI_CHUNK[stage]
    if res_t_cap is not None and res_t_cap < (res_chunk or c_res):
        res_chunk = _snap_pow2(res_t_cap)
    res_c = res_chunk or c_res
    dil_total = conv_l1(res_c, 7, resident_depth=rd, has_add=False)[-1]
    onexone_total = conv_l1(res_c, 1, resident_depth=rd, has_add=True)[-1]
    assert dil_total <= L1_BUDGET, f"stage {stage} residual dilated @ ci_chunk={res_chunk}: {dil_total} > {L1_BUDGET}"
    assert onexone_total <= L1_BUDGET, f"stage {stage} residual 1x1 @ ci_chunk={res_chunk}: {onexone_total} > {L1_BUDGET}"

    return dict(up_ci_chunk=up_chunk, res_ci_chunk=res_chunk, resident_depth=rd)


def head_plan():
    """ci_chunk is the MIN of the L1-derived and TDR-derived caps.

    HEAD_CI_CHUNK=128 fits L1 comfortably and does NOT fit the TDR window: at c_out=1536 it streams
    1536*(128*7+1)*4 = 5.5 MB in one dispatch, ~5.4 s against a 2000 ms watchdog. That is why the
    whole-decoder chain died on its first device op. Both budgets bind; only one was encoded."""
    c_in, c_out, k = head_shape()
    rd = NEW_RESIDENT_DEPTH
    t_cap = max_ci_chunk_time(c_out, k)
    ci = HEAD_CI_CHUNK if t_cap is None else _snap_pow2(min(HEAD_CI_CHUNK, t_cap))
    assert ci >= 1, f"head: no ci_chunk fits the TDR budget at c_out={c_out}, k={k}"
    total = conv_l1(ci, k, resident_depth=rd, has_add=False)[-1]
    assert total <= L1_BUDGET, f"head @ ci_chunk={ci}: {total} > {L1_BUDGET}"
    assert stream_bytes(c_out, ci, k) <= max_stream_bytes(), (
        f"head @ ci_chunk={ci}: {stream_bytes(c_out, ci, k)} B streamed > "
        f"{max_stream_bytes()} B TDR budget")
    return dict(ci_chunk=ci, resident_depth=rd)


def _report():
    print(f"GGUF {GGUF}")
    print(f"L1 budget {L1_BUDGET} bytes ({L1_BUDGET / 1024:.0f} KiB), RES_T={RES_T} UP_T={UP_T}\n")

    print("=== UPSAMPLE (snake -> conv_transpose), driver default resident_depth=2 (as shipped) ===")
    for stage in (1, 2, 3, 4):
        c_in, c_out, k, stride = stage_shape(stage)
        r, s, o, tot = conv_transpose_l1(c_in, k, stride, resident_depth=2)
        print(f"  stage {stage}: c_in={c_in:5d} c_out={c_out:4d} k={k:2d} stride={stride}  "
              f"c_in*k={c_in * k:6d}  resident={r:7d} stream={s:7d} out={o:5d} "
              f"total={tot:7d} ({tot / L1_BUDGET * 100:6.1f}%)  {'FITS' if tot <= L1_BUDGET else 'OVER'}")

    print("\n=== UPSAMPLE, resident_depth=1, unchunked (does depth alone save it?) ===")
    for stage in (1, 2, 3, 4):
        c_in, c_out, k, stride = stage_shape(stage)
        r, s, o, tot = conv_transpose_l1(c_in, k, stride, resident_depth=1)
        mx = max_ci_chunk(c_in, lambda c: conv_transpose_l1(c, k, stride, resident_depth=1))
        print(f"  stage {stage}: total={tot:7d} ({tot / L1_BUDGET * 100:6.1f}%) "
              f"{'FITS' if tot <= L1_BUDGET else 'OVER'}   max ci_chunk that fits (100% margin): "
              f"{mx if mx else 'unchunked'}")

    print("\n=== UPSAMPLE, CHOSEN policy (ci_chunk from UPSAMPLE_CI_CHUNK, resident_depth=1) ===")
    for stage in (1, 2, 3, 4):
        c_in, c_out, k, stride = stage_shape(stage)
        chunk = UPSAMPLE_CI_CHUNK[stage] or c_in
        r, s, o, tot = conv_transpose_l1(chunk, k, stride, resident_depth=1)
        nchunks = -(-c_in // chunk)
        print(f"  stage {stage}: ci_chunk={UPSAMPLE_CI_CHUNK[stage]}  chunks={nchunks}  "
              f"total={tot:7d} ({tot / L1_BUDGET * 100:6.1f}%)  {'FITS' if tot <= L1_BUDGET else 'OVER'}")

    print("\n=== RESIDUAL UNIT dilated conv (k=7), driver default resident_depth=2 (as shipped) ===")
    for stage in (1, 2, 3, 4):
        c = residual_channels(stage)
        r, s, o, tot = conv_l1(c, 7, resident_depth=2, has_add=False)
        print(f"  stage {stage}: c={c:4d}  resident={r:7d} stream={s:6d} out={o:4d} total={tot:7d} "
              f"({tot / L1_BUDGET * 100:6.1f}%)  {'FITS' if tot <= L1_BUDGET else 'OVER'}")

    print("\n=== RESIDUAL UNIT, CHOSEN policy (ci_chunk from RESIDUAL_CI_CHUNK, resident_depth=1) ===")
    for stage in (1, 2, 3, 4):
        c = residual_channels(stage)
        chunk = RESIDUAL_CI_CHUNK[stage] or c
        dil = conv_l1(chunk, 7, resident_depth=1, has_add=False)[-1]
        onexone = conv_l1(chunk, 1, resident_depth=1, has_add=True)[-1]
        nchunks = -(-c // chunk)
        print(f"  stage {stage}: ci_chunk={RESIDUAL_CI_CHUNK[stage]}  chunks={nchunks}  "
              f"dilated={dil:7d} ({dil / L1_BUDGET * 100:5.1f}%)  "
              f"1x1={onexone:7d} ({onexone / L1_BUDGET * 100:5.1f}%)")

    c_in, c_out, k = head_shape()
    hp = head_plan()
    total = conv_l1(hp["ci_chunk"], k, resident_depth=1)[-1]
    print(f"\n=== HEAD conv: c_in={c_in} c_out={c_out} k={k} ===")
    print(f"  unchunked, rd=2 (driver default): {conv_l1(c_in, k, resident_depth=2)[-1]} "
          f"({conv_l1(c_in, k, resident_depth=2)[-1] / L1_BUDGET * 100:.1f}%) OVER")
    print(f"  ci_chunk={hp['ci_chunk']} rd=1: {total} ({total / L1_BUDGET * 100:.1f}%) FITS  "
          f"chunks={-(-c_in // hp['ci_chunk'])}")

    c_in, c_out, k = tail_shape()
    tot = conv_l1(c_in, k, resident_depth=2)[-1]
    print(f"\n=== TAIL conv: c_in={c_in} c_out={c_out} k={k} ===")
    print(f"  unchunked, rd=2 (driver default, unchanged): {tot} ({tot / L1_BUDGET * 100:.1f}%)  "
          f"{'FITS' if tot <= L1_BUDGET else 'OVER'} -- same shape as stage 4, no new lever needed")

    print("\n(sanity) stage 4 numbers above must match the already-device-green driver defaults --")
    print("  resident_depth=2, unchunked, dilated conv ~84.0%, 1x1 ~77.7%, upsample ~47.3%.")


if __name__ == "__main__":
    _report()
