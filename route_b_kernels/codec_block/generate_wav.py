#!/usr/bin/env python3
"""END TO END: codes.bin -> latent -> audio -> a playable .wav, decoder half on the NPU.

    codec_codes.bin [10, T]  --(quantizer, host)-->  latent [1024, 4T]  --(decoder, DEVICE)-->  audio

This is generation, not a gate: unlike every verify_*.py here, it decodes the WHOLE latent stream (or
whatever --limit-frames leaves of it), not a fixed V_FINAL-sized window. That only works because
window_driver's ops already walk arbitrarily long input internally (window_driver.py's own T=64 /
UPSAMPLE_T=16 tiling) -- so decoder_chain.run_chain(z, g) needs nothing from this script beyond `z`
itself, however long it is. What this script DOES have to work out itself is where a windowed chain's
output lands in the TRUE (unwindowed) audio timeline, since window_driver's causal ops drop samples
that would need missing left context rather than zero-padding them the way codec_decoder_ref.decode()
does -- see decoder_chain.chain_offset, and its docstring for the exact formula. That offset is
`--limit-frames`-independent in direction (S is always 0 here: this script always starts from the
beginning of the codes it was given) and is what makes the rel-L2-vs-oracle check below correct
instead of silently comparing misaligned slices.

STEP 1 (codes -> latent) runs on HOST via codec_quantizer_ref.decode() -- the RVQ lookup + 8-layer
transformer + 2x upsample has no on-NPU port yet (--quantizer device raises NotImplementedError
naming the task that will add one, s2-quantizer-postmodule-assembly, rather than silently returning
host results under a device flag). STEP 2 (latent -> audio) is decoder_chain.run_chain, THE port this
task exists to exercise: head -> stage1..4 -> tail -> tanh, chained on device exactly as
verify_whole_decoder.py gates it (same decoder_chain.py functions, same weight cache convention),
just over the whole stream and without that script's iso/chain comparison.

SAMPLE RATE, derived not guessed (a prior agent shipped 16 kHz here and it was caught in review): the
codec's TOTAL upsample from one CODE frame to one audio sample is the decoder's own per-stage strides
(product 512, read from stage_shapes.DECODER_RATES = [8,8,4,2], not hardcoded) times the quantizer's
own downsample factors (product 4, codec_quantizer_ref.DOWNSAMPLE_FACTORS = [2,2]) = 2048x. Fish-audio's
DAC codec was trained at a fixed 44100 Hz output, which is SAMPLE_RATE_HZ below; dividing back through
the 2048x chain gives a code-frame rate of 44100/2048 ~= 21.53 Hz -- the same figure the oracle dump's
own shape file points at ("264 = 66 frames x 4"). 21.53 Hz x 4 x 512 ~= 44093, a few Hz off 44100
because 21.53 is itself already rounded; 44100 (not the back-multiplied 44093) is what this script
writes, since it is the exact, undivided constant the other direction rounds away from.

    python3 generate_wav.py <dump_dir> <out.wav> [--quantizer host|device] [--limit-frames N]

dump_dir needs codec_codes.{bin,shape} (the input) and codec_audio.{bin,shape} (used only for the
rel-L2 report below, never for generation itself -- this script never reads codec_latent.bin, it
recomputes the latent from codes). --limit-frames N caps input to the first N CODE frames (not latent
or audio samples) before the quantizer even runs: the RVQ transformer's attention mask is causal
(codec_quantizer_ref._causal_window_mask, k<=q) and every downstream op is causal too, so decoding a
prefix of the codes reproduces the same leading audio samples a full decode would, one code frame at
a time and cheaply, which is the point -- a first device run should not have to pay for all 66 frames.

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0 (same as verify_whole_decoder.py).
"""
import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import gguf_extract as gx  # noqa: E402
import codec_decoder_ref as R  # noqa: E402
import codec_quantizer_ref as Q  # noqa: E402
import stage_shapes as ss  # noqa: E402
import window_driver as wd  # noqa: E402
import decoder_chain as dc  # noqa: E402

GATE = 3.0e-2
GGUF = ss.GGUF

# Fish-audio's DAC codec's own fixed training-time output rate -- not derived from anything in this
# repo, but cross-checked against it below (module docstring SAMPLE RATE paragraph).
SAMPLE_RATE_HZ = 44100
TOTAL_UPSAMPLE = 1
for s in ss.DECODER_RATES:
    TOTAL_UPSAMPLE *= s
for f in Q.DOWNSAMPLE_FACTORS:
    TOTAL_UPSAMPLE *= f
FRAME_RATE_HZ = SAMPLE_RATE_HZ / TOTAL_UPSAMPLE   # ~21.53 Hz -- reported below, not just asserted


def alignment(cur, truth, V):
    """Same explicit shift-search every verify_*.py gate here uses: a one-sample misalignment reads
    as a small rel-L2 on smooth audio, so this has to be checked rather than trusted from the rel-L2
    alone."""
    return min(range(-2, 3),
               key=lambda s_: np.linalg.norm(cur[:, 2:V - 2] - truth[:, 2 + s_:V - 2 + s_]))


def write_wav(path, samples_f32, sample_rate):
    clipped = np.clip(samples_f32, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("out_wav", type=Path)
    ap.add_argument("--quantizer", choices=["host", "device"], default="host")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="cap input to the first N code frames (of codec_codes.bin's T)")
    args = ap.parse_args()

    if args.quantizer == "device":
        raise NotImplementedError(
            "codes -> latent on-NPU (RVQ lookup + post_module transformer + upsample x2) is not "
            "implemented -- see task s2-quantizer-postmodule-assembly. Use --quantizer host (the "
            "default) until that lands.")

    _cache = {}

    def g(name):
        if name not in _cache:
            _cache[name] = gx.load(GGUF, name).astype(np.float32)
        return _cache[name]

    t_start = time.time()

    codes = Q.load_codes(args.dump_dir)
    n_frames_avail = codes.shape[1]
    if args.limit_frames is not None:
        assert 1 <= args.limit_frames <= n_frames_avail, (
            f"--limit-frames {args.limit_frames} out of range 1..{n_frames_avail} "
            f"({args.dump_dir} has {n_frames_avail} code frames)")
        codes = codes[:, :args.limit_frames]
    n_frames = codes.shape[1]
    print(f"codes {codes.shape} (of {n_frames_avail} available)")

    latent = Q.decode(codes, g, verbose=True)
    print(f"latent {latent.shape}")

    # Fail on a too-short window BEFORE spending any device dispatches -- pure arithmetic, no I/O.
    audio_start, audio_len = dc.chain_offset(0, latent.shape[1])
    print(f"decoder chain: latent window [0, {latent.shape[1]}) -> "
          f"audio [{audio_start}, {audio_start + audio_len})  ({audio_len} samples)")

    t_host_done = time.time()
    wd.reset_stats()
    audio = dc.run_chain(latent, g)
    t_device_done = time.time()

    assert audio.shape == (1, audio_len), (
        f"device output {audio.shape} != predicted (1, {audio_len}) from chain_offset -- "
        "the windowing/left-context arithmetic and the device chain have desynced")
    audio_flat = audio.reshape(-1)

    write_wav(args.out_wav, audio_flat, SAMPLE_RATE_HZ)

    af, ashape = R.load_dump(args.dump_dir, "codec_audio")
    oracle_audio = af.reshape(-1)
    oracle_len = oracle_audio.shape[0]
    # "the overlapping region": --limit-frames can decode a PREFIX of what the dump's own oracle
    # covers, so clip to whatever actually overlaps rather than requiring the full audio_len to fit.
    overlap_len = min(audio_len, max(0, oracle_len - audio_start))
    if overlap_len > 0:
        oracle_slice = oracle_audio[audio_start:audio_start + overlap_len].reshape(1, -1).astype(np.float32)
        device_slice = audio[:, :overlap_len]
        rl2 = Q.rel_l2(device_slice, oracle_slice)
        shift = alignment(device_slice, oracle_slice, overlap_len) if overlap_len > 4 else 0
    else:
        rl2 = None
        shift = None

    stats = wd.stats()
    t_end = time.time()

    lines = []

    def rprint(s=""):
        print(s)
        lines.append(s)

    rprint("=" * 78)
    rprint("S2 TTS: codes -> latent (host) -> audio (NPU, decoder_chain.run_chain)")
    rprint(f"  dump dir              : {args.dump_dir}")
    rprint(f"  quantizer             : {args.quantizer}")
    rprint(f"  code frames decoded   : {n_frames} (of {n_frames_avail} available)")
    rprint(f"  latent shape          : {latent.shape}")
    rprint(f"  total upsample        : {TOTAL_UPSAMPLE}x (decoder x quantizer)  "
           f"-> implies {FRAME_RATE_HZ:.2f} Hz code-frame rate at {SAMPLE_RATE_HZ} Hz audio")
    rprint(f"  audio samples         : {audio_flat.shape[0]}  "
           f"({audio_flat.shape[0] / SAMPLE_RATE_HZ:.3f} s @ {SAMPLE_RATE_HZ} Hz)")
    rprint(f"  dispatch count        : {stats['dispatches']}  "
           f"(useful {stats['useful']}/{stats['computed']} -> {stats['overhead'] * 100:.1f}% recompute)")
    rprint(f"  wall time             : {t_end - t_start:.2f} s total  "
           f"({t_host_done - t_start:.2f} s quantizer/host, {t_device_done - t_host_done:.2f} s NPU decode)")
    if overlap_len > 0:
        gate_pass = rl2 <= GATE and shift == 0
        partial = f" (partial: only {overlap_len} of {audio_len} decoded samples overlap the " \
                  f"oracle's {oracle_len})" if overlap_len < audio_len else ""
        rprint(f"  rel-L2 vs codec_audio.bin[{audio_start}:{audio_start + overlap_len}){partial} : "
               f"{rl2:.6e}  (gate {GATE:.1e})  alignment {shift:+d}  "
               f"{'PASS' if gate_pass else 'FAIL'}")
    else:
        gate_pass = True
        rprint(f"  rel-L2 vs codec_audio.bin : NOT COMPUTED -- decoded region "
               f"[{audio_start}, {audio_start + audio_len}) does not overlap the oracle's "
               f"{oracle_len} samples at all")
    rprint(f"  wrote {args.out_wav}")

    report_path = args.out_wav.with_suffix(".report.txt")
    report_path.write_text("\n".join(lines) + "\n")
    rprint(f"  wrote {report_path}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
