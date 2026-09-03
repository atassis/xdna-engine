#!/usr/bin/env python3
"""Build a FOUR-speaker conversation, for the questions two speakers cannot answer.

`make_conversation_fixture.py` gives two speakers over ~45 s, which is enough to exercise
cross-window linking but not enough to price anything that scales with the SPEAKER COUNT. The
motivating case: raising `segmentation.step_s` cuts crops roughly linearly, and 3.1 reassigns any
cluster below `min_cluster_size` (12) into its nearest large one -- so the crop budget has to be
shared among more speakers before that floor bites, and a two-speaker clip cannot see it.

Speaker identity here is MEASURED, not assumed from filenames. Every wer_clip was embedded with the
shipped WeSpeaker graph and the pairwise cosine similarity computed; the groups below are what came
out (2026-08-28):

    {ru_01, ru_08, ru_09}   0.84-0.89 mutual  -> one voice
    {ru_06, ru_10, ru_11}   0.68-0.72 mutual  -> one voice
    en_01, en_02                              -> singletons, <=0.23 to anything

Those numbers are from the UNTRIMMED clips and are not what the guard sees. Trimming each clip to
its measured speech RAISES similarity, sometimes a lot -- en_01 vs en_02 goes 0.23 -> 0.261, en_03 vs
ru_08 goes 0.15 -> 0.349 -- because a short utterance embeds less distinctly. Two candidate line-ups
were rejected by the guard before this one, which is why the script re-measures on the trimmed audio
every run instead of trusting any table.

Two of the four speakers get several DIFFERENT utterances each, so recognising them across turns is
a real linking problem rather than the same waveform replayed.

The similarity guard is a FLOOR, not a proof. The real question is whether the shipped pipeline
recovers four speakers on this clip at the default hop; run `sweep_diarize_step` on it and check the
speaker count before trusting any sweep done with it.

The script re-measures those similarities every run and refuses to build a fixture whose speakers
are too close, with MARGIN rather than at the boundary -- speakers sitting exactly on the merge
threshold make the clustering answer a coin flip, so the fixture would measure itself instead of
whatever change is being scored against it. A fixture that quietly contains three speakers instead
of four is worse than no fixture.

Run: python3 scripts/make_conversation_4spk.py
"""
import json, os, sys, wave
import numpy as np
import onnxruntime as ort

OUT = "artifacts/pyannote/fixtures"
MODEL = "artifacts/pyannote/speaker-diarization-3.1"
SR = 16000
# Cosine SIMILARITY above which centroid linkage would merge two clusters, from 3.1's config
# threshold, which is a cosine DISTANCE: sim = 1 - 0.7045654963945799.
MERGE_SIM = 1.0 - 0.7045654963945799
# Required headroom below that boundary. Not a tuned number: 1.00 would accept a pair whose
# clustering outcome is a coin flip, which is the failure this guard exists to prevent.
MARGIN = 0.85

# Chosen by exhaustive search over voice groups against the TRIMMED similarity matrix, maximising
# total speech subject to every cross-speaker pair clearing the limit below. This is the best the
# wer_clips corpus admits: worst pair 0.245 against a 0.295 merge boundary. It is not comfortable,
# and that thinness is a fact about the corpus, not a tuning choice -- these are read-speech clips
# and several of them are the same voice.
SPEAKERS = {
    "A": ["en_02"],
    "B": ["ru_01", "ru_08", "ru_09"],
    "C": ["ru_12", "ru_07"],
    "D": ["ru_13"],
}

def read(stem):
    p = f"artifacts/wer_clips/{stem}.wav"
    with wave.open(p) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2), p
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

def sessions():
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() // 2 or 1
    seg = ort.InferenceSession(f"{MODEL}/segmentation.onnx", so, providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(f"{MODEL}/embedding.onnx", so, providers=["CPUExecutionProvider"])
    return seg, emb

def speech_bounds(seg, x):
    """First and last speaking frame, in seconds, from the segmentation model itself.

    The sibling fixture scripts take these bounds from a `npu diarize` run and paste the numbers in.
    Asking the graph directly keeps the fixture reproducible from the repo alone, and the bounds
    matter: both clips carry leading and trailing silence, and aligning on FILE extents is what made
    an earlier overlap fixture contain 0.03 s of overlapping speech instead of 3.
    """
    v = np.zeros(160000, dtype=np.float32)
    n = min(len(x), 160000)
    v[:n] = x[:n] / 32768.0
    logits = seg.run(None, {"waveform": v[None, None, :]})[0][0]      # [frames, 7]
    speech = logits.argmax(-1) != 0                                    # class 0 = non-speech
    frame_s = 10.0 / len(speech)
    on = np.flatnonzero(speech)
    if not len(on):
        raise SystemExit(f"a source clip has no speech at all")
    return on[0] * frame_s, min((on[-1] + 1) * frame_s, n / SR)

def embed(emb, x):
    v = np.zeros(160000, dtype=np.float32)
    n = min(len(x), 160000)
    v[:n] = x[:n] / 32768.0
    nf = (n - 400) // 160 + 1
    w = np.zeros((1, 998), dtype=np.float32)
    w[0, :min(nf, 998)] = 1.0
    e = emb.run(["embedding"], {"waveform": v[None, :], "weights": w})[0][0]
    return e / np.linalg.norm(e)

def main():
    os.makedirs(OUT, exist_ok=True)
    seg, emb = sessions()

    # --- gather sources, trimmed to their measured speech ------------------------------------
    clips = {}
    for spk, stems in SPEAKERS.items():
        for stem in stems:
            raw = read(stem)
            lo, hi = speech_bounds(seg, raw)
            clips[stem] = (spk, raw[int(lo * SR):int(hi * SR)])
            print(f"  {stem:>6} {spk}  speech {lo:5.2f}-{hi:5.2f}s")

    # --- check the speakers really are four ---------------------------------------------------
    vecs = {stem: embed(emb, seg_) for stem, (_, seg_) in clips.items()}
    worst = (0.0, None, None)
    for a in clips:
        for b in clips:
            if a >= b or clips[a][0] == clips[b][0]:
                continue
            sim = float(vecs[a] @ vecs[b])
            if sim > worst[0]:
                worst = (sim, a, b)
    limit = MARGIN * MERGE_SIM
    print(f"  worst cross-speaker similarity {worst[0]:.3f} ({worst[1]} vs {worst[2]}), "
          f"limit {limit:.3f} = {MARGIN} x merge boundary {MERGE_SIM:.3f}")
    if worst[0] >= limit:
        sys.exit(f"REFUSING: {worst[1]} and {worst[2]} are different speakers by construction but "
                 f"cos {worst[0]:.3f} >= {limit:.3f}, too close to centroid linkage's merge "
                 f"boundary {MERGE_SIM:.3f}. This fixture would be measuring its own speaker "
                 f"separation, not whatever is scored against it.")

    # Match loudness so no speaker is buried under another in the overlaps.
    ref = float(np.sqrt((clips["en_02"][1] ** 2).mean()))
    for k, (spk, x) in clips.items():
        clips[k] = (spk, x * (ref / max(float(np.sqrt((x ** 2).mean())), 1e-6)))

    # --- lay out the conversation --------------------------------------------------------------
    # Round-robin so every speaker recurs across many windows, with three deliberate overlaps.
    order = ["C", "A", "D", "B", "C", "D", "A", "C", "B", "D", "A", "C", "B", "D"]
    used = {s: 0 for s in SPEAKERS}
    t = 0.5
    placed = []
    for i, spk in enumerate(order):
        stems = SPEAKERS[spk]
        stem = stems[used[spk] % len(stems)]
        used[spk] += 1
        x = clips[stem][1]
        dur = len(x) / SR
        placed.append((spk, stem, t, dur, x))
        # Every third turn starts before the previous one ends, giving real overlapping speech.
        t += dur - (0.6 if i % 3 == 2 else -0.7)

    total = int((placed[-1][2] + placed[-1][3] + 1.0) * SR)
    mix = np.zeros(total, dtype=np.float32)
    truth = []
    for spk, stem, t0, dur, x in placed:
        i = int(t0 * SR)
        mix[i:i + len(x)] += x
        truth.append({"speaker": spk, "source": stem,
                      "start_s": round(t0, 3), "end_s": round(t0 + dur, 3)})

    peak = np.abs(mix).max()
    if peak > 32767.0:
        mix *= 32767.0 / peak

    wav_path = f"{OUT}/conversation_4spk.wav"
    with wave.open(wav_path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(mix.astype(np.int16).tobytes())

    meta = {"sample_rate": SR, "duration_s": round(total / SR, 3),
            "expect_speakers": len(SPEAKERS),
            "sources": {s: v for s, v in SPEAKERS.items()},
            "worst_cross_speaker_cosine": round(worst[0], 3),
            "merge_boundary_cosine": round(MERGE_SIM, 4),
            "turns": truth}
    with open(f"{OUT}/conversation_4spk.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{wav_path}  {meta['duration_s']}s, {len(truth)} turns, {len(SPEAKERS)} speakers")
    for t_ in truth:
        print(f"  {t_['speaker']} {t_['source']:>6} {t_['start_s']:>6.2f} - {t_['end_s']:>6.2f}")

main()
