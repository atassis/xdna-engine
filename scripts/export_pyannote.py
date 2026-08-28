#!/usr/bin/env python3
"""Export pyannote/speaker-diarization-3.1 into ONNX graphs + a provenance manifest.

Run with .venv-pyannote (scripts/setup_pyannote_venv.sh), NOT .venv-export -- pyannote.audio does
not work against the torch/torchaudio that .venv-export pins.

Two graphs, both taking raw 16 kHz mono waveform:
  segmentation.onnx  (waveform)          -> powerset logits [B, frames, 7]
  embedding.onnx     (waveform, weights) -> speaker embeddings [B, 256]

The embedder CANNOT be exported through WeSpeakerResNet34.forward: it computes features with
torch.vmap(torchaudio.compliance.kaldi.fbank), and both vmap and kaldi.fbank are untraceable
(pyannote-audio Discussion #1929). So we wrap `.resnet` with a traceable torch-op fbank and gate
that fbank against kaldi.fbank right here -- if this assertion fails the export is wrong and the
Rust side would never know.

The `weights` input is NOT optional: the shipped pipeline runs embedding_exclude_overlap=true and
pools WEIGHTED statistics over a speaker's non-overlapping frames.

Run: .venv-export/bin/python scripts/export_pyannote.py
Needs HF_TOKEN (pyannote/segmentation-3.0 is gated).
"""
import json, os, sys, urllib.request
import torch, torch.nn as nn, torchaudio
import pyannote.audio
import torchaudio.compliance.kaldi as kaldi
import yaml
from pyannote.audio import Model

OUT = "artifacts/pyannote"
os.makedirs(OUT, exist_ok=True)
SR = 16000
# Read from the installed package, never hardcoded: the manifest claims this as the
# provenance for values that live in NO config.yaml, so a stale literal here would
# attach real numbers to the wrong revision.
PYANNOTE_REV = pyannote.audio.__version__
# Ungated GitHub mirror of the gated HF pipeline repo: the hyperparameters need no token.
PIPELINE_CFG = "https://raw.githubusercontent.com/pyannote/hf-speaker-diarization-3.1/main/config.yaml"
EMBED_CFG = "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/config.yaml"

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN is required: pyannote/segmentation-3.0 is a gated repo. Accept its "
             "conditions on huggingface.co, then export HF_TOKEN=hf_...")

def fetch_yaml(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return yaml.safe_load(r.read())

pipe_cfg = fetch_yaml(PIPELINE_CFG)
emb_cfg = fetch_yaml(EMBED_CFG)   # fetched for provenance/diffing only; hparams win
clus = pipe_cfg["params"]["clustering"]
seg_params = pipe_cfg["params"]["segmentation"]
pipe_params = pipe_cfg["pipeline"]["params"]

# ---- segmentation: waveform in, exports cleanly ------------------------------------------------
seg = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token).eval()
dur = float(seg.specifications.duration)
n_samples = int(dur * SR)
x = torch.randn(1, 1, n_samples)
with torch.no_grad():
    seg_out = seg(x)
n_frames, n_classes = int(seg_out.shape[1]), int(seg_out.shape[2])
torch.onnx.export(seg, x, f"{OUT}/segmentation.onnx", input_names=["waveform"],
                  output_names=["logits"], opset_version=17,
                  dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}})
print(f"segmentation.onnx: {dur}s -> [{n_frames}, {n_classes}]")
# 589 frames is the derived expectation (SincNet strides); a mismatch means the checkpoint moved.
n_speakers = len(seg.specifications.classes)
max_per_frame = int(seg.specifications.powerset_max_classes)
# The Rust POWERSET_3 table hardcodes the 3-speaker / max-2-per-frame layout. Assert the checkpoint
# still matches it here, where the model is in hand -- otherwise the decode silently maps classes to
# the wrong speaker sets and every downstream number is quietly wrong.
from math import comb
expect = sum(comb(n_speakers, k) for k in range(max_per_frame + 1))
assert (n_speakers, max_per_frame) == (3, 2), (
    f"checkpoint is {n_speakers} speakers / max {max_per_frame} per frame; "
    f"npu-engine's POWERSET_3 table assumes 3/2 and must be regenerated")
assert n_classes == expect == 7, f"expected {expect} powerset classes, got {n_classes}"
print(f"powerset: {n_speakers} speakers, max {max_per_frame}/frame -> {n_classes} classes")

# ---- embedder: traceable fbank + resnet, weights in --------------------------------------------
emb = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=token).eval()
# Read the feature params off the LOADED model, not by parsing config.yaml: the published YAML
# nests them under a `model:` key, and more importantly the checkpoint's own hparams are the real
# provenance -- a config the loader ignored would be the wrong number attached to the right name.
_h = dict(emb.hparams)
NUM_MEL = int(_h["num_mel_bins"]); FLEN = float(_h["frame_length"])
FSHIFT = float(_h["frame_shift"]); WIN = str(_h["window_type"])
USE_ENERGY = bool(_h.get("use_energy", False)); DITHER = float(_h.get("dither", 0.0))
assert int(_h["sample_rate"]) == SR, f"embedder wants {_h['sample_rate']} Hz, pipeline is {SR}"

class TraceableFbank(nn.Module):
    """kaldi-compatible fbank in plain torch ops, so the graph traces.

    Mirrors `torchaudio.compliance.kaldi.fbank`'s reference path rather than approximating it. The
    three details that a from-scratch version gets wrong, all of which the assertion below caught:

      * `remove_dc_offset` subtracts each FRAME's mean, not the signal's.
      * the filterbank is kaldi's own `get_mel_banks`, NOT `melscale_fbanks` -- different triangle
        layout, and it is (n_mels, nfft//2), i.e. the Nyquist bin is DROPPED.
      * the log floor is kaldi's float32 epsilon, applied with max() before the log.

    The banks are baked in as a constant at construction, so the exported graph carries kaldi's
    exact filterbank and needs no DSP on the Rust side.
    """
    def __init__(s, n_mels, frame_len_ms, frame_shift_ms, sr, window_type):
        super().__init__()
        s.wlen = int(sr * frame_len_ms / 1000.0)
        s.wshift = int(sr * frame_shift_ms / 1000.0)
        s.nfft = 1
        while s.nfft < s.wlen:
            s.nfft *= 2                                  # round_to_power_of_two=True
        if window_type == "povey":
            w = torch.hamming_window(s.wlen, periodic=False).pow(0.85)
        elif window_type == "hanning":
            w = torch.hann_window(s.wlen, periodic=False)
        elif window_type == "rectangular":
            w = torch.ones(s.wlen)
        else:
            w = torch.hamming_window(s.wlen, periodic=False)
        s.register_buffer("win", w)
        # kaldi fbank defaults: low_freq=20, high_freq=0 (-> nyquist), vtln 100/-500, warp 1.0
        banks, _ = kaldi.get_mel_banks(n_mels, s.nfft, float(sr), 20.0, 0.0, 100.0, -500.0, 1.0)
        s.register_buffer("fb", banks.T.contiguous())     # (nfft//2, n_mels)
        s.eps = 1.1920928955078125e-07                    # kaldi's float32 epsilon
        # The real DFT as a baked matmul, NOT torch.fft.rfft: aten::fft_rfft has no ONNX lowering
        # (opset 17), so an rfft here exports as nothing at all. nfft is fixed, so the transform is
        # a constant matrix and the result is exact rather than approximate. Only the bins below
        # Nyquist are built, because kaldi drops the last one anyway.
        n_bin = s.nfft // 2
        t = torch.arange(s.nfft, dtype=torch.float64).unsqueeze(1)      # (nfft, 1)
        f = torch.arange(n_bin, dtype=torch.float64).unsqueeze(0)       # (1, n_bin)
        ang = 2.0 * torch.pi * t * f / s.nfft
        s.register_buffer("dft_cos", torch.cos(ang).float().contiguous())
        s.register_buffer("dft_sin", torch.sin(ang).float().contiguous())

    def forward(s, wav):                                  # wav [B, T] in [-1, 1]
        x = wav * 32768.0                                 # kaldi works in int16 units
        frames = x.unfold(1, s.wlen, s.wshift)            # snip_edges=True
        frames = frames - frames.mean(dim=-1, keepdim=True)   # remove_dc_offset, PER FRAME
        pre = torch.cat([frames[..., :1], frames[..., :-1]], dim=-1)
        frames = frames - 0.97 * pre                      # preemphasis, replicate-padded
        frames = frames * s.win
        frames = torch.nn.functional.pad(frames, (0, s.nfft - s.wlen))   # zero-pad to nfft
        re = frames @ s.dft_cos
        im = -(frames @ s.dft_sin)
        spec = re * re + im * im                           # use_power=True; Nyquist already absent
        mel = spec @ s.fb
        return torch.log(torch.clamp(mel, min=s.eps))

fbank_mod = TraceableFbank(NUM_MEL, FLEN, FSHIFT, SR, WIN).eval()

# GATE: our traceable fbank must match kaldi's, or every embedding is subtly wrong and the parity
# probe would blame the ResNet.
probe = torch.randn(1, n_samples) * 0.1
with torch.no_grad():
    ours = fbank_mod(probe)[0]
    theirs = kaldi.fbank(probe * 32768.0, num_mel_bins=NUM_MEL, frame_length=FLEN,
                         frame_shift=FSHIFT, sample_frequency=SR, window_type=WIN,
                         dither=DITHER, use_energy=USE_ENERGY, snip_edges=True)
n = min(ours.shape[0], theirs.shape[0])
rel = ((ours[:n] - theirs[:n]).abs().max() / theirs[:n].abs().max().clamp(min=1e-6)).item()
print(f"fbank max rel diff vs kaldi: {rel:.3e}  (ours {tuple(ours.shape)} kaldi {tuple(theirs.shape)})")
assert rel < 1e-3, (
    f"traceable fbank diverges from kaldi.fbank (rel {rel:.3e}). Do NOT export: every embedding "
    f"would be computed on the wrong features. Fix the constants, never the tolerance.")

class EmbedGraph(nn.Module):
    """waveform + per-frame weights -> 256-d embedding, with the MASK preserved."""
    def __init__(s, fbank, resnet):
        super().__init__(); s.fbank = fbank; s.resnet = resnet
    def forward(s, wav, weights):
        f = s.fbank(wav)
        f = f - f.mean(dim=1, keepdim=True)     # WeSpeaker applies CMN before the ResNet
        return s.resnet(f, weights=weights)[1]

n_emb_frames = int(fbank_mod(probe).shape[1])
graph = EmbedGraph(fbank_mod, emb.resnet).eval()
w0 = torch.ones(1, n_emb_frames)
with torch.no_grad():
    dim = int(graph(probe, w0).shape[-1])
torch.onnx.export(graph, (probe, w0), f"{OUT}/embedding.onnx",
                  input_names=["waveform", "weights"], output_names=["embedding"],
                  opset_version=17,
                  dynamic_axes={"waveform": {0: "batch"}, "weights": {0: "batch"},
                                "embedding": {0: "batch"}})
print(f"embedding.onnx: -> [{dim}], {n_emb_frames} fbank frames")
assert dim == 256, f"expected 256-d embeddings, got {dim}"

# ---- manifest: every value WITH its source ------------------------------------------------------
manifest = {
    "pyannote_audio_rev": PYANNOTE_REV,
    "sample_rate": SR,
    "segmentation": {
        "onnx": "segmentation.onnx", "duration_s": dur,
        # 0.1 x duration is a SpeakerDiarization.__init__ default; it is in no config.yaml.
        "step_s": round(0.1 * dur, 6),
        "max_speakers_per_chunk": n_speakers,
        "max_speakers_per_frame": max_per_frame,
        "powerset_classes": n_classes,
        "n_frames": n_frames,
        "source": "pyannote/segmentation-3.0 (HF, gated); step_s from pyannote-audio "
                  f"{PYANNOTE_REV} SpeakerDiarization.__init__",
    },
    "embedding": {
        "onnx": "embedding.onnx", "dim": dim, "num_mel_bins": NUM_MEL,
        "frame_length_ms": FLEN, "frame_shift_ms": FSHIFT, "n_frames": n_emb_frames,
        "source": "pyannote/wespeaker-voxceleb-resnet34-LM checkpoint hparams (fbank params); "
                  f"dim + TSTP pooling hardcoded in pyannote-audio {PYANNOTE_REV} source",
    },
    "clustering": {
        "method": clus["method"], "threshold": float(clus["threshold"]),
        "min_cluster_size": int(clus["min_cluster_size"]),
        "exclude_overlap": bool(pipe_params["embedding_exclude_overlap"]),
        "source": "pyannote/hf-speaker-diarization-3.1 config.yaml (ungated GitHub mirror)",
    },
    "min_duration_off_s": float(seg_params["min_duration_off"]),
}
with open(f"{OUT}/diarize.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
