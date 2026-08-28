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

PIPELINE = os.environ.get("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1")
SLUG = PIPELINE.split("/")[-1]
OUT = f"artifacts/pyannote/{SLUG}"
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

# `use_auth_token` was removed from huggingface_hub; pyannote 4.x takes `token`.
import inspect as _inspect
TOKEN_KW = ({"token": token} if "token" in _inspect.signature(Model.from_pretrained).parameters
            else {"use_auth_token": token})

def fetch_yaml(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return yaml.safe_load(r.read())

def pipeline_config():
    """The pipeline's own config.yaml. For 3.1 an ungated GitHub mirror serves it without a token;
    for anything else read the repo directly."""
    if SLUG == "speaker-diarization-3.1":
        return fetch_yaml(PIPELINE_CFG)
    import urllib.request
    url = f"https://huggingface.co/{PIPELINE}/resolve/main/config.yaml"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return yaml.safe_load(r.read())

pipe_cfg = pipeline_config()
emb_cfg = fetch_yaml(EMBED_CFG)   # fetched for provenance/diffing only; hparams win
clus = pipe_cfg["params"]["clustering"]
seg_params = pipe_cfg["params"]["segmentation"]
pipe_params = pipe_cfg["pipeline"]["params"]

# ---- segmentation: waveform in, exports cleanly ------------------------------------------------
# community-1 bundles its segmentation/embedding under subfolders of the pipeline repo; 3.1 names
# two separate repos. Same ARCHITECTURES either way (PyanNet + WeSpeakerResNet34), so everything
# below -- the traceable fbank, the weights input, the powerset assertion -- is shared.
BUNDLED = "community" in SLUG or "precision" in SLUG
def _load(sub, repo):
    if BUNDLED:
        return Model.from_pretrained(PIPELINE, subfolder=sub, **TOKEN_KW).eval()
    return Model.from_pretrained(repo, **TOKEN_KW).eval()

seg = _load("segmentation", "pyannote/segmentation-3.0")
dur = float(seg.specifications.duration)
n_samples = int(dur * SR)
x = torch.randn(1, 1, n_samples)
with torch.no_grad():
    seg_out = seg(x)
n_frames, n_classes = int(seg_out.shape[1]), int(seg_out.shape[2])
torch.onnx.export(seg, x, f"{OUT}/segmentation.onnx", input_names=["waveform"],
                  output_names=["logits"], opset_version=17, dynamo=False,
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
emb = _load("embedding", "pyannote/wespeaker-voxceleb-resnet34-LM")
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
                  opset_version=17, dynamo=False,
                  dynamic_axes={"waveform": {0: "batch"}, "weights": {0: "batch"},
                                "embedding": {0: "batch"}})
print(f"embedding.onnx: -> [{dim}], {n_emb_frames} fbank frames")
assert dim == 256, f"expected 256-d embeddings, got {dim}"


# ---- clustering: agglomerative (3.1) or VBx+PLDA (community-1) ---------------------------------
def clustering_block():
    """The clustering stage's parameters, and for VBx its LEARNED matrices.

    VBx needs a PLDA whose setup solves a generalized symmetric eigenproblem, `eigh(B, W)`. That is
    a ONE-TIME transform of two fixed files, so it is solved HERE and the results are shipped as
    plain .npy. The Rust side then needs only dense matmul -- no eigensolver, no new dependency --
    and the derived constants carry their provenance like every other number in this manifest.
    """
    method = str(pipe_cfg["pipeline"]["params"].get("clustering", "AgglomerativeClustering"))
    src = f"{PIPELINE} config.yaml"
    if "VBx" not in method:
        return {
            "method": "centroid", "threshold": float(clus["threshold"]),
            "min_cluster_size": int(clus["min_cluster_size"]),
            "exclude_overlap": bool(pipe_params["embedding_exclude_overlap"]),
            "source": src if SLUG != "speaker-diarization-3.1"
                      else "pyannote/hf-speaker-diarization-3.1 config.yaml (ungated GitHub mirror)",
        }

    import numpy as np
    from scipy.linalg import eigh
    from huggingface_hub import hf_hub_download
    tf_p = hf_hub_download(PIPELINE, "xvec_transform.npz", subfolder="plda", token=token)
    pl_p = hf_hub_download(PIPELINE, "plda.npz", subfolder="plda", token=token)
    x, pl = np.load(tf_p), np.load(pl_p)
    mean1, mean2, lda = x["mean1"], x["mean2"], x["lda"]
    mu, tr, psi = pl["mu"], pl["tr"], pl["psi"]

    # vbx_setup: whiten via the generalized eigenproblem, then reorder descending.
    W = np.linalg.inv(tr.T.dot(tr))
    B = np.linalg.inv((tr.T / psi).dot(tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1].copy()
    plda_tr = wccn.T[::-1].copy()

    d = int(lda.shape[1])
    np.save(f"{OUT}/xvec_mean1.npy", mean1.astype(np.float32))
    np.save(f"{OUT}/xvec_lda.npy", lda.astype(np.float32))
    np.save(f"{OUT}/xvec_mean2.npy", mean2.astype(np.float32))
    np.save(f"{OUT}/plda_mu.npy", mu.astype(np.float32))
    np.save(f"{OUT}/plda_tr.npy", plda_tr.astype(np.float32))
    np.save(f"{OUT}/plda_psi.npy", plda_psi[:d].astype(np.float32))
    print(f"plda: lda {lda.shape} tr {plda_tr.shape} psi[:{d}] -- eigenproblem solved at export")

    return {
        "method": "vbx",
        "threshold": float(clus["threshold"]),
        "fa": float(clus["Fa"]), "fb": float(clus["Fb"]),
        "max_iters": 20, "init_smoothing": 7.0, "lda_dim": d,
        "exclude_overlap": bool(pipe_params["embedding_exclude_overlap"]),
        "plda": {"xvec_mean1": "xvec_mean1.npy", "xvec_lda": "xvec_lda.npy",
                 "xvec_mean2": "xvec_mean2.npy", "plda_mu": "plda_mu.npy",
                 "plda_tr": "plda_tr.npy", "plda_psi": "plda_psi.npy"},
        # max_iters and init_smoothing are cluster_vbx() defaults in the library, not config keys.
        "source": f"{src}; max_iters/init_smoothing from pyannote-audio {PYANNOTE_REV} cluster_vbx()",
    }

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
        # Name where the checkpoint ACTUALLY came from: community-1 bundles its own segmentation
        # under a subfolder and does not use segmentation-3.0 at all.
        "source": (f"{PIPELINE} subfolder 'segmentation'" if BUNDLED
                   else "pyannote/segmentation-3.0 (HF, gated)")
                  + f"; step_s from pyannote-audio {PYANNOTE_REV} SpeakerDiarization.__init__",
    },
    "embedding": {
        "onnx": "embedding.onnx", "dim": dim, "num_mel_bins": NUM_MEL,
        "frame_length_ms": FLEN, "frame_shift_ms": FSHIFT, "n_frames": n_emb_frames,
        "source": (f"{PIPELINE} subfolder 'embedding'" if BUNDLED
                   else "pyannote/wespeaker-voxceleb-resnet34-LM")
                  + f" checkpoint hparams (fbank params); dim + TSTP pooling hardcoded in "
                    f"pyannote-audio {PYANNOTE_REV} source",
    },
    "clustering": clustering_block(),
    "min_duration_off_s": float(seg_params["min_duration_off"]),
}
with open(f"{OUT}/diarize.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
