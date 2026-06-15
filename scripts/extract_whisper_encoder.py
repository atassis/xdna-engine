#!/usr/bin/env python3
"""Export openai/whisper-small encoder weights + golden activations to npy.

Carries biases (Whisper's k_proj has no bias -> stored as zeros).
Linear weights are TRANSPOSED to [K_in, N_out] so Rust `x @ W` (ndarray .dot)
is correct.

Verified against transformers 5.12 / torch 2.12: encoder submodule attribute
paths are unchanged from 4.x (model.model.encoder, .layers[i].self_attn.{q,k,v,out}_proj,
.fc1/.fc2, .self_attn_layer_norm, .final_layer_norm, enc.conv1/conv2/embed_positions/layer_norm).
"""
import os, json, numpy as np, torch
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-small")
OUT = Path(os.environ.get("WHISPER_OUT", "artifacts/whisper-small"))
(OUT / "refs").mkdir(parents=True, exist_ok=True)
(OUT / "conv").mkdir(exist_ok=True)

torch.manual_seed(0)
m = WhisperForConditionalGeneration.from_pretrained(MODEL).eval()
enc = m.model.encoder
cfg = m.config

json.dump({"d_model": cfg.d_model, "n_layers": cfg.encoder_layers,
           "n_heads": cfg.encoder_attention_heads, "ffn": cfg.encoder_ffn_dim,
           "n_mels": cfg.num_mel_bins, "max_src": cfg.max_source_positions},
          open(OUT / "config.json", "w"), indent=2)


def save(p, t):
    np.save(OUT / p, t.detach().cpu().numpy().astype(np.float32))


# conv stem + learned positional embedding
save("conv/conv1.weight.npy", enc.conv1.weight)
save("conv/conv1.bias.npy", enc.conv1.bias)
save("conv/conv2.weight.npy", enc.conv2.weight)
save("conv/conv2.bias.npy", enc.conv2.bias)
save("conv/embed_positions.npy", enc.embed_positions.weight)

# per-layer linear weights (TRANSPOSED to [K_in, N_out]) + biases + layernorms
for i, blk in enumerate(enc.layers):
    d = OUT / f"L{i}"
    d.mkdir(exist_ok=True)
    for name, lin in [("q", blk.self_attn.q_proj), ("k", blk.self_attn.k_proj),
                      ("v", blk.self_attn.v_proj), ("out", blk.self_attn.out_proj),
                      ("fc1", blk.fc1), ("fc2", blk.fc2)]:
        np.save(d / f"{name}.weight.npy",
                lin.weight.detach().T.contiguous().numpy().astype(np.float32))
        b = lin.bias
        np.save(d / f"{name}.bias.npy",
                (b.detach().numpy() if b is not None
                 else np.zeros(lin.weight.shape[0])).astype(np.float32))
    for name, ln in [("ln1", blk.self_attn_layer_norm), ("ln2", blk.final_layer_norm)]:
        np.save(d / f"{name}.weight.npy", ln.weight.detach().numpy().astype(np.float32))
        np.save(d / f"{name}.bias.npy", ln.bias.detach().numpy().astype(np.float32))

# final encoder LayerNorm (post-stack)
save("refs/ln_post.weight.npy", enc.layer_norm.weight)
save("refs/ln_post.bias.npy", enc.layer_norm.bias)

# golden activations on a fixed clip
proc = WhisperProcessor.from_pretrained(MODEL)
import soundfile as sf
wav, sr = sf.read("artifacts/wer_clips/en_01.wav")
feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
np.save(OUT / "refs/input_features.npy", feats.numpy().astype(np.float32))

with torch.no_grad():
    acts = {}

    def mk(n):
        def h(mod, i, o):
            acts[n] = (o[0] if isinstance(o, tuple) else o)
        return h

    # after_conv = INPUT to encoder block 0, i.e. AFTER conv2 + GELU + permute
    # + positional-embedding add. We capture it via a forward_pre_hook on
    # layers[0] (its input), NOT a hook on conv2 (which would be pre-pos-embed,
    # pre-transpose). This makes after_conv.npy a clean conv-stem+pos golden
    # that the Rust conv-stem gate can compare against directly.
    def pre0(mod, args, kwargs=None):
        acts["after_conv"] = args[0]
    enc.layers[0].register_forward_pre_hook(pre0)

    for i, blk in enumerate(enc.layers):
        blk.register_forward_hook(mk(f"block_{i}"))
    out = enc(feats).last_hidden_state

# after_conv: block-0 input [1,T',d] -> [T',d]
np.save(OUT / "refs/after_conv.npy",
        acts["after_conv"].squeeze(0).numpy().astype(np.float32))
for i in range(cfg.encoder_layers):
    np.save(OUT / f"refs/block_{i}.npy",
            acts[f"block_{i}"].squeeze(0).numpy().astype(np.float32))
np.save(OUT / "refs/encoded.npy", out.squeeze(0).numpy().astype(np.float32))

print("wrote", OUT, "T'=", out.shape[1], "d_model=", cfg.d_model)
print("after_conv shape:", tuple(acts["after_conv"].squeeze(0).shape))
