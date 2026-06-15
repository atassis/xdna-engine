#!/usr/bin/env python3
"""Fixture generator for the general-conv2d ResNet-18 existence proof.

Uses transformers `microsoft/resnet-18` (torchvision is not installed and we won't perturb the
shared venv). Exports: resnet18.onnx (ORT oracle, BN baked in), a fixed-seed input, per-conv
BN-FOLDED weights (weight [Cout,Cin,kh,kw] + bias [Cout]) as .npy, and a FLAT op-list manifest so
the Rust forward is unambiguous (no residual-ordering guesswork).

BN fold: w' = w * (gamma/sqrt(var+eps)); b' = beta - mean*(gamma/sqrt(var+eps))  (conv has no bias).
HF ResNet-18 block: layer.0 = conv3x3+bn+ReLU, layer.1 = conv3x3+bn (no act), shortcut =
conv1x1+bn only when downsampling; ReLU applied AFTER the residual add.

Run: <repo>/.venv/bin/python scripts/export_resnet.py   (needs torch + transformers)
"""
import os, json, numpy as np, torch, torch.nn as nn
from transformers import ResNetForImageClassification

OUT = "artifacts/resnet18"; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
m = ResNetForImageClassification.from_pretrained("microsoft/resnet-18").eval()

class Wrap(nn.Module):
    def __init__(s, mm): super().__init__(); s.mm = mm
    def forward(s, x): return s.mm(x).logits

x = torch.randn(1, 3, 224, 224)
torch.onnx.export(Wrap(m), x, f"{OUT}/resnet18.onnx", input_names=["input"],
                  output_names=["logits"], opset_version=17, dynamo=False)
np.save(f"{OUT}/input.npy", x[0].detach().numpy().astype(np.float32))

def save(name, arr):
    np.save(f"{OUT}/{name}.npy", np.ascontiguousarray(arr.detach().numpy().astype(np.float32)))

def fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    wf = conv.weight * s.reshape(-1, 1, 1, 1)
    bf = bn.bias - bn.running_mean * s
    return wf, bf

manifest = []
# stem (embedder conv 7x7 s2 p3 + bn + relu, then maxpool 3x3 s2 p1)
emb = m.resnet.embedder.embedder
wf, bf = fold(emb.convolution, emb.normalization)
save("stem_w", wf); save("stem_b", bf)
manifest.append({"op": "conv", "name": "stem", "kh": 7, "kw": 7, "stride": 2, "pad": 3, "relu": True})
manifest.append({"op": "maxpool"})

for S, stage in enumerate(m.resnet.encoder.stages):
    for L, layer in enumerate(stage.layers):
        manifest.append({"op": "block_start"})
        c0, c1 = layer.layer[0], layer.layer[1]
        st = int(c0.convolution.stride[0])
        wf, bf = fold(c0.convolution, c0.normalization)
        save(f"s{S}l{L}c0_w", wf); save(f"s{S}l{L}c0_b", bf)
        manifest.append({"op": "conv", "name": f"s{S}l{L}c0", "kh": 3, "kw": 3, "stride": st, "pad": 1, "relu": True})
        wf, bf = fold(c1.convolution, c1.normalization)
        save(f"s{S}l{L}c1_w", wf); save(f"s{S}l{L}c1_b", bf)
        manifest.append({"op": "conv", "name": f"s{S}l{L}c1", "kh": 3, "kw": 3, "stride": 1, "pad": 1, "relu": False})
        sc = getattr(layer, "shortcut", None)
        if isinstance(getattr(sc, "convolution", None), nn.Conv2d):
            wf, bf = fold(sc.convolution, sc.normalization)
            save(f"s{S}l{L}sc_w", wf); save(f"s{S}l{L}sc_b", bf)
            manifest.append({"op": "downsample", "name": f"s{S}l{L}sc", "kh": 1, "kw": 1,
                             "stride": int(sc.convolution.stride[0]), "pad": 0})
        manifest.append({"op": "residual_relu"})

manifest.append({"op": "globalavgpool"})
fc = m.classifier[1]  # Sequential[0]=Flatten, [1]=Linear(512,1000)
save("fc_w", fc.weight); save("fc_b", fc.bias)
manifest.append({"op": "fc", "name": "fc"})
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print(f"exported {len(manifest)} ops + onnx + input -> {OUT}")
