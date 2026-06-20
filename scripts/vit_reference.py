#!/usr/bin/env python3
"""HF vit-base reference: fixed random image -> logits [1000] golden. Saves image + logits as .npy."""
import numpy as np, torch
from transformers import AutoModelForImageClassification
m=AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").eval()
rng=np.random.RandomState(0); img=rng.randn(1,3,224,224).astype(np.float32)
with torch.no_grad(): logits=m(pixel_values=torch.from_numpy(img)).logits[0].numpy()
np.save("artifacts/vit-base/ref_img.npy", img[0]); np.save("artifacts/vit-base/ref_logits.npy", logits)
print("argmax:", int(logits.argmax()), "logits[:5]:", logits[:5].round(3).tolist())
print("saved ref_img.npy [3,224,224] + ref_logits.npy [1000]")
