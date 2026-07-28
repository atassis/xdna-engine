#!/usr/bin/env python3
"""Measure |mean|/std per row on the REAL pre-norm activations.

ln_cheap_verdict.md's reason 2 is the only live objection left to fusing the LN into fc1's A
consumption, and the verdict itself calls it UNMEASURED: a single-pass prologue must pack RAW x
(pre-centering) to bf16, so a DC-heavy row loses its AC signal. The verdict's own table says the
normalize is fine at |mean|/std <~ 2 and breaks the 1e-2 gate by |mean|/std ~ 4-13.

So the question is entirely: what is |mean|/std on the activations we actually feed the LN?
"""
import glob, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(REPO, "artifacts", "refs", "*.npy")))

print(f"{'tensor':<34}{'rows':>7}{'median':>10}{'p95':>10}{'p99':>10}{'max':>10}   {'>2':>6}{'>4':>6}")
print("-" * 96)
allr = []
for f in files:
    a = np.load(f)
    x = np.squeeze(a)
    if x.ndim != 2 or x.shape[-1] not in (768, 1024, 3072, 4096):
        continue
    x = x.astype(np.float32)
    mu = x.mean(axis=-1)
    sd = x.std(axis=-1)
    r = np.abs(mu) / np.maximum(sd, 1e-12)
    allr.append(r)
    print(f"{os.path.basename(f):<34}{len(r):>7}{np.median(r):>10.3f}"
          f"{np.percentile(r,95):>10.3f}{np.percentile(r,99):>10.3f}{r.max():>10.3f}"
          f"{100*(r>2).mean():>5.1f}%{100*(r>4).mean():>5.1f}%")

if not allr:
    print("no 2-D activation tensors found"); sys.exit(1)
r = np.concatenate(allr)
print("-" * 96)
print(f"{'ALL ROWS':<34}{len(r):>7}{np.median(r):>10.3f}"
      f"{np.percentile(r,95):>10.3f}{np.percentile(r,99):>10.3f}{r.max():>10.3f}"
      f"{100*(r>2).mean():>5.1f}%{100*(r>4).mean():>5.1f}%")
print()
print(f"verdict's benign threshold  |mean|/std <~ 2 : {100*(r<=2).mean():.2f}% of rows")
print(f"verdict's break threshold   |mean|/std >~ 4 : {100*(r>4).mean():.2f}% of rows")
