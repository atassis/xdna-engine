#!/usr/bin/env python3
"""bge-base parity gate: NPU embeddings vs the HF f32 reference (mean-pooled, L2-normalized).

The engine serves bge-base with `pooling = "mean"`, so this reproduces exactly that -- mean over the
attention-masked tokens of `last_hidden_state`, then L2 normalize -- and compares per sentence.

Gate on cosine + rel-L2 against the reference, NOT on a downstream task metric: this is a numeric
parity check of one encoder, and the bf16 encoder's error floor is what it is measuring.

Usage (server must already be serving bge on --port):
    .venv-export/bin/python scripts/verify_bge_parity.py --port 11445 --model bge
    .venv-export/bin/python scripts/verify_bge_parity.py --port 11445 --min-cos 0.99
"""
import argparse
import json
import sys
import urllib.request

import numpy as np

SENTENCES = [
    "The capital of France is Paris.",
    "A cat sat on the mat.",
    "Transformer encoders map text to a dense vector.",
    "Le chat est assis sur le tapis.",
]


def npu_embeddings(port: int, model: str | None, sents: list[str]) -> np.ndarray:
    """One request per sentence, so a failure names the sentence that caused it."""
    out = []
    for s in sents:
        body = {"input": s}
        if model:
            body["model"] = model
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/embeddings",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = json.loads(r.read())
        if "error" in payload:
            sys.exit(f"engine error on {s!r}: {payload['error']}")
        out.append(np.asarray(payload["data"][0]["embedding"], dtype=np.float32))
    return np.stack(out)


def hf_reference(sents: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "BAAI/bge-base-en-v1.5"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    out = []
    for s in sents:
        enc = tok(s, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[0]              # [T, H]
        mask = enc["attention_mask"][0].unsqueeze(-1).float()  # [T, 1]
        pooled = (h * mask).sum(0) / mask.sum(0).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=0).numpy())
    return np.stack(out).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--model", default=None, help="model name to request (omit to use the default)")
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--max-rel-l2", type=float, default=0.15)
    a = ap.parse_args()

    got = npu_embeddings(a.port, a.model, SENTENCES)
    ref = hf_reference(SENTENCES)
    if got.shape != ref.shape:
        sys.exit(f"shape mismatch: engine {got.shape} vs reference {ref.shape}")

    fail = 0
    for s, g, r in zip(SENTENCES, got, ref):
        cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r)))
        rel = float(np.linalg.norm(g - r) / np.linalg.norm(r))
        norm = float(np.linalg.norm(g))
        bad = cos < a.min_cos or rel > a.max_rel_l2
        fail += bad
        print(f"[{'FAIL' if bad else 'ok  '}] cos={cos:.5f}  rel-L2={rel:.5f}  |v|={norm:.4f}  {s!r}")

    # A sanity check the reference cannot give us: unrelated sentences must stay farther apart than
    # a translated pair. Catches a pipeline that returns a constant or a shuffled vector.
    near = float(got[1] @ got[3])
    far = float(got[0] @ got[1])
    print(f"cos(en/fr same sentence)={near:.4f} vs cos(unrelated)={far:.4f}")
    if near <= far:
        print("FAIL: translated pair is not closer than an unrelated pair")
        fail += 1

    print("PARITY GREEN" if not fail else f"PARITY RED ({fail} failures)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
