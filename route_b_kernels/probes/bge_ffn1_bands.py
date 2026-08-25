#!/usr/bin/env python3
"""What band does a real encoder actually feed the GELU epilogue, and does the 1757x survive it?

probe_gelu_f32_arms.py scores the GELU ladder on a random-normal accumulator: A ~ N(0,1),
B ~ N(0,1)/sqrt(K), so fc1's pre-activation lands at unit variance. A real BERT fc1 does not --
its A operand is a LayerNorm output and its B operand is a trained weight, and the hardware tanh
LUT is worst near +/-0.5. The one place the model and the device disagreed (sw-tanh alone: ~10x
predicted on encoder bands, 2.18x measured on random-normal) is exactly band-dependent, so the
1757x has to be re-scored on the distribution a default flip would be argued against.

This runs bge-base-en-v1.5 (the engine's own K=768 embedding scenario) forward in numpy from the
extracted artifact weights, on real English prose, and writes each layer's fc1 A-operand in the
shipped K-augmented form -- A[:, :768] = bf16 activations, A[:, 768] = 1.0, B[768, :] = bias, the
same packing scripts/verify_k768_gelu_rail.py uses -- so the device probe consumes it unchanged.

Device-free. Writes <out>/bge_fc1_L{i}.npz, consumed by probe_gelu_f32_arms.py ENC_NPZ=...
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

ART = "artifacts/bge-base"
ENC = f"{ART}/encoder"
HID, DFF, NH, HD, NL = 768, 3072, 12, 64, 12
KRES, KAUG = 768, 800
LN_EPS = np.float32(1e-12)          # rust/npu-engine/src/bert/encoder.rs:15

f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(bfloat16)


# --- WordPiece, the tokenizer.json config: BertNormalizer(lowercase) + BertPreTokenizer ----------

class WordPiece:
    def __init__(self, vocab_txt):
        self.vocab = {}
        with open(vocab_txt, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                self.vocab[line.rstrip("\n")] = i
        self.unk = self.vocab["[UNK]"]

    def _words(self, text):
        # BertNormalizer(clean_text, lowercase) + BertPreTokenizer: split on whitespace, then peel
        # punctuation into its own tokens.
        text = re.sub(r"[\x00�]|\s", " ", text.lower())
        return [w for w in re.findall(r"\w+|[^\w\s]", text, re.UNICODE) if w]

    def encode(self, text):
        ids = []
        for w in self._words(text):
            if len(w) > 100:
                ids.append(self.unk)
                continue
            start, sub = 0, []
            while start < len(w):                      # greedy longest-match-first
                end, cur = len(w), None
                while start < end:
                    piece = ("##" if start else "") + w[start:end]
                    if piece in self.vocab:
                        cur = self.vocab[piece]
                        break
                    end -= 1
                if cur is None:
                    sub = None
                    break
                sub.append(cur)
                start = end
            ids.extend(sub if sub is not None else [self.unk])
        return ids


# --- the encoder, numpy, matching rust/npu-engine/src/bert/{frontend,encoder}.rs -----------------

def layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return f32((x - mu) / np.sqrt(var + LN_EPS) * w + b)


def load(rel):
    return np.load(f"{ENC}/{rel}.npy")


def embed(ids):
    x = load("emb/word_emb")[ids] + load("emb/pos_emb")[:len(ids)] + load("emb/type_emb")[0]
    return layer_norm(f32(x), load("emb/emb_ln_w"), load("emb/emb_ln_b"))


def mha(x, L):
    T = x.shape[0]
    q = x @ load(f"{L}/q_w") + load(f"{L}/q_b")
    k = x @ load(f"{L}/k_w") + load(f"{L}/k_b")
    v = x @ load(f"{L}/v_w") + load(f"{L}/v_b")
    out = np.empty((T, HID), np.float32)
    for h in range(NH):
        s = slice(h * HD, (h + 1) * HD)
        a = (q[:, s] @ k[:, s].T) / np.sqrt(HD)
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        out[:, s] = a @ v[:, s]
    return out @ load(f"{L}/attn_out_w") + load(f"{L}/attn_out_b")


def gelu(x):
    x = np.float64(x)
    return f32(0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3))))


def forward_capture(ids):
    """Runs all 12 layers, returning each layer's fc1 A-operand (the attn_ln output)."""
    x, caps = embed(ids), []
    for i in range(NL):
        L = f"L{i}"
        x = layer_norm(x + mha(x, L), load(f"{L}/attn_ln_w"), load(f"{L}/attn_ln_b"))
        caps.append(x.copy())                          # <-- fc1's input, the band we are after
        h = gelu(x @ load(f"{L}/ffn1_w") + load(f"{L}/ffn1_b"))
        x = layer_norm(x + (h @ load(f"{L}/ffn2_w") + load(f"{L}/ffn2_b")),
                       load(f"{L}/out_ln_w"), load(f"{L}/out_ln_b"))
    return caps


# --- band reporting ------------------------------------------------------------------------------

def bands(pre):
    a = np.abs(pre.ravel())
    return {
        "std": float(pre.std()), "absmax": float(a.max()),
        "frac_lt_0.5": float((a < 0.5).mean()), "frac_lt_1": float((a < 1.0).mean()),
        "frac_lt_2": float((a < 2.0).mean()),
        "q50": float(np.quantile(a, 0.5)), "q90": float(np.quantile(a, 0.9)),
        "q99": float(np.quantile(a, 0.99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad-m", type=int, default=512)
    ap.add_argument("--out", default="artifacts/bge_fc1_bands")
    ap.add_argument("--layers", default="0,6,11", help="layers to dump operands for ('all' ok)")
    args = ap.parse_args()

    if not os.path.isdir(ENC):
        sys.exit(f"no bge-base weights at {ENC}")
    tok = WordPiece(f"{ART}/vocab.txt")

    # Real English prose from the public docs -- in-repo and reproducible, unlike invented text.
    text = " ".join(open(p, encoding="utf-8").read() for p in sorted(glob.glob("docs/*.md")))
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # drop code fences, keep prose
    text = re.sub(r"[|`#*_>\[\]()-]", " ", text)
    ids = tok.encode(text)
    PAD_M = args.pad_m
    body = PAD_M - 2
    if len(ids) < body:
        sys.exit(f"corpus too short: {len(ids)} tokens < {body}")
    seq = [tok.vocab["[CLS]"]] + ids[:body] + [tok.vocab["[SEP]"]]
    print(f"corpus {len(ids)} tokens -> one {len(seq)}-token sequence (no padding, no mask)\n")

    caps = forward_capture(seq)
    want = range(NL) if args.layers == "all" else [int(s) for s in args.layers.split(",")]
    os.makedirs(args.out, exist_ok=True)

    print(f"{'layer':6s} {'std':>8s} {'absmax':>8s} {'|x|<0.5':>8s} {'|x|<1':>7s} "
          f"{'q50':>7s} {'q90':>7s} {'q99':>7s}")
    summary = {}
    for i in range(NL):
        xb = bf(caps[i])                                # the cast@768 output the rail feeds fc1
        W1, b1 = bf(load(f"L{i}/ffn1_w")), bf(load(f"L{i}/ffn1_b"))
        pre = f32(xb) @ f32(W1) + f32(b1)               # verify_k768_gelu_rail.py's own reference
        st = bands(pre)
        summary[f"L{i}"] = st
        print(f"L{i:<5d} {st['std']:8.4f} {st['absmax']:8.3f} {st['frac_lt_0.5']:8.3f} "
              f"{st['frac_lt_1']:7.3f} {st['q50']:7.4f} {st['q90']:7.4f} {st['q99']:7.4f}")

        if i in want:
            A = np.zeros((PAD_M, KAUG), np.float32)
            A[:len(seq), :KRES] = f32(xb)
            A[:, KRES] = 1.0                            # ones column carries the bias
            B = np.zeros((KAUG, DFF), np.float32)
            B[:KRES, :] = f32(W1)
            B[KRES, :] = f32(b1)
            np.savez(f"{args.out}/bge_fc1_L{i}.npz",
                     A=bf(A).view(np.uint16), B=bf(B).view(np.uint16),
                     pad_m=PAD_M, dff=DFF, layer=i, n_tokens=len(seq))

    # The random-normal accumulator the device probe used, for the same table.
    rng = np.random.default_rng(20260825)
    rn = (rng.standard_normal((PAD_M, KRES)) @ (rng.standard_normal((KRES, DFF)) / np.sqrt(KRES)))
    st = bands(f32(rn))
    summary["random_normal"] = st
    print(f"\n{'rand-N':6s} {st['std']:8.4f} {st['absmax']:8.3f} {st['frac_lt_0.5']:8.3f} "
          f"{st['frac_lt_1']:7.3f} {st['q50']:7.4f} {st['q90']:7.4f} {st['q99']:7.4f}"
          "   <- what the ladder was scored on")

    with open(f"{args.out}/bands.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.out}/ (operands for layers {sorted(want)})")


if __name__ == "__main__":
    main()
