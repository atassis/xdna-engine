#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-tokenize a corpus for eval_llm_perplexity.py's --ids path, OUTSIDE .venv-iron.

qwen_bpe.py is a from-scratch Qwen BPE encoder, written because .venv-iron (the toolchain env)
must not grow a `transformers`/`tokenizers` dependency. It hardcodes Qwen's pretokenizer shape
(`pre_tokenizer.pretokenizers[0].pattern.Regex`) and cannot read Gemma's tokenizer.json, whose
pretokenizer is a single `Split` on literal space plus a `▁` normalizer -- a KeyError, not a
wrong answer. Rather than write a second from-scratch encoder, this runs the real `tokenizers`
library (present in .venv-export, no torch needed) ONCE, offline, and hands eval_llm_perplexity.py
plain ids -- the same split gate_llm_reference.py already uses between its numpy and hf backends.

  .venv-export/bin/python designs/decode_fused/tokenize_corpus.py \\
      --tokenizer /mnt/data/xdna/artifacts/gemma4-12b/tokenizer/tokenizer.json \\
      --text some.txt --out some.ids.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    txt = open(a.text, encoding="utf-8").read()
    # No special tokens: matches hostlab/wq_eval.py's tokenize() (add_special_tokens=False) and
    # qwen_bpe.QwenBPE.encode, so an --ids run and a --text run score the same corpus the same way.
    ids = tok.encode(txt, add_special_tokens=False).ids
    json.dump(ids, open(a.out, "w"))
    print(f"[tokenize] {a.text}: {len(ids)} tokens -> {a.out}")


if __name__ == "__main__":
    main()
