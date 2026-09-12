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
      --text some.txt --out some.ids.json --prepend-bos

--prepend-bos is not cosmetic on Gemma. Its tokenizer.json carries a TemplateProcessing
post-processor whose `special_tokens` map is EMPTY, so neither add_special_tokens=True here nor
anything downstream ever emits <bos> -- while the model was trained with it on every sequence.
MEASURED on gemma4-12b, 48 layers, 96 positions of wikitext-2: mean NLL 6.698 with it against
16.530 without, i.e. without BOS the model scores worse than a uniform distribution over its own
262144-token vocabulary. This script warns when the tokenizer declares a bos_token the
post-processor will not add, because a missing BOS is silent everywhere else.
"""
import argparse
import json
import os


def _declared_bos(tokenizer_json, tok):
    """(token, id) of the tokenizer's declared BOS, or (None, None).

    Read from tokenizer_config.json beside tokenizer.json, because that is where the declaration
    lives; the post-processor is a separate question and is exactly what this catches.
    """
    cfg = os.path.join(os.path.dirname(os.path.abspath(tokenizer_json)), "tokenizer_config.json")
    if not os.path.isfile(cfg):
        return None, None
    bos = json.load(open(cfg)).get("bos_token")
    if isinstance(bos, dict):
        bos = bos.get("content")
    return (bos, tok.token_to_id(bos)) if bos else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepend-bos", action="store_true",
                    help="prepend the tokenizer_config bos_token's id (see the module docstring)")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    txt = open(a.text, encoding="utf-8").read()
    # No special tokens: matches hostlab/wq_eval.py's tokenize() (add_special_tokens=False) and
    # qwen_bpe.QwenBPE.encode, so an --ids run and a --text run score the same corpus the same way.
    ids = tok.encode(txt, add_special_tokens=False).ids

    bos_tok, bos_id = _declared_bos(a.tokenizer, tok)
    if bos_id is not None and not a.prepend_bos:
        print(f"[tokenize] WARNING: {bos_tok!r} (id {bos_id}) is declared as this tokenizer's "
              f"bos_token and its post-processor does not add it. Pass --prepend-bos unless you "
              f"mean to score the model off-distribution.")
    if a.prepend_bos:
        if bos_id is None:
            raise SystemExit("[tokenize] --prepend-bos but no bos_token is declared")
        ids = [bos_id] + ids
    json.dump(ids, open(a.out, "w"))
    print(f"[tokenize] {a.text}: {len(ids)} tokens"
          f"{f' (bos {bos_id} prepended)' if a.prepend_bos else ''} -> {a.out}")


if __name__ == "__main__":
    main()
