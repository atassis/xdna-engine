#!/usr/bin/env python3
"""Greedy-decode precomputed Whisper encoder hidden states through the CPU ONNX decoder and score
WER. Model-agnostic sibling of whisper_npu_wer.py: takes a directory of per-clip encoder hidden
state .npy files (produced by any source -- ONNX CPU, a PyTorch fake-quant sim, or an NPU dump) and
holds the decoder identical across every arm, so any WER delta is encoder-only.

Reports WER(hyp vs ground-truth ref) always, and WER(hyp vs a baseline hyp set) when --baseline-hyps
is given -- the second isolates encoder-format error from ground-truth transcription error (the
int8_wer_eval.py method, generalized to Whisper).

Usage:
  python scripts/whisper_wer_from_hidden.py --model whisper-turbo --clips artifacts/wer_clips_large \
      --hidden artifacts/whisper-turbo/enc_fp32_large --label turbo-fp32 --out /tmp/turbo_fp32.json \
      [--baseline-hyps /tmp/turbo_fp32.json]
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import WhisperProcessor

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(t):
    t = unicodedata.normalize("NFC", t or "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return (0.0 if not h else 1.0), 0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if rw == hw else 1)))
        prev = cur
    return prev[-1] / len(r), len(r)


def pooled(items, key):
    sum_edits = sum(round(v[key] * v["nref"]) for v in items)
    sum_ref = sum(v["nref"] for v in items)
    return (sum_edits / sum_ref) if sum_ref else 0.0


def greedy_decode_kv(dec_sess, dec_past_sess, n_layers, enc_hidden, start_ids, eot, max_new=200):
    """KV-cached greedy decode: O(1) work per step instead of O(n) (no re-feeding the whole
    prefix). Step 0 uses the no-past decoder graph (which also emits the encoder cross-attn KV,
    computed once); every step after uses decoder_with_past, feeding only the newest token and
    the running decoder self-attn KV -- the encoder KV from step 0 is reused unchanged, matching
    the with-past graph's own I/O (it has no encoder-KV outputs to update)."""
    enc = enc_hidden.astype(np.float32)
    ids = list(start_ids)
    input_ids = np.asarray([ids], dtype=np.int64)
    out = dec_sess.run(None, {"input_ids": input_ids, "encoder_hidden_states": enc})
    names = [o.name for o in dec_sess.get_outputs()]
    vals = dict(zip(names, out))
    logits = vals["logits"]
    nxt = int(np.argmax(logits[0, -1]))
    ids.append(nxt)
    past = {}
    for i in range(n_layers):
        for kind in ("decoder.key", "decoder.value", "encoder.key", "encoder.value"):
            past[f"past_key_values.{i}.{kind}"] = vals[f"present.{i}.{kind}"]

    for _ in range(max_new - 1):
        if nxt == eot:
            break
        feed = {"input_ids": np.asarray([[nxt]], dtype=np.int64)}
        feed.update(past)
        out = dec_past_sess.run(None, feed)
        names2 = [o.name for o in dec_past_sess.get_outputs()]
        vals2 = dict(zip(names2, out))
        logits = vals2["logits"]
        nxt = int(np.argmax(logits[0, -1]))
        ids.append(nxt)
        for i in range(n_layers):
            past[f"past_key_values.{i}.decoder.key"] = vals2[f"present.{i}.decoder.key"]
            past[f"past_key_values.{i}.decoder.value"] = vals2[f"present.{i}.decoder.value"]
            # encoder.key/value: no present.*.encoder.* in the with-past graph -- unchanged, reused.
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["whisper-small", "whisper-turbo"])
    ap.add_argument("--clips", required=True)
    ap.add_argument("--hidden", required=True, help="dir of per-clip [1500,d] encoder hidden .npy")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-hyps", default=None, help="json from a prior run; adds WER-vs-baseline")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--decoder-dir", default=None,
                     help="dir holding decoder_model[.onnx]/decoder_with_past_model.onnx; default "
                          "artifacts/<model>/onnx (fp32). Point at onnx_quant for a quantized decoder.")
    ap.add_argument("--decoder-suffix", default="",
                     help="e.g. '_int8' to load decoder_model_int8.onnx / decoder_with_past_model_int8.onnx")
    ap.add_argument("--checkpoint", default=None,
                     help="JSONL, one {'name':..., 'row':{...}} per clip, appended as computed. "
                          "On restart, clips already present are skipped and reused -- a killed run "
                          "loses at most the one clip in flight, never the whole arm.")
    args = ap.parse_args()

    onnx_dir = Path("artifacts") / args.model / "onnx"
    proc = WhisperProcessor.from_pretrained(str(onnx_dir))
    tok = proc.tokenizer
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    lang_ru = tok.convert_tokens_to_ids("<|ru|>")
    lang_en = tok.convert_tokens_to_ids("<|en|>")
    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    notimestamps = tok.convert_tokens_to_ids("<|notimestamps|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")

    dec_dir = Path(args.decoder_dir) if args.decoder_dir else onnx_dir
    sfx = args.decoder_suffix
    so = ort.SessionOptions()
    dec_path = dec_dir / f"decoder_model{sfx}.onnx"
    dec_past_path = dec_dir / f"decoder_with_past_model{sfx}.onnx"
    dec_sess = ort.InferenceSession(str(dec_path), so, providers=["CPUExecutionProvider"])
    dec_past_sess = ort.InferenceSession(str(dec_past_path), so, providers=["CPUExecutionProvider"])
    n_layers = (len(dec_sess.get_outputs()) - 1) // 4
    print(f"[{args.label}] decoder={dec_path.name} n_layers={n_layers} (KV-cached decode)")

    refs = json.load(open(Path(args.clips) / "refs.json", encoding="utf-8"))
    hidden_dir = Path(args.hidden)
    names = sorted(n for n in refs if (hidden_dir / f"{Path(n).stem}.npy").is_file())
    if args.limit:
        names = names[: args.limit]
    missing = len(refs) - len(names)
    if missing:
        print(f"[warn] {missing}/{len(refs)} clips have no hidden-state file in {hidden_dir}; scoring {len(names)}")

    baseline = None
    if args.baseline_hyps:
        b = json.load(open(args.baseline_hyps, encoding="utf-8"))
        baseline = {k: v["hyp"] for k, v in b["rows"].items()}

    rows = {}
    ckpt_fh = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.is_file():
            for line in open(ckpt_path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows[rec["name"]] = rec["row"]
            print(f"[{args.label}] resumed {len(rows)} clip(s) from {ckpt_path}", flush=True)
        ckpt_fh = open(ckpt_path, "a", encoding="utf-8")

    for name in names:
        if name in rows:
            continue  # checkpoint resume: already computed
        stem = Path(name).stem
        h = np.load(hidden_dir / f"{stem}.npy").astype(np.float32)
        enc_hidden = h[None, ...] if h.ndim == 2 else h
        lang = lang_ru if name.startswith("ru") else lang_en
        start_ids = [sot, lang, transcribe, notimestamps]
        ids = greedy_decode_kv(dec_sess, dec_past_sess, n_layers, enc_hidden, start_ids, eot)
        text = tok.decode(ids, skip_special_tokens=True)
        w, n = wer(normalize(refs[name]), normalize(text))
        row = {"hyp": text, "wer": w, "nref": n}
        if baseline and name in baseline:
            wb, nb = wer(normalize(baseline[name]), normalize(text))
            row["wer_vs_baseline"] = wb
            row["nref_baseline"] = nb
        rows[name] = row
        if ckpt_fh:
            ckpt_fh.write(json.dumps({"name": name, "row": row}, ensure_ascii=False) + "\n")
            ckpt_fh.flush()
        print(f"[{args.label}] {name}: WER={w:.3f}" + (f"  vsBaseline={row.get('wer_vs_baseline', float('nan')):.3f}" if baseline else ""), flush=True)
    if ckpt_fh:
        ckpt_fh.close()

    print(f"\n=== label={args.label} model={args.model} n={len(rows)} ===")
    for split in ("en", "ru"):
        sub = [v for k, v in rows.items() if k.startswith(split)]
        if sub:
            macro = sum(v["wer"] for v in sub) / len(sub)
            line = f"{split} pooled WER = {pooled(sub, 'wer'):.4f}  (macro {macro:.4f}, n={len(sub)})"
            if baseline:
                subb = [v for v in sub if "wer_vs_baseline" in v]
                if subb:
                    line += f"   vsBaseline pooled = {pooled(subb, 'wer_vs_baseline'):.4f} (n={len(subb)})"
            print(line)
    allv = list(rows.values())
    if allv:
        macro = sum(v["wer"] for v in allv) / len(allv)
        line = f"ALL pooled WER = {pooled(allv, 'wer'):.4f}  (macro {macro:.4f}, n={len(allv)})"
        if baseline:
            allvb = [v for v in allv if "wer_vs_baseline" in v]
            if allvb:
                line += f"   vsBaseline pooled = {pooled(allvb, 'wer_vs_baseline'):.4f} (n={len(allvb)})"
        print(line)

    if args.out:
        json.dump({"label": args.label, "model": args.model, "hidden": str(hidden_dir), "rows": rows},
                   open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
