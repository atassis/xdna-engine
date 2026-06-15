#!/usr/bin/env python3
"""Part 3 (P3): NPU-encoder -> ONNX whisper-small decoder (greedy) -> WER over 17 FLEURS clips.

Two encoder sources (--enc-source):
  onnx : run encoder_model.onnx on the mel (CPU) -> CONTROL. Isolates decode-harness error.
  npu  : load the NPU encoder hidden-state npy from whisper_encode_npu (Part 2) -> TEST.

Both feed the SAME greedy decode loop (decoder_model.onnx), so any delta is encoder-only.
Prompt token ids are derived PROGRAMMATICALLY from the tokenizer (never hardcoded).
"""
import argparse, json, re, unicodedata
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import WhisperProcessor

CLIPS = Path("artifacts/wer_clips")
ONNX = Path("artifacts/whisper-small/onnx")
MELS = Path("artifacts/whisper-small/mels")

# ---- normalize / wer : identical to scripts/whisper_cpu_oracle.py ----
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


def greedy_decode(dec_sess, enc_hidden, start_ids, eot, max_new=200):
    """Greedy argmax loop on decoder_model.onnx (no kv-cache; re-feed full prefix each step)."""
    ids = list(start_ids)
    enc = enc_hidden.astype(np.float32)  # [1,1500,768]
    for _ in range(max_new):
        input_ids = np.asarray([ids], dtype=np.int64)  # [1,L]
        logits = dec_sess.run(["logits"], {
            "input_ids": input_ids,
            "encoder_hidden_states": enc,
        })[0]  # [1,L,vocab]
        nxt = int(np.argmax(logits[0, -1]))
        ids.append(nxt)
        if nxt == eot:
            break
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-source", choices=["onnx", "npu"], required=True)
    ap.add_argument("--enc-dir", default="artifacts/whisper-small/enc_npu",
                    help="dir of NPU encoder hidden-state npys (for --enc-source npu)")
    args = ap.parse_args()

    proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    tok = proc.tokenizer
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    lang_ru = tok.convert_tokens_to_ids("<|ru|>")
    lang_en = tok.convert_tokens_to_ids("<|en|>")
    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    notimestamps = tok.convert_tokens_to_ids("<|notimestamps|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")

    so = ort.SessionOptions()
    dec_sess = ort.InferenceSession(str(ONNX / "decoder_model.onnx"), so,
                                    providers=["CPUExecutionProvider"])
    enc_sess = None
    if args.enc_source == "onnx":
        enc_sess = ort.InferenceSession(str(ONNX / "encoder_model.onnx"), so,
                                        providers=["CPUExecutionProvider"])

    refs = json.load(open(CLIPS / "refs.json"))
    out = {}
    for name in sorted(refs):
        stem = Path(name).stem
        if args.enc_source == "onnx":
            mel = np.load(MELS / f"{stem}.npy").astype(np.float32)  # [1,80,3000]
            enc_hidden = enc_sess.run(["last_hidden_state"], {"input_features": mel})[0]  # [1,1500,768]
        else:
            h = np.load(Path(args.enc_dir) / f"{stem}.npy").astype(np.float32)  # [1500,768]
            enc_hidden = h[None, ...]  # [1,1500,768]

        lang = lang_ru if name.startswith("ru") else lang_en
        start_ids = [sot, lang, transcribe, notimestamps]
        ids = greedy_decode(dec_sess, enc_hidden, start_ids, eot)
        text = tok.decode(ids, skip_special_tokens=True)
        w, n = wer(normalize(refs[name]), normalize(text))
        out[name] = {"hyp": text, "wer": w, "nref": n}
        print(f"{name}: WER={w:.3f}")

    print(f"\n=== enc-source={args.enc_source} ===")
    for split in ("en", "ru"):
        ws = [v["wer"] for k, v in out.items() if k.startswith(split)]
        print(f"{split} mean WER = {sum(ws) / len(ws):.3f}  (n={len(ws)})")

    dst = CLIPS / f"whisper_npu_wer_{args.enc_source}.json"
    json.dump(out, open(dst, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
