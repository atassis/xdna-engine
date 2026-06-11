#!/usr/bin/env python3
"""Drop-in HTTP ASR service — FLM/Whisper-compatible transcription API on a GigaAM-v3 backend.

Exposes  POST /v1/audio/transcriptions  (multipart/form-data, field `file` = WAV, optional
`model` text field ignored) and returns  {"text": "<transcript>"}  — exactly what voxd's
`AsrClient` sends and reads. Point voxd at it with:

    NPUVOX_ENDPOINT=http://127.0.0.1:11435/v1/audio/transcriptions

Pipeline: onnx-asr mel preprocessor + RNNT decode, with a *pluggable encoder backend* (env
`ASR_ENCODER`):
  - `onnx` (default): onnx-asr's native encoder. No NPU / no Rust needed.
  - `npu`           : our Rust NPU `encode_server` coprocess (single-tenant NPU, kept warm).

Run (from the repo root) — use the venv that has onnx_asr:

    # ONNX backend (default, no NPU):
    ASR_ENCODER=onnx ~/npuvox-asr-bench/.venv/bin/python scripts/asr_service.py

    # NPU backend (needs NPU free + `rust/target/release/encode_server` built):
    #   first stop flm-asr.service / voxd.service (NPU is single-tenant), then:
    ASR_ENCODER=npu  ~/npuvox-asr-bench/.venv/bin/python scripts/asr_service.py

    # CLI WER scoring (no server) against the oracle text:
    ~/npuvox-asr-bench/.venv/bin/python scripts/asr_service.py --wer <wav>

Env:
  NPUVOX_PORT   listen port (default 11435)
  ASR_ENCODER   onnx | npu  (default onnx)

Only the Python standard library is used for HTTP + multipart parsing (no pip deps).
"""
import glob
import json
import os
import subprocess
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HUB = os.path.expanduser("~/.cache/huggingface/hub")
SNAP = glob.glob(f"{HUB}/models--istupakov--gigaam-v3-onnx/snapshots/*")[0]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_TEXT_PATH = os.path.join(REPO, "artifacts", "asr_ref", "text.txt")

PORT = int(os.environ.get("NPUVOX_PORT", "11435"))
ENCODER = os.environ.get("ASR_ENCODER", "onnx").strip().lower()

# encode_server protocol constants (see rust/npu-asr/src/bin/encode_server.rs)
ENC_WIN = 1600            # 16 s mel window the NPU encoder accepts
ENC_T_OUT = 400          # fixed encoded time dim returned
ENC_D = 768              # encoder hidden dim
ENC_RESP_BYTES = 4 + ENC_D * ENC_T_OUT * 4   # u32 valid_len + f32*768*400


# --------------------------------------------------------------------------- audio I/O
def read_wav_16k(path):
    """Read a 16 kHz mono 16-bit PCM WAV into float32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x


def read_wav_16k_bytes(data):
    """Same as read_wav_16k but from in-memory WAV bytes."""
    import io
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getframerate() != 16000:
            raise ValueError(f"expected 16 kHz, got {w.getframerate()}")
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x


# --------------------------------------------------------------------------- model + encoders
print(f"[asr_service] loading gigaam-v3-rnnt from {SNAP} ...", file=sys.stderr)
import onnx_asr  # noqa: E402  (after the heavy header so --help is fast-ish)

_MODEL = onnx_asr.load_model("gigaam-v3-rnnt", path=SNAP)
_ASR = _MODEL.asr

# NPU coprocess (only spawned for ASR_ENCODER=npu)
_npu_proc = None
_npu_lock = threading.Lock()


def _spawn_npu():
    global _npu_proc
    _npu_proc = subprocess.Popen(
        ["rust/target/release/encode_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=REPO,
    )
    print("[asr_service] spawned encode_server NPU coprocess", file=sys.stderr)


def _read_exact(fh, n):
    """Read exactly n bytes from a pipe, looping over short reads."""
    buf = bytearray()
    while len(buf) < n:
        chunk = fh.read(n - len(buf))
        if not chunk:
            raise EOFError("encode_server closed the pipe")
        buf.extend(chunk)
    return bytes(buf)


def encode_onnx(features, feat_lens):
    """Default backend — onnx-asr's native encoder, unchanged."""
    return _ASR._encode(features, feat_lens)


def encode_npu(features, feat_lens):
    """NPU backend — delegate the encoder to the Rust encode_server coprocess.

    features: [1, 64, T] f32. We send mel features channel-major [64, T] and get back an
    encoded [768, 400] block + a valid_len. The encode_server truncates to a 1600-frame
    (16 s) window; if T > 1600 we send only the first 1600 frames (clip is ~12 s, so fine).
    """
    T = features.shape[2]
    Tsend = min(T, ENC_WIN)
    feats = features[0, :, :Tsend].astype("<f4")   # [64, Tsend], C-order = channel-major
    payload = np.uint32(Tsend).tobytes() + feats.tobytes()
    with _npu_lock:
        _npu_proc.stdin.write(payload)
        _npu_proc.stdin.flush()
        resp = _read_exact(_npu_proc.stdout, ENC_RESP_BYTES)
    valid_len = int(np.frombuffer(resp[:4], "<u4")[0])
    encoded = np.frombuffer(resp[4:], "<f4").reshape(ENC_D, ENC_T_OUT)  # [768, 400]
    enc_out = encoded.T[None, :valid_len, :]                            # [1, valid_len, 768]
    enc_lens = np.array([valid_len], np.int64)
    return enc_out.astype(np.float32), enc_lens


_ENCODE = encode_npu if ENCODER == "npu" else encode_onnx


def transcribe(waveform):
    """Full pipeline on a float32 [N] waveform -> transcript str. Encoder is pluggable."""
    n = waveform.shape[0]
    waveforms = waveform[None, :].astype(np.float32)
    wav_lens = np.array([n], np.int64)
    features, feat_lens = _ASR._preprocessor(waveforms, wav_lens)   # [1,64,T]
    enc_out, enc_lens = _ENCODE(features, feat_lens)               # [1,T',768]  (swapped backend)
    ids = []
    for tok_ids, ts, lp in _ASR._decoding(enc_out, enc_lens):
        ids = [int(x) for x in tok_ids]
    return _ASR._decode_tokens(ids, None, None).text


# --------------------------------------------------------------------------- WER helper
def wer(ref, hyp):
    """Word-level error rate via Levenshtein over word tokens. 0.0 == exact match."""
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(r)


# --------------------------------------------------------------------------- multipart parsing
def parse_multipart_file(body, content_type):
    """Extract the raw `file` part bytes from a multipart/form-data body.

    Robust to a quoted boundary and to arbitrary part headers. Returns the bytes between
    the headers (after the blank `\r\n\r\n` line) and the trailing `\r\n--<boundary>`.
    """
    # boundary=...   (may be quoted)
    boundary = None
    for tok in content_type.split(";"):
        tok = tok.strip()
        if tok.lower().startswith("boundary="):
            boundary = tok[len("boundary="):].strip().strip('"')
            break
    if not boundary:
        raise ValueError("no multipart boundary in Content-Type")
    delim = b"--" + boundary.encode("latin-1")

    parts = body.split(delim)
    for part in parts:
        # strip leading CRLF after the delimiter
        if part.startswith(b"\r\n"):
            part = part[2:]
        head_end = part.find(b"\r\n\r\n")
        if head_end < 0:
            continue
        header_blob = part[:head_end].decode("latin-1", "replace")
        if 'name="file"' not in header_blob and "name=file" not in header_blob:
            continue
        data = part[head_end + 4:]
        # drop the trailing CRLF that precedes the next boundary delimiter
        if data.endswith(b"\r\n"):
            data = data[:-2]
        return data
    raise ValueError('no `file` part found in multipart body')


# --------------------------------------------------------------------------- HTTP handler
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs to stderr
        sys.stderr.write("[asr_service] %s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/audio/transcriptions":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            ctype = self.headers.get("Content-Type", "")
            wav_bytes = parse_multipart_file(body, ctype)
            waveform = read_wav_16k_bytes(wav_bytes)
            text = transcribe(waveform)
            self._json(200, {"text": text})
        except Exception as e:  # noqa: BLE001 — return a clear error to the client
            sys.stderr.write(f"[asr_service] request error: {e!r}\n")
            self._json(400, {"error": str(e)})


# --------------------------------------------------------------------------- main
def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--wer":
        if len(sys.argv) < 3:
            print("usage: asr_service.py --wer <wav>", file=sys.stderr)
            return 2
        if ENCODER == "npu":
            _spawn_npu()
        wav = os.path.expanduser(sys.argv[2])
        ref = open(REF_TEXT_PATH, encoding="utf-8").read().strip()
        hyp = transcribe(read_wav_16k(wav)).strip()
        score = wer(ref, hyp)
        print(f"[--wer] backend={ENCODER} wav={wav}")
        print(f"[--wer] hyp: {hyp!r}")
        print(f"[--wer] ref: {ref!r}")
        print(f"[--wer] WER: {score:.4f}")
        return 0

    if ENCODER == "npu":
        _spawn_npu()

    print(
        f"[asr_service] listening on http://127.0.0.1:{PORT}/v1/audio/transcriptions  "
        f"backend={ENCODER}  model=gigaam-v3-rnnt",
        file=sys.stderr,
    )
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _npu_proc is not None:
            try:
                _npu_proc.stdin.write(np.uint32(0).tobytes())  # clean shutdown signal
                _npu_proc.stdin.flush()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
