#!/usr/bin/env python3
"""End-to-end test of the NPU encode path: Rust encode_server (our NPU encoder) -> onnx-asr
decode -> text, vs the oracle. Run in the onnx_asr venv from repo root:
  ~/npuvox-asr-bench/.venv/bin/python scripts/test_npu_pipeline.py
Requires the NPU free (stop flm-asr/voxd)."""
import glob, os, struct, subprocess, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--istupakov--gigaam-v3-onnx/snapshots/*"))[0]

def read_exact(f, n):
    b = b""
    while len(b) < n:
        chunk = f.read(n - len(b))
        if not chunk: raise EOFError("encode_server closed")
        b += chunk
    return b

def main():
    feats = np.load(f"{REPO}/artifacts/asr_ref/features.npy")  # [1,64,T]
    T = feats.shape[2]
    oracle = open(f"{REPO}/artifacts/asr_ref/text.txt").read().strip()

    p = subprocess.Popen([f"{REPO}/rust/target/release/encode_server"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=REPO)
    # request: u32 T + f32*64*T (channel-major [64,T])
    payload = np.ascontiguousarray(feats[0].astype("<f4")).tobytes()
    p.stdin.write(struct.pack("<I", T)); p.stdin.write(payload); p.stdin.flush()
    valid = struct.unpack("<I", read_exact(p.stdout, 4))[0]
    enc = np.frombuffer(read_exact(p.stdout, 768*400*4), "<f4").reshape(768, 400)
    p.stdin.write(struct.pack("<I", 0)); p.stdin.flush()  # shutdown
    print(f"[npu] valid_len={valid}  encoded[768,400] nan={np.isnan(enc).any()}")

    enc_out = enc.T[None, :valid, :].astype(np.float32)    # [1,valid,768]
    import onnx_asr
    model = onnx_asr.load_model("gigaam-v3-rnnt", path=SNAP)
    ids = None
    for tok_ids, ts, lp in model.asr._decoding(enc_out, np.array([valid], np.int64)):
        ids = [int(x) for x in tok_ids]
    text = model.asr._decode_tokens(ids, None, None).text

    # word-level WER
    def wer(ref, hyp):
        r, h = ref.split(), hyp.split()
        d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
        for i in range(len(r)+1): d[i][0]=i
        for j in range(len(h)+1): d[0][j]=j
        for i in range(1,len(r)+1):
            for j in range(1,len(h)+1):
                d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(r[i-1]!=h[j-1]))
        return d[len(r)][len(h)]/max(1,len(r))
    print(f"[oracle] {oracle!r}")
    print(f"[npu   ] {text!r}")
    print(f"[WER vs oracle] {wer(oracle, text)*100:.1f}%  ({len(ids)} tokens)")

if __name__ == "__main__":
    sys.exit(main())
