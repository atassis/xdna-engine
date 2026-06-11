#!/usr/bin/env python3
"""Rung 3 — export the GigaAM-v3 RNNT *encoder* to a clean static-shape ONNX.

NPUs (and AMD's VitisAI EP BF16 encoder flow) want static shapes. The istupakov
`gigaam-v3-onnx` already ships the encoder as a separate file with dynamic dims
`[batch_size, 64, seq_len]`; this script fixes batch=1 and seq_len=N, re-runs shape
inference so the symbolic output dim resolves, verifies the static graph is numerically
identical to the dynamic one on the same input, and saves it to models/.

BF16/INT8 quantization is intentionally NOT done here — that is AMD Quark's job inside
the Ryzen AI 1.7.1 toolchain (Rung 4). This script produces the clean fp32 static graph
that Quark/VitisAI consumes.

Usage:
  .venv/bin/python scripts/export_gigaam_encoder.py [--frames N] [--out PATH]
"""
import argparse, glob, os, sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed, fix_output_shapes

HUB = os.path.expanduser("~/.cache/huggingface/hub")

def find_encoder():
    hits = glob.glob(f"{HUB}/models--istupakov--gigaam-v3-onnx/snapshots/*/v3_rnnt_encoder.onnx")
    if not hits:
        sys.exit("GigaAM-v3 encoder ONNX not found in HF cache — run the bench once to fetch it.")
    return hits[0]

def io_shapes(model):
    def shp(t):
        d = t.type.tensor_type
        return TP.DataType.Name(d.elem_type), [(x.dim_param or x.dim_value) for x in d.shape.dim]
    return ({i.name: shp(i) for i in model.graph.input},
            {o.name: shp(o) for o in model.graph.output})

def probe_subsampling(sess, n_mels=64):
    """Run a few seq_lens through the dynamic encoder to learn time subsampling + feature dim.

    GigaAM-v3's encoder output is laid out [batch, feat, time] at runtime, but the ONNX
    graph declares it with a mislabeled symbolic middle dim — so we measure the real layout
    here and re-annotate the static graph's output explicitly.
    Returns (rows, feat_dim, subsample).
    """
    rows = []
    feat = sub = None
    for T in (256, 512, 1024):
        sig = np.random.randn(1, n_mels, T).astype(np.float32)
        ln = np.array([T], dtype=np.int64)
        enc, enc_len = sess.run(None, {"audio_signal": sig, "length": ln})
        rows.append((T, enc.shape, int(enc_len[0])))
        feat = enc.shape[1]
        sub = T // int(enc_len[0])
    return rows, feat, sub


def set_output_shape(model, name, dims):
    for o in model.graph.output:
        if o.name == name:
            del o.type.tensor_type.shape.dim[:]
            for d in dims:
                o.type.tensor_type.shape.dim.add().dim_value = d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1600,
                    help="static mel frames (16kHz, 10ms hop → 1600 = 16s window)")
    ap.add_argument("--out", default="models/gigaam_v3_encoder_static.onnx")
    args = ap.parse_args()

    src = find_encoder()
    print(f"[src] {src}")
    m = onnx.load(src)
    ins, outs = io_shapes(m)
    print(f"[dynamic] inputs={ins}\n           outputs={outs}")

    # reference: dynamic session
    dyn = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    rows, feat, sub = probe_subsampling(dyn)
    print("[probe] (T_in, enc.shape, enc_len):")
    for row in rows:
        print("   ", row)
    print(f"[probe] feat_dim={feat}  subsample={sub}  → output layout [1,{feat},N/{sub}]")

    N = args.frames
    if N % sub:
        sys.exit(f"--frames {N} not divisible by subsample {sub}; pick a multiple of {sub}")
    # fix dynamic dims → static
    make_dim_param_fixed(m.graph, "batch_size", 1)
    make_dim_param_fixed(m.graph, "seq_len", N)
    fix_output_shapes(m)
    # the original graph mislabels the encoded output dims — set the true static shape
    set_output_shape(m, "encoded", [1, feat, N // sub])
    set_output_shape(m, "encoded_len", [1])
    onnx.checker.check_model(m)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    onnx.save(m, args.out)
    sins, souts = io_shapes(m)
    print(f"[static]  inputs={sins}\n           outputs={souts}")

    # verify: static == dynamic on identical input (valid region)
    sig = np.random.randn(1, 64, N).astype(np.float32)
    ln = np.array([N], dtype=np.int64)
    ref, ref_len = dyn.run(None, {"audio_signal": sig, "length": ln})
    stat = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got, got_len = stat.run(None, {"audio_signal": sig, "length": ln})
    ok = np.allclose(ref, got, atol=1e-4, rtol=1e-3)
    md = float(np.abs(ref - got).max())
    print(f"[verify]  static vs dynamic: allclose={ok}  max|Δ|={md:.2e}  "
          f"enc_len ref={int(ref_len[0])} got={int(got_len[0])}")
    sz = os.path.getsize(args.out) / 1e6
    print(f"[done]    {args.out}  ({sz:.1f} MB, fp32, static [1,64,{N}])")
    print("[next]    Quark BF16/INT8 quantization happens in the Ryzen AI 1.7.1 env (Rung 4).")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
