#!/usr/bin/env python3
"""Host (CPU/NumPy) reference for the Parakeet-TDT-0.6b-v3 decoder.

This is the PORTABLE GOLDEN for the TDT decoder brick rebuild (spec
2026-06-28-parakeet-tdt-full-npu-brick-honoring.md, Tier-A task A6). It
re-implements the decoder math from scratch in NumPy -- the small prediction
network (token embedding + 2-layer LSTM) and the joint network (enc_t + pred_u
-> vocab + duration logits) -- plus the greedy token/duration emit loop with
frame-skipping. NO ONNXRuntime is used in the reference forward path; the ONNX
decoder_joint graph is loaded ONLY for its weights and used ONLY as the oracle
to gate against.

Why host-only: the TDT decoder is tiny (M=1, GEMV-shaped) and runs on the CPU
without the NPU. Validating it node-by-node here (rel-L2 <= 0.08 vs the ONNX
oracle) de-risks the later NPU port: the prediction GEMVs map to the M=1 GEMV
brick + embedding parallel_lookup, the joint argmax to max_cmp.

Bricks this reference stands in for (per aie2p-brick-catalog, NPU port):
  - embedding lookup        -> parallel_lookup (gather LUT)
  - LSTM / joint matmuls     -> GEMV (M=1 small) -- mac/accumulate, not mmul
  - token + duration argmax  -> max_cmp (value + index co-produced, free argmax)
  - greedy frame-skip loop   -> host-light control (few steps, frame-skipping)

Validation (all CPU, no NPU):
  1. Per-step joint-output golden: numpy prednet+joint vs onnxruntime
     decoder_joint.run over a real decode trajectory -> rel-L2 gate.
  2. Greedy decode token-sequence parity: our loop vs the onnx_asr NeMo TDT
     loop (mirrored exactly) driving the SAME ONNX graph -> exact token match.

Usage:
  .venv/bin/python scripts/parakeet_tdt_decoder_ref.py            # validate + golden
  .venv/bin/python scripts/parakeet_tdt_decoder_ref.py --bench    # + per-token timing
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DECODER_ONNX = ROOT / "artifacts/parakeet/decoder_joint.onnx"
ENC_REF = ROOT / "artifacts/parakeet/encoder/refs/encoded.npy"
VOCAB = ROOT / "artifacts/parakeet/vocab.txt"
ENCODER_ONNX = ROOT / "models/parakeet/encoder-model.onnx"
PREPROC_ONNX = ROOT / "artifacts/parakeet/preprocessor.onnx"

HIDDEN = 640
VOCAB_SIZE = 8193          # token logits incl <blk>; vocab.txt has ids 0..8192
NUM_DURATIONS = 5          # output 8198 = 8193 tokens + 5 durations
BLANK_IDX = 8192           # "<blk>"
MAX_TOKENS_PER_STEP = 10   # onnx_asr NeMo default


# --------------------------------------------------------------------------- #
# Weight loading (from the ONNX decoder_joint initializers)
# --------------------------------------------------------------------------- #
def load_weights(onnx_path: Path = DECODER_ONNX) -> dict:
    import onnx
    from onnx import numpy_helper

    g = onnx.load(str(onnx_path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    # Map opaque onnx::LSTM_* names by inspecting the two LSTM nodes in order.
    lstm_nodes = [n for n in g.node if n.op_type == "LSTM"]
    assert len(lstm_nodes) == 2, f"expected 2 LSTM layers, got {len(lstm_nodes)}"
    layers = []
    for n in lstm_nodes:
        # inputs: X, W, R, B, sequence_lens, initial_h, initial_c
        W = inits[n.input[1]][0]   # [4*hidden, input_size], gate order i,o,f,c
        R = inits[n.input[2]][0]   # [4*hidden, hidden]
        B = inits[n.input[3]][0]   # [8*hidden] = [Wb(iofc), Rb(iofc)]
        layers.append({"W": W.astype(np.float32),
                       "R": R.astype(np.float32),
                       "B": B.astype(np.float32)})

    # Joint MatMul weights by their (in,out) shapes.
    def find(shape):
        for k, v in inits.items():
            if k.startswith("onnx::MatMul") and v.shape == shape:
                return v.astype(np.float32)
        raise KeyError(shape)

    return {
        "embed": inits["decoder.prediction.embed.weight"].astype(np.float32),  # [8193,640]
        "lstm": layers,
        "enc_W": find((1024, HIDDEN)),                       # joint.enc proj
        "enc_b": inits["joint.enc.bias"].astype(np.float32),
        "pred_W": find((HIDDEN, HIDDEN)),                    # joint.pred proj
        "pred_b": inits["joint.pred.bias"].astype(np.float32),
        "joint_W": find((HIDDEN, VOCAB_SIZE + NUM_DURATIONS)),  # joint_net.2
        "joint_b": inits["joint.joint_net.2.bias"].astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# Prediction network: embedding + 2-layer LSTM (ONNX gate order i,o,f,c)
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_step(layer: dict, x: np.ndarray, h: np.ndarray, c: np.ndarray):
    """One ONNX-LSTM cell step. x,h,c: [hidden]. Returns (h_new, c_new)."""
    W, R, B = layer["W"], layer["R"], layer["B"]
    H = HIDDEN
    z = W @ x + R @ h + B[: 4 * H] + B[4 * H:]   # [4*hidden], gate order i,o,f,c
    i = _sigmoid(z[0 * H:1 * H])
    o = _sigmoid(z[1 * H:2 * H])
    f = _sigmoid(z[2 * H:3 * H])
    g = np.tanh(z[3 * H:4 * H])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def prednet_step(W: dict, token: int, state):
    """Embedding lookup + 2-layer LSTM. state = (h[2,hidden], c[2,hidden]).
    Returns (pred[hidden], new_state)."""
    h, c = state
    x = W["embed"][token]                       # parallel_lookup on NPU
    h0, c0 = lstm_step(W["lstm"][0], x, h[0], c[0])
    h1, c1 = lstm_step(W["lstm"][1], h0, h[1], c[1])
    new_h = np.stack([h0, h1]); new_c = np.stack([c0, c1])
    return h1, (new_h, new_c)


def init_state():
    return (np.zeros((2, HIDDEN), np.float32), np.zeros((2, HIDDEN), np.float32))


# --------------------------------------------------------------------------- #
# Joint network: enc_t + pred_u -> [vocab + durations] logits
# --------------------------------------------------------------------------- #
def joint(W: dict, enc_t: np.ndarray, pred_u: np.ndarray) -> np.ndarray:
    enc_proj = enc_t @ W["enc_W"] + W["enc_b"]      # [640]  (GEMV)
    pred_proj = pred_u @ W["pred_W"] + W["pred_b"]  # [640]  (GEMV)
    act = np.maximum(enc_proj + pred_proj, 0.0)     # ReLU
    return act @ W["joint_W"] + W["joint_b"]        # [8198] (GEMV)


# --------------------------------------------------------------------------- #
# TDT greedy decode (frame-skipping). Mirrors onnx_asr NeMo TDT loop exactly.
# --------------------------------------------------------------------------- #
def tdt_greedy_decode(W: dict, encodings: np.ndarray, enc_len: int):
    """encodings: [T, 1024]. Returns (tokens, timestamps, n_joint_calls)."""
    state = init_state()
    pred_u, _ = prednet_step(W, BLANK_IDX, state)  # priming step (token = blank)
    # NOTE: onnx_asr feeds prev_tokens[-1] (or blank if empty) each call and
    # only advances the prednet state when a non-blank is emitted. We mirror
    # that: keep (pred_u, state) for the "current" prednet output.
    tokens: list[int] = []
    timestamps: list[int] = []
    t = 0
    emitted = 0
    n_calls = 0
    # Track the prednet output for the current target (last emitted, or blank).
    cur_pred, cur_state = pred_u, state
    while t < enc_len:
        logits_full = joint(W, encodings[t], cur_pred)
        n_calls += 1
        token_logits = logits_full[:VOCAB_SIZE]
        dur_logits = logits_full[VOCAB_SIZE:]
        token = int(token_logits.argmax())          # max_cmp on NPU
        step = int(dur_logits.argmax())             # duration in {0..4}

        if token != BLANK_IDX:
            # emit + advance prednet on the newly emitted token
            cur_pred, cur_state = prednet_step(W, token, cur_state)
            tokens.append(token)
            timestamps.append(t)
            emitted += 1

        if step > 0:
            t += step
            emitted = 0
        elif token == BLANK_IDX or emitted == MAX_TOKENS_PER_STEP:
            t += 1
            emitted = 0
    return tokens, timestamps, n_calls


# --------------------------------------------------------------------------- #
# Oracle: drive the ONNX decoder_joint graph with the onnx_asr loop verbatim.
# --------------------------------------------------------------------------- #
def onnx_oracle_decode(encodings: np.ndarray, enc_len: int):
    import onnxruntime as rt

    sess = rt.InferenceSession(str(DECODER_ONNX), providers=["CPUExecutionProvider"])
    shapes = {x.name: x.shape for x in sess.get_inputs()}
    s1 = np.zeros((shapes["input_states_1"][0], 1, shapes["input_states_1"][2]), np.float32)
    s2 = np.zeros((shapes["input_states_2"][0], 1, shapes["input_states_2"][2]), np.float32)

    def _decode(prev_tokens, st, enc_t):
        outputs, n1, n2 = sess.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": enc_t[None, :, None],
                "targets": np.array([[prev_tokens[-1] if prev_tokens else BLANK_IDX]], np.int32),
                "target_length": np.array([1], np.int32),
                "input_states_1": st[0],
                "input_states_2": st[1],
            },
        )
        out = np.squeeze(outputs)
        return out[:VOCAB_SIZE], int(out[VOCAB_SIZE:].argmax()), (n1, n2)

    tokens, timestamps = [], []
    prev_state = (s1, s2)
    t, emitted = 0, 0
    while t < enc_len:
        logits, step, state = _decode(tokens, prev_state, encodings[t])
        token = int(logits.argmax())
        if token != BLANK_IDX:
            prev_state = state
            tokens.append(token)
            timestamps.append(t)
            emitted += 1
        if step > 0:
            t += step
            emitted = 0
        elif token == BLANK_IDX or emitted == MAX_TOKENS_PER_STEP:
            t += 1
            emitted = 0
    return tokens, timestamps


# --------------------------------------------------------------------------- #
# Per-step joint-output golden: numpy prednet+joint vs ONNX over a trajectory.
# --------------------------------------------------------------------------- #
def validate_joint_golden(W: dict, encodings: np.ndarray, enc_len: int):
    import onnxruntime as rt

    sess = rt.InferenceSession(str(DECODER_ONNX), providers=["CPUExecutionProvider"])
    s1 = np.zeros((2, 1, HIDDEN), np.float32)
    s2 = np.zeros((2, 1, HIDDEN), np.float32)

    # Walk a real trajectory of (token-history, enc_t) and compare full 8198
    # logits AND the prednet hidden state numpy-vs-onnx at each step.
    rng = np.random.default_rng(0)
    state = init_state()
    onnx_state = (s1, s2)
    prev_token = BLANK_IDX
    rels = []
    state_rels = []
    n_steps = min(enc_len, 24)
    for t in range(n_steps):
        enc_t = encodings[t]
        # numpy
        pred_np, new_state = prednet_step(W, prev_token, state)
        logits_np = joint(W, enc_t, pred_np)
        # onnx
        outputs, n1, n2 = sess.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": enc_t[None, :, None],
                "targets": np.array([[prev_token]], np.int32),
                "target_length": np.array([1], np.int32),
                "input_states_1": onnx_state[0],
                "input_states_2": onnx_state[1],
            },
        )
        logits_onnx = np.squeeze(outputs)
        rels.append(_rel_l2(logits_np, logits_onnx))
        # prednet hidden state golden (layer-1 h is the pred output)
        state_rels.append(_rel_l2(new_state[0], np.squeeze(n1)))
        # advance both with a sampled token to exercise embedding+LSTM
        prev_token = int(rng.integers(0, VOCAB_SIZE))
        state = new_state
        onnx_state = (n1, n2)
    return float(np.max(rels)), float(np.mean(rels)), float(np.max(state_rels))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    d = np.linalg.norm(a - b); n = np.linalg.norm(b)
    return float(d / n) if n > 0 else float(d)


def encode_clip(wav_path: Path):
    """Real oracle front-end: wav -> preprocessor -> encoder -> [T,1024]."""
    import wave
    import onnxruntime as rt

    w = wave.open(str(wav_path), "rb")
    n = w.getnframes()
    raw = w.readframes(n)
    wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    wav = wav[None, :]
    wl = np.array([wav.shape[1]], np.int64)
    pre = rt.InferenceSession(str(PREPROC_ONNX), providers=["CPUExecutionProvider"])
    feats, flens = pre.run(["features", "features_lens"],
                           {"waveforms": wav, "waveforms_lens": wl})
    enc = rt.InferenceSession(str(ENCODER_ONNX), providers=["CPUExecutionProvider"])
    out, elen = enc.run(["outputs", "encoded_lengths"],
                        {"audio_signal": feats, "length": flens})
    # encoder output is [1,1024,T]; decoder consumes [T,1024]
    return out[0].T.copy(), int(elen[0])


def dump_weights(W: dict, out_dir: Path):
    """Write decoder weights as .npy so the Rust host port (decoder.rs) can load them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embed.npy", W["embed"])
    for li, layer in enumerate(W["lstm"]):
        np.save(out_dir / f"lstm{li}_W.npy", layer["W"])
        np.save(out_dir / f"lstm{li}_R.npy", layer["R"])
        np.save(out_dir / f"lstm{li}_B.npy", layer["B"])
    np.save(out_dir / "enc_W.npy", W["enc_W"])
    np.save(out_dir / "enc_b.npy", W["enc_b"])
    np.save(out_dir / "pred_W.npy", W["pred_W"])
    np.save(out_dir / "pred_b.npy", W["pred_b"])
    np.save(out_dir / "joint_W.npy", W["joint_W"])
    np.save(out_dir / "joint_b.npy", W["joint_b"])
    print(f"[weights] dumped decoder npy -> {out_dir}")


def load_vocab():
    vocab = {}
    for line in VOCAB.read_text(encoding="utf-8").splitlines():
        tok, idx = line.rsplit(" ", 1)
        vocab[int(idx)] = tok.replace("▁", " ")
    return vocab


def detok(vocab, ids):
    import re
    s = "".join(vocab[i] for i in ids)
    return re.sub(r"\A\s|\s\B|(\s)\b", lambda m: " " if m.group(1) else "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--gate", type=float, default=0.08)
    ap.add_argument("--clip", default="artifacts/wer_clips/en_01.wav",
                    help="wav clip for a real non-trivial decode; '' uses the encoder ref npy")
    ap.add_argument("--dump-weights", action="store_true",
                    help="extract decoder weights to artifacts/parakeet/decoder/weights/ for the Rust port")
    args = ap.parse_args()

    W = load_weights()
    if args.dump_weights:
        dump_weights(W, ROOT / "artifacts/parakeet/decoder/weights")
    if args.clip:
        encodings, enc_len = encode_clip(ROOT / args.clip)
        print(f"real clip {args.clip}: T={enc_len} d={encodings.shape[1]}")
    else:
        enc = np.load(ENC_REF)              # [1,1024,32]
        encodings = enc[0].T.copy()        # [T,1024]
        enc_len = encodings.shape[0]
        print(f"encoder ref: T={enc_len} d={encodings.shape[1]}")

    # 1) per-step joint golden
    rmax, rmean, srmax = validate_joint_golden(W, encodings, enc_len)
    print(f"[golden] joint logits rel-L2: max={rmax:.3e} mean={rmean:.3e}")
    print(f"[golden] prednet state rel-L2: max={srmax:.3e}")
    joint_pass = rmax <= args.gate

    # 2) greedy token-sequence parity
    tok_np, ts_np, ncalls = tdt_greedy_decode(W, encodings, enc_len)
    tok_or, ts_or = onnx_oracle_decode(encodings, enc_len)
    seq_pass = tok_np == tok_or
    print(f"[greedy] numpy tokens ({len(tok_np)}): {tok_np}")
    print(f"[greedy] oracle tokens ({len(tok_or)}): {tok_or}")
    print(f"[greedy] sequence parity: {seq_pass}  joint_calls={ncalls} for T={enc_len}")

    vocab = load_vocab()
    print(f"[greedy] numpy text : {detok(vocab, tok_np)!r}")
    print(f"[greedy] oracle text: {detok(vocab, tok_or)!r}")

    # save golden artifact
    out = ROOT / "artifacts/parakeet/decoder/refs"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "tdt_decode_golden.npz",
             tokens=np.array(tok_or, np.int64),
             timestamps=np.array(ts_or, np.int64),
             encoded=encodings)
    print(f"[golden] saved {out/'tdt_decode_golden.npz'}")

    if args.bench:
        # per-token cost: time the full greedy decode, report ms/joint-call
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            tdt_greedy_decode(W, encodings, enc_len)
        dt = (time.perf_counter() - t0) / N
        print(f"[bench] full decode {dt*1e3:.3f} ms  ({ncalls} joint-calls)")
        print(f"[bench] per joint-call (prednet+joint GEMVs): {dt/ncalls*1e3:.4f} ms")

    ok = joint_pass and seq_pass
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(joint rel-L2 {rmax:.3e} <= {args.gate}: {joint_pass}, seq parity: {seq_pass})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
