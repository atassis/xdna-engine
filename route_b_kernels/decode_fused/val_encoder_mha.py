#!/usr/bin/env python3
"""Validate the static-shape encoder MHA op against a NON-CAUSAL golden reference, ON THE NPU.

NOTE: iron/operators/mha/reference.py hardcodes is_causal=True, so it CANNOT validate the encoder's
bidirectional (causal=False) attention. This builds its own non-causal golden: SDPA over the unpadded
Q/K/V (S=1500), then pads O to seq_pad — matching what the kernel produces (it masks the padded KV
columns via S_kv_effective=1500). Same seed/layout as generate_golden_reference.

Run under the device serialization protocol (stop npu-asr/voxd, verify fuser empty).
"""
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import newstack_compat  # noqa: F401
from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.mha.reference import pad_to_multiple_of_64
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from gen_encoder_mha import StaticMHA, HEADS, D, SEQ

PIPELINES = 8


def noncausal_golden(heads, S, d, num_pipeline, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    val_range = 4
    Q = torch.rand(heads, S, d, dtype=torch.bfloat16) * val_range
    K = torch.rand(heads, S, d, dtype=torch.bfloat16) * val_range
    V = torch.rand(heads, S, d, dtype=torch.bfloat16) * val_range
    K_orig, V_orig = K.clone(), V.clone()
    inv_scale = 1 / np.sqrt(d)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        O = torch.nn.functional.scaled_dot_product_attention(
            Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0),
            dropout_p=0.0, is_causal=False, scale=inv_scale,
        ).squeeze(0)
    Q = pad_to_multiple_of_64(Q, seq_dim=1, num_pipeline=num_pipeline)
    K_orig = pad_to_multiple_of_64(K_orig, seq_dim=1, num_pipeline=num_pipeline)
    V_orig = pad_to_multiple_of_64(V_orig, seq_dim=1, num_pipeline=num_pipeline)
    O = pad_to_multiple_of_64(O, seq_dim=1, num_pipeline=num_pipeline)
    return {"Q": Q, "K": K_orig, "V": V_orig, "O": O}


def main():
    golden = noncausal_golden(HEADS, SEQ, D, PIPELINES)
    ctx = AIEContext()
    op = StaticMHA(num_heads=HEADS, seq_len=SEQ, d=D, num_KV_heads=0, causal=False,
                   num_of_pipelines=PIPELINES, context=ctx)

    inputs = {"Q": golden["Q"].flatten(), "K": golden["K"].flatten(), "V": golden["V"].flatten()}
    outputs = {"O": golden["O"].flatten()}

    errors, latency_us, bw = run_test(op, inputs, outputs, rel_tol=4.0e-2, abs_tol=1.5e-1)
    nO = int(golden["O"].numel())
    nerr = len(errors["O"])
    valid = SEQ * D * HEADS
    thresh = int(valid * 0.005)
    print(f"\n[val] latency_us={latency_us:.1f}  bandwidth={bw:.3e} GB/s")
    print(f"[val] tight-tol errors={nerr} / {nO} elems (valid {valid})  (upstream max allowable {thresh})")

    # --- rigorous precision check: capture raw NPU output, compute rel-L2 over the VALID region ---
    seq_pad = op._calculate_seq_padding(SEQ, PIPELINES)
    spec = op.get_arg_spec()
    op_func = op.get_callable()
    qb = XRTTensor.from_torch(inputs["Q"]); kb = XRTTensor.from_torch(inputs["K"])
    vb = XRTTensor.from_torch(inputs["V"]); ob = XRTTensor(spec[3].shape, dtype=spec[3].dtype)
    op_func(qb, kb, vb, ob)
    got = ob.to_torch().to(torch.float32).numpy().reshape(HEADS, seq_pad, D)[:, :SEQ, :]
    exp = golden["O"].to(torch.float32).numpy().reshape(HEADS, seq_pad, D)[:, :SEQ, :]
    diff = got - exp
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(exp) + 1e-12))
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())
    print(f"[val] VALID-region rel-L2={rel_l2:.5f}  max_abs={max_abs:.4f}  mean_abs={mean_abs:.5f}")
    # Localize outliers: per-seq-position max abs error (across heads,d). Where do the big errors live?
    perpos = np.abs(diff).max(axis=(0, 2))  # [SEQ]
    bigpos = np.where(perpos > 0.5)[0]
    print(f"[val] #elems abs>0.5: {(np.abs(diff) > 0.5).sum()}; #seq-positions with maxerr>0.5: {len(bigpos)}")
    if len(bigpos):
        print(f"[val] outlier seq-positions (min={bigpos.min()}, max={bigpos.max()}): first/last 12 = "
              f"{bigpos[:12].tolist()} ... {bigpos[-12:].tolist()}")
    # rel-L2 excluding the last KV block boundary (>=1472) to test the padding-edge hypothesis
    diff_core = diff[:, :1472, :]; exp_core = exp[:, :1472, :]
    rel_l2_core = float(np.linalg.norm(diff_core) / (np.linalg.norm(exp_core) + 1e-12))
    print(f"[val] rel-L2 over seq[:1472] (excl last block) = {rel_l2_core:.5f}")
    # rel-L2 < 0.02 is the accepted bar for bf16-emulated GEMM kernels in this repo (cf. o6 0.01045).
    ok = rel_l2 < 0.02
    print("[val] RESULT:", "PASS (rel-L2 < 0.02)" if ok else "FAIL (rel-L2 >= 0.02)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
