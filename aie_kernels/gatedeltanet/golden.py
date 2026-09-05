#!/usr/bin/env python3
"""Numpy golden for the gated delta-rule linear-attention recurrence brick
(gatedeltanet.cc, op-TYPE `recurrence{delta,gated}`).

Math (single-scalar-gate Gated DeltaNet member -- see the header comment in
gatedeltanet.cc for the derivation and citation):

  State S_t in R^{DK x DV}, alpha_t in (0,1] (decay), beta_t in (0,1) (write
  strength) -- both ALREADY activated (this brick does not do the
  sigmoid/softplus itself; that is its own elementwise op-type).

    pred_t   = S_{t-1}^T k_t
    err_t    = v_t - pred_t
    S_t[i,:] = alpha_t * S_{t-1}[i,:] + beta_t * k_t[i] * err_t
    o_t      = S_t^T q_t

Two models, mirroring the dwconv1d_k9 golden's host_reference/kernel_model
split:
  - host_reference: full fp32 recurrence on fp32 k/v/q/gates/state (the
    golden a "correct" implementation should hit).
  - kernel_model: bit-faithful model of gatedeltanet.cc -- bf16 k/v/q (single
    round on the way in, matching the resident-stream bf16 contract), f32
    gates/state (accfloat, carried un-rounded across steps), f32 accumulate
    for every read/write step, one bf16 round on the per-step OUTPUT store
    only (the state itself never gets bf16-rounded mid-recurrence).

GATE: rel-L2(kernel_model, host_reference) over the [T,DV] output sequence.
Unlike dwconv1d (a bounded K-tap FIR, error does not compound across t), this
IS a T-step compounding recurrence, so the bf16-input rounding error at each
step feeds into every subsequent step's state and therefore its own error
grows with T. This golden reports rel-L2 vs T explicitly (see main()) -- the
finding is exactly how fast that curve rises, which is what the later
on-device rel-L2 gate threshold should be set from, not a single fixed
number.
"""
import numpy as np
from ml_dtypes import bfloat16


def bf16(x):
    return x.astype(bfloat16).astype(np.float32)


def make_inputs(rng, T, DK, DV, decay_lo=0.90, decay_hi=0.999):
    # Keys L2-normalized (standard delta-rule convention -- keeps pred_t/err_t
    # well-scaled so the recurrence doesn't blow up over T).
    k = rng.standard_normal((T, DK)).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True) + 1e-6
    v = rng.standard_normal((T, DV)).astype(np.float32) * 0.5
    q = rng.standard_normal((T, DK)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-6
    alpha = rng.uniform(decay_lo, decay_hi, size=T).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=T).astype(np.float32)
    gates = np.stack([alpha, beta], axis=-1)  # [T,2]
    return k, v, q, gates


def host_reference(k, v, q, gates, DK, DV):
    """Full fp32 recurrence -- the golden."""
    T = k.shape[0]
    S = np.zeros((DK, DV), dtype=np.float32)
    out = np.zeros((T, DV), dtype=np.float32)
    for t in range(T):
        alpha, beta = gates[t]
        pred = S.T @ k[t]                      # [DV]
        err = v[t] - pred
        S = alpha * S + beta * np.outer(k[t], err)
        out[t] = S.T @ q[t]
    return out, S


def kernel_model(k_f32, v_f32, q_f32, gates, DK, DV):
    """Bit-faithful model of gatedeltanet.cc: bf16 k/v/q (rounded once on
    entry), f32 gates/state (accfloat, un-rounded across steps), f32
    intermediate math, single bf16 round on the per-step output only."""
    T = k_f32.shape[0]
    kb = bf16(k_f32)
    vb = bf16(v_f32)
    qb = bf16(q_f32)
    S = np.zeros((DK, DV), dtype=np.float32)  # s_in seed (all-zero for a fresh sequence)
    out = np.zeros((T, DV), dtype=np.float32)
    for t in range(T):
        alpha, beta = gates[t]
        pred = S.T @ kb[t]
        err = vb[t] - pred
        S = alpha * S + beta * np.outer(kb[t], err)
        ov = S.T @ qb[t]
        out[t] = bf16(ov)  # one bf16 round on store (matches aie::store_v<bfloat16,DV>)
    return out, S


def rel_l2(a, b):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))


def main():
    DK, DV = 32, 32
    rng = np.random.default_rng(0)

    print(f"[gatedeltanet] DK={DK} DV={DV}  rel-L2(kernel_bf16-io, host_fp32) vs T")
    print(f"  {'T':>5}  {'rel-L2':>10}   note")
    worst_small_T = None
    for T in (16, 32, 64, 128, 256, 512):
        k, v, q, gates = make_inputs(rng, T, DK, DV)
        ref, s_ref = host_reference(k, v, q, gates, DK, DV)
        ker, s_ker = kernel_model(k, v, q, gates, DK, DV)
        err = rel_l2(ker, ref)
        s_err = rel_l2(s_ker, s_ref)
        print(f"  {T:>5}  {err:>10.5e}   state rel-L2={s_err:.3e}")
        if T == 64:
            worst_small_T = err

    gate = 3e-2
    print(f"\n  GATE (T=64, the GDN_T default in gatedeltanet.cc) <= {gate}: "
          f"{'PASS' if worst_small_T <= gate else 'FAIL'} (rel-L2={worst_small_T:.5e})")

    # Emit a single T=GDN_T tile for the later on-NPU node harness (Tier B),
    # matching the dwconv1d golden's DWCONV_TILE_OUT convention.
    import os
    sp = os.environ.get("GDN_TILE_OUT")
    if sp:
        T = 64
        os.makedirs(sp, exist_ok=True)
        k, v, q, gates = make_inputs(rng, T, DK, DV)
        ref, s_ref = host_reference(k, v, q, gates, DK, DV)
        bf16(k).astype(bfloat16).tofile(os.path.join(sp, "k.bin"))
        bf16(v).astype(bfloat16).tofile(os.path.join(sp, "v.bin"))
        bf16(q).astype(bfloat16).tofile(os.path.join(sp, "q.bin"))
        gates.astype(np.float32).tofile(os.path.join(sp, "gates.bin"))
        np.zeros((DK, DV), dtype=np.float32).tofile(os.path.join(sp, "s_in.bin"))
        ref.astype(bfloat16).tofile(os.path.join(sp, "out_ref.bin"))
        s_ref.astype(np.float32).tofile(os.path.join(sp, "s_out_ref.bin"))
        print(f"  wrote T={T} tile (k/v/q/gates/s_in/out_ref/s_out_ref .bin) -> {sp}")


if __name__ == "__main__":
    main()
