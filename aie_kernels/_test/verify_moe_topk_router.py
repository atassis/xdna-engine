#!/usr/bin/env python3
"""Device-verify config for the moe-topk-router brick (brick-wave1-device-verify).

BRICK: moe-topk-router (moe_topk_router.cc). Fused MoE gate GEMM + top-K expert
select: logits[EXPERTS] = hidden[1,HIDDEN] @ Wg[HIDDEN,EXPERTS] (bf16 in, f32 acc);
(top_values[K], top_indices[K]) = topk(logits, K) via iterated argmax (selection
sort). Same gemm+argmax skeleton as lm-head-argmax, N=EXPERTS instead of N=VOCAB,
extended from k=1 to k=K.

Entry:  extern "C" void moe_topk_router(const bfloat16* hidden,
                                        const bfloat16* gate_weight,
                                        float* top_values, int32_t* top_indices)

Two-output packing gotcha
-------------------------
bricklib.verify_oneshot has ONE output buffer, but the kernel co-produces
top_values (K f32) AND top_indices (K int32). The verify-shim packs BOTH into one
int32[2*K] buffer:  out[0..K-1] = bitcast<int32>(top_values[i]) ; out[K..2K-1] =
top_indices[i]. The host `unpack` views the first K back to f32 and reads the last
K as int32.

Gate (the authoritative correctness bar)
----------------------------------------
The PRIMARY gate is the selected-EXPERT-INDEX SET, not a raw rel-L2 on the packed
int32 buffer (bit patterns of f32 logits + integer ids are not an L2-meaningful
vector). unpack returns (indices, values); the decomposed authoritative gate is
  (a) set(device_indices) == set(reference_indices)   -- order may differ on ties
  (b) values rel-L2 (sorted desc) <= 3e-2             -- bf16-in / f32-acc slack
and is printed by `unpack`. verify_oneshot's own scalar rel-L2 (over the
[indices, values] pair vs the descending reference) is dominated by the index rows,
so it FAILS loudly on any index mismatch and PASSES when the set is right for the
tie-free random inputs used here -- a faithful proxy for gate (a). Gate (b) is
reported explicitly by unpack.

CPU-ONLY __main__: recomputes the reference top-K with the golden fns and, as a
strong recipe check, replays the kernel's exact tiled GEMM from the *packed* device
buffers in numpy (no device) -- validates BOTH packings. It never calls the device
do_ function.

Run the DEVICE gate under the NPU lock (do NOT import-time it):
    ./run.sh -c "import verify_moe_topk_router as m; m.do_moe_topk_router()"
"""
import importlib.util
from pathlib import Path
import numpy as np

BRICKS = Path(__file__).parent.parent
BRICK = "moe-topk-router"
CC = BRICKS / BRICK / "moe_topk_router.cc"

# Pure-buffer verify-shim: #included after the brick .cc (which pulls in aie_api,
# so `bfloat16` is a global type here). Packs the kernel's two outputs into one
# int32[2*MOE_TOPK] buffer. MOE_TOPK resolves to the .cc default (or a -D override).
SHIM_BODY = r'''
extern "C" void moe_topk_router_verify(const bfloat16 *hidden,
                                       const bfloat16 *gate_weight,
                                       int32_t *out) {
  float tv[MOE_TOPK];
  int32_t ti[MOE_TOPK];
  moe_topk_router(hidden, gate_weight, tv, ti);
  for (int i = 0; i < (int)MOE_TOPK; ++i) {
    int32_t bits;
    __builtin_memcpy(&bits, &tv[i], sizeof(bits));  // f32 logit -> raw int32 slot
    out[i] = bits;                                   // out[0..K-1]  = value bits
    out[(int)MOE_TOPK + i] = ti[i];                  // out[K..2K-1] = expert ids
  }
}
'''


def golden_mod():
    """Load the brick's numpy golden (pure numpy; no device import)."""
    p = BRICKS / BRICK / "golden.py"
    spec = importlib.util.spec_from_file_location("moe_topk_router_golden", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tile_pack(x, R, C):
    """[Rows,Cols] -> R x C block-tiled flat (block row-major, row-major in block).
    hidden A-operand layout: gemm loads load_v<MPAD*KT>(hidden + k*MPAD), so the
    [MPAD, HIDDEN] padded query packs with R=MPAD(8), C=KT(8)."""
    Rows, Cols = x.shape
    return np.ascontiguousarray(
        x.reshape(Rows // R, R, Cols // C, C).transpose(0, 2, 1, 3).reshape(-1))


def pack_gate_weight(W, K, N, KT=8, NT=8):
    """[K,N] gate weight -> the kernel's resident layout: N/NT expert column-tiles
    (nt-major), each [K, NT] with KT*NT contiguous per K-subtile. Mirrors
    gemm_expert_tile: tile nt at nt*(K*NT), subtile kb at kb*(KT*NT), elem (kk,nn)
    at kk*NT+nn. So W[kb*KT+kk, nt*NT+nn] -> offset nt*K*NT + kb*KT*NT + kk*NT + nn.
    Verified in __main__ by replaying the GEMM from the packed buffer."""
    return np.ascontiguousarray(
        W.reshape(K // KT, KT, N // NT, NT).transpose(2, 0, 1, 3).reshape(-1))


def do_moe_topk_router():
    """DEVICE verify config. Builds+runs the brick on aie2p via verify_oneshot and
    gates on the top-K expert-index SET (+ values rel-L2). DEVICE-run -- do NOT call
    at import/CPU time (triggers an iron.jit device build)."""
    import ml_dtypes
    import bricklib

    g = golden_mod()
    HIDDEN, EXPERTS = g.MOE_HIDDEN, g.MOE_EXPERTS
    TOPK, MPAD = g.MOE_TOPK, g.MOE_M_PAD
    # L1 fit: the gate weight streams as ONE [HIDDEN,EXPERTS] bf16 buffer; at
    # HIDDEN=256/EXPERTS=64 that is 32KB, and at objectFIFO depth 2 it overflows the
    # 64KB L1 bank (aiecc "allocated buffers exceeded available memory"). The brick is
    # -D-parameterized, so verify its numerics at a smaller, L1-fitting expert count
    # (still a valid top-2-of-16 MoE); EXPERTS=64 needs the gate weight tiled/streamed,
    # a separate dataflow concern, not a kernel-correctness question.
    EXPERTS = 16

    # Same inputs/scale as golden.py's self-check (small logits, gaussian).
    rng = np.random.default_rng(0)
    hidden_row = rng.standard_normal(HIDDEN).astype(np.float32) * 0.1
    gate_weight = rng.standard_normal((HIDDEN, EXPERTS)).astype(np.float32) * 0.1

    logits = g.gate_logits(hidden_row, gate_weight)           # [EXPERTS] f32
    ref_values, ref_indices = g.topk_selection_sort(logits, TOPK)  # descending

    # Device operand packing (bf16 host dtype = ml_dtypes.bfloat16).
    padded = g.make_padded_query(hidden_row, MPAD).astype(ml_dtypes.bfloat16)  # [MPAD,HIDDEN]
    gate_bf16 = gate_weight.astype(ml_dtypes.bfloat16)
    hidden_packed = tile_pack(padded, 8, 8)                   # A-tiled, MPAD*HIDDEN
    gate_packed = pack_gate_weight(gate_bf16, HIDDEN, EXPERTS)  # nt-major, HIDDEN*EXPERTS

    def unpack(dev_flat):
        flat = np.asarray(dev_flat).reshape(-1).astype(np.int32)
        values = flat[:TOPK].copy().view(np.float32)          # bitcast back to f32
        indices = flat[TOPK:2 * TOPK].astype(np.int32)
        # Authoritative decomposed gate (printed; the true correctness bar).
        set_ok = set(int(i) for i in indices) == set(int(i) for i in ref_indices)
        dv = np.sort(values.astype(np.float64))[::-1]
        rv = np.sort(ref_values.astype(np.float64))[::-1]
        vrl2 = float(np.linalg.norm(dv - rv) / (np.linalg.norm(rv) + 1e-12))
        verdict = "PASS" if (set_ok and vrl2 <= 3e-2) else "FAIL"
        print(f"  [moe-topk gate] index_set_match={set_ok} "
              f"dev_idx={indices.tolist()} ref_idx={ref_indices.tolist()} "
              f"values_rel_l2={vrl2:.3e} (gate 3e-2) -> {verdict}", flush=True)
        return indices, values

    # verify_oneshot's scalar rel-L2 compares (indices, values) vs the descending
    # reference; index rows dominate -> it enforces gate (a). gate (b) via unpack's
    # printed values_rel_l2.
    return bricklib.verify_oneshot(
        "moe-topk-router", str(CC), SHIM_BODY, "moe_topk_router_verify",
        inputs=[(hidden_packed, ml_dtypes.bfloat16), (gate_packed, ml_dtypes.bfloat16)],
        out_numel=2 * TOPK, out_shape=(2 * TOPK,),
        unpack=unpack,
        golden=(ref_indices.astype(np.float64), ref_values.astype(np.float64)),
        gate=3e-2, out_dt=np.int32,
        compile_flags=[f"-DMOE_HIDDEN={HIDDEN}", f"-DMOE_EXPERTS={EXPERTS}",
                       f"-DMOE_TOPK={TOPK}", f"-DMOE_M_PAD={MPAD}"])
do_moe_topk_router.brick_name = "moe-topk-router"


def _sim_gate_gemm_from_packed(hidden_packed, gate_packed, HIDDEN, EXPERTS, MPAD,
                               KT=8, NT=8):
    """Replay moe_topk_router's tiled gate GEMM (row 0) from the PACKED buffers,
    in f32 numpy -- validates tile_pack + pack_gate_weight without any device."""
    hp = np.asarray(hidden_packed, np.float32)
    gp = np.asarray(gate_packed, np.float32)
    logits = np.zeros(EXPERTS, np.float32)
    for nt in range(EXPERTS // NT):
        acc = np.zeros((MPAD, NT), np.float32)
        for kb in range(HIDDEN // KT):
            A = hp[kb * MPAD * KT: kb * MPAD * KT + MPAD * KT].reshape(MPAD, KT)
            base = nt * (HIDDEN * NT) + kb * KT * NT
            B = gp[base: base + KT * NT].reshape(KT, NT)
            acc += A @ B
        logits[nt * NT:(nt + 1) * NT] = acc[0]  # row 0 = live query
    return logits


if __name__ == "__main__":
    # CPU-ONLY cross-check. No device: golden reference + a numpy replay of the
    # kernel's tiled GEMM from the packed device buffers (validates the recipe).
    import ml_dtypes

    g = golden_mod()
    HIDDEN, EXPERTS = g.MOE_HIDDEN, g.MOE_EXPERTS
    TOPK, MPAD = g.MOE_TOPK, g.MOE_M_PAD

    rng = np.random.default_rng(0)
    hidden_row = rng.standard_normal(HIDDEN).astype(np.float32) * 0.1
    gate_weight = rng.standard_normal((HIDDEN, EXPERTS)).astype(np.float32) * 0.1

    logits = g.gate_logits(hidden_row, gate_weight)
    ref_values, ref_indices = g.topk_selection_sort(logits, TOPK)

    # (1) reference sanity: valid expert ids, finite + strictly descending values.
    assert ref_indices.shape == (TOPK,) and ref_values.shape == (TOPK,)
    assert np.all((ref_indices >= 0) & (ref_indices < EXPERTS)), "expert ids in range"
    assert len(set(ref_indices.tolist())) == TOPK, "selected experts distinct"
    assert np.all(np.isfinite(ref_values)), "values finite"
    assert np.all(np.diff(ref_values) <= 0), "values descending"
    assert ref_values[0] == logits.max(), "top value is the global max"

    # (2) recipe check: replay the tiled gate GEMM from the PACKED bf16 buffers.
    padded = g.make_padded_query(hidden_row, MPAD).astype(ml_dtypes.bfloat16)
    gate_bf16 = gate_weight.astype(ml_dtypes.bfloat16)
    hidden_packed = tile_pack(padded, 8, 8)
    gate_packed = pack_gate_weight(gate_bf16, HIDDEN, EXPERTS)
    sim_logits = _sim_gate_gemm_from_packed(hidden_packed, gate_packed, HIDDEN,
                                            EXPERTS, MPAD)
    # apples-to-apples: reference logits from the SAME bf16-rounded operands.
    ref_bf16_logits = (padded.astype(np.float32)[0] @ gate_bf16.astype(np.float32))
    num = np.linalg.norm(sim_logits - ref_bf16_logits)
    den = np.linalg.norm(ref_bf16_logits)
    pack_rl2 = float(num / den) if den else float(num)
    assert pack_rl2 < 1e-5, f"packed-GEMM replay must match bf16 reference, got {pack_rl2:.2e}"

    # top-K over the replayed logits must pick the same expert SET.
    sim_vals, sim_idx = g.topk_selection_sort(sim_logits, TOPK)
    assert set(sim_idx.tolist()) == set(ref_indices.tolist()), (
        "packed-replay top-K set must match the golden top-K set")

    print(f"packed-GEMM replay rel-L2 vs bf16 ref: {pack_rl2:.2e}")
    print(f"top{TOPK} expert ids={ref_indices.tolist()} values={ref_values.tolist()}")
    print(f"hidden{tuple(hidden_row.shape)} @ Wg{tuple(gate_weight.shape)} -> "
          f"logits[{EXPERTS}]; packed hidden={hidden_packed.size} "
          f"gate={gate_packed.size} out=int32[{2 * TOPK}]")
    print("verify_moe_topk_router.py __main__ cross-check OK: PASS")
