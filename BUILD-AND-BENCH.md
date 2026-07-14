# relpos_scores_softmax -- build + device-drive (step 1 of the resident MHA block)

Standalone rel-pos scores->softmax kernel. Given host-precomputed `AC[T,T]` f32
and `BD[T,P]` f32 (`P = 2T-1`), the NPU returns
`probs[T,T]` bf16 = `softmax_over_keys( rel_shift(BD) + AC , * 1/sqrt(DK) )`.
No matmul on device -- this de-risks the two hard rel-pos bricks (the
zero-arithmetic strided-relayout `rel_shift` + the vectorized-exp2 softmax)
before the full resident block is attempted.

- Kernel: `route_b_kernels/relpos_mha/relpos_mha.cc` -> `relpos_scores_softmax_bake`
  (thin zero-scalar wrapper baking `T`, `P`, `inv_scale=1/sqrt(128)` over the
  authored `relpos_scores_softmax`).
- Generator: `route_b_kernels/relpos_mha/relpos_scores_softmax_iron.py`
  (single core, 3-buffer ABI: AC in / BD in / probs out; `SequentialPlacer`).
- Makefile: `route_b_kernels/relpos_mha/Makefile` (synced into the mlir-aie
  sandbox by `scripts/sync_kernels.sh`; runs from there via `makefile-common`).
- Golden (CPU, no device): `scripts/parakeet_relpos_mha_golden.py` (G4a/G4b).
- Device runner: `scripts/run_npu_relpos_scores.py`.

`T = 32` (Parakeet block 0) is baked by default. **Single-tile design**: AC+BD+probs
share L1, so this de-risk variant is limited to small T (T=32 uses ~14 KB;
`RELPOS_TMAX=512` caps the per-row f32 scratch). Larger T needs the row-tiled
resident block, not this kernel. `T` MUST be identical in `make T=<n>` (it threads
both the generator `-T` and the kernel `-DRELPOS_T`).

## 0. Device discipline (NPU is single-tenant; serialize)

The NPU + the shared amd/IRON checkout are single-tenant. Before any timed/device
run: announce, quiesce `npu-serve`/`npu-asr`, and check nothing else holds the
device (`fuser`). Do NOT auto-restart services mid-run.

```bash
# from the PUBLIC checkout root (has the gitignored artifacts + toolchain):
cd $XDNA_ENGINE
fuser -v /dev/accel/accel0 2>&1 || true      # confirm the NPU is free
```

## (a) Build the xclbin from the main toolchain

The blessed FORK toolchain (`.venv-iron` + Peano + patched mlir-aie submodule)
lives ONLY in the main public checkout, NOT in the `xdna-engine-mha` worktree.
Build there. `sync_kernels.sh` copies the kernel .cc + generator + Makefile
FORWARD into the disposable mlir-aie sandbox (route_b_kernels is the source of
truth -- never edit the mlir-aie copy).

```bash
cd $XDNA_ENGINE
bash scripts/setup_route_b.sh            # idempotent: env + checkout + first sync
source scripts/iron_env.sh               # PEANO_INSTALL_DIR, MLIR_AIE_DIR, aiecc on PATH
bash scripts/sync_kernels.sh             # re-copy after any route_b_kernels edit

make -C mlir-aie/programming_examples/ml/relpos_mha NPU2=1 T=32
# -> mlir-aie/programming_examples/ml/relpos_mha/build/final.xclbin
# -> mlir-aie/programming_examples/ml/relpos_mha/build/insts.bin
```

Notes:
- If the kernel `.cc` changed but the `.o` is stale, `make clean` in that dir first
  (the wrapper bakes `T`/`inv_scale` at compile time, so a T change needs a rebuild).
- CPU-only step: `setup_route_b.sh`/`sync_kernels.sh`/`aiecc` touch no device.

## (b) Drive one head/block of AC+BD through it on device

The runner computes block-0 head-0 `AC[32,32]` f32 + `BD[32,63]` f32 from the
encoder weight artifacts, sends them (host_only BOs, group_ids 3/4), reads back
`probs[32,32]` bf16 (group_id 5). ABI: `opcode=3`,
`kernel(3, instr[gid1,cacheable], n_instr, AC[gid3], BD[gid4], PROBS[gid5])`.

Block-0 raw scores SATURATE (~one-hot softmax), which would only test rel_shift +
argmax. So by DEFAULT the runner pre-scales AC/BD by `1/std` host-side to land a
non-degenerate softmax that actually exercises the on-chip exp2 / bf16-reciprocal
path (the oracle uses the identical effective scale). Use `--raw` for the true
saturating regime.

```bash
cd $XDNA_ENGINE
.venv-iron/bin/python scripts/run_npu_relpos_scores.py \
    --xclbin mlir-aie/programming_examples/ml/relpos_mha/build/final.xclbin \
    --insts  mlir-aie/programming_examples/ml/relpos_mha/build/insts.bin \
    --block 0 --head 0
# add --raw to drive the saturating (one-hot) regime instead
```

(Use whichever python has system `pyxrt` visible -- `.venv-iron` is built
`--system-site-packages` for exactly this. `PARAKEET_ENC_DIR` overrides the
artifacts dir if not running from the public checkout.)

## (c) Compare to the golden (G3/G4)

CPU golden (no device) -- run from either checkout (falls back to the sibling
public artifacts if the worktree lacks them):

```bash
~/npuvox-asr-bench/.venv/bin/python scripts/parakeet_relpos_mha_golden.py
```

Expected CPU golden (verified this session):

```
G1  strided rel_shift == NeMo rel_shift : rel=0.000e+00  PASS
G2  f32 mirror (strided) == host mhsa   : rel=0.000e+00  PASS
G3  kernel bf16 model vs f32 host mhsa  : rel=4.235e-02  GATE<= 0.08  PASS
G4a standalone brick, real regime       : rel=0.000e+00  GATE<= 0.08  PASS  (worst of 8 heads; scores saturate -> ~one-hot)
G4b standalone brick, non-degenerate sm : rel=2.928e-03  GATE<= 0.08  PASS  (worst of 8 heads; exercises exp2 softmax)
```

Device PASS criteria (`run_npu_relpos_scores.py`): `rel-L2 <= 0.08` AND
`corr >= 0.99` vs the fp32 host softmax, with `probs` rowsums ~1.0. The host-side
numeric model of the device path was verified this session at rel-L2 ~2.5e-3
across all 8 heads (rescaled regime) and exactly 0 in the raw one-hot regime.

## What is device-gated (NOT run this session)

Everything above the device line is CPU-authored + numpy-validated. The device
gate (aiecc build of `final.xclbin` on the fork toolchain, and the pyxrt drive)
was deliberately NOT run -- the NPU is single-tenant and gated serially by the
orchestrator. First device action = step (a) `make ... NPU2=1 T=32`, then step (b).
