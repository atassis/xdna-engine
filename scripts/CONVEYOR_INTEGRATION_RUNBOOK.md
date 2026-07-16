# Conveyor -> Parakeet encoder integration: device runbook

Turnkey commands for the OWNER to run serially on the shared single-tenant NPU. This finishes the
8-head relpos-MHA conveyor integration whose host-side scaffold already landed on this branch
(`wt/conveyor-golden-prep`): the numpy BD-precision gate, the `relpos_mha_conveyor` host belt
packing, and the opt-in `PARAKEET_CONVEYOR_MHA` wiring. What remains is device-gated (xclbin build,
the dispatch wiring inside the TODO stub, and the WER/timing gate) and was deliberately NOT run here
because another human was using the box.

Paths are relative to the repo root (the checkout holding `scripts/`, `route_b_kernels/`, `artifacts/`).
`PY` below = the numpy/onnx venv, e.g. `~/npuvox-asr-bench/.venv/bin/python`.

## 0. Device hygiene preamble (EVERY on-device run)

The NPU (`/dev/accel/accel0`) is single-tenant and the toolchain checkout is shared. Before any build
or device run:

```
# announce in the shared channel first, then:
systemctl --user stop npu-asr npu-vox        # quiesce the background ASR/vox services
fuser /dev/accel/accel0                        # MUST print nothing (no other holder)
source scripts/iron_env.sh                      # fork toolchain env (place-tiles model, NOT the wheel)
python3 -c "import aie.iron" && echo IRON_OK    # confirm the fork instance is green
```

Build kernels/xclbins from the MAIN worktree env (the lever/worktree toolchain lives only in main).
When done, restart the services: `systemctl --user start npu-asr npu-vox`.

## 1. Numpy BD-precision gate (no device -- run anywhere, already validated on this branch)

```
PY scripts/conveyor_bd_precision_check.py
```

Verdict on this branch (block 0, T=32): PLAIN bf16 BD is sufficient (total ctx rel-L2 2.43e-3 == the
split-bf16 variant, ~2x under the 5e-3 bf16 gate; identical at the real operating point). So the belt
carries PLAIN bf16 BD by default. Flip only if the device WER (step 4) regresses vs 8.5:
`export PARAKEET_CONVEYOR_BD=split`.

## 2. Build the 8-head conveyor xclbin

Mirrors `relpos_prebuild.sh`. Real dims TQ=8 T=176 DK=128 N_QT=22 H=8 (172 padded to a VL multiple),
built by the validated recipe (grouped-MemTile split q+k, acquire-once weights, ctx JOIN, v-direct;
fork branch `conveyor-proto-real-dims`, example dir `mlir-aie/programming_examples/basic/conveyor_proto`).

```
# after the section-0 preamble:
scripts/conveyor_prebuild.sh          # -> artifacts/conveyor/single/{final.xclbin,insts.bin}
# FORCE=1 scripts/conveyor_prebuild.sh   # to rebuild
```

If the build errors, confirm the mlir-aie submodule is on the `conveyor-proto-real-dims` state and
that `KFLAGS` (`-DATTN_T=176 -DATTN_DK=128 -DATTN_SCALE=0.08838835f ...`) match the kernel `#define`s
in `conveyor_attn.cc`. The generator reads the `ATTN_*` dims from env; the kernel reads them via `-D`.

Toolchain note: the conveyor validated identically on the staged bump (fa85bb34 + local Peano) AND the
current pin (fb1f7095 + wheel Peano), bf16 exact-match. Prefer the current pin so no toolchain-bump WER
gate is coupled in.

## 3. Wire + verify the dispatch (finish the TODO stub, then on-device golden)

`npu.rs::relpos_mha_conveyor` already packs the host belts (qu bf16 + BD_shifted carriage; group-major
GJ=4; k/v head-major). Finish the stub (marked `unimplemented!`): lazy-load+cache the xclbin (mirror
`relpos_block()` -> a `ConveyorK`), upload qb/kb/vb, ONE dispatch, then de-interleave `bo_ctx`
(per-group `[N_QT, gsz, TQ, DK]` -> transpose H<->N_QT -> per-head `[T, DK]`, exactly
`run_conveyor_attn.py` lines 86-96).

Device sanity of the raw conveyor (random data, proves the xclbin + ABI):

```
cd mlir-aie/programming_examples/basic/conveyor_proto
ATTN_TQ=8 ATTN_T=176 ATTN_DK=128 ATTN_NQT=22 ATTN_HEADS=8 ATTN_RELPOS=1 \
  PY run_conveyor_attn.py conv        # expect per-head rel-L2 < 3e-2, prints ms/dispatch
```

Real-weights on-device golden (proves the encoder path == the shipped per-head relpos). Recommended:
add a `conveyor_parity` bin mirroring `src/bin/relpos_parity.rs` that feeds ONE real block's weights
through `relpos_mha_conveyor` and diffs ctx vs `npu.relpos_mha` per head. Gate rel-L2 <= ~5e-3.

```
cargo run --features npu --bin conveyor_parity      # (bin to be added; gate rel-L2 <= 5e-3)
```

## 4. PAYOFF gate: 17-clip WER + mhsa/encode wall-clock vs the shipped per-head loop

WER pipeline = dump mels (CPU frontend) -> Rust NPU encode -> TDT decode (CPU). 17 clips = 13 RU + 4 EN.
Baseline to beat/match: shipped WER 8.5 (MUST be <= 8.5). Run the shipped per-head path AND the
conveyor path, same mels, and compare WER + encode wall-clock.

```
# --- mels once ---
PY scripts/parakeet_npu_wer.py dump-mels artifacts/wer_mels

# --- A) shipped per-head relpos (baseline) ---
NPU_XCLBIN_ROOT=$PWD \
  cargo run --features npu --release --bin parakeet_encode_npu -- artifacts/wer_mels artifacts/wer_enc_shipped
PY scripts/parakeet_npu_wer.py decode-wer artifacts/wer_enc_shipped     # expect ~8.5

# --- B) 8-head conveyor (opt-in) ---
PARAKEET_CONVEYOR_MHA=1 NPU_XCLBIN_ROOT=$PWD \
  cargo run --features npu --release --bin parakeet_encode_npu -- artifacts/wer_mels artifacts/wer_enc_conveyor
PY scripts/parakeet_npu_wer.py decode-wer artifacts/wer_enc_conveyor    # gate: <= 8.5
```

Wall-clock A/B (phase timing splits out the `mhsa_conveyor` / `mhsa_resident` / host-score buckets):

```
PARAKEET_PHASE_TIMING=1 NPU_XCLBIN_ROOT=$PWD \
  cargo run --features npu --release --bin parakeet_encode_npu -- artifacts/wer_mels /tmp/enc_shipped
PARAKEET_PHASE_TIMING=1 PARAKEET_CONVEYOR_MHA=1 NPU_XCLBIN_ROOT=$PWD \
  cargo run --features npu --release --bin parakeet_encode_npu -- artifacts/wer_mels /tmp/enc_conveyor
# compare the per-clip encode wall time printed by each, and the mhsa bucket in the phase report.
```

Keep the conveyor OPT-IN until BOTH WER <= 8.5 AND encode-not-slower. Flipping the DEFAULT is an OWNER
gate. Restart `npu-asr npu-vox` when done.

## 5. Post-run bookkeeping

- If PLAIN BD held WER <= 8.5, record it and drop the `PARAKEET_CONVEYOR_BD=split` fallback note.
- Log the H=8 encode wall-clock vs the per-head loop as the conveyor payoff number.
- Open items still standing after this (see the integration/open-items handoffs): BD back on-chip
  (host BD precompute is a regression for the ~1-dispatch north-star), the q/k/v/linear_out projection
  boundary, the 4-tile out-projection fold.

Before committing any change under the public repo, run `scripts/check_no_private_refs.sh` (no
private-repo names / KB paths / absolute dev paths in tracked files).
