# conveyor_proto -- BD-on-chip MHSA attention conveyor (staged kernel sources)

Source-of-truth kernel + IRON generator for the **BD-on-chip** attention conveyor. These files are the
canonical copy (like the rest of `route_b_kernels/`); `scripts/conveyor_bd_prebuild.sh` copies them
forward into the disposable `mlir-aie/programming_examples/basic/conveyor_proto/` build sandbox before
each build. Always edit HERE, never the sandbox copy.

## What it is

A 4-stage FlashAttention-shaped functional pipeline per head (BD -> scores -> softmax -> ctx), replicated
data-parallel across heads. The 4th stage (BD) computes `BD = rel_shift((q + bias_v) @ p^T)` **on-chip**,
so the host no longer precomputes BD (the boundary that made the earlier host-BD conveyor a wall-clock
regression). Belt carries `qpv = qu || qv` per tile; `p`, `k`, `v` held resident per head; one 5-BO
dispatch `run_bd_conveyor(instr | qpv | p | k | v | ctx)`.

## Files

- `conveyor_attn.cc` -- the stage kernels. BD-on-chip bricks: `stage_bd*`, `bd_dot_block`,
  `bd_relshift_emit[_ta]`, `bd_emit_bake[_ta]`, `stage_scores_relpos_bd[_mask]`.
- `conveyor_attn_iron.py` -- the IRON generator. `--relpos-bd-onchip` builds the 4-stage columns;
  `--tactive-mask` wires the in-kernel `t_active` key-mask (variable-length clips); `--p-resident` is a
  device-iterated movement optimization (guarded off -- see the code comment).
- `Makefile` -- `make NPU2=1 VARIANT=attn BDON=1 MASK=1 ...` builds `build/final.xclbin` + `insts.bin`.
- `run_bd_onchip.py` -- standalone rel-L2 arithmetic gate vs a host relpos-MHA golden (gate <= 5e-3).

## t_active mask (why it exists)

BD-on-chip computes BD inside the kernel, so the host cannot pre-mask pad keys with a belt sentinel. The
`t_active` RTP register (int32[16], `use_write_rtp`, read as `rtp[0]`) drives the mask: the scores stage
nulls keys `j >= t_active`, and the BD emit stage uses `t_active` for the rel_shift base (so a MAX-T
xclbin serves any clip length `t <= T`). Default baked value is `t_active = T` (full-length passthrough);
the host patches it per dispatch for shorter clips.

## Build + gate

    scripts/conveyor_bd_prebuild.sh                 # builds H=4 BD-onchip xclbin (MASK=1 default)
    # standalone arithmetic gate (device):
    cd <sandbox>/conveyor_proto && python3 run_bd_onchip.py

Rust integration lives in `rust/npu-parakeet` (`relpos_mha_conveyor_bdonchip`, opt-in
`PARAKEET_CONVEYOR_MHA_BDONCHIP=1`).
