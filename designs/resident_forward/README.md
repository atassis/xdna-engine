# resident_forward -- the gemma4-12b resident 48-layer forward, one command a token

Builds the design that runs Gemma-4-12B's whole decoder stack (attention + MLP + LM head, all 48
layers) as ONE NPU command per token, with weights and K/V resident on-chip between tokens. No
CPU round trip between layers.

## Layout

- `rlayer_design.py` (+ `rattnh.py`, `rattn2_design.py`, `rglobal.py`, `rhead.py`, `rseg_design.py`,
  `rsplitl.py`, `rforward.py`, `m1_design.py`, `m2_design.py`, `rmlp_design.py`, `attn_layout.py`,
  `kvring.py`) -- the design generator: one sliding/global layer image, its attention, MLP, LM-head
  and whole-forward (`fwd=`) runtime sequences, and the kernel .cc sources it compiles
  (`rf_norm_z.cc`, `attn_standin.cc`, `rf_attn_qk_*.cc`, `rf_attn_pv_*.cc`, `rf_split_glue.cc`,
  `rf_pv_native.cc`, `attn_exp2_table.npz`).
- `rf_build.py` -- builds one generator invocation (one ladder "part") to a full ELF via IRON's
  `compile_cxx_core_function` + `aiecc`. `kobj_cache.py` content-addresses the ~19 compiled kernel
  objects so a ladder's parts, which share the same kernel set, compile them once.
- `fwd_ladder.sh` -- the whole forward is too large to lower as one aiecc design under a sane
  memory cap, so it is built as several parts (`rf_build.py` invocations, each `emit=`-restricted
  to a subset of runtime sequences) and packed into one ELF by `fwd_pack.py`.
- `chain_ref.py`, `bfp16_model.py`, `mlp_ref.py`, `qkv_ref.py`, `o_ref.py`, `rmlp_ref.py`,
  `attn_ref.py`, `head_ref.py`, `grun.py` -- bit-level CPU references of the device arithmetic
  (chain GEMM accumulation, bfp16 rounding, attention), used to validate a build, not to produce
  one.
- `weight_store.py`, `stack_prep.py` -- the data side: pack the checkpoint's shipped int4 weights
  into one content-addressed store, then the store into the per-layer device weight streams the
  built design's `%w` argument expects.
- `recipes/rf48C.sh` -- the one-command build of the served design (`rf48C`, the full ladder above).
- `recipes/gemma4_data.sh` -- recreates every data-side input (checkpoint, hf_config, quantized
  weights, towers, store, per-layer streams) from the upstream checkpoint, stage by stage, each
  skippable once its output's `.recipe-manifest.json` matches.

## Prerequisites

- An IRON checkout (`IRON_DIR`) whose HEAD is at or near `recipes/rf48C.sh`'s `IRON_PIN` --
  required, no default (a different IRON tree changes the generated MLIR/ELF bytes).
- The toolchain instance `scripts/toolchain_up.sh` resolves for this repo's `toolchain.lock`
  (`RF_INST`, resolved automatically by `env.sh`; override only to pin a specific instance).
- `RF_FUSED_ATTN_DIR` if `IRON_DIR` does not yet carry `aie_kernels/aie2p/fused_attn.cc` (the served
  build used a second, slightly newer checkout of the same IRON fork for that one file --
  `rf_paths.fused_attn_dir()`).
- A disk-backed `RF_BUILD` dir (never `/tmp` -- see `scripts/require_disk_backed.sh`).

## Build

    IRON_DIR=/path/to/iron RF_BUILD=/path/to/build/dir designs/resident_forward/recipes/rf48C.sh

Writes `$RF_BUILD/rf48C/design.elf.zst` (the repo's `<name>.elf.zst`-only convention, see
`designs/decode_fused/elf_zst.py`) plus `pack.json` (control-code sizes, the ELF's uncompressed
`sha256`, and IRON/toolchain-instance provenance) and `gen_args.txt`.

## Data prep

    designs/resident_forward/recipes/gemma4_data.sh --out /path/to/artifacts

Stages: `checkpoint` (downloads google/gemma-4-12B-it-qat-q4_0-unquantized, ~24 GB) `hf_config`
`weights` (int4 g32 planar dump) `towers` `store` (content-addressed pack) `rf_stack` (per-layer
device streams). Each is skipped once built; `--adopt` blesses an existing unmanaged dir instead of
rebuilding it, `--force` rebuilds regardless.
