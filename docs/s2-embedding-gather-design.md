# S2 AR embedding gather: design (contract (b) from gather-rows.cc / s2-bd-gather-feasibility.md)

Companion to `docs/s2-ar-graph-map.md` (op graph, item "embedding / codebook-table gather --
NEW") and `docs/s2-bd-gather-feasibility.md` (which already answered the mechanism question:
a host-written, per-dispatch `offset_parameter` on an objectFifo `fill()` is proven on device
by `aie_kernels/_test/probe_bd_gather_offsets.py` and already ships in this repo
for the Whisper decode KV-write offset). This doc is the brick DESIGN built on top of that
answer: what the kernel/generator actually are, the byte arithmetic, and why the shape chosen
avoids a defect class this project has paid for twice already.

Tags: **READ** (found in a cited source file/line) or **DERIVED** (arithmetic from READ facts).

## 1. Table shapes -- confirmed, not re-guessed

| table | rows | D | source |
|---|---|---|---|
| `embeddings.weight` (tied LM head) | 155,776 | 2,560 | `vocab_size` `scripts/s2_ar_ref.py:321`, `embedding_length` `:322` |
| `codebook_embeddings.weight` | 40,960 (= `num_codebooks`10 x `codebook_size`4096) | 2,560 | `num_codebooks` `s2_ar_ref.py:330`, `codebook_size` `:329`; row count per `docs/s2-ar-graph-map.md:82-84` (`codebook_embeddings.weight[2560,40960]` ggml `ne`, numpy `[40960,2560]`) |
| `fast_embeddings.weight` | 4,096 | 2,560 | `codebook_size` reused (`docs/s2-ar-graph-map.md:121`); `fast_embedding_length` `s2_ar_ref.py:338` == `embedding_length` |

Gather call sites: `ARWeights.embedding_rows` (`s2_ar_ref.py:465-469`), `.codebook_embedding_rows`
(`:471-475`), `.fast_embedding_rows` (`:477-481`) -- all three are `block[ids - lo]`, a pure row
index, no compute. `s2_ar_ref.py:452-454`'s own docstring: "a forward pass over a short prompt
never materializes the full 155776x2560 / 40960x2560 tables". No local `.gguf` exists in this
worktree to re-run `gguf_extract.py` against (it can't decode these tensors anyway -- they are
q6_k on disk, `docs/s2-ar-graph-map.md:17-22`); these numbers are `read_ar_hparams`'s (`s2_ar_ref.py:373-392`)
GGUF-KV-driven defaults, already cross-checked against `s2_model.cpp` and the GGUF's own tensor
table per `s2-ar-graph-map.md`'s own verification pass (§7).

Table bytes (bf16, the host-side load-time dequant target `s2-ar-graph-map.md:39-45`
recommends): `rows * 2560 * 2`.

| table | bf16 bytes | vs 64 KB core L1 | vs 4 MB MemTile aggregate (8 x 512 KB, `AIETargetModel.h`) |
|---|---|---|---|
| `embeddings.weight` | 797,573,120 B (797.6 MB) | 12,168x over | 190x over |
| `codebook_embeddings.weight` | 209,715,200 B (209.7 MB) | 3,200x over | 50x over |
| `fast_embeddings.weight` | 20,971,520 B (21.0 MB) | 320x over | 5x over |

Every table, including the smallest, is over the FULL 8-column MemTile aggregate -- not just
one core's L1. Contract (a) from `gather-rows.cc` (whole codebook resident) is categorically
inapplicable here; this is the numeric confirmation of what `s2-bd-gather-feasibility.md`
already established qualitatively. Contract (b) -- table stays in L3/DDR, one row crosses L1
per dispatch at a host-computed offset -- is the only option, and it is DERIVED here to be
size-independent: the mechanism is a register patch, not a transfer, so it costs the same
whether the table is 21 MB or 800 MB.

## 2. What actually crosses L1, per dispatch

One row, D=2560, is the unit that ever touches a core:

| dtype | row bytes | as a fraction of 64 KB L1 |
|---|---|---|
| bf16 | 5,120 B | 7.8% |
| f32 | 10,240 B | 15.6% |

Both fit trivially in isolation. The design question is not "does a row fit" -- it does, by a
wide margin, in either dtype -- it is what SHAPE of kernel call moves that row, because of a
specific, previously-paid-for toolchain defect, measured 2026-07-31: a loop with a fixed OR
runtime multi-iteration trip count
**inside an AIE kernel body** can silently miscompile on this toolchain pin, independent of
whether the per-iteration body is correct in isolation. `gelu-erf` and `sin` both failed this
way (not on their maths: `gelu-erf` 9.538e+00 -> 1.138e-03, `sin` 7.091e-01 -> 1.275e-04, fixed
by moving the loop's trip count OUT of the kernel body and into the objectFifo WORKER loop,
which the same KB entry's grid confirms is a different, unaffected call-site shape). This
project's own `prefill-attn` hit the same class from a different angle: "rows 0-5 clean, 6+
destroyed."

`gather-rows.cc`'s own kernel loops `T_TILE` times inside one call (`gather_rows.cc:118-133`)
-- exactly the flagged shape -- and its own header says as much: "This has NOT been
device-verified (no NPU access for this task)" (`gather_rows.cc:85-88`). That is not a defect
in this brick's design note; it is direct evidence that the shape this brick's kernel is about
to choose has an open, unresolved question mark on it elsewhere in this exact catalog, and this
design does not need to inherit that mark when the toolchain gives a cleaner path.

**Design decision: one N=16-wide bf16 vector copy per kernel call, zero loop inside the
kernel.** D=2560/16 = 160 chunks per row, and the repeat count of 160 lives in the objectFifo
WORKER loop (`for _ in range_(160): kern(...)` in IRON's Python-level design, compiled to ONE
call site) -- the shape the KB entry's own operating rule names as the fix ("let the objectFIFO
worker loop supply volume... this only has to cover the case ... where n_tiles==1 ... leaves
cross-dispatch rotation as the only path", `bricklib.py:660-664`, matching this exact
construction). N=16 is not a new choice: it is this catalog's established bf16 row-vector width
(`rmsnorm.cc:79`, `cast_quant_bf16_int8.cc:120/124/127/130`, `swiglu.cc:74`, all instantiate at
N=16 for a bf16-or-f32 row).

Byte arithmetic for the chosen shape (chunk_n=16, objectFifo depth 2, both in and out fifos):

    2 * (16*2 + 16*2) = 128 B  ->  0.20% of 64 KB L1

L1 occupancy is not the constraint for this design at all -- it is trivially small either way.
The constraint that decided the shape is the internal-loop hazard above, not capacity.

For contrast, the REJECTED "whole row, one call" shape (matching `gather-rows.cc`'s own
T_TILE-loop pattern, D=2560 as one tile):

| dtype | depth-2, in+out, D=2560 whole-row tile | vs 64 KB L1 |
|---|---|---|
| bf16 | `2*(5120+5120)` = 20,480 B | 31.25% |
| f32 | `2*(10240+10240)` = 40,960 B | 62.5% (23.4 KB left for code+stack+spill) |

Both would still FIT L1. Both are rejected on the internal-loop-miscompile risk, not on
capacity -- worth stating plainly since "does it fit" is the wrong question for this decision.

**Open, not closed, and named as a follow-on, not assumed:** whether a whole-row internal loop
is actually unsafe for a body that carries NO state across iterations (this copy has none --
each 16-wide chunk is independent, no accumulator). A probe over a similarly stateless
per-chunk body (abs -> min) found it bit-exact at 1/2/4/8 chunks, which is evidence AGAINST a
blanket "any internal loop is unsafe" reading. This design does not rely on that evidence
holding, because it does not need to: pushing chunking into the worker loop costs nothing (the
worker loop is IRON's own proven-safe construct) and sidesteps an argument that is still
unexplained.

## 3. The `offset_parameter` alignment hazard -- checked, not assumed

Measured 2026-08-25: a runtime `*_offset_parameter` is in ELEMENT units of the bound objectFifo's own
dtype; the firmware scales by `elemBytes` and patches a BD base-address register that addresses
32-bit words. A resulting byte offset not divisible by 4 does not error -- it silently
TRUNCATES the low bit(s), landing a whole correct row in the WRONG slot with no crash and no
error. The static path has a compile-time check for this (`AIEXDialect.cpp:653`, `offset % 4
!= 0` -> hard error); the runtime scratchpad path has none.

Byte offset for row `idx` at element width `elem_bytes`: `idx * D * elem_bytes`. For this to be
a multiple of 4 for EVERY possible `idx` (not just even ones), the row STRIDE itself must be:
`(D * elem_bytes) % 4 == 0`.

- bf16 (`elem_bytes=2`), D=2560: `2560*2 = 5120`; `5120 % 4 == 0` (`5120/4 = 1280` exactly).
  Holds for every row index, on all three AR tables (all share D=2560). No truncation risk.
- General case: this only requires `D` even, since `elem_bytes=2` fixes the factor-of-2 half.
  D=2560 is even; this is a property of the checkpoint's `embedding_length`, not a general
  guarantee -- the generator asserts it (`(d * 2) % 4 == 0`) rather than assuming it, per this
  project's "hanging numbers are bugs" rule.
- f32 (`elem_bytes=4`): `D*4` is a multiple of 4 for ANY integer D -- if a future caller
  dequantizes to f32 instead of bf16, this specific hazard disappears entirely regardless of D.
  Recorded here because it is a real, checkable fact, not because f32 is the recommended format
  (bf16 remains the recommendation per `s2-ar-graph-map.md:39-45`).

## 4. What the brick does and does not do

**Does:** one dispatch = one row read from an L3-resident `[n_rows, D]` bf16 table at a
host-written offset, delivered via `chunk_n=16` grouped objectFifo tiles patched with ONE
`offset_parameter` covering the whole row (see `gen_embedding_gather.py`'s docstring for why a
single grouped-tap `fill()` call correctly offsets all 160 chunks, not just the first -- the
patch lands on the BD's base-address register once; the group tiler's own strides walk from
there, an orthogonal mechanism).

**Does not:** clamp the index. `gather-rows.cc` bakes an unconditional `[0,n_rows)` clamp into
the KERNEL because the kernel computes its own row pointer from an index it reads off a
streamed tile. This kernel never sees an index -- the DMA has already selected the row before
the core runs, via a host-computed offset. So an out-of-range `idx` here is a silent OOB L3
read with **no chance for the kernel to intervene**; clamping is the HOST DRIVER's
responsibility (`verify_embedding_gather.py`'s test driver does this; a production Rust driver
must too). This also happens to match the actual AR data: `s2_ar_ref.py:628`'s codebook-index
construction (`ids = np.where(is_semantic, raw_ids + cb*codebook_size, cb*codebook_size)`)
never produces an out-of-range index by construction, unlike the RVQ codec's `clamp_code`
lambda that `gather-rows.cc` matches.

**Does not:** batch multiple independent rows into one dispatch. A slow-AR step needs up to 11
row gathers (1 embedding + up to 10 codebook rows, `s2-bd-gather-feasibility.md`'s own count);
`probe_bd_gather_offsets.py` Part 2 proved T=2 independent `offset_parameter`s can fire in one
dispatch, up to the 32-slot state-table cap (`AIEX.td:1091`). This brick issues one dispatch
per row (the mechanism Part 1 proves, the lower-risk of the two per that doc's own ranking).
Batching is a named, unbuilt follow-on, not assumed here.

**Does not:** integrate with the Rust weight-arena loader or the AR driver loop. This is the
brick in isolation, gated against a synthetic table, matching `gather-rows.cc`'s own scope.

## 5. Combination not previously exercised in this repo -- named honestly

`gen_embedding_gather.py`'s `fill(table, tap=<TensorTiler2D.group_tiler(...) result>,
offset_parameter=row_off)` combines two mechanisms each independently proven on this toolchain
pin -- `bricklib._build_streamed`'s multi-tile grouped `tap` (every green brick in this catalog)
and `probe_bd_gather_offsets.py`'s single-tile `offset_parameter` (device-confirmed) -- but
**not proven together**. `ObjectFifoHandle.fill`/`._emit_transfer`
(`mlir-aie/python/iron/dataflow/objectfifo.py:804-839,692-700`) accept `tap` and
`offset_parameter` as independent, orthogonal keyword arguments feeding the same `DMATask`, and
the offset patches the BD's base-address register once while the tap's own strides govern the
multi-chunk walk from that base -- structurally independent mechanisms, which is why this
composition is expected to work. `verify_embedding_gather.py` is the test of exactly this
composition on real hardware; nobody has run it before this task.

## 6. Checked without the device (2026-09-02) -- what this rules out and what it does not

No device access for this task, per the owning session's exclusivity. Everything below is
compile-time/IR-generation only -- no `pyxrt.device()`, no `hw_context`, no `run.start()`.

- `embedding_gather.cc` compiles clean under the pinned Peano (`compile_check.sh`), confirming
  `event0/event1`/`aie::load_v`/`aie::store_v` on `bfloat16` at `GATHER_CHUNK_N=16` is valid
  AIE2P C++ on this toolchain instance.
- `gen_embedding_gather.build_design()` runs against the CURRENT `toolchain.lock` pin
  (`435f2cbb`, instance `f91864accc81`, resolved fresh via `scripts/toolchain_up.sh` -- this
  worktree's own `.venv-iron` resolved a DIFFERENT, presumably stale instance by default, so
  `PYTHONPATH` was forced explicitly rather than trusting the venv's own `aie` import) and
  emits valid MLIR at both a toy shape and the real D=2560/n_rows=64 shape, with a worker loop
  trip count of exactly 160 (`D // GATHER_CHUNK_N`) and `offset_parameter = @row_off` attached
  to the INPUT `dma_bd` only, never the output -- both match the design as specified above, not
  assumed.
- The generated MLIR, at the real D=2560 shape, **compiles all the way to a working AIE2P ELF**
  (`aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters`, 43/43 steps green,
  `aie.elf` + `params.txt` containing `row_off 0 i32 addr`). This is the strongest check
  available without a device: placement, address materialization, the core's own IR->object
  compile (containing the worker loop, confirmed to contain no in-kernel loop by inspecting the
  emitted `scf.for` structure), CDO/BIF/PDI packaging, and final ELF assembly via `aiebu-asm`
  all accept this design with zero errors. It does not prove the DMA moves the right bytes at
  the right offset -- only a real dispatch (`verify_embedding_gather.py`, on device) proves
  that -- but it retires "does this even lower" as a source of doubt.
- Two corrections to prior material found while doing this, both because the toolchain moved
  since they were written, not because they were wrong when written:
  1. `probe_bd_gather_offsets.py`'s `_compile()` docstring claims `--no-xchesscc --no-xbridge`
     mirror mlir-aie's own `scratchpad_addr_offset` RUN line verbatim. Neither flag exists in
     the aiecc built from the CURRENT pin (`aiecc --help` has no such options -- Peano is
     already the default backend, so there is nothing to negate) and passing them is a hard
     error. The RUN line as checked out at this pin (`test/python/npu-xrt/
     scratchpad_addr_offset/test.py:9`) has never had them either.
  2. A raw `aiecc <file>.mlir` CLI call does **not** compile an `ExternalFunction`'s source --
     that staging is normally done by `iron.jit`'s build orchestration
     (`mlir-aie/python/iron/kernel.py:293-295`). Bypassing `iron.jit` (required here, since
     `offset_parameter` dispatch needs the raw pyxrt path) means the caller must compile
     `embedding_gather.cc` to `embedding_gather_chunk_bf16.o` and stage it next to `aie.mlir`
     itself, or linking fails with `cannot open ... embedding_gather_chunk_bf16.o`.
     `verify_embedding_gather.py`'s `_compile()` does this.

Both corrections are folded into `verify_embedding_gather.py` itself, which is why its `_compile`
differs from the probe's in exactly these two ways.
