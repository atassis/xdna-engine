# S2 embedding gather: does AIE2P's DMA/BD engine support a runtime offset?

Companion to `docs/s2-ar-graph-map.md` sections 4/6, which already established WHAT this op
is (`get_rows`, an indexed row-gather from a huge table) and flagged it as "a genuinely new
MOVEMENT pattern, not a reparameterization of an existing one" without answering whether the
hardware/toolchain can actually express it. This doc answers that one question, from the
toolchain sources, with file:line citations, and ships a device probe
(`route_b_kernels/bricks/_verify/probe_bd_gather_offsets.py`) that confirms it on real
hardware.

Every claim below is tagged **READ** (found in a source file, cited) or **INFERRED**
(reasoned from READ facts, not itself read off a line). Toolchain paths are relative to the
`mlir-aie` fork repo root (e.g. `include/aie/Dialect/AIEX/IR/AIEX.td`); repo paths with no
prefix are this repo.

## The question

The S2 AR half needs three embedding-table gathers (`docs/s2-ar-graph-map.md:82-84,121`):

| table | shape | rows | q6_k on disk | bf16 resident (recommended, `s2-ar-graph-map.md:39-45`) | row size |
|---|---|---|---|---|---|
| `embeddings.weight` | `[2560, 155776]` | 155,776 | ~327 MB | **797.6 MB** | 5120 B |
| `codebook_embeddings.weight` | `[2560, 40960]` | 40,960 | ~86 MB | **209.7 MB** | 5120 B |
| `fast_embeddings.weight` | `[2560, 4096]` | 4,096 | ~8.6 MB | **21.0 MB** | 5120 B |

(q6_k figures are INFERRED from GGUF's ~6.5625 bit/element block format; the bf16 column is
the size after the host-side load-time dequant `s2-ar-graph-map.md:39-45` already
recommends and this repo's weight-arena loader ships resident to the device. All three tables
are far larger than the WHOLE array's aggregate MemTile capacity — READ, `AIETargetModel.h`;
this project's own doctrine already established that as 8 columns x 512 KB = 4 MB total, one
`getLocalMemorySize()` per MemTile row/column — let alone a single core's 64 KB L1
(`getLocalMemorySize() = 0x00010000` on the core-tile branch, `AIETargetModel.h:640`), so none
of these tables can ever be L1- or MemTile-resident, individually or all together. The gather
must be driven by the DMA/BD engine reading rows at a runtime-computed L3 offset.)

Per-dispatch row-gather count (READ, `s2-ar-graph-map.md:82-84,121,226`): a slow-AR step
gathers 1 embedding row plus up to 10 codebook rows (11 total, masked to whether the token is
semantic); a fast-AR frame calls `get_rows` 9 times against a growing 0-to-8-row prefix from
the 4096-row `fast_embeddings` table. **No call needs more than 11 independent row offsets.**

**Is this expressible at all?** `route_b_kernels/bricks/gather-rows/gather_rows.cc:57-60`
already flags it as open: "it is UNVERIFIED whether AIE2P's BD engine even supports
data-dependent offsets at all."

## What the sources say

### 1. The dialect ops have SSA-operand offset slots — READ

`aiex.npu.dma_memcpy_nd` (`include/aie/Dialect/AIEX/IR/AIEX.td:568-724`) declares its offsets
as BOTH a static attribute and a variadic SSA operand:

```
Variadic<I64>:$offsets, Variadic<I64>:$sizes, Variadic<I64>:$strides,
ConfinedAttr<DenseI64ArrayAttr, ...>:$static_offsets, ...
```
(`AIEX.td:656-662`). The op carries a dedicated verifier hook, `verifyDynamicSizesStrides`,
described as "Verifies the supported scope and bounds for a transfer carrying runtime (SSA)
sizes/strides" (`AIEX.td:706-710`; implementation `lib/Dialect/AIEX/IR/AIEXDialect.cpp:495`).

The core-level `aie.dma_bd` (`include/aie/Dialect/AIE/IR/AIEOps.td:871-1014`) mirrors this:
`offset`/`len` are each `Optional<I32>` SSA operands OR a static attribute
(`AIEOps.td:989-992`). The Python builder makes the split explicit: "a Python int becomes the
static attribute, an SSA Value becomes the runtime operand" (instance
`python/aie/dialects/aie.py:130-139`, `dma_bd()` wrapper at `142-184`).

**So the IR shape genuinely supports a runtime offset. This is not fictional or aspirational
in the dialect design.**

### 2. But the toolchain's own correctness harness says genuine data-dependent BD content is not shipped yet — READ

`test/Targets/NPU/static_vs_dynamic/README.md` is the harness that proves a compiled
`aie.runtime_sequence` produces byte-identical TXN streams whether its scalar fields are
compile-time constants or runtime arguments — i.e. it is the toolchain's own definition of
"this SSA-operand path actually works end to end." It states outright (`README.md:48-52`):

> Only `rtp_write` carries the runtime value: it is the most a runtime argument can drive
> without pushing the BD onto the per-register `write32` path, which would make the static
> and dynamic streams differ structurally rather than in value. **Genuinely runtime-valued
> DMA sizes/strides arrive with the Phase-2 dynamic BD-word encoder, which will extend this
> harness.**

This matches the fork's own `ROADMAP.md:17` (and `docs/ROADMAP.md:17`), which lists "Dynamic
runtime sequences" as an **unchecked** (`- [ ]`) item: "The core milestone (standalone TXN
encoding, AIEX->EmitC codegen, SSA-operand control ops, dynamic BD allocation, ...) is
merged; what remains is reworking the IRON `Runtime` Python API into an eager callback body."

Consistent with this, `docs/aie2p-brick-catalog.md:180-182` (this repo, written before this
investigation) already recorded: "BD-chain on-chip loop: the hardware brick exists, but
in-tree authoring is limited (npu-insts is flat / loop-incapable) -> exposing it is a
toolchain opportunity. The dynamic-runtime-sequences work upstream is already moving in this
direction." That prediction is now confirmed from the harness source directly.

**No example anywhere in the fork** (`programming_examples/`, `test/`) shows a compute core
reading a data value (e.g. a token id it just streamed in) and using it to reprogram a BD
field within the same dispatch. `programming_examples/basic/custom_dma/custom_dma.py`'s
`ScatterReadDMA` builds a 3-BD chain with three DIFFERENT offsets — but `offset_a`/`offset_b`/
`offset_c` are plain Python ints fixed at build time (`custom_dma.py:165-167`,
`offset_a = 0 * cols`, etc.), passed straight into `dma_bd(src_buf, offset=self._offset_a,
...)` (`custom_dma.py:130`). This demonstrates non-uniform **static** gather (several
different offsets in one chain), not data-dependent gather. `python/aie/helpers/taplib/tap.py`
confirms the same limit one layer up: `TensorAccessPattern.__init__`'s `offset` parameter is
typed as a plain `int` (`tap.py:30`, validated by `validate_offset` at `tap.py:44`) — matching
`gather_rows.cc:52-56`'s claim that "every design in bricklib.py drives a fixed,
index-INDEPENDENT TensorTiler2D access pattern."

### 3. The one runtime-offset mechanism that IS proven end-to-end — READ

`aiex.scratchpad_parameter` (`include/aie/Dialect/AIEX/IR/AIEX.td` area; generated Python
docstring at instance `python/aie/dialects/_aiex_ops_gen.py:348-388`) declares a named
parameter the host writes and the firmware copies into device registers:

> Parameters can alternatively offset BD addresses when used as the `offset_parameter`
> attribute in `aiex.dma_bd` and `aiex.dma_memcpy_nd`. ... If used as an address offset on a
> BD, the parameter is a multiple of the BD's element size.

The flow (`aiex.npu.create_scratchpad` docstring, `_aiex_ops_gen.py:3792-3821`, mirroring
`AIEX.td:1054-1094`):

```
[Host memory] --create_scratchpad--> [Command processor memory] --update_from_scratchpad--> [BD register]
```

`npu.update_from_scratchpad` (`AIEX.td:1097-1172`) is **always additive** — it adds a
StateTable-derived delta to whatever is already in the target 8-byte register pair — and the
StateTable is capped at **32 entries / 128 bytes total** (`AIEX.td:1091`,
`_aiex_ops_gen.py:3820`). 32 slots comfortably covers the AR gather's worst case of 11
independent row offsets per dispatch (section "The question" above).

This is not a paper mechanism. It is CI-gated in the fork today:
`test/python/npu-xrt/scratchpad_addr_offset/{aie_design.py,test.py}` (mirrored in
`test/npu-xrt/scratchpad_addr_offset/{aie.mlir,test.cpp}`) builds a passthrough kernel whose
input DMA start offset is controlled entirely by a host-written `ScratchpadParameter`, and
re-dispatches the SAME compiled ELF three times with three different offsets, asserting exact
output each time (`test.py:53-78`). And it is **already shipping in this exact repo**: the
per-token KV-cache-append offset `kv_off` used by `route_b_kernels/decode_fused/gen_decode.py:
193-198`, `gen_gemma_decode.py:123-128`, and gated by
`decode_fused/verify_fused_decode_sp.py:110-111,218` (`sp.write("kv_off", step * HD)`) is the
exact same `offset_parameter` mechanism, already proven on-device for a different per-token
runtime offset. `docs/aie2p-brick-catalog.md:102` records it: "RTP scratchpad | per-token
params, constant ELF | OK | kv_off + sm_mask, 2 words/token (replaced a 27MB ELF patch)".

**Correction to `gather_rows.cc:52-56`'s claim that "bricklib AS IT EXISTS TODAY CANNOT
EXPRESS THIS."** That is true of `bricklib.py` specifically (confirmed: it never passes
`offset_parameter` anywhere in `_build_streamed`/`_build_rowwise`/`_build_oneshot`,
`route_b_kernels/bricks/_verify/bricklib.py`). It is not true of the underlying IRON API one
layer down: `ObjectFifoHandle.fill()`/`.drain()` already accept `offset_parameter=` directly
(instance `python/aie/iron/dataflow/objectfifo.py:726-798`, kwarg at lines 732/769) — this
probe uses that layer directly rather than extending bricklib, precisely because bricklib's
`TensorAccessPattern`-only convention is the part that doesn't reach it, not the toolchain.

### 4. Adjacent mechanisms, and why they don't solve this alone — READ

- `aiex.npu.address_patch` (`AIEX.td:1006-1032`) patches a **host launch argument's** buffer
  address — known before the dispatch starts, same "host, not chip, decides" class as
  `offset_parameter`.
- `aiex.npu.push_queue` (`AIEX.td:779-806`) lets a runtime-sequence value pick **which**
  pre-built BD slot to launch (`bd_id`/`repeat_count` are SSA operands) — but the hardware BD
  table only holds 16 slots per core/shim tile or 48 per MemTile
  (`AIETargetModel.h`, `getNumBDs`, npu2 branch: `tileType == AIETileType::MemTile ? 48 : 16`).
  Nowhere near 155,776/40,960 possible rows — "N pre-built descriptors, one per possible row"
  is not viable standalone; it only helps combined with mechanism #3's content-patching, which
  collapses back to the same answer.
- `aiex.npu.writebd`/`npu.write32` (`AIEX.td:1196-1239`, `807-...`) are the raw, all-attribute
  fallback the static_vs_dynamic README calls "the per-register write32 path" — every field is
  a compile-time `I32Attr`, no SSA operands at all. This is the path a truly dynamic BD write
  would fall onto today, and it is host/runtime-sequence-issued, not core-issued.

## VERDICT: supported-with-caveats

A **host-written, per-dispatch (or per compile-time-unrolled-slot) offset** patched into a
`dma_bd`/`dma_memcpy_nd` via `aiex.scratchpad_parameter` + `offset_parameter=` **is
supported**, proven in upstream CI, and already shipping in this repo for an analogous
per-token offset (`kv_off`). It works against a table of arbitrary size in L3 — the mechanism
doesn't care whether the table is 32 bytes (the CI test) or 800 MB (the S2 embedding table);
only the row being read has to fit through L1 en route, and a 5 KB row does trivially.

A **device-computed offset — a value the chip derives from data it just read or produced,
fed into a BD field without any host write in between** — is **not supported today**. The
dialect's IR shape admits it (section 1), but the toolchain's own correctness harness says the
lowering for genuinely runtime-valued BD content is a named, unshipped "Phase-2" milestone
(section 2), and no example anywhere in the fork exercises it. This is the literal reading of
"UNVERIFIED whether AIE2P's BD engine even supports data-dependent offsets at all" from
`gather_rows.cc` — narrowed from "unverified" to "not supported as of this toolchain instance,
pending a named upstream milestone."

For the S2 gather specifically, this caveat is close to moot: the row indices being gathered
(token ids, codebook ids) are already host-visible at the point the AR loop issues the next
dispatch (they come from the previous step's sampled/argmax token, which the host reads back
to drive the greedy/sampling loop per `docs/s2-ar-graph-map.md:134-137`), so "the offset must
be known to the host before dispatch" costs nothing extra beyond what the AR loop already
does.

## Alternatives, ranked by cost

### A. Host-side gather + upload (cheapest for large/batched T, e.g. prefill)

Host reads the T needed rows directly (from its own copy of the table, or by reading back the
NPU-resident copy once at load time) and DMAs one dense `[T, 2560]` bf16 buffer to the device.

- Cost: `T * 5120` bytes over the host link — for a single decode step (T<=11) that's under 56
  KB, negligible next to this project's own measured per-dispatch overhead (`docs/
  aie2p-brick-catalog.md`'s "~91% inter-op dispatch overhead" figure for decode).
- Fits this project's own front-to-back doctrine's allowance for a host-uploaded **resident
  head** at the start of a pipeline segment.
- Downside: needs the table ALSO readable from the host per step. `s2-ar-graph-map.md:39-45`'s
  own recommendation is a single NPU-resident weight blob (matching this project's existing
  weight-arena loader), so this option either re-reads NPU LPDDR back to host each step (an
  extra round trip) or keeps a second host-RAM copy — a real cost, not free.

### B. Scratchpad-parameter-patched BD, per-row, NPU-side only (matches existing kv_off precedent, no host-RAM duplicate)

Host writes each row's byte offset (`idx * 2560 * sizeof(bf16)`, element-units per
`s2_ar_ref.py`'s / `kv_off`'s own convention) into a distinct `ScratchpadParameter` slot before
the dispatch; up to 32 slots / 128 bytes total (`AIEX.td:1091`), which comfortably covers the
AR gather's 11-row worst case in a single dispatch.

- Cost: T separate small BD executions (each ~5 KB), no batching across rows into one
  descriptor since the offsets are non-uniform — likely below this project's measured
  small-transfer efficiency floor (`resadd` at 6.0 GB/s, `docs/aie2p-brick-catalog.md`'s
  movement-floor numbers) per individual row, but the absolute bytes are tiny (11 rows x 5 KB
  = 55 KB) so the dominant cost is dispatch/BD-reprogram overhead, not bandwidth.
- Advantage over A: never leaves the NPU's own LPDDR — no host-RAM duplicate of an 800+200+21
  MB weight set, consistent with the single-resident-copy weight-arena loader this project
  already ships.
- `route_b_kernels/bricks/_verify/probe_bd_gather_offsets.py` Part 2 tests exactly this: T=2
  independent offsets patched and fired in ONE dispatch, to check whether the AR step's whole
  11-row gather can ride in a single xclbin call rather than needing 11 separate dispatches.

### C. N pre-built descriptors selected by `push_queue`'s runtime `bd_id` — ruled out standalone

The hardware BD table (16 slots/core-or-shim, 48/MemTile — `AIETargetModel.h`) is orders of
magnitude short of 155,776/40,960 possible rows. Only useful jammed together with B's
content-patching, at which point it IS B.

### D. Chunked full-table streaming (stream the whole codebook, mask-accumulate matching rows) — ruled out

Reading the entire 797.6 MB / 209.7 MB table to extract 11 rows of 5 KB each moves
~70,000-140,000x more bytes than needed. At this box's measured achievable LPDDR floor
(47-57 GB/s), just the semantic-embedding table read alone costs ~14-17 ms — worse than any
plausible per-dispatch overhead from options A/B by 2-3 orders of magnitude. Not viable except
as a correctness fallback of last resort.

**Ranking:** A and B are not mutually exclusive — A suits bulk gathers (prefill, where T can
be the whole prompt length and a single big host-computed upload amortizes well) while B suits
the steady-state single-token AR decode step (T<=11, NPU-resident-only, no host-RAM
duplicate). C only exists combined with B. D is ruled out.

## What the probe proves, and what it cannot

`route_b_kernels/bricks/_verify/probe_bd_gather_offsets.py`, run via
`cd route_b_kernels/bricks/_verify && ./run.sh probe_bd_gather_offsets.py`:

- **Part 1** (primary, low-risk): one `offset_parameter`-patched `dma_bd` reads a single row
  from a 256 KiB ramp table (4096 rows x 16 x i32, far past the 64 KiB L1), re-dispatched 3x
  with 3 different host-written offsets (row 0 / 2047 / 4095) against the SAME compiled ELF,
  no recompilation between runs. A wrong offset is unmistakable because row `r`'s data is `r`
  broadcast across the row. This is a scaled-up, gather-shaped variant of mlir-aie's own
  CI-gated `scratchpad_addr_offset` test.
- **Part 2** (secondary, exploratory): two INDEPENDENT `offset_parameter`s, each on its own
  `dma_bd`/tile pair, both patched by the host and fired in a single dispatch — a new
  combination not covered by any existing in-tree test, since each half (offset_parameter
  singly; multi-tile IRON designs) is proven separately but not together. This is the direct
  probe for "can the AR step's whole per-dispatch row set ride in one xclbin call."

**What it cannot determine:** whether a value computed ON THE CHIP mid-dispatch (e.g. a token
id sampled by an on-NPU argmax, with no host write in between) can drive a BD offset. No
mechanism to even attempt this exists anywhere in this toolchain instance (section 2) — that
question is closed by source reading, not by a device probe, and stays open pending the
toolchain's own named "Phase-2 dynamic BD-word encoder" milestone. It also cannot determine
real-world latency/throughput of options A vs B (that needs the actual S2 weight blob and a
timed run, out of scope for a feasibility probe).
