# Cascade-FFN Phase 0 -- bf16 sibling structure + Whisper-FFN port mapping

Task 2 (Steps 2-4) deliverable: the "own the primitive" structural map that Tasks 3 (kernel) and 4
(generator) consume. Read this before writing `mv_bf16_gelu.cc` or `ffn_cascade.py`.

All `air/` paths below are READ-ONLY references in the shared checkout, rooted at:
`~/mlir-air/programming_examples/` (abbreviated `PE/` here).
Our repo paths are absolute under `$REPO/`.

---

## 0. HEADLINE FINDING -- the named "PORT BASE" is NOT a cascade (verify result)

The task named `PE/llms/llama32_1b_int4/multi_launch_builder/o_ffn_bfp16_multi.py` as the bf16 cascade
sibling. It is **NOT a cascade**. Verified by reading it end-to-end:

- `o_ffn_bfp16_multi.py` is an **8-launch PREFILL GEMM stitcher** for Llama (seq_len=2048). It builds 8
  independent sub-kernels (O GEMM, Res Add, RMSNorm, Gate GEMM, Up GEMM, SwiGLU, Down GEMM, FFN Add) as
  separate IR strings and **text-concatenates** them into one `func.func` via
  `_extract_between_func_and_return` / `_fix_launch_func_args` (file lines 472-530). Intermediates
  (`proj`, `res1`, `normed2`, `gate`, `up`, `swiglu`, `down`) are **L3/DDR buffers** passed as func args
  arg2/arg4/arg6/arg8/arg10/arg11/arg13 (the 15-arg ABI, lines 314-327). There is **no `npu_cascade`,
  no `ChannelPut/Get`, no W->E shift** in this file -- it is dispatch-stitch, the very thing we are
  trying to beat. `grep -n "cascade\|npu_cascade" o_ffn_bfp16_multi.py` -> 0 hits.
- `grep -rln "npu_cascade\|cascade"` across `multi_launch_builder/` returns exactly ONE generator:
  `o_gemv_ffn_int4_fused.py` (the int4 cousin) + its `.lit`. No bf16 file in that dir has a cascade.

The real cascade lives in two places:

  (A) `PE/llms/llama32_1b_int4/multi_launch_builder/o_gemv_ffn_int4_fused.py` -- the int4 DECODE
      cascade the task described (3 herds LA/LGU/LD, two `npu_cascade` chains). int4-AWQ weights.
      Cascade is used for **M-ASSEMBLY** (concatenate per-core output slabs), NOT K-reduction.

  (B) `PE/matrix_vector_multiplication/bf16_cascade/` -- a generic **bf16** cascade GEMV primitive:
      `matvec_cascade.py` (plain), `matvec_cascade_add.py` (fused residual add `D = A@B + R`), kernel
      `mv_bf16.cc`. Cascade is used for **K-REDUCTION** (sum partial dot products down the cascade).
      THIS is bf16, already shaped like our fc2 (K-reduction + residual), and is the cleanest port base.

**Decision (not a blocker):** Phase 0 is NOT blocked. The mapping is viable and in fact cleaner than the
literal task framing: port the bf16 K-reduction cascade (B) for fc2, borrow the single-launch multi-stage
fusion idea from the int4 (A), and base the kernel on `bf16_cascade/mv_bf16.cc` (B) rather than the int4
packed kernel. Sections A/B below document (A) and (B); the int4 packed weights / SwiGLU / RMSNorm
machinery is dropped (we are bf16 + GELU + LayerNorm). The literal `o_ffn_bfp16_multi.py` is used only as
the bf16 numerics/`-Os`/legalization reference, not as the dataflow skeleton.

---

## SECTION A -- the cascade structures (cited)

Two cascade idioms exist. Our FFN uses the K-reduction idiom (A.2) for fc2; the int4 fused (A.1) is the
single-launch multi-stage chaining reference.

### A.1 -- int4 fused DECODE cascade (M-assembly): `o_gemv_ffn_int4_fused.py`

This is the file whose mechanism the task pre-described; confirmed accurate. It is int4, so it is a
STRUCTURAL reference only (we do not reuse its packed kernel).

Dim constants / where set (`build_module`, lines 172-228):
- `K=2048` (= emb_dim), `M_LA=2048`, `M_LGU=2*hidden=16384`, `K_LD=hidden=8192`, `GS=128` (int4 group
  size), `M_TILE=8`, `K_CHUNK=2048`, `N_LA=N_LGU=N_LD=8` cores. `assert K == K_CHUNK` (line 185) -> all
  stages link one `mv_int4_bf16.o` built at DIM_K=2048, DIM_M=8 (docstring lines 42-45).
- Derived: `M_la_per_core = M_LA/N_LA = 256` (207), `M_lgu_per_core=2048`, `half_M_per_core=1024`,
  `M_OUT = M_LGU/2 = 8192` (assembled swiglu), `K_LD_div = K_LD/K_CHUNK = 4` (218).

Herd layout (3 herds, explicit `x_loc`/`y_loc`, each `sizes=[N,1]` or 1x1 per-col):
- **LA** -- `N_LA` single-core herds `la_{col}` at `y_loc=4`, `x_loc=col` (lines 467-661, attrs 658-659).
  Computes O-proj + residual #1: per-core int4 matvec (`matvec_int4_bf16_packed_store`, 538-540) ->
  inline `partial + R` 8-wide bf16 add (544-571) -> scatter into `l1_local_la` at `global_off` (575-579).
- **LGU** -- one herd `lgu_h` `sizes=[N_LGU,1]` at `y_loc=2`, `x_loc=0` (663-956, attrs 954-956).
  Receives res1 + gamma; inline **RMSNorm** in-place into res1 (693-784); int4 gate/up matvec (809-837);
  vectorized **SwiGLU** (840-872) -> per-core 1024 slab.
- **LD** -- one herd `ld_herd` `sizes=[N_LD,1]` at `y_loc=3`, `x_loc=0` (959-1073, attrs 1069-1072).
  Receives 4 swiglu chunks + res1; int4 down matvec as K_LD=8192 reduction in K_CHUNK=2048 slices via
  `matvec_int4_bf16_packed_b_offset` (1016-1026) -> inline `partial + R` residual #2 (1030-1052) -> per-col
  `ldOutD` write to L3 (1063).

W->E cascade mechanism (M-ASSEMBLY -- this is the key int4 idiom):
- Channel decls: `chan_cascade_la` `size=[N_LA-1]`, `channel_type="npu_cascade"` (299);
  `chan_cascade_lgu` `size=[N_LGU-1]` (313). Two broadcast packet channels assembled from the eastmost
  core: `res1ToCons` `broadcast_shape=[N_LGU+N_LD,1]` (16 dests, 307-308) and `swigluToLd`
  `broadcast_shape=[N_LD,1]` (320-321).
- Per-core put/get/accumulate/forward (LA, lines 585-649): `is_first` (col 0) `ChannelPut("chan_cascade_la",
  l1_local, indices=[col])`; else `ChannelGet(..., indices=[col-1])` the assembled-so-far, **vector-add it
  into l1_local across the full M_LA** (605-637), then if `is_last` (col N-1) `ChannelPut("res1ToCons", ...)`
  broadcast to all consumers (640-642), else forward `ChannelPut("chan_cascade_la", ..., indices=[col])`
  (644-649). Each core wrote only its own M-slab (rest zero-filled, 490-503), so the cascade add
  effectively **concatenates** the 8 slabs into the full vector. Same idiom in LGU (882-946).
- Eastmost broadcast: LGU's last core slices the assembled `M_OUT=8192` swiglu into `K_LD_div=4` K_CHUNK
  chunks and `ChannelPut("swigluToLd", ...)` them FIFO on ONE packet channel (927-943) -- collapsed to 4
  chunks on one channel to stay under the 4-msel stream-switch multicast limit (comment 316-322).
- Rides the cascade vs touches L3: `res1` and `swiglu` ride the cascade/broadcast, **never L3**. Only L3:
  the three packed-weight inputs, the final `D_ld` output (1063), and a **deletable debug copy**
  `D_dbg`/`laResDebug` (40-41, 250-252, 309-310, 412, 640-642). DELETE the debug copy in our port.

### A.2 -- bf16 K-reduction cascade (THE port base for fc2): `matvec_cascade_add.py`

`PE/matrix_vector_multiplication/bf16_cascade/matvec_cascade_add.py` -- `D[M] = A[M,K]@B[K] + R[M]`, bf16
in/out, accfloat accumulate. This is our fc2 shape exactly (K-reduction + residual). Cleaner than the int4
fused: bf16, no packing, residual injected at the cascade head.

Dims / where set (`build_module(m, k, tile_m, m_input, herd_cols, n_cascade, ...)`, lines 89-107; defaults
in `__main__` 645-651): `M=2048, K=8192, TILE_M=2, M_INPUT=1, HERD_COLS=8, N_CASCADE=4`. Key split:
`k_chunk = k // n_cascade` (96). `herd_cols` partitions the **output M** (each col independent), `n_cascade`
partitions **K** (the reduction). Asserts: `n_cascade>=2` (94), `m % (tile_m*herd_cols)==0` (98),
`k % n_cascade==0`, `k_chunk % 64 == 0` (vector width, 105).

Herd layout: ONE herd `herd_0` `sizes=[herd_cols, n_cascade]` (342-344). The cascade runs along **ty**
(the `n_cascade` axis); `tx` (herd_cols) is the independent M-partition.

Cascade roles (per-(col,ty), lines 498-588) -- note the convention HEAD = ty==n_cascade-1, TAIL = ty==0:
- **HEAD** (`ty == last_ty`, 503-524): compute partial dot `A_slice @ B_slice` over this ty's k_chunk
  (`compute_partial_dot`, 46-86), `init_acc = partial + R` (R injected here, 510-513),
  `ChannelPut("chan_cascade", scratch, indices=[tx, ty-1])` (518-523).
- **MIDDLE** (else, 556-584): `ChannelGet("chan_cascade", recv, indices=[tx,ty])` (558-562), `total =
  recv + partial` (564-576), `ChannelPut("chan_cascade", scratch, indices=[tx, ty-1])` (578-583).
- **TAIL** (`ty == 0`, 530-554): `ChannelGet` cascade (533-537), `total = recv + partial`, truncf to bf16
  (548-549), write D (550-551). No R here (R added at head). Only ty==0 drains L1->L2->L3 (591-629).
- Channel decl: `chan_cascade` `size=[herd_cols, n_cascade-1]`, `channel_type="npu_cascade"` (203-207).
  Partials ride the cascade; only the final D touches L2/L3.

The dot product is computed INLINE in MLIR (`compute_partial_dot`, 46-86: bf16 load -> extf f32 -> `fma` ->
f32 reduce). **`matvec_cascade_add.py` links NO `.cc` kernel** (the cascade is pure MLIR vector ops). The
`.cc` kernel (`mv_bf16.cc`) is linked only by the SEPARATE `matvec_2tile_add.py` design (Makefile 102-120).
For our port we will link a `.cc` (faster matvec + the GELU epilogue), so Section A.3's kernel ABI applies.

### A.3 -- the bf16 GEMV microkernel ABI (what `mv_bf16_gelu.cc` must match)

Base: `PE/matrix_vector_multiplication/bf16_cascade/mv_bf16.cc` (76 lines, clean bf16). Entries
(`extern "C"`, lines 62-75), templated on `DIM_M`/`DIM_K` (19-24, default DIM_M=8, DIM_K=512), inner
vector `r=32`:
- `matvec_vectorized_bf16(bfloat16* a, bfloat16* b, bfloat16* c)` -- **accumulating** `c[0..m] += a[m,k]@b[k]`
  (28-44, 64-66). `set_rounding(conv_even)` (32); per row: accfloat acc, `aie::mac` over k in steps of
  r=32 (36-39), `reduce_add` -> `c[row] += partial` (41-42). Caller must zero c first (or K is one chunk).
- `zero_vectorized_bf16(bfloat16* c)` -- `c[0..m]=0` (46-50, 68).
- `partial_plus_r_bf16(bfloat16* partial, bfloat16* r_full, int offset, bfloat16* d)` -- residual add
  `d[i] = partial[i] + r_full[offset+i]` (52-60, 70-73).
L1 tile sizing: `a` is `[DIM_M, DIM_K]` bf16, `b` is `[DIM_K]` bf16, `c` is `[DIM_M]` bf16. `r=32` ->
DIM_K must be a multiple of 32 (768 and 384 both qualify).

The int4 kernel `PE/matrix_vector_multiplication/int4_awq/mv_int4_bf16.cc` documents the `_store` and
`_b_offset` variant pattern we will mirror in bf16 (lines 399-470):
- `_packed` = accumulating (line 403); `_packed_store` = **overwriting** (`c[row]=s`, saves a zero call when
  K is a single chunk, 427-434); `_packed_b_offset(packed, b, int b_offset, c)` = accumulating with a base
  offset into `b` so a caller can keep one big B buffer and tile K via `scf.for` (414-423, used by LD's
  K=8192/K_CHUNK loop 1016-1026). The int4 packed-tile byte layout `[Q | S | Z]` (docstring 16-28) does NOT
  apply to us (we are dense bf16, no Q/S/Z) -- ignore it.

So `mv_bf16_gelu.cc` (Task 3) should provide, in bf16 (no packing):
  - `matvec_vectorized_bf16(a,b,c)`            accumulate  (from mv_bf16.cc verbatim)
  - `matvec_vectorized_bf16_store(a,b,c)`      overwrite   (new: `c[row] = partial`, drop the `+c[row]`)
  - `matvec_vectorized_bf16_b_offset(a,b,off,c)` accumulate with b-offset (new; only if fc2 K is chunked)
  - `zero_vectorized_bf16(c)`                  zero
  - `partial_plus_r_bf16(p,r,off,d)`           residual add
  - `gelu_tile_bf16(uint32_t n, bfloat16* c)`  GELU(tanh) epilogue over the FULL m_output slab (Section B.2)
`link_with` wiring: every `FuncOp` / herd that calls these sets
`attributes["link_with"] = StringAttr.get("mv_bf16_gelu.o")` and `llvm.emit_c_interface` (pattern:
`o_gemv_ffn_int4_fused.py` 330-354 for func decls, 655-657 / 954 / 1069 for herd-body link). Compile flags
(Makefile 108-110, 21): `clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG -DDIM_M=<m_input>
-DDIM_K=<k_chunk> -I <aieopt>/include -c mv_bf16_gelu.cc -o mv_bf16_gelu.o`. Use `-Os` if program memory
is tight (see AIE2P gotchas).

### A.4 -- "one air.launch" / the dispatch boundary

- int4 fused: ONE `@launch` (line 367) wrapping ONE `@segment` (439) holding all 3 herds + L2 staging. The
  whole post-attention block is one dispatch. Crosses DDR: 3 packed weight BOs in (streamed per launch, no
  residency), 2 activation inputs (`attn_out`, `x_residual`), `output` out, `D_dbg` debug out (delete).
  Intermediates res1/swiglu cross NO DDR.
- bf16 cascade: ONE `@launch` (214) -> ONE `@segment` (272) -> ONE herd. Weights (A) + B + R stream in;
  D out. Partials cross no DDR.
- Our target: ONE `@launch`. Crosses DDR: `x` in, `Wfc1`/`bfc1`/`Wfc2`/`bfc2` streamed per launch
  (residency = Phase 2, not now), `out` written. The 3072 intermediate `h` and the fc2 partials stay in
  L1 / ride the cascade.

### A.5 -- AIE2P gotchas (called out in comments)

- **16-wide bf16 legalization:** 8-wide bf16 does not legalize on AIE2P -- inner vectors must be 16 or 32
  wide. int4 fused pairs two outer iters so the bf16 add is 16-wide (comment 194-197); `mv_bf16.cc` uses
  r=32 (65); `matvec_cascade_add.py` asserts `k_chunk % 64 == 0` (105) and uses 16-wide f32 dot (409).
  The GELU epilogue MUST run over a slab whose width is a multiple of 16 (our 384-slab is fine), applied
  ONCE per C-tile, NOT per `m_input` matvec tile (the ru-2.05 bug, see B.2).
- **`air.shrinkage = False`:** the cascade scratch/partial L1 allocs set
  `attributes["air.shrinkage"] = BoolAttr.get(False)` so the buffer keeps its declared (padded) width and
  is not shrunk to the used M_TILE (int4 fused 518-520, 528-530, 1000-1002, 1008). Cascade buffers are
  padded to CASCADE_WIDTH (16 in bf16 cascade line 187; 32 in int4 line 270). Carry this attr on our
  cascade scratch.
- **In-place norm to save L1:** int4 RMS writes back into `res1` in place (`l1_normed = l1_res1`,
  comment 676-678 / 781-782) -- saves 4 KB L1. SwiGLU keeps a per-core 2 KB output copied into the cascade
  buffer at the hop to avoid a second 16 KB scratch (comment 681-686 / 262-268). Do the analogous in-place
  LayerNorm.
- **`-Os` to fit < 16 KB core program memory:** the spec/plan call for `-Os`; the example Makefiles use
  `-O2` for these small kernels (Makefile 21). Start `-O2`; fall back to `-Os` if the core ELF overflows
  the ~16 KB program-memory budget at link (Task 4 Step 3).
- **stack_size / lock fix:** int4 fused compiles with `stack_size=4096`, `use_lock_race_condition_fix=False`
  (1146-1148); bf16 cascade `stack_size` default, `use_lock_race_condition_fix=True`,
  `runtime_loop_tiling_sizes=[2,2]` (713, 717). Match the bf16 cascade backend settings for our bf16 port.

### A.6 -- build / run / time (pointer; Tasks 4/5/6 reuse)

- Kernel (Peano): Makefile `compile-mv-bf16` (102-110) -- `clang++ $PEANOWRAP2P_FLAGS -DDIM_M -DDIM_K -c
  mv_bf16.cc -o mv_bf16.o`.
- air -> xclbin + insts: `python3 <gen>.py --output-format xclbin --compile-mode compile-and-xclbin`
  (`matvec_cascade_add.py` 729-738 via `XRTBackend.compile`); or `--compile-mode compile-and-run` for the
  built-in `XRTRunner` correctness check (698-727).
- C++ chrono harness: `PE/matrix_vector_multiplication/bf16_cascade/test_add.cpp` (built by Makefile
  `build-test-add-exe` 82-83 / `build-test-exe-impl` 122-149), or the richer warmup+iters profiler
  `PE/llms/.../test_o_gemv_ffn_int4_fused.cpp` (lines 117-153: `--warmup`/`--iterations`,
  `high_resolution_clock` around `run.start()/run.wait2()`, reports avg/min/max us). Task 5/6's
  `test_cascade_ffn.cpp` adapts this (the plan also cites `attention_decode/test_xclbin_decode.cpp`).

---

## SECTION B -- Whisper-FFN port mapping (concrete decisions)

Our decode FFN (`$REPO/route_b_kernels/decode_fused/gen_ffn.py`):
`x -> LayerNorm -> fc1(768->3072) -> +bias -> GELU(tanh) -> fc2(3072->768) -> +bias`, D=768, FF=3072,
`num_aie_columns=8`, LN affine folded into fc1 (`mat_fc1=(gf[:,None]*Wfc1).T`, `bias_fc1=bf@Wfc1+b_fc1`,
gen_ffn lines 75-78). bf16, M=1.

### B.1 -- dim + structure mapping

| sibling (int4 fused / bf16 cascade)            | Whisper FFN                                  |
|------------------------------------------------|----------------------------------------------|
| emb_dim 2048 / K                               | **D = 768**                                  |
| hidden_dim 8192 (inter), M_OUT 8192            | **FF = 3072** (fc1 out / fc2 K)              |
| gate+up GEMV (M_LGU=16384) + SwiGLU            | **single fc1 GEMV(768->3072) + GELU(tanh)**  |
| RMSNorm (mean-of-squares, rstd)                | **LayerNorm** (mean-subtract + var-normalize)|
| down GEMV K_LD=8192 reduction + residual #2    | **fc2 GEMV(3072->768) K-reduction + residual x** |
| N=8 cores, K_CHUNK=2048, K_LD_div=4            | **8 cores, FF=3072 = 8 x 384, no K-chunking (3072 fits)** |

### B.2 -- replace gate/up+SwiGLU with single fc1 + GELU(tanh)

Drop the gate/up split and the SwiGLU nonlinearity. Each of the 8 cores computes its 384-row slab of the
3072 intermediate: `h_c[384] = Wfc1_slab_c[384,768] @ x_norm[768] + bias_fc1_slab_c[384]`, then GELU(tanh)
**in place over the full 384 slab**. GELU math = reuse our shipped epilogue
`$REPO/patches/iron-gemv-gelu-epilogue.patch` lines 7-31
(the `gelu_inplace_bf16` body: tanh approx, 16-wide bf16, constants 0.5/1.0/0.79788456/0.044715) and 37-40
(the `gelu_tile_bf16(uint32_t n, bfloat16* c)` entry). **Critical (ru-2.05 bug):** apply GELU ONCE per
C-tile over the full `m_output` slab (a multiple of 16), AFTER the matvec inner-loop -- NOT per `m_input`
matvec tile (per-tile m_input can be < 16 and overruns the 16-wide vector). This is the same lesson the
patch header (lines 7-9) and `decode-microop-fusion-map.md:108-116` record. fc2 uses the PLAIN matvec (no
epilogue) -- gate the GELU behind a separate symbol/flag so fc1 links the GELU variant and fc2 does not.

### B.3 -- replace RMSNorm with LayerNorm; LN placement

RMS lacks mean-centering; LayerNorm = `(x - mean) / sqrt(var + eps)` then affine `* gamma + beta`. Adapt
the int4 inline RMS (lines 693-784: vectorized sum -> reduce -> rsqrt -> in-place multiply) to also compute
the **mean** and subtract it (`var = mean_sq - mean^2`). Two sub-options:

- **Recommended: fold the affine into fc1 (mirror gen_ffn), keep only the normalization on-chip.** On host,
  pre-bake `mat_fc1 = (gamma[:,None] * Wfc1).T` and `bias_fc1 = beta @ Wfc1 + b_fc1` (gen_ffn 75-76). The
  on-chip step computes only `x_norm = (x - mean) / sqrt(var + eps)` (non-affine LayerNorm). Reasons: (1)
  it reproduces gen_ffn's golden EXACTLY (same fold) so the rel-L2 <= 0.08 gate is apples-to-apples; (2)
  the on-chip code is a minimal delta on the existing inline-RMS pattern; (3) no extra gamma/beta L1 buffer
  or broadcast. D=768 is tiny, so compute LN per-core redundantly at each fc1 core (cheap; avoids a
  one-core-computes-then-broadcast dependency like the int4 res1ToCons). Place it at the head of each core
  before the fc1 matvec.
- Alternative (inline full affine LN in-herd, no fold): simpler host side but diverges from gen_ffn's golden
  numerics and needs gamma/beta in L1. Not recommended for the gate.

### B.4 -- post-fc2 add: ONLY `b_fc2` (NO decode-block `+x` in the Phase-0 gate)

> **CORRECTION (2026-06-26):** an earlier draft had the cascade add the decode-block residual `+x`. That is
> WRONG for this gate. The IRON baseline `gen_ffn.py` computes `LN -> fc1 -> +bias_fc1 -> GELU -> fc2 ->
> +b_fc2` and does NOT add `+x` (the decode-block residual is a SEPARATE op, `op_add768` at
> gen_decode.py:419, added OUTSIDE the FFN in the full decode). The Phase-0 cascade must compute the EXACT
> SAME FUNCTION as gen_ffn so the rel-L2 gate is apples-to-apples and the dispatch A/B compares the same op
> set. So: **no `+x`.** The only post-fc2 add is `b_fc2`.

The cascade computes gen_ffn's function, fusing the bias adds into the dataflow:
- **bias_fc1**: added per-core to `h_ty[384]` AFTER the fc1 tiling loop and BEFORE GELU (gen_ffn's `add1`
  feeds the GELU, so it cannot be deferred). Use `partial_plus_r_bf16(384, h_ty, bias_fc1_full, ty*384, h_ty)`.
- **b_fc2**: the cascade-head injected R. `matvec_cascade_add.py` head-injection (lines 505-524): HEAD
  (ty==n_cascade-1) seeds the accumulator with `b_fc2` (static, length 768), the 8 per-core fc2 partials sum
  down the cascade, TAIL (ty==0) writes `out`. One add covers the fc2 bias for the whole reduction.

(The decode-block `+x` belongs to Phase 1, when the whole layer is assembled; it is explicitly OUT of the
Phase-0 FFN-isolated gate -- adding it here would diverge from gen_ffn's golden and add an op the IRON
baseline's FFN span does not have.)

### B.5 -- core count / slab width

8 cores (match gen_ffn `num_aie_columns=8`). The elegant single-herd fused design: `herd sizes=[1, 8]`
(1 M-column x `n_cascade=8` K-reduction rows, the `matvec_cascade_add` axis convention). Core `ty`
(0..7) holds `Wfc1_slab[384,768]` and `Wfc2_slab[768,384]` and does, in one launch:
1. `x_norm = LN(x)` (B.3),
2. `h_ty[384] = Wfc1_slab @ x_norm + bias_fc1_slab`, GELU in place (B.2),
3. `partial_ty[768] = Wfc2_slab[768,384] @ h_ty[384]` (fc2 K-chunk = this core's 384),
4. cascade-reduce `partial_ty` over the 8 rows -> `out[768]`, head/tail inject residual (B.4), tail writes.
fc1 M-slab 384 and fc2 K-chunk 384 coincide on the same core -> `h_ty` stays in L1, never L3 (the whole
point). 3072 % 16 == 0 and 384 % 16 == 0 satisfy the bf16 legalization + GELU-slab constraints.

> **CORRECTION (2026-06-26, Task 4 build-proven):** a core does NOT hold its full weight slab resident.
> `Wfc1_slab[384,768]` = 576 KB and `Wfc2_slab[768,384]` = 576 KB, but AIE2P (NPU2) L1 is **64 KB/core**
> (`AIE2TargetModel::getLocalMemorySize()=0x10000`). aiecc fails `allocated buffers exceeded available
> memory` (buf 0x90000 > L1 0x10000). So WEIGHTS STREAM IN TILES from L3 (ping-pong), exactly like every
> reference (`matvec_cascade_add` holds `[tile_m=2,k_chunk]`; int4 fused streams `M_TILE=8` row tiles). Per
> core: loop `for t in 384//M_INPUT: ChannelGet [M_INPUT,768] Wfc1 tile -> matvec_fc1_tile_store -> h_ty
> slice`; GELU ONCE over the full 384 `h_ty` after the loop (ru-2.05 rule holds); same M-tiling for fc2's 768
> output rows (`[M_INPUT,384]` tiles). Only `h_ty[384]` (768 B) + `partial_ty[768]` (1.5 KB) + cascade scratch
> stay resident -- the INTERMEDIATES are on-chip (the premise holds); only the (already-streamed) weights tile.
> Kernel needs tiled small-M entries (`matvec_fc1_tile_bf16_store<M_INPUT,768>`,
> `matvec_fc2_tile_bf16_store<M_INPUT,384>`, M_INPUT~8); the fixed large-M `<384,768>`/`<768,384>` entries from
> the first Task-3 cut cannot be called on a sub-slab (they read OOB). This is still ONE air.launch.

### B.6 -- sibling machinery to DROP (candidates; keep Phase 0 faithful first)

Flag, do not prematurely strip (right-sizing is the SOFT-gate fallback, not now):
- **int4 packing (Q/S/Z) + dequant + `GS` group-scale** -- gone; we are dense bf16 (`mv_bf16.cc` path).
- **SwiGLU + gate/up concat** -- replaced by single fc1 + GELU (B.2).
- **`swigluToLd` 4-chunk multicast** (int4 316-321, 927-943) -- existed only because K_LD=8192 > one
  K_CHUNK; our fc2 K=3072 fits without chunking and the intermediate never leaves the herd (single-herd
  design) -> drop the multicast entirely.
- **`res1ToCons` 16-dest broadcast** (int4 307-308) -- existed only because LA/LGU/LD were 3 SEPARATE
  herds; our single fused herd keeps `h` in L1 -> no inter-herd broadcast needed.
- **Debug L3 copy** `D_dbg` / `laResDebug` (int4 40-41, 250-252, 412, 640-642) -- delete.
- **L2 bulk-A staging** (`matvec_cascade_add.py` 281-332): exists for prefill-scale A; at our tiny weight
  sizes it MAY be droppable (stream weights L3->L1 directly), but KEEP it for the first faithful build and
  only strip if the SOFT-gate right-sizing pass calls for it.

---

## One-paragraph summary for the next tasks

Port base for the fc2 cascade = `PE/matrix_vector_multiplication/bf16_cascade/matvec_cascade_add.py`
(bf16 K-reduction cascade, residual at head); single-launch multi-stage chaining idea = the int4
`o_gemv_ffn_int4_fused.py`; kernel base = `bf16_cascade/mv_bf16.cc` extended with `_store`, optional
`_b_offset`, and a `gelu_tile_bf16` epilogue from our `patches/iron-gemv-gelu-epilogue.patch`. The named
`o_ffn_bfp16_multi.py` is a dispatch-stitch prefill GEMM, NOT a cascade -- use it only as a bf16 numerics
reference. Target: ONE `air.launch`, one fused herd `sizes=[1,8]`, each core does LN + fc1-384-slab + GELU
+ fc2-384-K-partial, cascade-reduce to out[768] + residual, gated rel-L2 <= 0.08 vs gen_ffn's golden.
