#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decoder-LLM whole decode stack as ONE fused ELF, built from an `LlmSpec`.

Generalises gen_gemma_decode.py from one checkpoint to the spec vocabulary in llm_decode_spec.py:
a MODEL is a spec plus its weights, not a generator. Same deep-C mechanism as the shipped Whisper
fused decode (gen_decode.py): the ELF is CONSTANT across tokens; per token the host writes `x`, the
RoPE angle row, and two scratchpad params (`kv_off`, `sm_mask`), then dispatches ONCE.

Per block, all projections bias-free:
  h  = RMSNorm_in(x)
  q,k,v = Wq/Wk/Wv @ h
  q,k   = RMSNorm_qk(per head, head_dim)          # if spec.qk_norm
  q,k   = RoPE(theta)                             # dual-theta if spec.rope_theta_local
  KV cache append at n_past; GQA broadcast kv -> q heads
  scores = (K @ q) * spec.attn_scale ; softmax(width n_past+1)
  ctx = V^T @ scores ; a = Wo @ ctx
  [sandwich] a = RMSNorm_post_attn(a)             # Gemma-3 only
  x1 = x + a
  hf = RMSNorm_pre_ffn(x1)
  d  = Wdown @ (act(Wgate @ hf) * (Wup @ hf))     # act = gelu_tanh | silu
  [sandwich] d = RMSNorm_post_ffn(d)              # Gemma-3 only
  x2 = x1 + d
then RMSNorm_final + tied lm-head -> logits.

Run INSIDE the fork IRON env (scripts/toolchain_up.sh), never the wheel python. Example:
  python designs/decode_fused/gen_llm_decode.py --spec qwen3-0.6b \
      --weights artifacts/qwen3-0.6b/weights --out artifacts/qwen3-0.6b/decode --layers 28
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import SPECS  # noqa: E402

# Dataflow switches, read ONCE at module scope. They are consumed in three different functions
# (graph construction, the runlist, and the meta writer), and defining them next to their first
# use put them out of scope in the others -- four times in one session, because Python does not
# complain until the single run that matters.
# DEFAULTS FLIPPED ON 2026-09-07 (owner call) after a four-arm A/B, interleaved, 28 layers:
#
#   arm                        MB/token   ms/token   tok/s
#   base, no flags              3105.99     137.90    7.25
#   GQA_GROUPED_K+V             2166.47     100.03   10.00
#   TMV_CTX alone               2036.18      98.50   10.15
#   TMV_CTX + GQA_GROUPED_K     1566.42      79.13   12.64   <- the default now
#
# ~1.25x over the previous best arm, and 1.20x AHEAD of AMD's own mlir-air Qwen3 example (95.2
# ms/token on this box), which we were 1.44x behind on 2026-09-05. Correctness: 8/8 vs the bf16
# oracle at every step but one, and that one is a THREE-WAY EXACT bf16 tie (279/9625/15344 all at
# 16.7500, argmax broken by index) which the un-grouped arm happens to win by exactly one ulp.
# Determinism is bit-identical across passes. Full record: the 2026-09-07 log note in the journal.
#
# Each flag still takes "0" to turn it OFF, so every arm above is still reachable for A/B.
GROUPED_K = os.environ.get("GQA_GROUPED_K", "1") == "1"
# NOT flipped: TMV_CTX subsumes the v-side grouping (TMatVec reads vc per kv head itself, so the
# Repeat would materialise a `vr` nothing consumes). Setting this with TMV_CTX on is a no-op, and
# reading a null result from toggling it as evidence about grouping would be a mistake.
GROUPED_V = os.environ.get("GQA_GROUPED_V", "0") == "1"
# Context step as a transposed-A reduction over the V cache rows, deleting op_trv outright.
TMV_CTX = os.environ.get("TMV_CTX", "1") == "1"
TMV_RPC_DEFAULT = 64
TMV_RPC = int(os.environ.get("TMV_RPC", str(TMV_RPC_DEFAULT)))

import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemv.op import GEMV  # noqa: E402
from iron.operators.gemv.quant import quantize_weight  # noqa: E402
from iron.operators.rms_norm.op import RMSNorm  # noqa: E402
from iron.operators.rope.op import RoPE  # noqa: E402
from iron.operators.elementwise_add.op import ElementwiseAdd  # noqa: E402
from iron.operators.elementwise_mul.op import ElementwiseMul  # noqa: E402
from iron.operators.softmax.op import Softmax  # noqa: E402
from iron.operators.strided_copy.op import StridedCopy  # noqa: E402
from iron.operators.transpose.op import Transpose  # noqa: E402
from iron.operators.tmatvec.op import TMatVec  # noqa: E402
from iron.operators.gelu.op import GELU  # noqa: E402
from iron.operators.silu.op import SiLU  # noqa: E402
from iron.operators.repeat.op import Repeat  # noqa: E402

BF16 = ml_dtypes.bfloat16
COLS = 8      # num_aie_columns for every GEMV; the spec's check() is written against this
TSI = 4       # tile_size_input


def bf16(a):
    return np.asarray(a).astype(BF16)


# Engineering-check MLP weight quantization axis (Wg/Wu/Wd -- the "MLP weights" byte class), gated
# by env vars so build/verify/bench need no CLI plumbing to A/B it, matching DECODE_PLACER_FLAGS'
# convention. QUANT_MLP_DTYPE="bf16" (default) is a no-op: every GEMV byte-for-byte unchanged.
# NOT a quality claim: this axis is validated as a byte-stream + determinism engineering check on
# Qwen3-0.6B, not a token-quality gate (tests/refs/qwen3-0.6b/bf16_oracle.json is 1 prompt / 8
# free-running tokens with knife-edge logit margins -- too small to see quantization damage).
QUANT_MLP_DTYPE = os.environ.get("QUANT_MLP_DTYPE", "bf16")
QUANT_MLP_GROUP = int(os.environ.get("QUANT_MLP_GROUP", "128"))

DECODE_PLACER_FLAGS_DEFAULT = "--cores-per-col 1"

# INSTRUMENT, not a feature. Alternates the per-head qk-norm between two IDENTICAL RMSNorm
# instances. RMSNorm has no design_key, so two instances are two DESIGNS: the 24 consecutive runs
# stop sharing one aiex.configure and become 24. Runs, bytes and output are unchanged, so it
# isolates the cost of a CHEAP configure (18 KB of views) the way share_designs isolated an
# expensive one. Predicted +644 configures/token; at the measured 61.9 us for a big configure that
# is +39.9 ms if the cost is flat, and ~0 if it tracks the view count.
SPLIT_QKNORM = os.environ.get("SPLIT_QKNORM", "0") == "1"


def weight_bytes(arr):
    """Bytes for one weight buffer exactly as written into the .bin / device arena.

    A quantize_weight() packed array (np.int8, opaque on-wire bytes) must NOT be value-cast to
    bf16 like every other weight here -- that would renumber the packed bytes instead of copying
    them.
    """
    a = np.asarray(arr)
    if a.dtype == np.int8:
        return a.tobytes()
    return np.asarray(a, BF16).tobytes()


def load_weight_buffer(buf, arr):
    """Load one weight into its device buffer, mirroring weight_bytes()'s dtype split.

    `buf.data` is always a bfloat16-dtype view (iron.common.sequence's shared bf16-granule arena,
    FusedFullELFCallable.get_buffer) regardless of the argument's declared dtype, so a packed
    int8 array must be written via a raw byte view, never `np.asarray(arr, BF16)` (which would
    numerically reinterpret the packed byte VALUES as floats).
    """
    a = np.asarray(arr)
    if a.dtype == np.int8:
        dst = buf.data.view(np.uint8)
        assert dst.nbytes == a.nbytes, f"weight byte-size mismatch: buf {dst.nbytes} vs arr {a.nbytes}"
        dst[:] = a.view(np.uint8)
    else:
        np.copyto(buf.data, np.asarray(a, BF16).reshape(-1))



def sequence_name(sp, NL, S, placer_flags):
    """Name the fused sequence after everything that changes its graph, not just the model.

    IRON keys the cached artifact by this name. Every knob below produces a DIFFERENT ELF, so
    without them two arms share one filename and a later run executes the earlier arm's binary --
    see isolate_build_dir() for what that cost twice on 2026-09-07. Isolating the build dir hides
    the collision; naming the arm removes it, and only naming it makes an arm's artifact
    identifiable after the fact.

    Suffixes are emitted only for NON-DEFAULT values, so the shipped default keeps the bare
    `<spec>_decode` name and its existing artifact stays valid. Same convention as TMatVec's
    `_ak{alloc_K}`.
    """
    base = f"{sp.name.replace('-','_').replace('.','_')}_decode"
    parts = []
    if not TMV_CTX:
        parts.append("noctx")
    if not GROUPED_K:
        parts.append("nogk")
    if GROUPED_V:
        parts.append("gv")
    if TMV_CTX and TMV_RPC != TMV_RPC_DEFAULT:
        parts.append(f"rpc{TMV_RPC}")
    if QUANT_MLP_DTYPE != "bf16":
        parts.append(f"{QUANT_MLP_DTYPE}g{QUANT_MLP_GROUP}")
    if SPLIT_QKNORM:
        parts.append("splitqk")
    if NL != sp.n_layers:
        parts.append(f"l{NL}")
    if S != 2048:
        parts.append(f"s{S}")
    if placer_flags != DECODE_PLACER_FLAGS_DEFAULT.split():
        parts.append("p" + hashlib.sha1(" ".join(placer_flags).encode()).hexdigest()[:6])
    return "_".join([base, *parts])


def isolate_build_dir(tag):
    """chdir into a private build dir, because IRON writes build/ intermediates under CWD.

    IRON keys cached operator artifacts by NAME, and the name encodes SHAPES but not the dataflow
    flags -- so two arms of this graph produce the same filenames. Any entry point that runs in a
    shared directory can therefore assemble an ELF partly from another arm's operator binaries. The
    result is not a crash and not noise: it is a deterministic, reproducible wrong answer that looks
    exactly like a numerical bug.

    MEASURED COST 2026-09-07, twice in one day. Once here (an A/B in a shared dir produced a
    different step-0 token for a FIXED graph, which is impossible), and once in a parallel session
    that spent ~3 hours on a full-depth decode returning the constant token 3972 at every step,
    deterministic across runs, because its runner never left the shared xdna-engine/build.

    build_llm_decode.sh already does `WORK=$(mktemp -d); cd "$WORK"` for exactly this reason. This
    gives the Python entry points the same protection instead of trusting the caller's cwd.

    Set DECODE_WORK=<dir> to use a specific directory (kept, not deleted) when you need the
    intermediates -- a byte census needs the fused MLIR, which is otherwise discarded.
    """
    import atexit
    import shutil
    import tempfile

    explicit = os.environ.get("DECODE_WORK")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        os.chdir(explicit)
        print(f"[{tag}] build dir {explicit} (DECODE_WORK, kept)", flush=True)
        return explicit
    work = tempfile.mkdtemp(prefix=f"{tag}-")
    atexit.register(shutil.rmtree, work, ignore_errors=True)
    os.chdir(work)
    print(f"[{tag}] build dir {work} (private, removed on exit; "
          f"set DECODE_WORK=<dir> to keep)", flush=True)
    return work


def repo_root():
    # gen_llm_decode.py -> decode_fused -> designs -> repo root (toolchain.lock lives there).
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def toolchain_provenance():
    """Best-effort build provenance for meta.json: the toolchain.lock semantic hash this ELF was
    just compiled against, plus the instance dir the build used, if resolvable.

    Sources scripts/kernel_sandbox.sh's `current_toolchain_hash` (the canonical derivation --
    comment/blank-stripped toolchain.lock, sha256, first 12 hex; matches
    kernel_registry.rs::current_toolchain_hash and toolchain_up.sh's LOCKHASH) rather than
    reimplementing it a fifth time. Returns {} rather than raising: provenance is a record, not a
    gate, and a build must not fail because this is unresolvable.
    """
    repo = repo_root()
    sandbox = os.path.join(repo, "scripts", "kernel_sandbox.sh")
    if not os.path.isfile(sandbox):
        return {}
    try:
        out = subprocess.run(
            ["bash", "-c", f'source "{sandbox}" && current_toolchain_hash "$1"', "_", repo],
            capture_output=True, text=True, timeout=10,
        )
    except OSError:
        return {}
    h = out.stdout.strip()
    if out.returncode != 0 or not h:
        return {}
    prov = {"hash": h}
    inst = os.environ.get("MLIR_AIE_INSTANCE")
    if inst:
        prov["instance"] = os.path.basename(inst.rstrip("/"))
    return prov


def report_artifact_freshness(weights_dir):
    """Print-only freshness check for the sibling `decode/` artifact a `--weights` dir implies
    (`artifacts/<spec>/weights` -> `artifacts/<spec>/decode`).

    verify_llm_decode.py and bench_llm_decode.py always recompile the graph fresh via build_graph,
    so THIS run never dispatches stale bytes -- but the shipped Rust engine loads decode.elf/
    meta.json directly (npu-engine::LlmArtifact) and does not rebuild. A stale artifact sitting next
    to fresh weights is exactly the silent-wrong-token hole this closes: the decode ELF that
    shipped 2026-09-03 returned a wrong token at one margin step after the 2026-09-04 re-pin, with
    nothing to say so. Never raises and never affects the caller's exit code -- this is a report
    about a DIFFERENT consumer, not a gate on the graph this process just verified.
    """
    art_dir = os.path.join(os.path.dirname(os.path.normpath(weights_dir)), "decode")
    meta_path = os.path.join(art_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return
    meta = json.load(open(meta_path))
    built = meta.get("toolchain", {}).get("hash")
    tag = "[freshness]"
    if not built:
        print(f"{tag} {meta_path}: no toolchain provenance recorded (built before this check "
              f"existed) -- the Rust-loaded artifact's freshness is UNVERIFIED", file=sys.stderr)
        return
    cur = toolchain_provenance().get("hash")
    if not cur:
        print(f"{tag} {meta_path}: built against {built}, but the current toolchain.lock could not "
              f"be resolved here -- the Rust-loaded artifact's freshness is UNVERIFIED", file=sys.stderr)
    elif cur != built:
        print(f"{tag} STALE: {meta_path} was built against toolchain {built}, current toolchain.lock "
              f"is {cur} -- rebuild with scripts/build_llm_decode.sh before trusting what the Rust "
              f"engine would load from {art_dir}", file=sys.stderr)
    else:
        print(f"{tag} {meta_path}: OK (toolchain {cur})", file=sys.stderr)


L1_BYTES = 65536      # AIE2P core local memory (getLocalMemorySize(), AIETargetModel.h)
L1_RESERVE = 8192     # stack + the allocator's own slack; measured headroom, not a guess (see below)
C_TILE_GRANULE = 8    # tile_size_output must be a multiple of this (16 bytes of bf16); see gemv_tile_output


def gemv_tile_output(M, K, cols=None, tsi=None):
    """Largest legal `tile_size_output` for a GEMV that also FITS L1.

    Two independent constraints, and only the first is checked by the toolchain:

    1. design.py asserts `m_output <= M//cols` and `(M//cols) % m_output == 0`, plus
       `m_output % m_input == 0`.
    2. NOTHING checks L1 capacity. The generated core holds, per the linker map of a failing build:
       the C output tile DOUBLE-buffered (2 x m_output x 2B), the A input tile double-buffered
       (2 x m_input x K x 2B) and the B vector double-buffered (2 x K x 2B). Exceed 64 KB and aiecc
       dies with "'aie.tile' op Basic sequential allocation also failed" -- which names a tile, not a
       tile SIZE, so it reads as a placement bug rather than "your output tile is too big".

    Taking m_output = M//cols (the largest the asserts allow) blows constraint 2 on the lm-head:
    vocab 151936 -> 18992 elements -> 37984 B, double-buffered 76 KB against 64 KB of L1. Both the
    tracked gen_gemma_decode.py (vocab//8 = 32768) and a naive port hit this.

    Returns (tile_size_input, tile_size_output).
    """
    cols = COLS if cols is None else cols
    per_col = M // cols
    # m_input must shrink too: the A tile is m_input x K, so at K=3072 (the FFN down projection)
    # A+B double-buffered already exceed L1 at m_input=4 and leave the C tile nothing. Search
    # m_input downward and take the first that admits any legal C tile.
    # Search every m_input and keep the largest legal C TILE, not the largest m_input. Preferring
    # m_input first is a trap: at the lm-head shape it accepts (4, 16) -- legal, correct, and 1187
    # tiles per column -- while (2, 9496) exists one step down and is 593x fewer tiles. A smaller
    # m_input frees L1 budget quadratically faster than it costs, because A is m_input x K while C
    # is just m_output.
    best_pair = (0, 0)
    for cand_tsi in ([tsi] if tsi is not None else (4, 2, 1)):
        if per_col % cand_tsi:
            continue
        budget = L1_BYTES - L1_RESERVE - 2 * (cand_tsi * K * 2) - 2 * (K * 2)
        if budget <= 0:
            continue
        cap = budget // 4                  # C is double-buffered, 2 bytes per element
        # THIRD constraint, and nothing in the toolchain checks it: the C tile must be a multiple of
        # C_TILE_GRANULE elements. MEASURED 2026-09-03 with a standalone one-GEMV repro at the
        # lm-head shape -- M=151936 K=1024 with tso=4748 (4748 % 8 == 4) returns a PERMUTATION of the
        # right answer: values correct (sorted rel-L2 7.9e-3) in wrong positions (rel-L2 1.40). The
        # SAME M and K with tso=9496 is correct at 2.8e-3, and 1024/512/256 are correct at M=8192.
        # Every passing tile is a multiple of 8; the one failing tile is not. It is silent -- the
        # build succeeds and the argmax is simply wrong -- so it has to be refused here.
        best = max((d for d in range(cand_tsi, per_col + 1, cand_tsi)
                    if per_col % d == 0 and d <= cap and d % C_TILE_GRANULE == 0), default=0)
        if best > best_pair[1]:
            best_pair = (cand_tsi, best)
    if best_pair[1]:
        return best_pair
    raise ValueError(f"GEMV M={M} K={K}: no (tile_size_input, tile_size_output) fits L1 "
                     f"({L1_BYTES} B) with M//cols={per_col} and tile_size_output a multiple of "
                     f"{C_TILE_GRANULE}")


def gemv(M, K, ctx, **kw):
    """GEMV tiled as large as both the design asserts AND L1 allow."""
    tsi, tso = gemv_tile_output(M, K)
    return GEMV(M=M, K=K, num_aie_columns=COLS, tile_size_input=tsi,
                tile_size_output=tso, context=ctx, **kw)


def build_graph(spec_name, weights_dir, layers=None, max_seq=2048):
    """Construct the fused decode graph + its weight dict for a spec.

    Shared by the generator CLI and verify_llm_decode.py so the harness drives the SAME graph the
    artifact was built from, rather than a re-typed copy that can drift from it.
    Returns (spec, fused, weights, meta_dims).
    """
    sp = SPECS[spec_name]
    sp.check(cols=COLS, tsi=TSI)
    sp.check_seq(max_seq)
    NL = layers if layers is not None else sp.n_layers
    S = max_seq
    D, FF, HD = sp.d_model, sp.ffn, sp.head_dim
    Hq, Hkv, QD, KVD, VOCAB = sp.n_q_heads, sp.n_kv_heads, sp.q_dim, sp.kv_dim, sp.vocab

    def npy(name):
        return np.load(os.path.join(weights_dir, f"{name}.npy")).astype(np.float32)

    def load_norm(name):
        # Gemma-3 stores RMSNorm gain as w with the kernel computing x_hat*(1+w); Qwen3 stores it
        # already absolute. IRON's weighted RMSNorm always does x_hat*w', so Gemma folds the +1 here.
        w = npy(name)
        return bf16(1.0 + w) if sp.norm_gain == "one_plus_w" else bf16(w)

    ctx = AIEContext()

    # ---- op vocabulary: created ONCE, reused across every layer (same dims per layer) ----
    # 1 column, and this is FORCED by the operator's semantics, not a placement budget.
    # RMSNorm's `tile_size` is the NORMALISED WIDTH, not a work split: its own test reads
    # `rows = input_length // tile_size, cols = tile_size`, and design_weighted.py carries one
    # weight ObjectFifo of `tile_size` elements shared across every column. num_aie_columns splits
    # ROWS. A decode step normalises ONE vector of d_model, so rows == 1 and there is nothing to
    # spread. Setting tile_size = D // 4 to buy width would compute four independent 256-wide
    # normalisations instead of one 1024-wide one -- a different function, not a faster one. The
    # buffer-size gate caught it first ("L0_n_in.bin is 2048 bytes, layout declares 512"), which is
    # luck: the sizes happened to disagree. Had D//4 divided evenly into the dumped weight it would
    # have run and been quietly wrong.
    op_norm = RMSNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D,
                      weighted=True, epsilon=sp.eps, context=ctx)
    op_qk_norm = RMSNorm(size=HD, num_aie_columns=1, num_channels=1, tile_size=HD,
                         weighted=True, epsilon=sp.eps, context=ctx) if sp.qk_norm else None
    op_qk_norm_b = (RMSNorm(size=HD, num_aie_columns=1, num_channels=1, tile_size=HD,
                            weighted=True, epsilon=sp.eps, context=ctx)
                    if (sp.qk_norm and SPLIT_QKNORM) else op_qk_norm)
    op_q = gemv(QD, D, ctx)
    op_kv = gemv(KVD, D, ctx)
    op_o = gemv(D, QD, ctx)
    op_rope_q = RoPE(rows=Hq, cols=HD, angle_rows=1, context=ctx)
    op_rope_k = RoPE(rows=Hkv, cols=HD, angle_rows=1, context=ctx)
    # KV append: deep-C scratchpad offset "kv_off" (element units = n_past*HD), constant ELF.
    sc = dict(input_sizes=(Hkv, HD), input_strides=(HD, 1), input_offset=0,
              output_sizes=(1, Hkv, HD), output_strides=(0, S * HD, 1), output_offset=0,
              input_buffer_size=Hkv * HD, output_buffer_size=Hkv * S * HD, num_aie_channels=1)
    op_sck = StridedCopy(**sc, output_offset_parameter="kv_off", context=ctx)
    # V stays [S][HD]. A transposed append would delete op_trv, but a SINGLE-token transposed write
    # is 1024 isolated bf16 elements (h*HD*S + d*S + p) and the shim address generator steps in
    # 4-byte granules: the BD silently halves the innermost dimension (measured on the emitted
    # descriptor -- d0_size 64 for 128 elements, d0_stride 1023, i.e. 64 granules of two ADJACENT
    # elements). The runtime offset has the same granule floor, so an odd `p` truncates down.
    # The working shape is a PAIR write on an even offset, whose staging cannot itself be a DMA.
    op_scv = StridedCopy(**sc, output_offset_parameter="kv_off", context=ctx)
    # GQA broadcast. Correctness-first; the byte-free form is a batch-stride-0 GEMV read of the kv
    # head (0 ops, 0 bytes) -- at Hq=16 x 28 layers this Repeat plus the V transpose are 41% of the
    # per-token DDR budget, so it is the first optimisation after parity, not an afterthought.
    #
    # S below is deliberately ONE value shared by kc/vc/kr/vr/vt/sc/sw, op_scores, op_rep_k/v, op_trv
    # AND op_ctx -- not the op_ctx-excluded 4-of-5 split llm-decode-attention-pads-to-full-window.md
    # scoped out device-free. That split needs op_trv to write a bucket-wide `vt` while op_ctx reads
    # it at full max_seq width, and symmetrically op_rep_k/v to read a bucket-wide prefix of a kc/vc
    # row whose true stride is max_seq*HD (Hkv=8 here, not a degenerate single-row case where prefix
    # == whole buffer). Neither holds with today's operators: Repeat's input TensorAccessPattern
    # ties its row stride directly to `cols` (repeat/design.py: strides=[0, cols, cols_split, 1]),
    # and Transpose's output stride is tied to its own `M` (transpose/design.py: taps_out_L1L3
    # strides derive from M) -- neither exposes a stride independent of its own declared size, so
    # "read/write a narrower window of a wider-strided buffer" is new IRON capability, not a
    # generator change. Bucketing S UNIFORMLY (this build already takes it as `max_seq`) is the
    # route that needs none.
    op_rep_k = Repeat(rows=Hkv, cols=S * HD, repeat=sp.gqa_group, transfer_size=HD, context=ctx)
    op_rep_v = Repeat(rows=Hkv, cols=S * HD, repeat=sp.gqa_group, transfer_size=HD, context=ctx)
    op_scores = gemv(S, HD, ctx, num_batches=Hq,
                     batch_group=sp.gqa_group if GROUPED_K else 1)
    op_scale = ElementwiseMul(size=Hq * S, tile_size=S // COLS, num_aie_columns=COLS, context=ctx)
    op_softmax = Softmax(rows=Hq, cols=S, num_aie_columns=COLS, num_channels=1, rtp_vector_size=S,
                         vector_size_parameter="sm_mask", context=ctx)
    # num_batches=Hq, not Hq separate invocations. transpose/design.py's L3 tensors already hold
    # "num_batches contiguous (M,N) matrices stacked along the row dimension", which is EXACTLY what
    # vr/vt are; calling it per head issued 448 configure+run pairs per token to do 28 ops' work.
    #
    # It is NOT a speed fix and must not be quoted as one. Isolated on device against the same
    # placer: -0.01 ms/token, 0.0%. Dropping 420 dispatches per token is worth nothing measurable,
    # because these are mode selections inside ONE hardware context. Kept because it is correct,
    # free, and 2.2 MB smaller in the ELF -- not because it is faster.
    # 4, not COLS: Transpose splits N across columns as `N // num_columns // n`, and at N=HD=128
    # with n=32 that is 4 tiles, so 8 columns divides to ZERO. Its __post_init__ does not catch it
    # -- it checks M*N % (m*n*cols*channels), which 8 satisfies -- and the failure surfaces deep in
    # taplib as "All sizes must be >= 1, but got [8, 0, 256, 32]", naming no operator and no
    # parameter. 4 is the real ceiling at this n; raising it needs n=16.
    # GQA broadcast as an ACCESS PATTERN instead of a materialised copy. gqa_group query heads
    # attend to one kv head; with batch_group the consumer reads that head directly and the Repeat
    # that duplicated it into DDR disappears. Opt-in per side because k and v are not the same edit
    # (k: Repeat feeds the GEMV; v: Repeat feeds a Transpose that feeds the GEMV), so they must be
    # gated separately to stay attributable in an A/B ladder.
    op_trv = Transpose(M=S, N=HD, num_aie_columns=4, num_channels=1, m=256, n=32, s=8,
                       num_batches=Hq, batch_group=sp.gqa_group if GROUPED_V else 1, context=ctx)
    # CONTEXT STEP. op_trv exists only because gemv reduces ALONG a row: it wants [HD][S] and the
    # cache is [S][HD], so the whole cache is rearranged every token -- 16.777 MB/layer measured, at
    # 0% compute. TMatVec reduces DOWN the rows instead and reads `vc` as it is stored, so the
    # transpose has nothing left to do. One kv head per column (n_matrices == cols == Hkv), so each
    # column streams its own head ONCE and applies both query heads' softmax rows out of L1 -- the
    # stride-0 group re-read goes too. rows_per_chunk=64 puts the L1 A tile at 16 KB double-buffered.
    if TMV_CTX:
        op_ctx = TMatVec(M=HD, K=S, num_aie_columns=Hkv, num_batches=Hq,
                         batch_group=sp.gqa_group,
                         rows_per_chunk=TMV_RPC, context=ctx)
    else:
        op_ctx = gemv(HD, S, ctx, num_batches=Hq)
    # MLP weight-stream dtype axis (Wg/Wu/Wd -- "MLP weights" in the byte breakdown, the largest
    # single weight class). bf16 (default) is byte-for-byte the pre-existing path; QUANT_MLP_DTYPE
    # is an engineering-check toggle (see its definition above), not a quality-validated default.
    mlp_quant_kw = (dict(weight_dtype=QUANT_MLP_DTYPE, group_size=QUANT_MLP_GROUP)
                    if QUANT_MLP_DTYPE != "bf16" else {})
    op_gate = gemv(FF, D, ctx, **mlp_quant_kw)
    op_up = gemv(FF, D, ctx, **mlp_quant_kw)
    if sp.act == "silu":
        op_act = SiLU(size=FF, num_aie_columns=COLS, tile_size=FF // COLS, context=ctx)
    else:
        op_act = GELU(size=FF, num_aie_columns=COLS, num_channels=1, tile_size=FF // COLS, context=ctx)
    op_mul_ffn = ElementwiseMul(size=FF, tile_size=FF // COLS, num_aie_columns=COLS, context=ctx)
    op_down = gemv(D, FF, ctx, **mlp_quant_kw)
    op_add = ElementwiseAdd(size=D, tile_size=D // COLS, num_aie_columns=COLS, context=ctx)
    op_head = gemv(VOCAB, D, ctx)

    weights, bufsz, cache_names, rl = {}, {}, [], []
    cur = "x"

    for l in range(NL):
        p = f"L{l}_"
        nm = sp.norm_weight_names(l)
        for key, tensor in nm.items():
            weights[p + key] = load_norm(tensor)
        mlp_keys = {"Wg", "Wu", "Wd"}
        for key, tensor in (("Wq", "self_attn.q_proj"), ("Wk", "self_attn.k_proj"),
                            ("Wv", "self_attn.v_proj"), ("Wo", "self_attn.o_proj"),
                            ("Wg", "mlp.gate_proj"), ("Wu", "mlp.up_proj"), ("Wd", "mlp.down_proj")):
            w = npy(f"model.layers.{l}.{tensor}.weight")  # [M, K], f32
            if key in mlp_keys and QUANT_MLP_DTYPE != "bf16":
                weights[p + key] = quantize_weight(w, QUANT_MLP_GROUP, QUANT_MLP_DTYPE)
            else:
                weights[p + key] = bf16(w).reshape(-1)
        weights[p + "kc"] = np.zeros(Hkv * S * HD, BF16)
        weights[p + "vc"] = np.zeros(Hkv * S * HD, BF16)
        cache_names += [p + "kc", p + "vc"]
        ang = "rope_global" if sp.is_global(l) else "rope_local"

        bufsz.update({
            p + "q": QD * 2, p + "k": KVD * 2, p + "v": KVD * 2,
            p + "kc": Hkv * S * HD * 2, p + "vc": Hkv * S * HD * 2,
            p + "kr": Hq * S * HD * 2, p + "vr": Hq * S * HD * 2, p + "vt": Hq * S * HD * 2,
            p + "sc": Hq * S * 2, p + "sw": Hq * S * 2,
            p + "cx": QD * 2, p + "a": D * 2,
            p + "g": FF * 2, p + "u": FF * 2, p + "gh": FF * 2, p + "d": D * 2,
            p + "hn": D * 2, p + "hf": D * 2,
        })
        nxt = f"x{l+1}"
        qk = []
        if sp.qk_norm:
            qk = [*[((op_qk_norm if h % 2 == 0 else op_qk_norm_b),
                     f"{p}q[{h*HD*2}:{(h+1)*HD*2}]", p + "n_qn",
                     f"{p}q[{h*HD*2}:{(h+1)*HD*2}]") for h in range(Hq)],
                  *[((op_qk_norm if h % 2 == 0 else op_qk_norm_b),
                     f"{p}k[{h*HD*2}:{(h+1)*HD*2}]", p + "n_kn",
                     f"{p}k[{h*HD*2}:{(h+1)*HD*2}]") for h in range(Hkv)]]
        rl += [
            (op_norm, cur, p + "n_in", p + "hn"),
            (op_q, p + "Wq", p + "hn", p + "q"),
            (op_kv, p + "Wk", p + "hn", p + "k"),
            (op_kv, p + "Wv", p + "hn", p + "v"),
            *qk,
            (op_rope_q, p + "q", ang, p + "q"),
            (op_rope_k, p + "k", ang, p + "k"),
            (op_sck, p + "k", p + "kc"),
            (op_scv, p + "v", p + "vc"),
            *([] if GROUPED_K else [(op_rep_k, p + "kc", p + "kr")]),
            # TMV_CTX subsumes the v-side grouping: TMatVec reads vc per kv head itself, so a
            # Repeat would materialise a `vr` nothing consumes.
            *([] if (GROUPED_V or TMV_CTX) else [(op_rep_v, p + "vc", p + "vr")]),
            (op_scores, p + ("kc" if GROUPED_K else "kr"), p + "q", p + "sc"),
            (op_scale, p + "sc", "attn_scale", p + "sc"),
            (op_softmax, p + "sc", p + "sw"),
            *([] if TMV_CTX else [(op_trv, p + ("vc" if GROUPED_V else "vr"), p + "vt")]),
            (op_ctx, p + ("vc" if TMV_CTX else "vt"), p + "sw", p + "cx"),
            (op_o, p + "Wo", p + "cx", p + "a"),
        ]
        if sp.sandwich_norms:
            rl.append((op_norm, p + "a", p + "n_pa", p + "a"))
        rl += [
            (op_add, cur, p + "a", p + "x1"),
            (op_norm, p + "x1", p + "n_pf", p + "hf"),
            (op_gate, p + "Wg", p + "hf", p + "g"),
            (op_up, p + "Wu", p + "hf", p + "u"),
            (op_act, p + "g", p + "g"),
            (op_mul_ffn, p + "g", p + "u", p + "gh"),
            (op_down, p + "Wd", p + "gh", p + "d"),
        ]
        if sp.sandwich_norms:
            rl.append((op_norm, p + "d", p + "n_pff", p + "d"))
        rl.append((op_add, p + "x1", p + "d", nxt))
        bufsz[p + "x1"] = D * 2
        cur = nxt

    weights["n_final"] = load_norm("model.norm.weight")
    weights["W_head"] = bf16(npy("model.embed_tokens.weight")).reshape(-1)   # tied
    weights["attn_scale"] = np.full(Hq * S, sp.attn_scale, BF16)
    rl += [(op_norm, cur, "n_final", "xf"), (op_head, "W_head", "xf", "logits")]
    bufsz["xf"] = D * 2
    bufsz["logits"] = VOCAB * 2

    if os.environ.get("DUMP_OPS"):
        from collections import Counter
        c = Counter(type(e[0]).__name__ for e in rl)
        print(f"# {sp.name}: runlist {len(rl)} entries over NL={NL} "
              f"({(len(rl)-2)//NL}/layer + 2 tail)")
        for k, v in c.most_common():
            print(f"  {k:16} {v:5}")
        raise SystemExit(0)

    inputs = ["x", "rope_global"] + (["rope_local"] if sp.rope_theta_local is not None else [])
    # cores-per-col=1 spreads each operator's workers one per column instead of stacking them four
    # deep in two columns, which is what the default column-major SequentialPlacer does. Every op
    # here has <= 8 workers, so one per column fits the 8-column array. Overridable because this is
    # a placement experiment, not a settled default.
    #
    # THIS is the whole measured win: -11.14 ms/token, -7.1%, isolated on device with the transpose
    # batching held constant and DDR bytes identical at 3105.99 MB in every arm.
    placer_flags = os.environ.get("DECODE_PLACER_FLAGS", DECODE_PLACER_FLAGS_DEFAULT).split()
    # Two designs where one would do: gate/up are the same GEMV shape and adjacent, as are the two
    # KV StridedCopys. Each duplicate pair costs an extra aiex.configure PER LAYER -- 56 per token
    # against a measured ~40 us each. SHARE_DESIGNS=0 restores the unshared build for an A/B.
    share = os.environ.get("SHARE_DESIGNS", "1") == "1"
    fused = OperatorSequence(sequence_name(sp, NL, S, placer_flags), rl,
                              input_args=inputs, output_args=["logits"],
                              buffer_sizes=bufsz, context=ctx, extra_flags=placer_flags,
                              share_designs=share)
    fused.compile()
    return sp, fused, weights, dict(NL=NL, S=S, inputs=inputs, cache_names=cache_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS), help="model spec name")
    ap.add_argument("--weights", required=True, help="dir of dumped .npy weights (see dump_llm_weights.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (bring-up)")
    ap.add_argument("--max-seq", type=int, default=2048, help="KV-cache padded capacity S")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)
    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, a.max_seq)
    NL, S, inputs, cache_names = md["NL"], md["S"], md["inputs"], md["cache_names"]
    D, HD, Hq, Hkv, VOCAB = sp.d_model, sp.head_dim, sp.n_q_heads, sp.n_kv_heads, sp.vocab
    FF = sp.ffn
    elf = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    wnames = list(weights.keys())
    # Build the layout over what the graph DECLARES, never a hand-written list. The literal this
    # replaces omitted `rope_global` -- a declared input -- so every emitted meta.json described a
    # buffer set the ELF did not have, and a consumer placing buffers by layout could not find it.
    # IRON already computed the answer: `subbuffer_layout` covers every input, output and scratch
    # arg, and `calculate_buffer_layout` raises if a declared arg is missing from the runlist.
    lay = {n: fused.get_layout_for_buffer(n) for n in [*inputs, "logits", *wnames]}

    import glob
    import shutil
    # The project dir suffix moved from `.mlir.prj` to `.mlir.d`; match params.txt wherever aiecc
    # put it under the build tree rather than pinning a suffix that has already changed once.
    _pp = sorted(glob.glob("**/params.txt", recursive=True), key=os.path.getmtime)
    scratchpad_params = {}
    if _pp:
        shutil.copy(_pp[-1], os.path.join(a.out, "params.txt"))
        for line in open(_pp[-1]).read().splitlines()[1:]:
            if line.strip():
                n_, idx, ty, kind = line.split()
                scratchpad_params[n_] = {"byte_offset": int(idx) * 4, "kind": kind, "dtype": ty}

    bdir = os.path.join(a.out, "buffers")
    for n_, arr in weights.items():
        open(os.path.join(bdir, f"{n_}.bin"), "wb").write(weight_bytes(arr))
    open(os.path.join(a.out, "decode.elf"), "wb").write(elf)

    meta = {
        "spec": sp.name, "elf": "decode.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": inputs, "weights": wnames, "output": "logits",
        "scratchpad": {"params": scratchpad_params, "kv_param": "kv_off", "mask_param": "sm_mask",
                       "head_dim": HD, "kv_heads": Hkv},
        "dims": {"layers": NL, "d_model": D, "q_heads": Hq, "kv_heads": Hkv, "head_dim": HD,
                 "ffn": FF, "vocab": VOCAB, "S": S,
                 "sliding_window": sp.sliding_window, "sw_pattern": sp.sw_pattern},
        # Per-token host protocol (the ELF is constant; only these change):
        #   x        = embed[token], scaled by sqrt(d_model) iff embed_scale == "sqrt_d_model"
        #   rope_*   = precomputed [S,HD] angle tables; the row for n_past is used
        #   kv_off   = n_past * head_dim   (addr kind, element units, raw)
        #   sm_mask  = n_past + 1          (core kind, causal width; host writes it <<2)
        "host_protocol": {"embed_scale": sp.embed_scale, "attn_scale": float(sp.attn_scale),
                          "act": sp.act, "norm_gain": sp.norm_gain, "eps": sp.eps,
                          "rope_theta_global": sp.rope_theta_global,
                          "rope_theta_local": sp.rope_theta_local},
        "layer_types": ["global" if sp.is_global(l) else "sliding" for l in range(NL)],
        "cache_buffers": cache_names,
        # Engineering-check axis (see QUANT_MLP_DTYPE above), not a validated model default.
        "weight_quant": {"mlp_dtype": QUANT_MLP_DTYPE, "mlp_group_size": QUANT_MLP_GROUP},
    }
    prov = toolchain_provenance()
    if prov:
        meta["toolchain"] = prov
    else:
        print("[build] WARNING: could not record toolchain provenance in meta.json "
              "(no toolchain.lock / kernel_sandbox.sh resolvable) -- this artifact will read as "
              "unstamped to any freshness check", file=sys.stderr)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"\nwrote {NL}-layer {sp.name} decode ELF ({len(elf)}B, scratch {scr/1e6:.1f}MB) to {a.out}")


if __name__ == "__main__":
    main()
