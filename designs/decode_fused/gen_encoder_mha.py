#!/usr/bin/env python3
"""Build the Whisper-encoder full-attention (MHA) ELF — roadmap part-3 (encoder MHA on NPU).

Uses a STATIC-SHAPE variant of the IRON MHA operator. The stock IRON MHA op reads its flash-attention
loop bounds (num_kv_blocks) from a runtime RTP buffer and guards the last-KV-block matmul with
`with if_(loop_idx_kv > 1)`, which puts an objectfifo acquire/release *inside* an scf.if. Our vendored
mlir-aie 1.3.2 `AIEObjectFifoStatefulTransform` deliberately REFUSES conditional acquire/release
(see internal notes), so the stock op does not build here
for ANY config.

For the encoder the loop bounds are compile-time constants (seq_len=1500 fixed -> num_kv_blocks=24), so
`mha_static_design.py` replaces the runtime RTP reads with those constants and turns the `with if_(...)`
guards into plain Python `if`s -> only the live branch is emitted, no scf.if, no conditional acquire.

causal=False (non-causal / bidirectional = the encoder's full attention), d=64, heads per --heads, seq_len=1500
(pads to 1536 = 24x64). Q/K/V/O are each [heads*d*seq_pad] bf16 flat. Replaces the ~300 ms/utt host MHA.

Usage (iron env, like build_projout_elf.sh): python gen_encoder_mha.py --out <dir> [--pipelines 8]
"""
import argparse, glob, json, os, shutil
from pathlib import Path
import newstack_compat  # noqa: F401 — MUST precede iron imports
from iron.common import AIEContext
from iron.common import PythonGeneratedMLIRArtifact, DesignGenerator
from iron.operators.mha.op import MHA
import aie.utils as aie_utils
from mha_static_design import MEMTILE_DMA_CHANNELS, join_distribute_width

# whisper-small's shape; every Whisper size shares d=64 and s=1500 (max_source_positions) and
# differs only in head count -- large-v3-turbo is 20 heads, not 12. Overridable per build.
HEADS, D, SEQ = 12, 64, 1500
STATIC_DESIGN = Path(__file__).resolve().parent / "mha_static_design.py"


class StaticMHA(MHA):
    """MHA whose MLIR comes from the local static-shape design (no conditional objectfifo acquire).

    Also builds mha.o NON-CAUSALLY. IRON's mha.cc masks the upper triangle at four sites; the
    encoder is bidirectional, so all four must be compiled out. Causality is a property of the
    KERNEL, not of the design -- mha_static_design.py has no causal branch, which is why the
    masking survived every review of the design file.
    """

    def get_kernel_artifacts(self):
        arts = super().get_kernel_artifacts()
        # Default ON: the encoder is bidirectional. ENC_MHA_NONCAUSAL=0 restores the stock causal
        # kernel, which exists only as the control arm for measuring what the masking costs.
        if os.environ.get("ENC_MHA_NONCAUSAL", "1") != "0":
            for a in arts:
                # the mha.o translation unit -- identified by its own defines, not by filename
                if "-Dbf16_bf16_ONLY" in getattr(a, "extra_flags", []):
                    a.extra_flags = list(a.extra_flags) + ["-DMHA_NONCAUSAL"]
        # TIMING PROBE, opt-in and never a default. ENC_MHA_NOOP_STAGES is a comma-separated
        # subset of {qk, softmax, pv}: each named stage's kernel compiles to an immediate return, so
        # its COMPUTE leaves the mha stage while the design's objectFIFO handshakes stay. The
        # ENC_PEROP delta against an otherwise identical build is that stage's arithmetic share,
        # which is how the three-row pipeline's limiter gets named without a trace.
        # Requires an IRON tree whose mha.cc carries the matching MHA_<STAGE>_NOOP guards.
        # Numerically wrong by construction -- an artifact built with any of these must never ship.
        _stages = [x.strip().lower() for x in os.environ.get("ENC_MHA_NOOP_STAGES", "").split(",") if x.strip()]
        _known = {"qk": "MHA_QK_NOOP", "softmax": "MHA_SOFTMAX_NOOP", "pv": "MHA_PV_NOOP"}
        for _st in _stages:
            if _st not in _known:
                raise SystemExit(f"ENC_MHA_NOOP_STAGES: unknown stage {_st!r}; want a subset of {sorted(_known)}")
        if _stages:
            for a in arts:
                if "-Dbf16_bf16_ONLY" in getattr(a, "extra_flags", []):
                    a.extra_flags = list(a.extra_flags) + [f"-D{_known[x]}" for x in _stages]
        return arts

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                STATIC_DESIGN,
                "fused_mha",
                (),
                {
                    "dev": aie_utils.DefaultNPURuntime.device(),
                    "heads": self.num_heads,
                    "S_q": self.seq_len,
                    "S_kv": self.seq_len,
                    "d": self.d,
                    "B_q": self.B_q,
                    "B_kv": self.B_kv,
                    "num_KV_heads": self.num_KV_heads,
                    "number_of_pipelines": self.num_of_pipelines,
                    "emulate_bf16_mmul_with_bfp16": True,
                    "trace_size": 0,
                    "verbose": False,
                },
            ),
        )


def _validate_pipelines(n):
    """Reject a --pipelines the array cannot realise, naming the knob and the budget.

    Both limits below are enforced by aiecc anyway, but only after generation and against a TILE:
    `--pipelines 6` reports "tile (6, 1) requires 6 input/1 output DMA channels" and `--pipelines 10`
    reports "column index (8) must be less than the number of columns in the device (8)". Neither
    names --pipelines, and a caller reading either has to reconstruct which knob produced the tile.

    The rule: a tiling must FIT, not merely divide, and the place to check that is where the value
    is chosen.
    """
    dev = aie_utils.DefaultNPURuntime.device()
    ok = [k for k in range(1, dev.cols + 1) if _pipelines_problem(k, dev) is None]
    problem = _pipelines_problem(n, dev)
    if problem is None:
        return
    raise SystemExit(f"--pipelines {n}: {problem}. Valid here: {', '.join(map(str, ok))}.")


def _pipelines_problem(n, dev):
    """Why n pipelines cannot be realised, or None. Three independent limits, measured 2026-09-08.

    The valid set is not a range -- 6 fails between working 5 and 8 -- because the two-MemTile path
    above 6 pipelines changes which limit binds.
    """
    if n < 1 or n > dev.cols:
        return (f"the design places one AIE column per pipeline and this device has {dev.cols} "
                f"({type(dev).__name__})")
    if n > 6 and n % 2:
        # Above 6 the split feeds two MemTiles of n//2 outputs each, so memQ holds 2*(n//2) entries
        # while the worker loop runs to n. Odd n indexes past the end (IndexError in fused_mha).
        return f"above 6 the design splits over two MemTiles and needs an even count, not {n}"
    join = join_distribute_width(n)
    if join + 1 > MEMTILE_DMA_CHANNELS:
        return (f"the inQ split and memO join share one MemTile, needing {join + 1} DMA channels in "
                f"each direction ({join}-way fan-out plus the shim side) against "
                f"{MEMTILE_DMA_CHANNELS} per MemTile")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipelines", type=int, default=8,
                    help="AIE columns used (tested upstream at 4/8; static design needs no specific value)")
    ap.add_argument("--heads", type=int, default=HEADS,
                    help="attention heads (whisper-small 12, large-v3-turbo 20)")
    ap.add_argument("--d", type=int, default=D, help="head dim (64 for every Whisper)")
    ap.add_argument("--seq", type=int, default=SEQ, help="encoder sequence length")
    a = ap.parse_args()
    heads, d, seq = a.heads, a.d, a.seq
    _validate_pipelines(a.pipelines)
    os.makedirs(a.out, exist_ok=True)

    ctx = AIEContext()
    # `causal` is not a constructor field -- the current IRON MHA dropped it. That is NOT the same
    # as being non-causal: mha.cc masks unconditionally, so non-causality comes from StaticMHA's
    # -DMHA_NONCAUSAL above, not from the absence of a `causal` argument.
    op = StaticMHA(num_heads=heads, seq_len=seq, d=d, num_KV_heads=0,
                   num_of_pipelines=a.pipelines, context=ctx)
    seq_pad = op._calculate_seq_padding(seq, a.pipelines)
    bufelems = heads * d * seq_pad
    print(f"StaticMHA(h={heads}, s={seq}->pad{seq_pad}, d={d}, causal=False, pipelines={a.pipelines}); "
          f"Q/K/V/O = {bufelems} bf16 each, name={op.name}")
    op.compile()
    bd = str(ctx.build_dir)
    xclbins = glob.glob(os.path.join(bd, "*.xclbin"))
    instss = glob.glob(os.path.join(bd, "*insts*")) + glob.glob(os.path.join(bd, "*.bin"))
    for f in xclbins + instss:
        shutil.copy(f, a.out)
    meta = {
        "kernel_name": "main:sequence", "op_name": op.name,
        "heads": heads, "d": d, "seq": seq, "seq_pad": seq_pad, "causal": False,
        "pipelines": a.pipelines, "buf_elems": bufelems,
        "io": "Q,K,V in + O out, each [heads*d*seq_pad] bf16", "design": "mha_static_design.py",
        "xclbin": [os.path.basename(f) for f in xclbins], "insts": [os.path.basename(f) for f in instss],
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"wrote encoder MHA artifacts ({len(xclbins)} xclbin, {len(instss)} insts) to {a.out}")


if __name__ == "__main__":
    main()
