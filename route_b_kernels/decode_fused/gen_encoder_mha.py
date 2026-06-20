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

causal=False (non-causal / bidirectional = the encoder's full attention), d=64, heads=12, seq_len=1500
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

HEADS, D, SEQ = 12, 64, 1500
STATIC_DESIGN = Path(__file__).resolve().parent / "mha_static_design.py"


class StaticMHA(MHA):
    """MHA whose MLIR comes from the local static-shape design (no conditional objectfifo acquire)."""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipelines", type=int, default=8,
                    help="AIE columns used (tested upstream at 4/8; static design needs no specific value)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ctx = AIEContext()
    op = StaticMHA(num_heads=HEADS, seq_len=SEQ, d=D, num_KV_heads=0, causal=False,
                   num_of_pipelines=a.pipelines, context=ctx)
    seq_pad = op._calculate_seq_padding(SEQ, a.pipelines)
    bufelems = HEADS * D * seq_pad
    print(f"StaticMHA(h={HEADS}, s={SEQ}->pad{seq_pad}, d={D}, causal=False, pipelines={a.pipelines}); "
          f"Q/K/V/O = {bufelems} bf16 each, name={op.name}")
    op.compile()
    bd = str(ctx.build_dir)
    xclbins = glob.glob(os.path.join(bd, "*.xclbin"))
    instss = glob.glob(os.path.join(bd, "*insts*")) + glob.glob(os.path.join(bd, "*.bin"))
    for f in xclbins + instss:
        shutil.copy(f, a.out)
    meta = {
        "kernel_name": "main:sequence", "op_name": op.name,
        "heads": HEADS, "d": D, "seq": SEQ, "seq_pad": seq_pad, "causal": False,
        "pipelines": a.pipelines, "buf_elems": bufelems,
        "io": "Q,K,V in + O out, each [heads*d*seq_pad] bf16", "design": "mha_static_design.py",
        "xclbin": [os.path.basename(f) for f in xclbins], "insts": [os.path.basename(f) for f in instss],
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"wrote encoder MHA artifacts ({len(xclbins)} xclbin, {len(instss)} insts) to {a.out}")


if __name__ == "__main__":
    main()
