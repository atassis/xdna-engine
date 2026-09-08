#!/usr/bin/env python3
"""Check an LlmSpec against the weight dump it claims to describe.

Why this exists. A spec is a hand-typed transcription of a model, and the two ways it goes wrong are
both silent. Copying an axis from a sibling model builds and runs -- `norm_gain="one_plus_w"`,
inherited from Gemma-3 by family resemblance, would have been wrong on all 193 of Gemma-4's norm
tensors and no build gate could have seen it, because load_norm folds the gain into the weight
buffer. And copying an axis from `config.json` trusts a file that is sometimes silent about what the
checkpoint actually contains: `layer_scalar` is a register_buffer, so it is in the weights and not in
the config at all.

So the dump is the authority and this asks it directly. Every check is an arithmetic identity
between a spec field and a byte count or an array shape on disk, which means a wrong axis is a
failed identity rather than a plausible number.

It is deliberately runnable against EVERY spec, not just the one it was written for. A checker that
only passes on the case that motivated it has not been tested -- run it on qwen3-0.6b and
gemma3-270m too, where it must be equally green.

    scripts/check_llm_spec_against_dump.py --spec gemma4-12b --weights artifacts/gemma4-12b/weights

Needs the iron venv on PYTHONPATH: the packed row stride is `iron.operators.gemv.quant`'s to
compute, and a second copy of that arithmetic here is exactly the seam (K014) this file is checking
for elsewhere.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "designs", "decode_fused"))
from llm_decode_spec import SPECS, k_chunks_for  # noqa: E402

COLS = 8   # gen_llm_decode.py's GEMV column count; k_chunks_for is written against it


class Checker:
    def __init__(self, sp, wdir):
        self.sp, self.wdir = sp, wdir
        self.ok, self.bad = [], []
        qmf = os.path.join(wdir, "quant.json")
        self.manifest = json.load(open(qmf)) if os.path.isfile(qmf) else {}
        self.packed = set(self.manifest.get("packed", []))

    # -- primitives -------------------------------------------------------------------------
    def path(self, name):
        return os.path.join(self.wdir, name + ".npy")

    def exists(self, name):
        return os.path.isfile(self.path(name))

    def shape(self, name):
        """Shape, or None with a named disagreement if the tensor is not there.

        Raising instead would report a PATH where the checker already knows the AXIS: a wrong
        `weight_prefix` or a wrong `v_from_k_on_global` both present as a missing file, and a
        traceback naming a .npy is the dual of the bug this file exists to catch. Found by the
        negative control -- perturbing either axis crashed rather than reported.
        """
        if not self.exists(name):
            self.bad.append((f"{name}: tensor present in the dump", False, True))
            return None
        return np.load(self.path(name), mmap_mode="r").shape

    def layer(self, l, leaf):
        return f"{self.sp.weight_prefix}layers.{l}.{leaf}"

    def check(self, label, got, want):
        (self.ok if got == want else self.bad).append((label, got, want))

    # -- the axes ---------------------------------------------------------------------------
    def rows_of(self, name, K):
        """Rows in a weight tensor, whether it is packed on the wire or a plain [M, K] array.

        The packed case is the reason this file needs the iron import: the dump is a flat byte
        array, so the only way back to a row count is the operator's own row stride.
        """
        s = self.shape(name)
        if s is None:
            return None
        if name in self.packed:
            from iron.operators.gemv.quant import row_stride_bytes
            stride = row_stride_bytes(K, self.manifest["group_size"], self.manifest["dtype"])
            n = s[0]
            if n % stride:
                self.bad.append((f"{name}: bytes divisible by row stride", n % stride, 0))
                return None
            return n // stride
        return s[0]

    def run(self):
        sp = self.sp
        # d_model and vocab, from tensors whose shape is unambiguous in every dump format.
        s = self.shape(f"{sp.weight_prefix}norm.weight")
        self.check("d_model (final norm)", s and s[0], sp.d_model)
        s = self.shape(f"{sp.weight_prefix}embed_tokens.weight")
        self.check("vocab, d_model (embedding)", s and tuple(s), (sp.vocab, sp.d_model))
        # n_layers: the first layer index with no input_layernorm is the depth.
        depth = 0
        while self.exists(self.layer(depth, "input_layernorm.weight")):
            depth += 1
        self.check("n_layers", depth, sp.n_layers)

        # Per-layer attention geometry. head_dim comes from q_norm, which is head_dim-wide and
        # never packed; that is what makes it the anchor for the packed row-stride arithmetic below.
        for l in range(min(depth, sp.n_layers)):
            hd, kvh = sp.head_dim_for(l), sp.n_kv_heads_for(l)
            if sp.qk_norm:
                for side in ("q", "k"):
                    s = self.shape(self.layer(l, f"self_attn.{side}_norm.weight"))
                    self.check(f"L{l} head_dim ({side}_norm)", s and s[0], hd)
            self.check(f"L{l} q_dim (q_proj rows)",
                       self.rows_of(self.layer(l, "self_attn.q_proj.weight"), sp.d_model),
                       sp.n_q_heads * hd)
            self.check(f"L{l} kv_dim (k_proj rows)",
                       self.rows_of(self.layer(l, "self_attn.k_proj.weight"), sp.d_model),
                       kvh * hd)
            # attention_k_eq_v: v_proj must be absent EXACTLY on the global layers, and present
            # everywhere else. Both halves matter -- absence alone would also be satisfied by a
            # truncated dump.
            want_v = not (sp.v_from_k_on_global and sp.is_global(l))
            self.check(f"L{l} v_proj present",
                       self.exists(self.layer(l, "self_attn.v_proj.weight")), want_v)
            if want_v:
                self.check(f"L{l} kv_dim (v_proj rows)",
                           self.rows_of(self.layer(l, "self_attn.v_proj.weight"), sp.d_model),
                           kvh * hd)
            self.check(f"L{l} layer_scalar present",
                       self.exists(self.layer(l, "layer_scalar")), sp.layer_scalar)
            # ffn, and the K-split the L1 fit model predicts. The dump made its own chunking
            # decision; if ours disagrees the generator looks for weights that are not there.
            self.check(f"L{l} ffn (gate_proj rows)",
                       self.rows_of(self.layer(l, "mlp.gate_proj.weight"), sp.d_model), sp.ffn)
            self.chunks(l, "mlp.down_proj.weight", k_chunks_for(sp.d_model, sp.ffn, COLS))
            self.chunks(l, "self_attn.o_proj.weight",
                        k_chunks_for(sp.d_model, sp.n_q_heads * hd, COLS))
        return self

    def unverifiable(self):
        """Axes this instrument CANNOT check, named -- because a silent gap reads as coverage.

        `norm_gain` is the important one, and it is the axis most likely to be wrong: it is the one
        a family-resemblance copy gets backwards. The negative control caught that perturbing it
        produced ZERO disagreements here.

        It cannot be fixed by looking harder at the weights. The obvious statistic -- "under (1+w)
        trained weights sit near 0" -- is FALSE, measured on our own two Gemma checkpoints:
        gemma3-270m IS `one_plus_w` and its raw norm weights run mean 12.5-25.4 (max 300), LARGER
        than gemma4-12b's 6.6-20.3 which is `w`. Meanwhile qwen3-0.6b, also `w`, sits at 0.14-0.19.
        The statistic does not separate the conventions in either direction, so no threshold on it
        is sound. `norm_gain` is a property of the modeling CODE (Gemma3RMSNorm: zeros(dim) and
        `output * (1.0 + weight)`; Gemma4UnifiedRMSNorm: ones(dim) and `normed * weight`), and code
        is where it has to be read.
        """
        return [
            "norm_gain: not derivable from weights -- see this method's docstring; read "
            "<Model>RMSNorm.__init__/forward in the modeling source instead",
            "eps, rope_theta_*, sliding_window, act, embed_scale: config/code axes with no "
            "signature in the dumped tensors",
        ]

    def chunks(self, l, leaf, want):
        """How many K-chunks the dump actually shipped for one weight."""
        base = self.layer(l, leaf)
        got = 1 if self.exists(base) else sum(1 for i in range(64) if self.exists(f"{base}.kchunk{i}"))
        self.check(f"L{l} {leaf.split('.')[-2]} K-chunks", got, want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--verbose", action="store_true", help="print the passing identities too")
    a = ap.parse_args()
    c = Checker(SPECS[a.spec], a.weights).run()
    if a.verbose:
        for label, got, _ in c.ok:
            print(f"  ok   {label}: {got}")
    for label, got, want in c.bad:
        print(f"  FAIL {label}: dump says {got}, spec says {want}")
    for u in c.unverifiable():
        print(f"  n/a  {u}")
    print(f"{a.spec}: {len(c.ok)} identities hold, {len(c.bad)} disagree, "
          f"{len(c.unverifiable())} axis group(s) NOT checkable here")
    return 1 if c.bad else 0


if __name__ == "__main__":
    sys.exit(main())
