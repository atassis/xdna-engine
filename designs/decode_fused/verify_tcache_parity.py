#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-arm gate for the M0.5 transposed self-V cache (`gen_decode --coalesce-self-tr`).

Nothing has ever driven the 4-parameter contract the tr arm declares. `verify_fused_decode_sp.py`
cannot: it REBUILDS the op graph in-process, so it only ever speaks the 2-parameter baseline, and
mirroring gen_decode's graph there would duplicate the generator. This CONSUMES THE ARTIFACT
instead -- decode.elf plus meta.json plus buffers/ -- so whatever the generator emitted is what
gets driven.

THE CONTRACT, per token n_self (= num_preceding + step):

    kv_off      addr    n_self * head_dim   K cache stays [H,S,HD]
    sm_mask     core    n_self + 1          self-softmax width
    vcache_off  addr    n_self & ~1         EVEN column base of the transposed V cache
    v_par       core    n_self & 1          which slot of the staged pair this token writes

`vcache_off` is the failure to look for: write `n_self` and even tokens still land while odd ones
overwrite their predecessor, silently. The 4-byte bf16 granule spans columns (n & ~1, n | 1) and an
odd element offset truncates down into it, which is why the pair is staged core-side at all.

THE GATE is two-arm parity rather than an f32 V reference: drive the tr and non-tr ELFs with
identical inputs over consecutive tokens and require the same hidden state out of each at every
step. Same weights, same random encoder, same seeded k_past/v_past -- the arms differ only in how V
reaches the cache, so any divergence is the cache. Token argmax parity rides on top when the
decoder weights are available; it is the weaker check (a projection of the hidden state) and is
reported, not relied on.

Build the two arms from ONE generator invocation each, differing in exactly one flag:

    GEN_EXTRA=--coalesce-self-tr bash scripts/build_deepc_decode.sh 2 /tmp/tcache_tr
    bash scripts/build_deepc_decode.sh 2 /tmp/tcache_base
    python verify_tcache_parity.py --tr /tmp/tcache_tr --base /tmp/tcache_base --steps 443

Do NOT pair an arm built with --coalesce-cross against one without: that moves the cross-attention
V layout too and the gate stops isolating the self cache.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

import newstack_compat  # noqa: F401 -- MUST precede iron imports (new-mlir-aie port shim)
import pyxrt
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from elf_dispatch_compat import FullELFCallable
from param_scratchpad_compat import get_parameter_scratchpad

BF16 = ml_dtypes.bfloat16
SOT = 50258
# A value no kernel produces, so "the device never wrote here" is distinguishable from "the device
# wrote a zero". Reading zeros is what a dead dispatch and a correct one look like alike, and the
# whole three-week pre-fill misattribution came from not being able to tell them apart.
SENTINEL = np.float32(7168.0)


def ln(x, g=None, b=None, eps=1e-5):
    x = np.asarray(x, np.float32)
    y = (x - x.mean()) / np.sqrt(x.var() + eps)
    return y if g is None else y * g + b


class Arm:
    """One decode ELF, driven straight from its build directory."""

    def __init__(self, path):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.params = self.meta["scratchpad"]["params"]
        dev, seq = self.meta["kernel_name"].split(":")
        elf = np.frombuffer((self.dir / self.meta["elf"]).read_bytes(), dtype=np.uint32)
        self.callable_ = FullELFCallable(elf, device_name=dev, sequence_name=seq)

        isz, osz, ssz = (self.meta[k] for k in ("input_size", "output_size", "scratch_size"))
        item = np.dtype(BF16).itemsize
        self.arena = {
            "input": XRTTensor((isz // item,), dtype=BF16),
            "output": XRTTensor((osz // item,), dtype=BF16),
            "scratch": XRTTensor((ssz // item,), dtype=BF16),
        }
        self.x = self._view(self.meta["inputs"][0])
        self.y = self._view(self.meta["output"])

        # Weights, K/V caches and the seeded k_past/v_past all arrive as buffers/*.bin already in
        # the layout the ELF expects -- including the vcache, which the generator writes
        # transposed [H,HD,S] for the tr arm. Loading them rather than recomputing them is what
        # keeps the two arms bit-identical up to the one thing under test.
        for name in self.meta["weights"]:
            raw = (self.dir / "buffers" / f"{name}.bin").read_bytes()
            _, _, length = self._layout(name)
            if len(raw) != length:
                raise ValueError(f"{name}.bin is {len(raw)} bytes, layout says {length}")
            with self._view(name).overwrite() as dst:
                np.copyto(dst, np.frombuffer(raw, dtype=BF16))
        self.arena["scratch"].device = "cpu"
        self.arena["scratch"].to("npu")

        self.run = pyxrt.run(self.callable_.xrt_kernel)
        for i, kind in enumerate(("input", "output", "scratch")):
            self.run.set_arg(i, self.arena[kind].buffer_object())
        self.sp = get_parameter_scratchpad()(self.run, str(self.dir / "params.txt"))

    def _layout(self, name):
        e = self.meta["layout"][name]
        return e["type"], e["offset"], e["len"]

    def _view(self, name):
        kind, offset, length = self._layout(name)
        item = np.dtype(BF16).itemsize
        return self.arena[kind].subview(offset, (length // item,), BF16)

    def step(self, x, n_self):
        """One token. Returns (hidden, unwritten) -- unwritten counts surviving sentinel words."""
        with self.x.overwrite() as dst:
            np.copyto(dst, np.asarray(x, BF16).reshape(-1))
        self.arena["input"].device = "cpu"
        self.arena["input"].to("npu")

        # Pre-filling the OUTPUT arena is only safe with the flush that follows: an unflushed
        # pre-fill leaves dirty host lines over the bytes the DMA is about to write, the
        # device-to-host sync does not discard them, and the caller reads back its own pre-fill.
        # That is the defect IRON e49245f fixed inside FusedFullELFCallable; this harness drives
        # pyxrt directly, so it owns the flush itself.
        with self.y.overwrite() as dst:
            dst[:] = np.asarray(SENTINEL, BF16)
        self.arena["output"].device = "cpu"
        self.arena["output"].to("npu")

        for name, value in self.token_params(n_self).items():
            self.sp.write(name, value)
        self.sp.sync()
        self.run.start()
        self.run.wait2()

        # Dispatching through pyxrt directly means nothing marks the output device-written, so
        # `to("cpu")` would no-op against the host copy this loop just wrote and every token after
        # the first would read stale bytes.
        self.arena["output"].device = "npu"
        self.arena["output"].to("cpu")
        got = np.array(self.y.data, copy=True)
        unwritten = int((np.asarray(got, BF16).astype(np.float32) == SENTINEL).sum())
        return got, unwritten

    def vcache(self, layer=0):
        """The layer's self-V cache as [H, S, HD], whichever way the arm stores it.

        Reads back the whole scratch arena, so this is a diagnostic, not something to call in the
        measured loop. It is what turns "the hidden state diverged" into "column c is wrong".
        """
        self.arena["scratch"].device = "npu"
        self.arena["scratch"].to("cpu")
        H, HD, S = 12, self.meta["scratchpad"]["head_dim"], self.meta["dims"]["S"]
        v = np.array(self._view(f"L{layer}_vcache").data, copy=True).astype(np.float32)
        return v.reshape(H, HD, S).transpose(0, 2, 1) if self.meta.get("coalesce_self_tr") \
            else v.reshape(H, S, HD)

    def token_params(self, n_self):
        hd = self.meta["scratchpad"]["head_dim"]
        p = {"kv_off": n_self * hd, "sm_mask": n_self + 1}
        if self.meta.get("coalesce_self_tr"):
            p[self.meta["vcache_param"]] = n_self & ~1
            p[self.meta["vparity_param"]] = n_self & 1
        if set(p) != set(self.params):
            raise ValueError(f"{self.dir}: drives {sorted(p)}, ELF declares {sorted(self.params)}")
        return p


def check_pair(tr, base):
    """Refuse a comparison whose two arms differ in more than the self-V cache."""
    if not tr.meta.get("coalesce_self_tr"):
        raise SystemExit(f"--tr {tr.dir} was not built with --coalesce-self-tr")
    if base.meta.get("coalesce_self_tr"):
        raise SystemExit(f"--base {base.dir} IS a tr arm; it is the baseline")
    for k in ("coalesce_cross", "coalesce_self", "int8_cross_k", "int8_cross_v", "int8_ffn",
              "int8_attn_w", "npu_logits"):
        if tr.meta.get(k) != base.meta.get(k):
            raise SystemExit(
                f"arms differ in {k} ({tr.meta.get(k)} vs {base.meta.get(k)}) as well as the "
                "self-V cache -- rebuild them differing in --coalesce-self-tr alone"
            )
    if tr.meta["dims"] != base.meta["dims"]:
        raise SystemExit(f"arms differ in dims: {tr.meta['dims']} vs {base.meta['dims']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr", required=True, help="build dir of the --coalesce-self-tr arm")
    ap.add_argument("--base", required=True, help="build dir of the baseline arm")
    ap.add_argument("--steps", type=int, default=64, help="tokens to advance (0 = the full width)")
    ap.add_argument("--weights", default="/mnt/data/models/xdna-artifacts/whisper-small/"
                                         "whisper_decoder",
                    help="decoder weights, for the token-argmax report; skipped when absent")
    # The cache is seeded with num_preceding columns, so the first token lands at an ODD column
    # whenever num_preceding is odd. That is a resume, not a cold start, and it is the case a
    # sequential fill from column 0 never reaches -- hence the override.
    ap.add_argument("--first-nself", type=int, default=None,
                    help="column the first token writes (default: num_preceding)")
    ap.add_argument("--vcache-check", type=int, default=0, metavar="N",
                    help="after each of the first N steps, diff the two arms' L0 self-V caches "
                         "column by column (reads the whole scratch arena; slow)")
    a = ap.parse_args()

    base = Arm(a.base)
    tr = Arm(a.tr)
    check_pair(tr, base)
    dims = base.meta["dims"]
    P = a.first_nself if a.first_nself is not None else base.meta["scratchpad"]["num_preceding"]
    steps = a.steps or dims["S"] - P
    if P + steps > dims["S"]:
        raise SystemExit(f"{steps} steps from n_self={P} runs past the S={dims['S']} cache width")
    print(f"[tcache] tr={tr.dir} base={base.dir}")
    print(f"[tcache] layers={dims['layers']} S={dims['S']} P={P} steps={steps} "
          f"(n_self {P}..{P + steps - 1})", flush=True)

    wdir = Path(a.weights)
    proj = None
    if (wdir / "proj_out.weight.npy").exists():
        emb_t = np.load(wdir / "embed_tokens.npy").astype(np.float32)
        emb_p = np.load(wdir / "embed_positions.npy").astype(np.float32)
        lnp_w = np.load(wdir / "ln_post.weight.npy").astype(np.float32)
        lnp_b = np.load(wdir / "ln_post.bias.npy").astype(np.float32)
        proj = np.load(wdir / "proj_out.weight.npy").astype(np.float32)
        print(f"[tcache] token report on: embeddings {emb_t.shape}, proj_out {proj.shape}")
    else:
        print(f"[tcache] no decoder weights at {wdir}; hidden-state parity only")

    D = base.meta["input_size"] // 2
    rng = np.random.default_rng(0)

    # The baseline drives the token stream and the tr arm REPLAYS its input, so the two arms see
    # one identical stream. A greedy loop per arm would fork at the first divergence and then hide
    # the cause behind different inputs.
    exact_steps, worst, first_bad = 0, 0.0, None
    dead = dead_tr = tok_match = 0
    tok = SOT
    for n in range(steps):
        x = (np.asarray(emb_t[tok] + emb_p[P + n], BF16) if proj is not None
             else np.asarray(rng.standard_normal(D), BF16))
        ref, unwritten = base.step(x, P + n)
        dead += unwritten
        h, unwritten = tr.step(x, P + n)
        dead_tr += unwritten

        same = int((np.asarray(h, BF16) == np.asarray(ref, BF16)).sum())
        if same == h.size:
            exact_steps += 1
        elif first_bad is None:
            first_bad = (n, P + n, (P + n) & 1, same, h.size)
        worst = max(worst, float(np.abs(h.astype(np.float32) - ref.astype(np.float32)).max()))
        if proj is not None:
            tok = int(np.argmax(ln(ref[0:D], lnp_w, lnp_b) @ proj))
            tok_match += tok == int(np.argmax(ln(h[0:D], lnp_w, lnp_b) @ proj))

        if n < a.vcache_check:
            vb, vt = base.vcache(0), tr.vcache(0)
            bad = [c for c in range(dims["S"]) if not np.array_equal(vb[:, c, :], vt[:, c, :])]
            print(f"  step {n} n_self={P + n} parity={(P + n) & 1} "
                  f"vcache_off={(P + n) & ~1}: columns differing = "
                  f"{bad if len(bad) <= 8 else str(bad[:8]) + f' (+{len(bad) - 8})'}", flush=True)

    odd = sum(1 for n in range(steps) if (P + n) & 1)
    print(f"\n[tcache] hidden-state parity: {exact_steps}/{steps} steps bit-identical, "
          f"max |diff| {worst:g}")
    print(f"[tcache] parities exercised: {steps - odd} even, {odd} odd; "
          f"sentinel words surviving: base {dead}, tr {dead_tr}")
    if proj is not None:
        print(f"[tcache] token parity: {tok_match}/{steps} argmax tokens match")
    if first_bad:
        n, n_self, par, same, size = first_bad
        print(f"[tcache] first divergence at step {n} (n_self={n_self}, parity={par}): "
              f"{same}/{size} elements identical")

    ok = exact_steps == steps and dead == 0 and dead_tr == 0
    if ok and proj is not None:
        ok = tok_match == steps
    print("\n*** M0.5 PARITY PASS -- the transposed cache advances one token per dispatch, both "
          "parities, and the arms agree bit-for-bit ***" if ok else
          "*** M0.5 PARITY FAIL -- read the first divergence; an odd-only failure means "
          "vcache_off is carrying n_self instead of n_self & ~1 ***")
    del tr, base
    gc.collect()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
