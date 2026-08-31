#!/usr/bin/env python3
"""Was gatedeltanet's NaN the same copy-then-read ordering fault as rope-lut's?

WHY. rope-lut's pinned-aie_api damage turned out to be ORDERING, not arithmetic: a scalar copy loop
that writes a buffer, immediately followed by a vector loop that reads it, had the reads scheduled
ahead of the writes.
`gatedeltanet_core` opens with exactly that shape:

    for (i) { row = load_v(&s_in[i]); store_v(&s_out[i], row); }   // seed s_out from s_in
    for (t = 0; t < T; ++t) { ... gdn_state_read(s_out, ...) ... } // first iteration READS s_out

If the same fault applies, step 0 reads unwritten state and poisons the recurrence -- which is what
the NaN looked like, entering at t=2 with steps 0-1 clean. It would also explain why hoisting one
scalar multiply "fixed" it with no mechanism ever found: the edit changed the schedule.

TWO PRIOR MECHANISMS FOR THIS BRICK ALREADY FAILED A DIRECT TEST (the scalar multiply is fine in
isolation; the iteration count does not degrade it), so this one earns nothing without one either.

THREE ARMS, all on the PRE-FIX kernel (the one that went NaN), so the copy is the only variable:
  1 copy + static scratch  -- the original shape. Expect NaN; this is the control that proves the
                              arm reproduces the bug at all.
  2 copy + host-zeroed s_out -- same copy, but s_out is an INPUT buffer the host filled with zeros.
                              Isolates "where s_out lives" from "is there a copy".
  3 NO copy + host-zeroed s_out -- the seeding loop deleted. Semantically identical because s_in is
                              all zeros and the host already zeroed s_out, so the copy was a no-op.

3 green while 1 and 2 are NaN => the copy-then-read ordering IS the mechanism.
3 still NaN                   => it is not; the hypothesis dies like the previous two.

Run:  ./run.sh probe_gdn_copy_then_read.py
"""
import importlib.util
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron
import bricklib

HERE = Path(__file__).parent
GEN = HERE / "gen"
GEN.mkdir(exist_ok=True)
BRICK = (HERE.parent / "gatedeltanet").resolve()
BF = ml_dtypes.bfloat16
T, DK, DV = 64, 32, 32
PRE_FIX_REV = "a6d6b47~1"   # the commit before the vector-domain gate fix


def _load_golden():
    spec = importlib.util.spec_from_file_location("gdn_golden", BRICK / "golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- materialise the PRE-FIX kernel, and a no-seeding-copy variant of it ---
src = subprocess.run(["git", "show", f"{PRE_FIX_REV}:route_b_kernels/bricks/gatedeltanet/gatedeltanet.cc"],
                     cwd=HERE.parent.parent.parent, capture_output=True, text=True, check=True).stdout
assert "float bk = beta * k[i];" in src, "this is not the pre-fix kernel"
pre = GEN / "gdn_prefix.cc"
pre.write_text(src)

SEED = """  for (unsigned i = 0; i < DK * DV; i += DV) {
    aie::vector<float, DV> row = aie::load_v<DV>(&s_in[i]);
    aie::store_v(&s_out[i], row);
  }
"""
assert SEED in src, "seeding loop text not found -- re-read the pre-fix source"
nocopy = GEN / "gdn_prefix_nocopy.cc"
nocopy.write_text(src.replace(SEED, "  // seeding copy REMOVED: s_in is all zeros and the host\n"
                                    "  // already zeroed s_out, so this loop was a no-op.\n"))

g = _load_golden()
rng = np.random.default_rng(0)
k, v, q, gates = g.make_inputs(rng, T, DK, DV)
ker, _ = g.kernel_model(k, v, q, gates, DK, DV)
kb, vb, qb = (np.ascontiguousarray(x).astype(BF) for x in (k, v, q))
gf = np.ascontiguousarray(gates).astype(np.float32)
k_off, v_off = 0, T * DK * 2
q_off = v_off + T * DV * 2
g_off = q_off + T * DK * 2
packed = np.concatenate([kb.reshape(-1).view(np.int8), vb.reshape(-1).view(np.int8),
                         qb.reshape(-1).view(np.int8), gf.reshape(-1).view(np.int8)])
zeros_state = np.zeros(DK * DV, np.float32)

PTRS = (f'  bfloat16* k = (bfloat16*)(packed + {k_off});\n'
        f'  bfloat16* v = (bfloat16*)(packed + {v_off});\n'
        f'  bfloat16* q = (bfloat16*)(packed + {q_off});\n'
        f'  float*    g = (float*)(packed + {g_off});\n')

ARMS = [
    ("1 copy + static scratch", pre,
     'extern "C" void gdn_probe(int8_t* packed, float* s_in, bfloat16* o) {\n' + PTRS +
     '  static float s_out_scratch[GDN_DK * GDN_DV];\n'
     '  gatedeltanet_step(k, v, q, g, s_in, o, s_out_scratch);\n}\n'),
    ("2 copy + host-zeroed s_out", pre,
     'extern "C" void gdn_probe(int8_t* packed, float* s_state, bfloat16* o) {\n' + PTRS +
     '  gatedeltanet_step(k, v, q, g, s_state, o, s_state);\n}\n'),
    ("3 NO copy + host-zeroed s_out", nocopy,
     'extern "C" void gdn_probe(int8_t* packed, float* s_state, bfloat16* o) {\n' + PTRS +
     '  gatedeltanet_step(k, v, q, g, s_state, o, s_state);\n}\n'),
]

for label, cc, body in ARMS:
    shim = GEN / f"gdn_ctr_{label[0]}.cc"
    shim.write_text(f'#include <stdint.h>\n#include "{cc}"\n{body}')
    try:
        design = bricklib._build_oneshot("gdn_probe", shim, [packed.size, DK * DV], T * DV,
                                         [np.int8, np.float32], BF,
                                         [f"-DGDN_T={T}", f"-DGDN_DK={DK}", f"-DGDN_DV={DV}"])
        i0 = iron.tensor(np.ascontiguousarray(packed), dtype=np.int8, device="npu")
        i1 = iron.tensor(np.ascontiguousarray(zeros_state), dtype=np.float32, device="npu")
        o = iron.zeros((T * DV,), dtype=BF, device="npu")
        design(i0, i1, o)
        got = np.asarray(np.array(o.numpy(), copy=True), np.float32).reshape(T, DV)
        finite = np.isfinite(got)
        rl2 = float(np.linalg.norm((got - ker).ravel()) / (np.linalg.norm(ker.ravel()) + 1e-12))
        print(f"{label:32s} rel_l2={rl2:.3e}  non-finite {int((~finite).sum())}/{got.size}",
              flush=True)
    except Exception as e:
        print(f"{label:32s} ERROR {type(e).__name__}: {str(e)[:110]}", flush=True)
