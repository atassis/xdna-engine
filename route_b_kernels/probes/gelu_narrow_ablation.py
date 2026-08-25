"""Ablate every bf16 narrow in the GELU epilogue, device-free, against a device control.

The chain in tanh_ab.cc / mm_silu_epilogue.cc is re-implemented here with
round-nearest-even bf16 (the kernel sets conv_even). It is only trustworthy because
it REPRODUCES the measured gelu_sw column from a device dump -- that agreement is
printed first and is the reason to believe the counterfactual arms.

  python3 gelu_narrow_ablation.py /path/to/dump.npz

The .npz is what run_tanh_ab.py --dump writes: <domain>_{x64,raw,hwtanh,swtanh,gelu_hw,gelu_sw}.
"""
import sys
import numpy as np


def bf16(x):
    """f32 -> bf16 -> f32, round-nearest-even."""
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float32)


C0_EXACT = np.float64(0.7978845608028654)
C0, C1 = np.float32(0.7978845608), np.float32(0.044715)
C0B, C1B = bf16(C0), bf16(C1)


def gelu_ref(x64):
    return 0.5 * x64 * (1.0 + np.tanh(C0_EXACT * (x64 + 0.044715 * x64**3)))


def rel_l2(approx, ref):
    return float(np.linalg.norm(np.float64(approx) - ref) / (np.linalg.norm(ref) + 1e-30))


def cube(xin, narrow):
    """c0*(xin + c1*xin^3). narrow: every aie::mul/add narrows its accumulator back to bf16."""
    if narrow:
        x2 = bf16(xin * xin)
        x3 = bf16(x2 * xin)
        return np.float32(C0B * bf16(xin + bf16(C1B * x3)))
    return np.float32(C0 * (xin + C1 * xin * xin * xin))


def chain(x32, narrow_x_cube, narrow_cube, narrow_x_tail, tanh_of):
    xc = bf16(x32) if narrow_x_cube else x32
    t = tanh_of(cube(xc, narrow_cube))
    xt = bf16(x32) if narrow_x_tail else x32
    return np.float32(np.float32(np.float32(0.5) * xt) * np.float32(np.float32(1.0) + t))


ARMS = [
    ("shipped (bf16 x, bf16 cube)", True, True, True),
    ("f32 cube internals, bf16 x", True, False, True),
    ("f32 x into the cube only", False, False, True),
    ("f32 x in the tail only", True, True, False),
    ("f32 x everywhere", False, False, False),
]


def main(path):
    d = np.load(path)
    doms = [k[:-4] for k in d.files if k.endswith("_x64")]
    exact = lambda v: np.tanh(np.float64(v)).astype(np.float32)

    print("CONTROL -- model vs the device's own gelu_sw (must be small, else stop reading)")
    for dm in doms:
        x64 = np.float64(d[f"{dm}_x64"])
        m = chain(x64.astype(np.float32), True, True, True, exact)
        dev = np.float64(d[f"{dm}_gelu_sw"])
        print(f"  {dm:>10}  {np.linalg.norm(np.float64(m)-dev)/(np.linalg.norm(dev)+1e-30):.2e}")

    print("\nABLATION -- rel-L2 vs float64 GELU, exact tanh throughout")
    print(f"{'arm':<30}" + "".join(f"{dm:>13}" for dm in doms))
    base = {}
    for label, a, b, c in ARMS:
        line = f"{label:<30}"
        for dm in doms:
            x64 = np.float64(d[f"{dm}_x64"])
            v = rel_l2(chain(x64.astype(np.float32), a, b, c, exact), gelu_ref(x64))
            base.setdefault(dm, v)
            line += f"{v:>13.4e}"
        print(line)

    print("\nBOTH LEVERS -- device tanh samples as an oracle, where inner(x) stays in range")
    print(f"{'domain':>12} {'shipped':>12} {'f32x+hwLUT':>12} {'f32x+swtanh':>13} {'gain':>9}")
    for dm in doms:
        x64 = np.float64(d[f"{dm}_x64"])
        x32 = x64.astype(np.float32)
        inner = np.float64(C0 * (x32 + C1 * x32**3))
        if inner.min() < x64.min() or inner.max() > x64.max():
            print(f"{dm:>12}  inner(x) leaves the sampled range -- needs a device arm")
            continue
        g = lambda t: np.float32(0.5 * x64 * (1.0 + t))
        ref = gelu_ref(x64)
        hw = rel_l2(g(np.interp(inner, x64, np.float64(d[f"{dm}_hwtanh"]))), ref)
        sw = rel_l2(g(np.interp(inner, x64, np.float64(d[f"{dm}_swtanh"]))), ref)
        now = rel_l2(np.float32(d[f"{dm}_gelu_hw"]), ref)
        print(f"{dm:>12} {now:>12.4e} {hw:>12.4e} {sw:>13.4e} {now/max(sw,1e-30):>8.0f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tanh2/dump_jli.npz")
