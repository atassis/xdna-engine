"""SPAN x2 as the exact integer program the NPU bricks run, and its quality.

Quantizes the collapsed inference convs of a SPAN checkpoint export (a manifest.json plus one
<layer>.weight.npy / <layer>.bias.npy per conv, train-time branches already folded) into the
parameters the aie_kernels bricks take --
per-channel int8 weights, int32 bias, per-channel requant multipliers and shifts, SiLU / sigmoid
tables, gate constants -- and runs the whole network with the bricks' own integer goldens. Its
output is therefore the device's output bit for bit, and its PSNR is the quality the device can
deliver.

Layer plan. Only one tensor is int16: block_1's first SiLU output, whose values are heavy-tailed
(median magnitude far below an int8 step at max-abs scale); int8 there costs ~2 dB.
  conv_1         conv3x3_u8i8   uint8 RGB (3 of 8 channels), mean folded into the bias
  block_1.c1_r   conv3x3_i8_lut16  -> SiLU as int16
  block_1.c2_r   conv3x3_i16i8_lut -> SiLU int8
  block_b.c1_r/c2_r (b >= 2)  conv3x3_i8_lut (SiLU)
  block_b.c3_r   conv3x3_i8_gate   (c3 + x) * (sigmoid(c3) - 0.5)
  conv_2         conv3x3_i8
  conv_cat       conv1x1_cat_i8 over [conv_1, conv_2, block_1, block_6.c1 SiLU], each source's
                 scale folded into its weight slice
  upsampler      conv3x3_i8u8, output channels permuted to (sub-row, sub-col, R, G, B, A) so each
                 8-channel block of a row IS an RGBA row of the x2 image: no pixel shuffle
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parent.parent / "aie_kernels"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c3 = _load(KERNELS / "conv2d-3x3-u8" / "golden.py", "c3g")
c1 = _load(KERNELS / "conv2d-1x1-cat" / "golden.py", "c1g")

silu = lambda t: t / (1 + np.exp(-t))
sig = lambda t: 1 / (1 + np.exp(-t))


def fconv(x, w, b, pad_value=None):
    c, h, wd = x.shape
    o, _, k, _ = w.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    if pad_value is not None:
        border = np.ones((h + 2 * p, wd + 2 * p), bool)
        border[p:p + h, p:p + wd] = False
        xp[:, border] = np.asarray(pad_value, np.float64).reshape(-1, 1)
    cols = np.stack([xp[:, dy:dy + h, dx:dx + wd] for dy in range(k) for dx in range(k)], 1)
    return (w.reshape(o, -1) @ cols.reshape(c * k * k, h * wd)).reshape(o, h, wd) + b[:, None, None]


class Span:
    def __init__(self, export_dir):
        self.dir = Path(export_dir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.mean255 = np.array(self.man["rgb_mean"]) * 255

    def W(self, name):
        return (np.load(self.dir / f"{name}.weight.npy").astype(np.float64),
                np.load(self.dir / f"{name}.bias.npy").astype(np.float64))

    # ---------------- float reference, with a hook on every tensor ----------------
    def forward_float(self, rgb01, rec=None):
        rec = {} if rec is None else rec
        put = lambda k, t: rec.__setitem__(k, np.maximum(rec.get(k, 0), np.abs(t).max())) or t
        x = np.round(rgb01 * 255)                                   # uint8 RGB
        w, b = self.W("conv_1")
        feat = put("conv_1", fconv(x - self.mean255[:, None, None], w, b))
        xb, outs = feat, {}
        for i in range(1, 7):
            t = xb
            for j in (1, 2, 3):
                w, b = self.W(f"block_{i}.c{j}_r")
                t = put(f"b{i}.c{j}", fconv(t, w, b))
                if j < 3:
                    t = put(f"b{i}.c{j}.silu", silu(t))
                if j == 1:
                    outs[f"b{i}.c1.silu"] = t
            att = sig(t) - 0.5
            s = put(f"b{i}.sum", t + xb)
            xb = put(f"b{i}.out", s * att)
            if i == 1:
                outs["b1.out"] = xb
        w, b = self.W("conv_2")
        c2 = put("conv_2", fconv(xb, w, b))
        cat = np.concatenate([feat, c2, outs["b1.out"], outs["b6.c1.silu"]])
        w, b = self.W("conv_cat")
        cc = put("conv_cat", fconv(cat, w, b))
        w, b = self.W("upsampler.0")
        y = fconv(cc, w, b)
        return y, rec

    # ---------------- quantization ----------------
    def quantize(self, calib_rgb01):
        _, amax = self.forward_float(calib_rgb01)
        S = {k: v / 127 for k, v in amax.items()}                   # int8 scales
        S["b1.c1.silu"] = amax["b1.c1.silu"] / 32767                # the one int16 tensor
        S["att"] = 0.5 / 127
        self.S, L = S, {}

        def conv_params(name, s_in, s_out, cin_pad=None, w=None, b=None, qmax=127):
            if w is None:
                w, b = self.W(name)
            if cin_pad and w.shape[1] < cin_pad:
                w = np.pad(w, ((0, 0), (0, cin_pad - w.shape[1]), (0, 0), (0, 0)))
            s_in = np.broadcast_to(np.asarray(s_in, np.float64), (w.shape[1],))
            weff = w * s_in[None, :, None, None]                    # input scales folded in
            sw = np.abs(weff).reshape(w.shape[0], -1).max(1) / 127
            # an all-zero channel (the upsampler's alpha) takes the largest real scale: a scale of
            # 1 would make its ratio dominate the shared requant shift and starve the others
            sw[sw == 0] = sw.max() if sw.max() > 0 else 1
            wq = np.clip(np.round(weff / sw[:, None, None, None]), -127, 127).astype(np.int8)
            bq = np.round(b / sw).astype(np.int64)
            r = sw / s_out                                          # acc -> output units
            return dict(w=wq, b=bq.astype(np.int32), **requant_consts(r, qmax))

        def requant_consts(r, qmax):
            # pre: an output at full scale (qmax) must still fit int16 after acc >> pre, on the
            # channel with the most acc units per output unit; mult < 2^15
            need = np.abs(1.0 / r).max()
            pre = int(max(0, np.ceil(np.log2(max(need * (qmax + 1), 1) / 32767))))
            shift = int(np.floor(np.log2(32767 / (r.max() * 2 ** pre))))
            mult = np.round(r * 2 ** (pre + shift)).astype(np.int64)
            assert mult.max() < 32768 and shift >= 0, (pre, shift, mult.max())
            return dict(pre=pre, shift=shift, mult=mult)

        def table(fn, s_in, s_out, lo=-128, hi=127):
            k = np.arange(256) - 128
            return np.clip(np.round(fn(k * s_in) / s_out), lo, hi).astype(np.int64)

        # conv_1: uint8 RGB in units of 1/255 image, mean folded into the bias
        w, b = self.W("conv_1")
        b_fold = b - (w * self.mean255[None, :, None, None]).sum((1, 2, 3))
        L["conv_1"] = conv_params("conv_1", 1.0, S["conv_1"], cin_pad=8, w=w, b=b_fold)
        for i in range(1, 7):
            s_x = S["conv_1"] if i == 1 else S[f"b{i - 1}.out"]
            s_c1silu = S[f"b{i}.c1.silu"]
            L[f"b{i}.c1"] = conv_params(f"block_{i}.c1_r", s_x, S[f"b{i}.c1"])
            if i == 1:  # SiLU to int16 via hi/lo tables
                L["b1.c1"]["lut16"] = table(silu, S["b1.c1"], s_c1silu, -32768, 32767)
            else:
                L[f"b{i}.c1"]["lut"] = table(silu, S[f"b{i}.c1"], s_c1silu)
            L[f"b{i}.c2"] = conv_params(f"block_{i}.c2_r", s_c1silu, S[f"b{i}.c2"])
            L[f"b{i}.c2"]["lut"] = table(silu, S[f"b{i}.c2"], S[f"b{i}.c2.silu"])
            L[f"b{i}.c3"] = conv_params(f"block_{i}.c3_r", S[f"b{i}.c2.silu"], S[f"b{i}.c3"])
            # gate: sum = (c*ga + x*gb) >> gs1 at scale S[sum]; out = sum*att*gc >> gs2
            ra, rb = S[f"b{i}.c3"] / S[f"b{i}.sum"], s_x / S[f"b{i}.sum"]
            gs1 = int(np.floor(np.log2(127 / max(ra, rb))))
            rc = S[f"b{i}.sum"] * S["att"] / S[f"b{i}.out"]
            gs2 = int(np.floor(np.log2(32767 / rc)))
            L[f"b{i}.c3"].update(ga=int(round(ra * 2 ** gs1)), gb=int(round(rb * 2 ** gs1)), gs1=gs1,
                                 gc=int(round(rc * 2 ** gs2)), gs2=gs2,
                                 lut=table(lambda t: sig(t) - 0.5, S[f"b{i}.c3"], S["att"]))
        L["conv_2"] = conv_params("conv_2", S["b6.out"], S["conv_2"])
        s_cat = np.concatenate([np.full(48, S[k]) for k in ("conv_1", "conv_2", "b1.out", "b6.c1.silu")])
        w, b = self.W("conv_cat")
        L["conv_cat"] = conv_params("conv_cat", s_cat, S["conv_cat"], w=w, b=b)
        # upsampler: 12 channels (colour-major, then i, j) -> 16 in (i, j, R, G, B, A) order,
        # output in 1/255 pixel units, alpha = saturated 255
        w, b = self.W("upsampler.0")
        wp = np.zeros((16,) + w.shape[1:])
        bp = np.zeros(16)
        for i in range(2):
            for j in range(2):
                for col in range(3):
                    wp[(i * 2 + j) * 4 + col] = w[col * 4 + i * 2 + j]
                    bp[(i * 2 + j) * 4 + col] = b[col * 4 + i * 2 + j]
                bp[(i * 2 + j) * 4 + 3] = 1e3                      # alpha: saturates to 255
        L["up"] = conv_params("upsampler.0", S["conv_cat"], 1.0 / 255, w=wp, b=bp, qmax=255)
        self.L = L
        return L

    # ---------------- the integer program ----------------
    def _conv(self, x, p, signed=True):
        return c3.conv3x3_u8_ref(x, p["w"], p["b"], p["shift"], signed=signed,
                                 pre_shift=p["pre"], mult=p["mult"])

    def block_int(self, i, xb):
        """SPAB block i on its int8 input [48,H,W] -> (int8 output, c1 SiLU output)."""
        p1, p2, p3 = self.L[f"b{i}.c1"], self.L[f"b{i}.c2"], self.L[f"b{i}.c3"]
        t = self._conv(xb, p1)
        if i == 1:
            t = p1["lut16"][t.astype(np.int64) + 128].astype(np.int16)
        else:
            t = p1["lut"][t.astype(np.int64) + 128].astype(np.int8)
        c1silu = t
        t = self._conv(t, p2)
        t = p2["lut"][t.astype(np.int64) + 128].astype(np.int8)
        t = self._conv(t, p3)
        out = c3.gate_ref(t, xb, p3["lut"], p3["ga"], p3["gb"], p3["gs1"], p3["gc"], p3["gs2"])
        return out, c1silu

    def int_tensors(self, rgb01):
        """Every integer tensor the network produces, keyed like the float recorder."""
        L, m = self.L, self.mean255
        T = {}
        x8 = np.round(rgb01 * 255).astype(np.int64)
        # the producer pads the frame with the (rounded) mean colour: emulate by padding and
        # cropping, since the brick's zero padding is torch's padding of the NORMALIZED input
        pad = np.round(m).astype(np.int64)
        xp = np.stack([np.pad(x8[c], 1, constant_values=pad[c]) for c in range(3)])
        xp = np.concatenate([xp, np.zeros((5,) + xp.shape[1:], np.int64)]).astype(np.uint8)
        T["conv_1"] = feat = self._conv(xp, L["conv_1"])[:, 1:-1, 1:-1]
        xb = feat
        for i in range(1, 7):
            T[f"b{i}.in"] = xb
            xb, T[f"b{i}.c1.silu"] = self.block_int(i, xb)
            T[f"b{i}.out"] = xb
        T["conv_2"] = self._conv(xb, L["conv_2"])
        pc = L["conv_cat"]
        T["conv_cat"] = c1.conv1x1_cat_ref(
            [feat, T["conv_2"], T["b1.out"], T["b6.c1.silu"]], pc["w"].reshape(48, -1), pc["b"],
            pc["shift"], pre_shift=pc["pre"], mult=pc["mult"])
        T["rgba"] = self._conv(T["conv_cat"], L["up"], signed=False)
        return T

    def forward_int(self, rgb01):
        rgba = self.int_tensors(rgb01)["rgba"]
        # channel block (i*2+j) of LR pixel (y, x) is pixel (2y+i, 2x+j), RGBA
        h, w = rgba.shape[1:]
        return rgba.reshape(2, 2, 4, h, w).transpose(3, 0, 4, 1, 2).reshape(2 * h, 2 * w, 4)

    def core_params(self, i):
        """Per-core inputs for block i's three conv cores: packed params blob, table, constants."""
        P = {}
        for j, key in ((1, "c1"), (2, "c2"), (3, "c3")):
            p = self.L[f"b{i}.c{j}"]
            d = dict(blob=c3.pack_params(p["w"], p["b"], p["mult"]), pre=p["pre"],
                     shift=p["shift"], table=p.get("lut"))
            if j == 3:
                d.update({k: p[k] for k in ("ga", "gb", "gs1", "gc", "gs2")})
            P[key] = d
        return P

    def net_params(self):
        """Every stage's device inputs, keyed by net_layout.STAGES name: packed params blob,
        requant pre/shift, tables (int8 table, or hi/lo halves of block 1's int16 SiLU) and the
        gate constants."""
        L, P = self.L, {}

        def conv(p, **extra):
            return dict(blob=c3.pack_params(p["w"], p["b"], p["mult"]), pre=p["pre"],
                        shift=p["shift"], **extra)

        P["conv_1"] = conv(L["conv_1"])
        for i in range(1, 7):
            p1, p2, p3 = (L[f"b{i}.c{j}"] for j in (1, 2, 3))
            P[f"b{i}c1"] = conv(p1, tables=list(c3.split_lut16(p1["lut16"])) if i == 1
                                else [p1["lut"]])
            P[f"b{i}c2"] = conv(p2, tables=[p2["lut"]])
            P[f"b{i}c3"] = conv(p3, tables=[p3["lut"]],
                                **{k: p3[k] for k in ("ga", "gb", "gs1", "gc", "gs2")})
        P["conv_2"] = conv(L["conv_2"])
        pc = L["conv_cat"]
        P["conv_cat"] = dict(blob=c1.pack_params(pc["w"].reshape(48, -1), pc["b"], 4, pc["mult"]),
                             pre=pc["pre"], shift=pc["shift"])
        P["up"] = conv(L["up"])
        return P


def _y(rgb):
    return (65.481 * rgb[0] + 128.553 * rgb[1] + 24.966 * rgb[2] + 16) / 255


def psnr(a, b, border=2):
    a, b = _y(a)[border:-border, border:-border], _y(b)[border:-border, border:-border]
    return 10 * np.log10(1 / np.mean((a - b) ** 2))


def main():
    from PIL import Image
    import os
    if len(sys.argv) < 2:
        sys.exit("usage: span_int.py <span-export-dir> [demo-dir with hr.png]")
    export = Path(sys.argv[1])
    demo = Path(sys.argv[2] if len(sys.argv) > 2 else
                os.environ.get("SPAN_DEMO_DIR", HERE.parent.parent / "artifacts/edsr/demo"))
    net = Span(export)
    hr_img = Image.open(demo / "hr.png").convert("RGB")
    w0, h0 = hr_img.size
    hr_img = hr_img.crop((0, 0, w0 - w0 % 4, h0 - h0 % 2))
    lr_img = hr_img.resize((hr_img.size[0] // 2, hr_img.size[1] // 2), Image.BICUBIC)
    hr = np.asarray(hr_img, np.float64).transpose(2, 0, 1) / 255
    lr = np.asarray(lr_img, np.float64).transpose(2, 0, 1) / 255
    half = lr.shape[2] // 2
    lr_cal, lr_test, hr_test = lr[:, :, :half], lr[:, :, half:], hr[:, :, 2 * half:]
    bic = np.asarray(lr_img.crop((half, 0, lr_img.size[0], lr_img.size[1])).resize(
        (hr_test.shape[2], hr_test.shape[1]), Image.BICUBIC), np.float64).transpose(2, 0, 1) / 255
    fp, _ = net.forward_float(lr_test)
    fp = np.clip(fp, 0, 1)
    fp_img = fp.reshape(3, 2, 2, *fp.shape[1:]).transpose(0, 3, 1, 4, 2).reshape(3, 2 * fp.shape[1], 2 * fp.shape[2])
    net.quantize(lr_cal)
    img = net.forward_int(lr_test)
    sr = img[..., :3].transpose(2, 0, 1) / 255.0
    print(f"test half, Y PSNR vs HR: bicubic {psnr(bic, hr_test):.2f}  fp {psnr(fp_img, hr_test):.2f}  "
          f"integer program {psnr(sr, hr_test):.2f}  (vs fp {psnr(sr, fp_img):.2f})")
    print(f"alpha channel all 255: {bool((img[..., 3] == 255).all())}")
    for k in ("conv_1", "b1.c1", "b1.c3", "conv_cat", "up"):
        p = net.L[k]
        print(f"  {k:9s} pre {p['pre']:2d} shift {p['shift']:2d} mult [{p['mult'].min()}, {p['mult'].max()}]"
              + (f"  gate ga {p['ga']} gb {p['gb']} gs1 {p['gs1']} gc {p['gc']} gs2 {p['gs2']}" if "ga" in p else ""))


if __name__ == "__main__":
    main()
