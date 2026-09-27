"""Probe: what does int8-quantizing block_1's c1 SiLU output (currently the network's one int16
tensor, `S["b1.c1.silu"]`) cost in PSNR? Read-only wrt span_int.py -- subclasses Span, does not
edit it. See INT8_BLOCK1.md for the measured table this produces.

usage: python3 _probe_int8_block1.py <export-dir> <demo-dir>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_int as S  # noqa: E402

silu, sig, fconv, c3, c1 = S.silu, S.sig, S.fconv, S.c3, S.c1


class SpanProbe(S.Span):
    """b1_mode selects how S["b1.c1.silu"] and its LUT(s) are built:
      "int16"        -- span_int.py's shipped scheme (amax/32767, hi/lo table), for parity check
      "int8"         -- variant (a): amax/127, ordinary 8-bit table (blocks 2-6's scheme)
      "int8_perchan" -- variant (b): per-output-channel amax/127, one table per channel, s_in to
                        b1.c2's conv becomes the per-channel vector
      "int8_pct"     -- variant (c): scalar percentile clip (self.pct) instead of amax, /127
    """

    def __init__(self, export_dir, b1_mode="int16", pct=99.9):
        super().__init__(export_dir)
        self.b1_mode = b1_mode
        self.pct = pct

    def forward_float_ext(self, rgb01, rec=None):
        """forward_float, but also returns b1's c1 SiLU tensor at full float precision, [48,H,W],
        for per-channel/percentile calibration (forward_float itself only returns a scalar
        amax dict, not the tensor)."""
        rec = {} if rec is None else rec
        put = lambda k, t: rec.__setitem__(k, np.maximum(rec.get(k, 0), np.abs(t).max())) or t
        x = np.round(rgb01 * 255)
        w, b = self.W("conv_1")
        feat = put("conv_1", fconv(x - self.mean255[:, None, None], w, b))
        xb, outs, b1c1silu = feat, {}, None
        for i in range(1, 7):
            t = xb
            for j in (1, 2, 3):
                w, b = self.W(f"block_{i}.c{j}_r")
                t = put(f"b{i}.c{j}", fconv(t, w, b))
                if j < 3:
                    t = put(f"b{i}.c{j}.silu", silu(t))
                if j == 1:
                    outs[f"b{i}.c1.silu"] = t
                    if i == 1:
                        b1c1silu = t
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
        return y, rec, b1c1silu

    def quantize(self, calib_rgb01):
        _, amax, b1c1silu_f = self.forward_float_ext(calib_rgb01)
        S_ = {k: v / 127 for k, v in amax.items()}
        S_["att"] = 0.5 / 127

        if self.b1_mode == "int16":
            sc = amax["b1.c1.silu"] / 32767
        elif self.b1_mode == "int8":
            sc = amax["b1.c1.silu"] / 127
        elif self.b1_mode == "int8_perchan":
            sc = np.abs(b1c1silu_f).reshape(48, -1).max(1) / 127   # [48]
        elif self.b1_mode == "int8_pct":
            sc = np.percentile(np.abs(b1c1silu_f), self.pct) / 127
        else:
            raise ValueError(self.b1_mode)
        S_["b1.c1.silu"] = sc
        self.S, L = S_, {}

        def conv_params(name, s_in, s_out, cin_pad=None, w=None, b=None, qmax=127):
            if w is None:
                w, b = self.W(name)
            if cin_pad and w.shape[1] < cin_pad:
                w = np.pad(w, ((0, 0), (0, cin_pad - w.shape[1]), (0, 0), (0, 0)))
            s_in = np.broadcast_to(np.asarray(s_in, np.float64), (w.shape[1],))
            weff = w * s_in[None, :, None, None]
            sw = np.abs(weff).reshape(w.shape[0], -1).max(1) / 127
            sw[sw == 0] = sw.max() if sw.max() > 0 else 1
            wq = np.clip(np.round(weff / sw[:, None, None, None]), -127, 127).astype(np.int8)
            bq = np.round(b / sw).astype(np.int64)
            r = sw / s_out
            return dict(w=wq, b=bq.astype(np.int32), **requant_consts(r, qmax))

        def requant_consts(r, qmax):
            need = np.abs(1.0 / r).max()
            pre = int(max(0, np.ceil(np.log2(max(need * (qmax + 1), 1) / 32767))))
            shift = int(np.floor(np.log2(32767 / (r.max() * 2 ** pre))))
            mult = np.round(r * 2 ** (pre + shift)).astype(np.int64)
            assert mult.max() < 32768 and shift >= 0, (pre, shift, mult.max())
            return dict(pre=pre, shift=shift, mult=mult)

        def table(fn, s_in, s_out, lo=-128, hi=127):
            k = np.arange(256) - 128
            return np.clip(np.round(fn(k * s_in) / s_out), lo, hi).astype(np.int64)

        w, b = self.W("conv_1")
        b_fold = b - (w * self.mean255[None, :, None, None]).sum((1, 2, 3))
        L["conv_1"] = conv_params("conv_1", 1.0, S_["conv_1"], cin_pad=8, w=w, b=b_fold)
        for i in range(1, 7):
            s_x = S_["conv_1"] if i == 1 else S_[f"b{i - 1}.out"]
            s_c1silu = S_[f"b{i}.c1.silu"]
            L[f"b{i}.c1"] = conv_params(f"block_{i}.c1_r", s_x, S_[f"b{i}.c1"])
            if i == 1 and self.b1_mode == "int16":
                L["b1.c1"]["lut16"] = table(silu, S_["b1.c1"], s_c1silu, -32768, 32767)
            elif i == 1 and self.b1_mode == "int8_perchan":
                L["b1.c1"]["lut_perchan"] = np.stack(
                    [table(silu, S_["b1.c1"], s_c1silu[c]) for c in range(48)])
            else:
                L[f"b{i}.c1"]["lut"] = table(silu, S_[f"b{i}.c1"], s_c1silu)
            L[f"b{i}.c2"] = conv_params(f"block_{i}.c2_r", s_c1silu, S_[f"b{i}.c2"])
            L[f"b{i}.c2"]["lut"] = table(silu, S_[f"b{i}.c2"], S_[f"b{i}.c2.silu"])
            L[f"b{i}.c3"] = conv_params(f"block_{i}.c3_r", S_[f"b{i}.c2.silu"], S_[f"b{i}.c3"])
            ra, rb = S_[f"b{i}.c3"] / S_[f"b{i}.sum"], s_x / S_[f"b{i}.sum"]
            gs1 = int(np.floor(np.log2(127 / max(ra, rb))))
            rc = S_[f"b{i}.sum"] * S_["att"] / S_[f"b{i}.out"]
            gs2 = int(np.floor(np.log2(32767 / rc)))
            L[f"b{i}.c3"].update(ga=int(round(ra * 2 ** gs1)), gb=int(round(rb * 2 ** gs1)), gs1=gs1,
                                 gc=int(round(rc * 2 ** gs2)), gs2=gs2,
                                 lut=table(lambda t: sig(t) - 0.5, S_[f"b{i}.c3"], S_["att"]))
        L["conv_2"] = conv_params("conv_2", S_["b6.out"], S_["conv_2"])
        s_cat = np.concatenate([np.full(48, S_[k]) for k in ("conv_1", "conv_2", "b1.out", "b6.c1.silu")])
        w, b = self.W("conv_cat")
        L["conv_cat"] = conv_params("conv_cat", s_cat, S_["conv_cat"], w=w, b=b)
        w, b = self.W("upsampler.0")
        wp = np.zeros((16,) + w.shape[1:])
        bp = np.zeros(16)
        for i in range(2):
            for j in range(2):
                for col in range(3):
                    wp[(i * 2 + j) * 4 + col] = w[col * 4 + i * 2 + j]
                    bp[(i * 2 + j) * 4 + col] = b[col * 4 + i * 2 + j]
                bp[(i * 2 + j) * 4 + 3] = 1e3
        L["up"] = conv_params("upsampler.0", S_["conv_cat"], 1.0 / 255, w=wp, b=bp, qmax=255)
        self.L = L
        return L

    def block_int(self, i, xb):
        p1, p2, p3 = self.L[f"b{i}.c1"], self.L[f"b{i}.c2"], self.L[f"b{i}.c3"]
        t = self._conv(xb, p1)
        if i == 1 and self.b1_mode == "int16":
            t = p1["lut16"][t.astype(np.int64) + 128].astype(np.int16)
        elif i == 1 and self.b1_mode == "int8_perchan":
            idx = t.astype(np.int64) + 128                        # [48,H,W]
            lut = p1["lut_perchan"]                                # [48,256]
            flat = idx.reshape(48, -1)
            t = np.take_along_axis(lut, flat, axis=1).reshape(idx.shape).astype(np.int8)
        else:
            t = p1["lut"][t.astype(np.int64) + 128].astype(np.int8)
        c1silu = t
        t = self._conv(t, p2)
        t = p2["lut"][t.astype(np.int64) + 128].astype(np.int8)
        t = self._conv(t, p3)
        out = c3.gate_ref(t, xb, p3["lut"], p3["ga"], p3["gb"], p3["gs1"], p3["gc"], p3["gs2"])
        return out, c1silu


def run(export, demo, b1_mode, pct=99.9):
    from PIL import Image
    net = SpanProbe(export, b1_mode=b1_mode, pct=pct)
    hr_img = Image.open(Path(demo) / "hr.png").convert("RGB")
    w0, h0 = hr_img.size
    hr_img = hr_img.crop((0, 0, w0 - w0 % 4, h0 - h0 % 2))
    lr_img = hr_img.resize((hr_img.size[0] // 2, hr_img.size[1] // 2), Image.BICUBIC)
    hr = np.asarray(hr_img, np.float64).transpose(2, 0, 1) / 255
    lr = np.asarray(lr_img, np.float64).transpose(2, 0, 1) / 255
    half = lr.shape[2] // 2
    lr_cal, lr_test, hr_test = lr[:, :, :half], lr[:, :, half:], hr[:, :, 2 * half:]
    net.quantize(lr_cal)
    img = net.forward_int(lr_test)
    sr = img[..., :3].transpose(2, 0, 1) / 255.0
    return S.psnr(sr, hr_test)


def main():
    export, demo = sys.argv[1], sys.argv[2]
    for mode, pct in [("int16", None), ("int8", None), ("int8_perchan", None),
                      ("int8_pct", 99.9), ("int8_pct", 99.99), ("int8_pct", 99.5)]:
        p = psnr = run(export, demo, mode, pct if pct else 99.9)
        label = mode if pct is None else f"{mode}(p{pct})"
        print(f"{label:20s} PSNR {psnr:.3f} dB")


if __name__ == "__main__":
    main()
