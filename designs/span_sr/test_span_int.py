"""CPU checks for span_int's block-level API. usage: python3 test_span_int.py <export> [demo]"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_int as S  # noqa: E402


def load_split(demo):
    hr = Image.open(Path(demo) / "hr.png").convert("RGB")
    w0, h0 = hr.size
    hr = hr.crop((0, 0, w0 - w0 % 4, h0 - h0 % 2))
    lr = np.asarray(hr.resize((hr.size[0] // 2, hr.size[1] // 2), Image.BICUBIC),
                    np.float64).transpose(2, 0, 1) / 255
    half = lr.shape[2] // 2
    return lr[:, :, :half], lr[:, :, half:]


def main():
    export = sys.argv[1]
    demo = sys.argv[2] if len(sys.argv) > 2 else os.environ["SPAN_DEMO_DIR"]
    cal, test = load_split(demo)
    net = S.Span(export)
    net.quantize(cal)
    T = net.int_tensors(test)
    # block_int reproduces what the full forward computed, block by block
    for i in range(1, 7):
        out, c1silu = net.block_int(i, T[f"b{i}.in"])
        assert np.array_equal(out, T[f"b{i}.out"]), f"block {i} output differs"
        assert np.array_equal(c1silu, T[f"b{i}.c1.silu"]), f"block {i} c1 SiLU differs"
    # core_params packs what the conv3x3 brick reads
    P = net.core_params(2)
    assert set(P) == {"c1", "c2", "c3"}
    wbytes = 48 * 3 * 48 * 3                      # [COUT/8][3][CIN/8][3][8][8] for 48x48
    assert P["c1"]["blob"].size == wbytes + 6 * 64 * 4 + 6 * 64 * 2
    assert {"ga", "gb", "gs1", "gc", "gs2"} <= set(P["c3"])
    print("test_span_int: OK")


if __name__ == "__main__":
    main()
