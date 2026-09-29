"""CPU checks for net_layout and Span.net_params. usage: python3 test_net_layout.py <export> [demo]"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_int as S  # noqa: E402
import net_layout as NL  # noqa: E402
import test_span_int as TS  # noqa: E402

W, H = 32, 24


def main():
    export = sys.argv[1]
    demo = sys.argv[2] if len(sys.argv) > 2 else os.environ["SPAN_DEMO_DIR"]
    cal, test = TS.load_split(demo)
    net = S.Span(export)
    net.quantize(cal)
    NP = net.net_params()
    assert len(NP) == 22, sorted(NP)
    assert NP["conv_1"]["blob"].size == 48 * 3 * 8 * 3 + 6 * 64 * 6       # CIN 8
    assert NP["b2c1"]["blob"].size == 48 * 3 * 48 * 3 + 6 * 64 * 6
    assert NP["conv_cat"]["blob"].size == 48 * 4 * 48 + 6 * 64 * 6        # 1x1 over 4 x 48
    assert NP["up"]["blob"].size == 16 * 3 * 48 * 3 + 2 * 64 * 6          # COUT 16
    assert len(NP["b1c1"]["tables"]) == 2 and len(NP["b2c1"]["tables"]) == 1
    assert "tables" not in NP["conv_2"]
    assert {"ga", "gb", "gs1", "gc", "gs2"} <= set(NP["b4c3"])

    names = [n for n, _ in NL.STAGES]
    assert len(names) == 22 and set(names) == set(NP), "stage table and params disagree"
    # rows each skip source must have emitted before conv_cat can emit its first row
    assert [NL.rows_ahead(s) for s, _ in NL.CAT_SOURCES] == [20, 1, 17, 4]
    assert [len(g) for g in NL.weight_groups(names)] == [6, 6, 6, 4]
    # conv_1's host rows, seen the way the kernel sees them (zero beyond the one margin pixel),
    # reproduce the oracle's mean-colour padding exactly
    crop = test[:, :H, :W]
    T = net.int_tensors(crop)
    rows = NL.conv1_rows(crop, net.mean255)
    assert rows.dtype == np.uint8 and rows.size == (H + 2) * NL.layout("conv1", W).in_bytes
    xp = S.c3.unpack_rows(rows.reshape(-1), 8, H + 2, W, margins=True)
    got = net._conv(xp, net.L["conv_1"])[:, 1:-1, NL.PAD:NL.PAD + W]
    assert np.array_equal(got, T["conv_1"]), "conv_1 edge padding differs from the oracle"
    # unpack_out inverts the device row layout for every gated stage
    for upto, (key, ch, dt) in NL.GOLDEN.items():
        ref = T[key]
        out_bytes = NL.layout(dict(NL.STAGES)[upto], W).out_bytes
        own = S.c3.pack_rows(ref.astype(dt)).reshape(H, -1).view(np.int8)
        y = np.zeros((H, out_bytes), np.int8)
        y[:, :own.shape[1]] = own
        assert np.array_equal(NL.unpack_out(y.reshape(-1), upto, H, W), ref), upto
    blob = NL.weights_blob(NP, names)
    assert blob.dtype == np.int8 and blob.size == sum(NP[n]["blob"].size for n in names)
    print("test_net_layout: OK")


if __name__ == "__main__":
    main()
