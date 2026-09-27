import os, sys, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa
import net_design as N
import net_layout as NL
import span_int as S
import test_span_int as TS

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
W = 32

def try_build(tag, **kw):
    try:
        N.build(W, 64, NP, HERE / "gen" / f"cs1g_{tag}", tag=f"cs1g{tag}", **kw)
        print(f"[{tag}] OK  kw={kw}")
        return True
    except Exception as e:
        msg = str(e).splitlines()[-1] if str(e) else repr(e)
        print(f"[{tag}] FAIL kw={kw}  -- {msg}")
        return False

print("== prod_depth sweep ==")
for pd in (2, 4, 8, 16):
    try_build(f"pd{pd}", prod_depth=pd)

print("== cat_cons_depth sweep ==")
for cd in (2, 3, 4):
    try_build(f"cd{cd}", cat_cons_depth=cd)

print("== skip_cons_depths[conv_1] sweep ==")
for d in (3, 4):
    try_build(f"sc1_{d}", skip_cons_depths={"conv_1": d})

print("== skip_cons_depths[b1c3]/[b6c1] sweep (main_depth+1=5) ==")
for src in ("b1c3", "b6c1"):
    try_build(f"sc_{src}5", skip_cons_depths={src: 5})

print("== combined: prod_depth=8 + cat_cons_depth=3 ==")
try_build("combo", prod_depth=8, cat_cons_depth=3)

print("== combined: prod_depth=8 + cat_cons_depth=3 + skip_cons_depths b1c3=5,b6c1=5 ==")
try_build("combo2", prod_depth=8, cat_cons_depth=3, skip_cons_depths={"b1c3": 5, "b6c1": 5})
