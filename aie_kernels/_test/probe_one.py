import sys, importlib, traceback
mod, fn = sys.argv[1], sys.argv[2]
m = importlib.import_module(mod)
try:
    r = getattr(m, fn)()
    print("RESULT:", r if isinstance(r,dict) else type(r).__name__)
except Exception as e:
    print("RUN-FAIL:", e); traceback.print_exc()
