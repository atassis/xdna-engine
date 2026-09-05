#!/usr/bin/env python3
"""Per-dispatch DDR bytes for the fused decode, from the shim BDs in its own MLIR.

Method is gemm_ddr_bytes.py's, extended for the fused form: the top-level runtime_sequence
issues `aiex.configure @opN_X { aiex.run @sequence(...) }` per runlist step, so total DDR
bytes = sum over invocations of the invoked device's own shim-BD bytes.

A BD's `len` is the product of the INNER THREE sizes; the OUTER entry is a repeat count.
A stride-0 outer dim re-reads the same address range and the shim DMA has no cache, so those
are real repeated DDR reads -- which is exactly why unique bytes != DDR bytes.
"""
import collections, re, sys

ELEM = {"bf16": 2, "f32": 4, "i8": 1, "i32": 4}
# The shape is `(\d+x)+`, not `\d+x`: a BD on a MULTI-dimensional memref is still a shim BD.
# Accepting only 1-D silently dropped 212 of 2508 BDs in the Qwen3 fused decode -- every one of
# them on a memref<NxMxbf16> -- and among them BOTH GQA Repeats, which the tool then reported as
# 0.000 MB while their own BDs carry len=2097152 with an outer repeat of 2. A regex that skips
# what it cannot parse reports a clean total for a subset it never names.
BD = re.compile(r"aie\.dma_bd\(%(\w+)\s*:\s*memref<((?:\d+x)+)(bf16|f32|i8|i32)>[^)]*?"
                r"len\s*=\s*(\d+)\s+sizes\s*=\s*\[([^\]]*)\]\s+strides\s*=\s*\[([^\]]*)\]")
DEV = re.compile(r"aie\.device\(\w+\)\s*@(\w+)\s*\{")
CFG = re.compile(r"aiex\.configure\s+@(\w+)\s*\{")

src = open(sys.argv[1]).read()
lines = src.splitlines()

# --- device blocks by name (brace matching from each header) ---
devs, spans = {}, []
for m in DEV.finditer(src):
    name, i, depth = m.group(1), m.end() - 1, 0
    for j in range(m.end() - 1, len(src)):
        if src[j] == "{": depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0: break
    devs[name] = src[m.end():j]

def dev_bytes(body):
    """shim DDR bytes for one invocation of this device, per named memref arg."""
    per = collections.Counter()
    for b in BD.finditer(body):
        arg, _n, ty, ln, sizes, _st = b.groups()
        sz = [int(x.strip()) for x in sizes.split(",")]
        outer = sz[0] if len(sz) == 4 else 1
        inner = 1
        for v in sz[1:]: inner *= v
        if inner != int(ln):
            print(f"  WARN len={ln} != product(inner)={inner} on {arg}", file=sys.stderr)
        per[arg] += outer * int(ln) * ELEM[ty]
    return per

# --- top-level orchestrator = the device with no @name ---
top = src[src.rindex("aie.device(npu2) {"):]
# One EXECUTION is one `aiex.run`, not one `aiex.configure`: a configure block may hold many runs
# (op7_Transpose issues 12 inside a single block), so attribute each run to its enclosing configure.
invocations, cur = [], None
for line in top.splitlines():
    m = CFG.search(line)
    if m:
        cur = m.group(1)
    elif "aiex.run" in line and cur:
        invocations.append(cur)

print(f"operator invocations per dispatch: {len(invocations)}")
by_op = collections.Counter(invocations)
total = 0
rows = []
for op, n in by_op.most_common():
    if op not in devs:
        print(f"  MISSING device body for {op}", file=sys.stderr); continue
    b = sum(dev_bytes(devs[op]).values())
    rows.append((op, n, b, n * b))
    total += n * b

print(f"\n{'op':28} {'runs':>5} {'MB/run':>9} {'MB/dispatch':>12}")
for op, n, b, tot in sorted(rows, key=lambda r: -r[3]):
    print(f"{op:28} {n:5} {b/1e6:9.3f} {tot/1e6:12.2f}")
print(f"\nTOTAL DDR bytes/dispatch: {total/1e6:.2f} MB")
# Read peak measured on THIS box, 8 columns; a design placed on fewer columns does not get it.
print(f"transport floor at the 52.69 GB/s 8-column read peak: {total/52.69e9*1e3:.2f} ms/token")
# The two lines that used to follow divided this total by 271.52 MB of scratch and by 39.59 ms of
# array time -- both constants measured on the WHISPER decode, printed unlabelled next to any
# design handed to this script. On the Qwen3 fused decode they read 11.44x and 148.9% of ceiling,
# which is not a result, it is another model's denominator. Pass the step time you actually
# measured for THIS design instead, or get no ratio.
if len(sys.argv) > 2:
    ms = float(sys.argv[2])
    print(f"against the {ms:.2f} ms/token you measured: {total/(ms*1e-3)/1e9:.2f} GB/s effective"
          f" = {total/(ms*1e-3)/52.69e9*100:.1f}% of the 8-column read peak")
else:
    print("(pass a measured ms/token as argv[2] for an effective-bandwidth line)")
