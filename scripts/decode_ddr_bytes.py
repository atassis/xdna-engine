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
BD = re.compile(r"aie\.dma_bd\(%(\w+)\s*:\s*memref<(\d+)x(bf16|f32|i8|i32)>[^)]*?"
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
print(f"vs meta.json mapped scratch total: 271.52 MB  -> ratio {total/271.52e6:.2f}x")
print(f"transport floor at 52.69 GB/s: {total/52.69e9*1e3:.2f} ms/token")
print(f"against 39.59 ms of array time: {total/39.59e-3/1e9:.2f} GB/s effective"
      f" = {total/39.59e-3/52.69e9*100:.1f}% of ceiling")
