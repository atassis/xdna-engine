"""For each `vtanh <D>, <S>` in an aie2p .S, report whether D's value is READ or
CLOBBERED first.

Register model, confirmed from the reduced repro (`vbcst.32 x2, r0` / `vmov wh2, wl2`
/ `vtanh wl2, bmll0` / `vmul.f ..., x2, ...`): x<N> is the 512-bit pair
{wl<N>, wh<N>}. So a read of x<N> READS the tanh result; a write to x<N> or to the
same half CLOBBERS it; a write to the OTHER half does not.
"""
import re, sys

VREG = re.compile(r'\b(?:x|wl|wh)\d+\b')

def touched_by(d):
    m = re.fullmatch(r'(w[lh])(\d+)', d)
    if not m:
        return {d}
    return {d, 'x' + m.group(2)}

def slots(line):
    return [s.strip() for s in line.split('//')[0].split(';') if s.strip()]

def dest_and_srcs(s):
    """Split a slot into (dest_or_None, [source regs]). Stores have no dest."""
    m = re.match(r'(\S+)\s+(.*)', s)
    if not m:
        return None, []
    mnem, rest = m.group(1), m.group(2)
    ops = [o.strip() for o in rest.split(',')]
    if re.match(r'^v?st', mnem):           # vst/st: operand 1 is a SOURCE
        return None, VREG.findall(rest)
    dest = ops[0] if ops and VREG.fullmatch(ops[0]) else None
    srcs = VREG.findall(','.join(ops[1:]))
    return dest, srcs

def analyse(path):
    lines = open(path).read().splitlines()
    out = []
    for i, l in enumerate(lines):
        m = re.search(r'vtanh\s+(\w+),', l)
        if not m:
            continue
        d = m.group(1)
        touch = touched_by(d)
        verdict, where = 'dead(never-referenced)', ''
        for j in range(i + 1, len(lines)):
            # One VLIW bundle issues together: every slot reads the PRE-bundle
            # value, so a read in the same bundle as a write is a read, not a loss.
            reads, writes, text = set(), set(), []
            for s in slots(lines[j]):
                dest, srcs = dest_and_srcs(s)
                reads.update(srcs)
                if dest:
                    writes.add(dest)
                text.append(s)
            if reads & touch:
                verdict, where = 'read', f'+{j-i}: {" ; ".join(text)[:70]}'
                break
            if writes & touch:
                verdict, where = 'CLOBBER', f'+{j-i}: {" ; ".join(text)[:70]}'
                break
        out.append((d, verdict, where))
    return out

if __name__ == '__main__':
    r = analyse(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    print(f'{tag}: {len(r)} vtanh')
    for d, v, w in r:
        print(f'   vtanh {d:>6} -> {v} {w}')
