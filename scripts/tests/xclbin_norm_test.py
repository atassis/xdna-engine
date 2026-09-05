#!/usr/bin/env python3
"""CPU gate for xclbin_norm.py: positive control (identity-only mutation hashes equal) and
negative control (a real program byte hashes different), both device-free.

The positive control here is SYNTHETIC. Two on-disk candidates that looked like a real same-
design pair (`armA_oldpin` under `.cache/peano-gate-2026-09-01` vs `.cache/repin-gate-2026-09-04`)
turned out, per their `.toolchain-stamp`, to be built from two DIFFERENT toolchain instances
(f37308d2b719 vs 9da6356ac521) -- not a valid pair, and their normalized hashes correctly still
differ. So this gate proves the mask is complete against the KNOWN identity fields
([[an-xclbin-hash-answers-same-build-not-same-program]]); it does not prove those are the only
fields two real builds can vary in. The task record carries the honest gap.

Run: python3 scripts/tests/xclbin_norm_test.py
"""
import glob
import hashlib
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import xclbin_norm as N  # noqa: E402

failures = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        failures.append(msg)


def find_fixture():
    if os.environ.get("XCLBIN_NORM_TEST_FIXTURE"):
        return os.environ["XCLBIN_NORM_TEST_FIXTURE"]
    cands = sorted(glob.glob(str(REPO / "artifacts" / "**" / "*.xclbin"), recursive=True))
    if not cands:
        print("no .xclbin under artifacts/ -- build a kernel first (device-free build is fine)",
              file=sys.stderr)
        sys.exit(1)
    return cands[0]


def locate_sections(raw):
    """Independent structural walk over format constants only -- does not call into
    xclbin_norm's own masking logic, so it cannot rubber-stamp a bug there."""
    header_off = N.AXLF_PREFIX.size
    fields = N.AXLF_HEADER.unpack_from(raw, header_off)
    num_sections = fields[12]
    table_off = header_off + N.AXLF_HEADER.size
    out = []
    for i in range(num_sections):
        off = table_off + i * N.SECTION_HEADER.size
        kind, _name, s_off, s_size = N.SECTION_HEADER.unpack_from(raw, off)
        out.append((kind, s_off, s_size))
    return out


def pdi_entries(raw):
    """[(uuid_off, pdi_image_off, pdi_image_size), ...] across every AIE_PARTITION section."""
    out = []
    for kind, s_off, s_size in locate_sections(raw):
        if kind != N.AIE_PARTITION_KIND:
            continue
        *_rest, pdi_count, pdi_off = N.AIE_PARTITION_PREFIX.unpack_from(raw, s_off)
        for i in range(pdi_count):
            entry_off = s_off + pdi_off + i * N.AIE_PDI.size
            img_size, img_off = struct.unpack_from("<II", raw, entry_off + 16)
            out.append((entry_off, s_off + img_off, img_size))
    return out


def swap_mirror_text(buf, start, end, key, old_val_hex, new_val_hex):
    old_pat = f'"{key}":"{old_val_hex}"'.encode()
    new_pat = f'"{key}":"{new_val_hex}"'.encode()
    i = buf.find(old_pat, start, end)
    check(i >= 0, f"mirror carries a matching {key} to update")
    if i >= 0:
        buf[i:i + len(old_pat)] = new_pat


fixture = find_fixture()
print(f"fixture: {fixture}")
orig = bytearray(Path(fixture).read_bytes())

uid_off = N.AXLF_PREFIX.size - 8
ts_off = N.AXLF_PREFIX.size + N.AXLF_HEADER_TIMESTAMP_OFF
huuid_off = N.AXLF_PREFIX.size + N.AXLF_HEADER_UUID_OFF

print("\n--- positive control: mutate ONLY the fields xclbin_norm masks ---")
mutated = bytearray(orig)
rng = random.Random(0)
mutated[uid_off:uid_off + 8] = bytes(rng.randrange(256) for _ in range(8))
# Keep the new timestamp the same decimal WIDTH as a real one (10 digits, current era) -- a wider
# random 64-bit value would change the mirror text's byte LENGTH, which is a test-harness bug
# (bytearray slice assignment resizing the buffer), not anything xclbin_norm.py needs to handle.
mutated[ts_off:ts_off + 8] = struct.pack("<Q", rng.randrange(1_000_000_000, 2_000_000_000))
mutated[huuid_off:huuid_off + 16] = bytes(rng.randrange(256) for _ in range(16))
for entry_off, _img_off, _img_size in pdi_entries(mutated):
    mutated[entry_off:entry_off + 16] = bytes(rng.randrange(256) for _ in range(16))

# A real rebuild's mirror trailer always agrees with its own binary fields -- xclbinutil writes
# both from the same values -- so the synthetic mutation must keep them in sync too, or this
# would only prove the binary mask works and say nothing about the text mask.
start = mutated.find(N.MIRROR_START)
check(start >= 0, "fixture carries a MIRROR_DATA trailer (exercises the text-masking path)")
if start >= 0:
    end = mutated.find(N.MIRROR_END, start)
    old_uid = struct.unpack_from("<Q", orig, uid_off)[0]
    new_uid = struct.unpack_from("<Q", mutated, uid_off)[0]
    old_ts = struct.unpack_from("<Q", orig, ts_off)[0]
    new_ts = struct.unpack_from("<Q", mutated, ts_off)[0]
    old_huuid, new_huuid = bytes(orig[huuid_off:huuid_off + 16]), bytes(mutated[huuid_off:huuid_off + 16])
    swap_mirror_text(mutated, start, end, "UniqueID",
                      old_uid.to_bytes(8, "little").hex(), new_uid.to_bytes(8, "little").hex())
    swap_mirror_text(mutated, start, end, "TimeStamp", str(old_ts), str(new_ts))
    swap_mirror_text(mutated, start, end, "XclBinUUID", old_huuid.hex(), new_huuid.hex())

check(bytes(mutated) != bytes(orig), "the raw bytes actually changed")
h_orig = hashlib.sha256(N.normalize(bytes(orig), fixture)).hexdigest()
h_mut = hashlib.sha256(N.normalize(bytes(mutated), fixture)).hexdigest()
check(h_orig == h_mut, f"identity-only mutation hashes EQUAL ({h_orig[:12]}...)")

print("\n--- negative control: flip a byte inside pdi_image (real program content) ---")
entries = pdi_entries(orig)
check(len(entries) > 0, "fixture has at least one AIE_PARTITION PDI entry")
_uuid_off0, img_off, img_size = entries[0]
target = img_off + img_size // 2
neg = bytearray(orig)
neg[target] ^= 0xFF
check(neg[target] != orig[target], "the flipped byte actually changed")

# Fail-loud guard (method-negative-control-must-fail-loud): confirm the flip landed somewhere
# xclbin_norm does NOT mask, using the independent walk above rather than trusting normalize()'s
# own bookkeeping about what it touched.
masked_ranges = [(uid_off, uid_off + 8), (ts_off, ts_off + 8), (huuid_off, huuid_off + 16)]
masked_ranges += [(e[0], e[0] + 16) for e in entries]
check(img_off <= target < img_off + img_size, "flip target lands inside pdi_image")
check(not any(a <= target < b for a, b in masked_ranges), "flip target is OUTSIDE every masked range")

h_orig2 = hashlib.sha256(N.normalize(bytes(orig), fixture)).hexdigest()
h_neg = hashlib.sha256(N.normalize(bytes(neg), fixture)).hexdigest()
check(h_orig2 == h_orig, "sanity: hashing the same bytes twice is stable")
check(h_orig2 != h_neg, "a real content byte hashes DIFFERENT")

print("\n--- malformed input is rejected, not silently mis-hashed ---")
try:
    N.normalize(b"not an xclbin at all", "garbage")
    check(False, "garbage bytes raise XclbinFormatError")
except N.XclbinFormatError:
    check(True, "garbage bytes raise XclbinFormatError")

r = subprocess.run(
    [sys.executable, str(REPO / "scripts" / "xclbin_norm.py"), "/nonexistent/path.xclbin"],
    capture_output=True, text=True,
)
check(r.returncode != 0, "CLI exits non-zero on a missing file")
check("/nonexistent/path.xclbin" in r.stderr, "CLI names the failing file on stderr")

print()
if failures:
    print(f"GATE RED ({len(failures)} failed)")
    sys.exit(1)
print("GATE GREEN")
