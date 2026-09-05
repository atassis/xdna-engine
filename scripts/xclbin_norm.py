#!/usr/bin/env python3
"""Normalised content hash of an xclbin: mask per-BUILD identity, hash what's left.

`sha256(xclbin)` answers "same build run", never "same program" -- an xclbin embeds per-build
identity in four fields plus a text duplicate of three of them, all regenerated fresh by a
rebuild of the IDENTICAL design (measured 2026-09-01, two builds of one design on one toolchain:
68 bytes differ, all of them identity -- [[an-xclbin-hash-answers-same-build-not-same-program]]):

  1. `axlf.m_uniqueId`            (8 B,  container header) -- "use it to skip redownload etc",
                                    regenerated per build so the driver can tell builds apart.
  2. `axlf_header.m_timeStamp`    (8 B,  container header) -- unix seconds when xclbin was built.
  3. `axlf_header.uuid`           (16 B, container header) -- fresh xclbin UUID every build.
  4. `aie_pdi.uuid`               (16 B, per AIE_PARTITION section entry) -- fresh PDI container
                                    UUID every build. There is one per PDI; some designs pack more
                                    than one PDI into a partition, so every entry is masked.
  5. xclbinutil's MIRROR_DATA trailer (ASCII JSON appended after the formal sections, used to
     support `--replace-section` without the original recipe) duplicates #1-#3 as text:
     `"UniqueID"`, `"TimeStamp"`, `"XclBinUUID"`. Left unmasked, this text alone would make two
     builds of one design hash differently even with the binary fields above masked.

Struct layouts are taken from the installed XRT header (`/usr/include/xrt/detail/xclbin.h`,
XRT 2.21), reproduced here as `struct.Struct` objects with the SAME size assertions that header
carries as `XCLBIN_STATIC_ASSERT` -- so a layout change this tool does not know about fails loud
here too, instead of silently masking (or missing) the wrong bytes. Only the outer axlf/section
layout is a format constant; every SECTION offset, PDI count and PDI offset is read from the file
being hashed, never assumed.

What is deliberately NOT masked, because it is not per-build: `m_featureRomTimeStamp` and
`m_interface_uuid` are unused by every design measured so far (always 0) -- masking a field that
never varies would be over-masking without evidence. `kernel_commit_id` is the KERNEL SOURCE's
git commit, which is a property of the program, not the build.

CLI:  xclbin_norm.py FILE [FILE ...]     -- prints "<hash>  <path>" per file, sha256sum-style.
API:  normalized_sha256(path) -> str     -- for other tooling (kernel_registry.rs, eventually).
"""
import argparse
import hashlib
import struct
import sys
from pathlib import Path

MAGIC = b"xclbin2\x00"
AIE_PARTITION_KIND = 32  # enum axlf_section_kind, xclbin.h

# axlf, through m_uniqueId: magic, m_signature_length, reserved[28], m_keyBlock[256], m_uniqueId.
AXLF_PREFIX = struct.Struct("<8si28s256sQ")
assert AXLF_PREFIX.size == 304, "axlf prefix (through m_uniqueId) no longer 304 bytes"

# axlf_header: m_length, m_timeStamp, m_featureRomTimeStamp, versionPatch/Major/Minor, mode,
# actionMask, interface_uuid[16], platformVBNV[64], uuid[16] (union w/ m_next_axlf), debug_bin[16],
# numSections, +4 pad (struct's own uint64 members force its size up to a multiple of 8).
AXLF_HEADER = struct.Struct("<QQQHBBHH16s64s16s16sI4x")
assert AXLF_HEADER.size == 152, "axlf_header no longer 152 bytes (xclbin.h XCLBIN_STATIC_ASSERT)"
AXLF_HEADER_TIMESTAMP_OFF = 8    # relative to the header, right after m_length
AXLF_HEADER_UUID_OFF = 112       # relative to the header: length+timestamp+featureRom+ver*4+iface+vbnv

# axlf_section_header: kind, name[16], 4B pad (align the two following uint64), offset, size.
SECTION_HEADER = struct.Struct("<I16s4xQQ")
assert SECTION_HEADER.size == 40, "axlf_section_header no longer 40 bytes"

# aie_partition, through the aie_pdi array_offset: schema_version+pad[3], mpo_name, ops_per_cycle,
# pad[4], inference_fingerprint, pre_post_fingerprint, aie_partition_info (88B, opaque -- no field
# in it is per-build so it is never unpacked further), aie_pdi{count,offset}.
AIE_PARTITION_PREFIX = struct.Struct("<B3xII4xQQ88sII")
assert AIE_PARTITION_PREFIX.size == 128, "aie_partition prefix through aie_pdi changed size"

# aie_pdi: uuid[16] (the field masked here), pdi_image{size,offset}, cdo_groups{size,offset},
# reserved[64]. Only the leading 16 bytes are read.
AIE_PDI = struct.Struct("<16s8s8s64s")
assert AIE_PDI.size == 96, "aie_pdi no longer 96 bytes"

MIRROR_START = b"XCLBIN_MIRROR_DATA_START"
MIRROR_END = b"XCLBIN_MIRROR_DATA_END"


class XclbinFormatError(Exception):
    """The input is not a well-formed xclbin2 container, or is truncated."""


def _mask_aie_partition(raw, sec_off, sec_size, path):
    if sec_size < AIE_PARTITION_PREFIX.size:
        raise XclbinFormatError(f"{path}: AIE_PARTITION section is {sec_size} B, too small")
    *_rest, pdi_count, pdi_off = AIE_PARTITION_PREFIX.unpack_from(raw, sec_off)
    for i in range(pdi_count):
        entry_off = sec_off + pdi_off + i * AIE_PDI.size
        if entry_off + AIE_PDI.size > sec_off + sec_size:
            raise XclbinFormatError(f"{path}: PDI[{i}] runs past its AIE_PARTITION section")
        raw[entry_off:entry_off + 16] = b"\0" * 16  # aie_pdi.uuid
    return pdi_count


def _mask_mirror_text(raw, unique_id, timestamp, xclbin_uuid):
    start = raw.find(MIRROR_START)
    if start < 0:
        return  # no mirror trailer (older/foreign xclbinutil) -- nothing to mask here
    end = raw.find(MIRROR_END, start)
    end = len(raw) if end < 0 else end
    # Built from the ALREADY-PARSED binary values, not a generic key regex: this only matches
    # (and masks) text that the binary fields themselves say should be there, so a mirror whose
    # copy has drifted from the binary is left alone rather than guessed at.
    for pattern in (
        b'"UniqueID":"' + unique_id.to_bytes(8, "little").hex().encode() + b'"',
        b'"TimeStamp":"' + str(timestamp).encode() + b'"',
        b'"XclBinUUID":"' + xclbin_uuid.hex().encode() + b'"',
    ):
        i = raw.find(pattern, start, end)
        if i >= 0:
            raw[i:i + len(pattern)] = b"\0" * len(pattern)


def normalize(raw_bytes, path="<bytes>"):
    """Return a copy of raw_bytes with every per-build identity field zeroed."""
    if len(raw_bytes) < AXLF_PREFIX.size + AXLF_HEADER.size:
        raise XclbinFormatError(f"{path}: {len(raw_bytes)} B is smaller than the axlf header")
    raw = bytearray(raw_bytes)

    magic, _sig_len, _reserved, _keyblock, unique_id = AXLF_PREFIX.unpack_from(raw, 0)
    if magic != MAGIC:
        raise XclbinFormatError(f"{path}: bad magic {magic!r}, not an xclbin2 container")

    header_off = AXLF_PREFIX.size
    header_fields = AXLF_HEADER.unpack_from(raw, header_off)
    m_length, m_timeStamp = header_fields[0], header_fields[1]
    xclbin_uuid = header_fields[10]  # uuid (union w/ m_next_axlf), 11th field
    num_sections = header_fields[12]
    if m_length != len(raw):
        raise XclbinFormatError(f"{path}: m_length={m_length} != file size {len(raw)}")

    raw[AXLF_PREFIX.size - 8:AXLF_PREFIX.size] = b"\0" * 8  # axlf.m_uniqueId
    ts_off = header_off + AXLF_HEADER_TIMESTAMP_OFF
    raw[ts_off:ts_off + 8] = b"\0" * 8  # axlf_header.m_timeStamp
    uuid_off = header_off + AXLF_HEADER_UUID_OFF
    raw[uuid_off:uuid_off + 16] = b"\0" * 16  # axlf_header.uuid

    section_table_off = header_off + AXLF_HEADER.size
    if section_table_off + num_sections * SECTION_HEADER.size > len(raw):
        raise XclbinFormatError(f"{path}: section table ({num_sections} entries) runs past EOF")

    for i in range(num_sections):
        off = section_table_off + i * SECTION_HEADER.size
        kind, _name, s_offset, s_size = SECTION_HEADER.unpack_from(raw, off)
        if s_offset + s_size > len(raw):
            raise XclbinFormatError(f"{path}: section {i} (kind {kind}) runs past EOF")
        if kind == AIE_PARTITION_KIND:
            _mask_aie_partition(raw, s_offset, s_size, path)

    _mask_mirror_text(raw, unique_id, m_timeStamp, xclbin_uuid)
    return bytes(raw)


def normalized_sha256(path):
    raw = Path(path).read_bytes()
    return hashlib.sha256(normalize(raw, str(path))).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xclbin", nargs="+", help="one or more .xclbin files")
    args = ap.parse_args(argv)

    failed = False
    for p in args.xclbin:
        try:
            print(f"{normalized_sha256(p)}  {p}")
        except (OSError, XclbinFormatError) as e:
            print(f"xclbin_norm: {p}: {e}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
