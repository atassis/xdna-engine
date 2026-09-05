#!/usr/bin/env python3
"""Normalised content hash of an xclbin: mask per-BUILD identity, hash what's left.

`sha256(xclbin)` answers "same build run", never "same program" -- an xclbin embeds per-build
identity in five fields plus a text duplicate of three of them, all regenerated fresh by a
rebuild of the IDENTICAL design (measured 2026-09-01, two builds of one design on one toolchain:
68 bytes differ, all of them identity -- [[an-xclbin-hash-answers-same-build-not-same-program]]):

  1. `axlf.m_uniqueId`            (8 B,  container header) -- "use it to skip redownload etc",
                                    regenerated per build so the driver can tell builds apart.
  2. `axlf_header.m_timeStamp`    (8 B,  container header) -- unix seconds when xclbin was built.
  3. `axlf_header.uuid`           (16 B, container header) -- fresh xclbin UUID every build.
  4. `aie_pdi.uuid`               (16 B, per AIE_PARTITION section entry) -- fresh PDI container
                                    UUID every build. There is one per PDI; some designs pack more
                                    than one PDI into a partition, so every entry is masked.
  5. bootgen's `ImageHeader.metaHdrRevokeId` + `.checksum` (4 B each, INSIDE the PDI blob
     `aie_pdi.pdi_image` embeds) -- see "The bootgen field" below. Found 2026-09-05 after a real
     same-toolchain two-build pair (`armB_newpin` vs `armB2_newpin`,
     `.cache/peano-gate-2026-09-01`, both toolchain-stamp `4b8464eac495`) still hashed unequal
     post-mask on 4 of 5 designs.
  6. xclbinutil's MIRROR_DATA trailer (ASCII JSON appended after the formal sections, used to
     support `--replace-section` without the original recipe) duplicates #1-#3 as text:
     `"UniqueID"`, `"TimeStamp"`, `"XclBinUUID"`. Left unmasked, this text alone would make two
     builds of one design hash differently even with the binary fields above masked.

Struct layouts are taken from the installed XRT header (`/usr/include/xrt/detail/xclbin.h`,
XRT 2.21), reproduced here as `struct.Struct` objects with the SAME size assertions that header
carries as `XCLBIN_STATIC_ASSERT` -- so a layout change this tool does not know about fails loud
here too, instead of silently masking (or missing) the wrong bytes. Only the outer axlf/section
layout is a format constant; every SECTION offset, PDI count and PDI offset is read from the file
being hashed, never assumed.

The bootgen field (#5). `aie_pdi.pdi_image` is not our format -- it is a Xilinx Versal boot image
written by `bootgen` (third_party/bootgen, vendored; aiecc.cpp statically links
`bootgen_generate_pdi`). `bootgen -read <extracted pdi_image> -arch versal` names the two fields
directly: `mHdr_revoke_id (0x08)` inside the Image Header, and its `checksum (0x3c)` (which covers
bytes [0x00,0x3c) of that header, i.e. it is DERIVED FROM mHdr_revoke_id). Grepping
`third_party/bootgen` for `metaHdrRevokeId` (the field's real name) finds FOUR hits, and the
reachability is what matters rather than the count: the declaration
(`versal/include/imageheadertable-versal.h:144`), the `-read` display code
(`versal/src/readimage-versal.cpp:699`), and two Versal setters
(`versal/src/imageheadertable-versal.cpp:1899,4699`). Those setters exist only to satisfy a pure
virtual on the common base (`common/include/imageheadertable.h:523`); their only call sites are in
the SPARTAN-UP backend (`spartanup/src/imageheadertable-spartanup.cpp:1141,2971`), a different device
family. So on the Versal path they are never invoked -- bootgen never WRITES
it when building an image, so it is uninitialized memory, not a value with meaning for an
unauthenticated single-partition CDO image (revocation IDs are a secure/authenticated-boot
concept; our BIF requests neither). Confirmed independently of any real build: two `aiecc`
invocations 4 SECONDS apart, byte-identical MLIR/kernel-object/toolchain inputs, produced two
different `mHdr_revoke_id` values and reproduced the exact same 75-byte raw diff (same offsets,
same widths) as the real 20-minutes-apart pair. The Image Header's OTHER covered fields
(pht_offset, section_count, name, id, memcpy addresses) are constant across every design in this
repo's `artifacts/`; only mHdr_revoke_id moves, which is why masking it (and the checksum riding
on it) does not risk hiding a real content change -- the Partition Header Table's OWN checksum,
which DOES cover content-derived fields (encrypted_length, partition_offset, ...), is untouched
and stays stable across the same real pair.
`aie_pdi.pdi_image` starts with a 16-byte boot-image sync/width-detect preamble (present even when
bootgen's own `-read` reports "NO BOOT HEADER") -- confirmed byte-identical across 97 PDI entries
in every xclbin under `artifacts/` at the time of writing. The Image Header Table's `ih_offset`
field, 8 bytes into that preamble, gives the Image Header's position as a 32-bit WORD count from
the start of `pdi_image` (bootgen convention, verified against `-read`'s own byte offsets) --
`normalize()` reads that field rather than hardcoding the Image Header's position, so only the
16-byte preamble size is an assumed constant, matching the offset-derivation discipline used
everywhere else in this file.

What is deliberately NOT masked, because it is not per-build: `m_featureRomTimeStamp` and
`m_interface_uuid` are unused by every design measured so far (always 0) -- masking a field that
never varies would be over-masking without evidence. `kernel_commit_id` is the KERNEL SOURCE's
git commit, which is a property of the program, not the build. The Partition Header Table's own
checksum (see above) is left alone for the same reason.

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

# aie_pdi: uuid[16] (masked here), pdi_image{size,offset}, cdo_groups{size,offset}, reserved[64].
AIE_PDI = struct.Struct("<16s8s8s64s")
assert AIE_PDI.size == 96, "aie_pdi no longer 96 bytes"

# bootgen's PDI layout (see module docstring "The bootgen field"): a 16-byte sync/width-detect
# preamble, then the Image Header Table, whose ih_offset field (8 bytes into the preamble) is a
# 32-bit WORD offset of the Image Header from pdi_image byte 0. Confirmed byte-identical across
# every PDI entry in this repo's artifacts/ at the time of writing.
PDI_PREAMBLE = bytes.fromhex("dd0000004433221188776655ccbbaa99")
IHT_IH_OFFSET_FIELD = len(PDI_PREAMBLE) + 0x08  # ImageHeaderTable.ih_offset (word count)
IH_REVOKE_ID_OFF = 0x08   # ImageHeader.metaHdrRevokeId -- bootgen never writes it (uninitialized)
IH_CHECKSUM_OFF = 0x3c    # ImageHeader.checksum -- covers [0x00,0x3c), so it rides on revoke_id

MIRROR_START = b"XCLBIN_MIRROR_DATA_START"
MIRROR_END = b"XCLBIN_MIRROR_DATA_END"


class XclbinFormatError(Exception):
    """The input is not a well-formed xclbin2 container, or is truncated."""


def _mask_bootgen_image_header(raw, img_off, img_size, path, pdi_index):
    """Zero bootgen's uninitialized mHdr_revoke_id + the checksum riding on it (see module
    docstring). Returns True if masked. A layout that does not match the preamble/bounds we have
    verified across every xclbin in this repo is left UNMASKED with a warning -- under-masking an
    unrecognised layout is safer than guessing at its offsets."""
    if img_size < IHT_IH_OFFSET_FIELD + 4:
        print(f"xclbin_norm: {path}: PDI[{pdi_index}] pdi_image too small for a bootgen IHT, "
              "not masking its per-build header field", file=sys.stderr)
        return False
    if bytes(raw[img_off:img_off + len(PDI_PREAMBLE)]) != PDI_PREAMBLE:
        print(f"xclbin_norm: {path}: PDI[{pdi_index}] pdi_image has an unrecognised preamble, "
              "not masking its per-build header field", file=sys.stderr)
        return False
    ih_off_words = struct.unpack_from("<I", raw, img_off + IHT_IH_OFFSET_FIELD)[0]
    ih_base = img_off + ih_off_words * 4
    if ih_base < img_off or ih_base + IH_CHECKSUM_OFF + 4 > img_off + img_size:
        print(f"xclbin_norm: {path}: PDI[{pdi_index}] Image Header offset runs outside "
              "pdi_image, not masking its per-build header field", file=sys.stderr)
        return False
    raw[ih_base + IH_REVOKE_ID_OFF:ih_base + IH_REVOKE_ID_OFF + 4] = b"\0" * 4
    raw[ih_base + IH_CHECKSUM_OFF:ih_base + IH_CHECKSUM_OFF + 4] = b"\0" * 4
    return True


def _mask_aie_partition(raw, sec_off, sec_size, path):
    if sec_size < AIE_PARTITION_PREFIX.size:
        raise XclbinFormatError(f"{path}: AIE_PARTITION section is {sec_size} B, too small")
    *_rest, pdi_count, pdi_off = AIE_PARTITION_PREFIX.unpack_from(raw, sec_off)
    for i in range(pdi_count):
        entry_off = sec_off + pdi_off + i * AIE_PDI.size
        if entry_off + AIE_PDI.size > sec_off + sec_size:
            raise XclbinFormatError(f"{path}: PDI[{i}] runs past its AIE_PARTITION section")
        raw[entry_off:entry_off + 16] = b"\0" * 16  # aie_pdi.uuid
        img_size, img_off = struct.unpack_from("<II", raw, entry_off + 16)
        if img_off + img_size > sec_off + sec_size:
            raise XclbinFormatError(f"{path}: PDI[{i}] pdi_image runs past its section")
        _mask_bootgen_image_header(raw, sec_off + img_off, img_size, path, i)
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
