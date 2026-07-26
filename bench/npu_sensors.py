#!/usr/bin/env python3
"""Read the NPU's own sensors via DRM_AMDXDNA_QUERY_SENSORS.

The energy scoreboard the decode work is judged on has no working instrument: package RAPL
cannot resolve NPU per-token decode energy (the whole-SoC idle floor swamps the signal and
drifts within a single sweep). The driver already exposes a power sensor through the UAPI --
it is simply not surfaced through hwmon, so nothing reads it. This is that reader.

Usage:
    python bench/npu_sensors.py            # one sample, all sensors
    python bench/npu_sensors.py --watch 20 # sample once a second for 20 s (does it TRACK load?)

Exposing the same sensor through hwmon is a separable upstream follow-on; it is NOT a
prerequisite for using the number here.
"""
import argparse
import ctypes
import fcntl
import os
import struct
import time

DRM_COMMAND_BASE = 0x40
DRM_AMDXDNA_GET_INFO = 7
DRM_AMDXDNA_QUERY_SENSORS = 4

# _IOWR('d', DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO, struct amdxdna_drm_get_info)
_IOC_WRITE, _IOC_READ = 1, 2
_GET_INFO_SIZE = 16  # __u32 param; __u32 buffer_size; __u64 buffer;
DRM_IOCTL_AMDXDNA_GET_INFO = (
    ((_IOC_READ | _IOC_WRITE) << 30)
    | (_GET_INFO_SIZE << 16)
    | (ord("d") << 8)
    | (DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO)
)

# struct amdxdna_drm_query_sensor
_SENSOR_FMT = "64sIIII64s16sbB6x"
_SENSOR_SIZE = struct.calcsize(_SENSOR_FMT)
assert _SENSOR_SIZE == 168, _SENSOR_SIZE

SENSOR_TYPE = {0: "power", 1: "column_utilization"}


def _cstr(b):
    return b.split(b"\0", 1)[0].decode("utf-8", "replace")


def read_sensors(dev="/dev/accel/accel0"):
    """Return a list of sensor dicts. `value` is already unit-scaled (pow(10, unitm) * input)."""
    # Size the request with headroom rather than probing: a zero-length buffer makes the
    # driver copy to a null pointer and the ioctl comes back EFAULT. This is what the XRT
    # shim does too (32 entries of headroom, then trust the returned buffer_size).
    max_entries = 32
    fd = os.open(dev, os.O_RDWR)
    try:
        buf = ctypes.create_string_buffer(max_entries * _SENSOR_SIZE)
        req = bytearray(struct.pack("=IIQ", DRM_AMDXDNA_QUERY_SENSORS, len(buf),
                                    ctypes.addressof(buf)))
        fcntl.ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, req, True)
        _, got, _ = struct.unpack("=IIQ", bytes(req))
        if got > len(buf):
            raise RuntimeError(f"sensor buffer too small: driver wants {got} B")
    finally:
        os.close(fd)

    out = []
    raw = buf.raw[:got]
    for off in range(0, (len(raw) // _SENSOR_SIZE) * _SENSOR_SIZE, _SENSOR_SIZE):
        (label, inp, mx, avg, high, status, units,
         unitm, stype) = struct.unpack_from(_SENSOR_FMT, raw, off)
        scale = 10.0 ** unitm
        out.append(dict(
            label=_cstr(label), type=SENSOR_TYPE.get(stype, f"type{stype}"),
            status=_cstr(status), units=_cstr(units), unitm=unitm,
            value=inp * scale, max=mx * scale, average=avg * scale, highest=high * scale,
            raw_input=inp,
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default="/dev/accel/accel0")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="sample once a second for N seconds instead of once")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    if not args.watch:
        for s in read_sensors(args.dev):
            print(f"{s['label']:24s} {s['type']:20s} {s['value']:10.4f} {s['units']:6s} "
                  f"(avg {s['average']:.4f} max {s['max']:.4f} high {s['highest']:.4f}) "
                  f"raw={s['raw_input']} unitm={s['unitm']} status={s['status']}")
        return

    t0 = time.monotonic()
    print(f"{'t(s)':>7s}  " + "  ".join(f"{s['label']}({s['units']})"
                                        for s in read_sensors(args.dev)))
    while time.monotonic() - t0 < args.watch:
        row = read_sensors(args.dev)
        print(f"{time.monotonic() - t0:7.1f}  " + "  ".join(f"{s['value']:.4f}" for s in row),
              flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
