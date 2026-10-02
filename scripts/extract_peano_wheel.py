#!/usr/bin/env python3
"""Extract a checksum-verified compiler wheel with its executable permissions."""
import os
import sys
import zipfile


def extract(wheel: str, destination: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            path = archive.extract(member, destination)
            mode = (member.external_attr >> 16) & 0o777
            if member.create_system == 3 and mode:
                os.chmod(path, mode)


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
