"""Extract selected members from a PyPI wheel without downloading the whole file.

A wheel is a zip, so ranged GETs let us read the central directory and then pull only the
members we want. The robosuite wheel is 150 MB and we need about 8 MB of it.
"""

from __future__ import annotations

import json
import struct
import urllib.request
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass

_EOCD = b"PK\x05\x06"
_CENTRAL = b"PK\x01\x02"


@dataclass(frozen=True)
class Member:
    name: str
    header_offset: int
    compressed_size: int
    compress_type: int


class RemoteZip:
    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size

    def _range(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request) as response:
            return response.read()

    def members(self) -> dict[str, Member]:
        tail = self._range(max(0, self.size - 100_000), self.size - 1)
        eocd = tail.rfind(_EOCD)
        if eocd < 0:
            raise RuntimeError(f"no zip end-of-central-directory in {self.url}")
        directory_size, directory_offset = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
        if directory_offset == 0xFFFFFFFF:
            raise RuntimeError("zip64 archives are not supported")
        blob = self._range(directory_offset, directory_offset + directory_size - 1)

        found: dict[str, Member] = {}
        cursor = 0
        while cursor < len(blob) and blob[cursor : cursor + 4] == _CENTRAL:
            compress_type = struct.unpack("<H", blob[cursor + 10 : cursor + 12])[0]
            compressed_size = struct.unpack("<I", blob[cursor + 20 : cursor + 24])[0]
            name_len, extra_len, comment_len = struct.unpack(
                "<HHH", blob[cursor + 28 : cursor + 34]
            )
            header_offset = struct.unpack("<I", blob[cursor + 42 : cursor + 46])[0]
            name = blob[cursor + 46 : cursor + 46 + name_len].decode("utf-8", "replace")
            found[name] = Member(name, header_offset, compressed_size, compress_type)
            cursor += 46 + name_len + extra_len + comment_len
        return found

    def read(self, member: Member) -> bytes:
        # The local header repeats the name and extra field with its own lengths.
        head = self._range(member.header_offset, member.header_offset + 29)
        name_len, extra_len = struct.unpack("<HH", head[26:30])
        start = member.header_offset + 30 + name_len + extra_len
        raw = self._range(start, start + member.compressed_size - 1)
        if member.compress_type == 0:
            return raw
        return zlib.decompress(raw, -zlib.MAX_WBITS)


def newest_wheel(project: str) -> RemoteZip:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{project}/json") as response:
        metadata = json.load(response)
    wheels = [f for f in metadata["urls"] if f["filename"].endswith(".whl")]
    if not wheels:
        raise RuntimeError(f"{project} publishes no wheel")
    chosen = wheels[0]
    return RemoteZip(chosen["url"], chosen["size"])


def select(
    members: dict[str, Member], predicate: Callable[[str], bool]
) -> Iterator[Member]:
    for name in sorted(members):
        if predicate(name):
            yield members[name]
