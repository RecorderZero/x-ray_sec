"""Canonical byte encoding shared by every v2 structure (§8.4.1 (1)-(3)).

``LP(x) = u32_BE(len(x)) || x``. Decoding is strict: a length that runs past the
buffer, an over-limit field, or trailing bytes after a structure is rejected
before any large allocation.
"""

from __future__ import annotations

import re
import struct

from .errors import ShareFormatError


PREAMBLE_MAGIC = b"SOTERIA\x00"
CONTAINER_VERSION = 2
PREAMBLE = PREAMBLE_MAGIC + struct.pack(">H", CONTAINER_VERSION)

PROFILE_M1_PRIVATE_RANGE_V2 = "soteria-share/m1-private-range/v2"
ALLOWED_PROFILES = frozenset({PROFILE_M1_PRIVATE_RANGE_V2})

MAX_H_BYTES = 4096
META_CT_BYTES = 32
MAX_PNG_BYTES = 1_048_576
TAG_BYTES = 32
SIG_BYTES = 64
DIGEST_BYTES = 32

_PROFILE_RE = re.compile(r"^soteria-share/[a-z0-9-]+/v[0-9]+$")
_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_U32 = struct.Struct(">I")


def lp(value: bytes) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise ShareFormatError("LP field must be bytes")
    if len(value) >= 2**32:
        raise ShareFormatError("LP field too long")
    return _U32.pack(len(value)) + bytes(value)


def lp_join(*fields: bytes) -> bytes:
    return b"".join(lp(field) for field in fields)


class Reader:
    """Strict cursor over one buffer; call :meth:`done` to reject trailing bytes."""

    def __init__(self, data: bytes):
        if not isinstance(data, (bytes, bytearray)):
            raise ShareFormatError("expected bytes")
        self._data = bytes(data)
        self._pos = 0

    def fixed(self, size: int) -> bytes:
        if size < 0 or self._pos + size > len(self._data):
            raise ShareFormatError("truncated input")
        out = self._data[self._pos:self._pos + size]
        self._pos += size
        return out

    def lp(self, max_len: int, *, exact: int | None = None) -> bytes:
        (length,) = _U32.unpack(self.fixed(4))
        if exact is not None and length != exact:
            raise ShareFormatError(f"field must be exactly {exact} bytes")
        if length > max_len:
            raise ShareFormatError("field exceeds its size limit")
        return self.fixed(length)

    def done(self) -> None:
        if self._pos != len(self._data):
            raise ShareFormatError("trailing bytes after structure")


def ascii_field(raw: bytes, pattern: re.Pattern[str], what: str) -> str:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ShareFormatError(f"{what} must be ASCII") from exc
    if not pattern.fullmatch(text):
        raise ShareFormatError(f"invalid {what}")
    return text


def require_profile(profile_id: str) -> str:
    if not isinstance(profile_id, str) or not _PROFILE_RE.fullmatch(profile_id):
        raise ShareFormatError("profile_id must match soteria-share/<name>/v<N>")
    if profile_id not in ALLOWED_PROFILES:
        raise ShareFormatError("profile_id is not on the allowlist")
    return profile_id


def require_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ShareFormatError(f"invalid {what}")
    return value


def decode_profile(raw: bytes) -> str:
    return require_profile(ascii_field(raw, _PROFILE_RE, "profile_id"))


def decode_id(raw: bytes, what: str) -> str:
    return ascii_field(raw, _ID_RE, what)
