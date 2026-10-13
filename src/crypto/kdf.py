"""D5.1/D5.2 key creation, password mode, and HKDF domain separation.

The default is a uniformly random 256-bit master key. Password mode is an
explicit deployment option and uses stdlib OpenSSL-backed scrypt with a random
128-bit salt. No API accepts an integer seed or silently converts text into a
master key.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .errors import CryptoConfigurationError


MASTER_KEY_BYTES = 32
NONCE_BYTES = 16
SCRYPT_SALT_BYTES = 16
HKDF_BYTES = 32
LABEL_PREFIX = "cxr-transform/v1"
MAC_LABEL = f"{LABEL_PREFIX}/mac"
PURPOSES = ("sign-pre", "perm", "sign-post")


def require_master_key(master_key: bytes) -> bytes:
    if not isinstance(master_key, bytes) or len(master_key) != MASTER_KEY_BYTES:
        raise CryptoConfigurationError("master key must be exactly 32 bytes")
    return master_key


def require_nonce(nonce: bytes) -> bytes:
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
        raise CryptoConfigurationError("nonce must be exactly 16 bytes")
    return nonce


def generate_master_key() -> bytes:
    """Return a new random 256-bit key from the operating-system CSPRNG."""
    return secrets.token_bytes(MASTER_KEY_BYTES)


@dataclass(frozen=True)
class ScryptParameters:
    n: int = 1 << 15
    r: int = 8
    p: int = 1
    dklen: int = MASTER_KEY_BYTES
    maxmem: int = 128 * 1024 * 1024

    def validate(self) -> None:
        if self.n < (1 << 14) or self.n & (self.n - 1):
            raise CryptoConfigurationError("scrypt n must be a power of two >= 2^14")
        if self.r < 1 or self.p < 1 or self.dklen != MASTER_KEY_BYTES:
            raise CryptoConfigurationError("invalid scrypt r/p/dklen")

    def manifest(self) -> dict[str, int | str]:
        return {"algorithm": "scrypt", "n": self.n, "r": self.r, "p": self.p,
                "dklen": self.dklen, "salt_bytes": SCRYPT_SALT_BYTES}


@dataclass(frozen=True, repr=False)
class PasswordKeyResult:
    master_key: bytes = field(repr=False)
    salt: bytes
    parameters: ScryptParameters

    def __repr__(self) -> str:
        return (f"PasswordKeyResult(master_key=<redacted>, salt={self.salt.hex()!r}, "
                f"parameters={self.parameters!r})")


def master_key_from_password(
    password: str,
    *,
    salt: bytes | None = None,
    parameters: ScryptParameters = ScryptParameters(),
) -> PasswordKeyResult:
    """Derive 32 bytes using UTF-8(password); password entropy remains decisive."""
    if not isinstance(password, str) or not password:
        raise CryptoConfigurationError("password mode requires a non-empty string")
    parameters.validate()
    salt = secrets.token_bytes(SCRYPT_SALT_BYTES) if salt is None else salt
    if not isinstance(salt, bytes) or len(salt) != SCRYPT_SALT_BYTES:
        raise CryptoConfigurationError("scrypt salt must be exactly 16 bytes")
    key = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=parameters.n, r=parameters.r,
        p=parameters.p, dklen=parameters.dklen, maxmem=parameters.maxmem,
    )
    return PasswordKeyResult(key, salt, parameters)


def hkdf_sha256(ikm: bytes, *, salt: bytes, info: bytes, length: int) -> bytes:
    """Generic RFC 5869 helper, exposed so official vectors can test the primitive."""
    if not isinstance(ikm, bytes) or not isinstance(salt, bytes) or not isinstance(info, bytes):
        raise CryptoConfigurationError("HKDF inputs must be bytes")
    if length < 1 or length > 255 * hashes.SHA256.digest_size:
        raise CryptoConfigurationError("invalid HKDF output length")
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def round_label(round_index: int, purpose: str) -> str:
    if not isinstance(round_index, int) or round_index < 0:
        raise CryptoConfigurationError("round index must be a non-negative integer")
    if purpose not in PURPOSES:
        raise CryptoConfigurationError(f"unknown transform purpose: {purpose}")
    return f"{LABEL_PREFIX}/round/{round_index}/{purpose}"


def derive_subkey(master_key: bytes, nonce: bytes, label: str) -> bytes:
    """Derive one 256-bit subkey; the full 128-bit image nonce is the HKDF salt."""
    require_master_key(master_key)
    require_nonce(nonce)
    allowed = label == MAC_LABEL or any(
        label.endswith(f"/{purpose}") and label.startswith(f"{LABEL_PREFIX}/round/")
        for purpose in PURPOSES
    )
    if not isinstance(label, str) or not allowed:
        raise CryptoConfigurationError("label is outside the frozen v1 domain set")
    return hkdf_sha256(master_key, salt=nonce, info=label.encode("ascii"), length=HKDF_BYTES)
