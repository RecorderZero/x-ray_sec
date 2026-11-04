"""Per-recipient image key: HPKE wrap/unwrap, v2 subkeys, and buffer hygiene.

§8.3 steps 2-4 / §8.4.1 (4)-(5). The image key is never derived from a master
key: it is fresh CSPRNG output wrapped to the recipient's X25519 key with
RFC 9180 HPKE (``cryptography`` 50.0.2 single-shot ``Suite``), so only the
holder of that private key can recover it in the normal protocol.

"Not landing" (§8.5) is an application-level property: secrets live in
:class:`SecretKey` buffers that are zeroed on every exit path and are never
logged, repr'd or pickled. Python cannot guarantee that immutable copies made
inside ``cryptography`` or by the interpreter are erased.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, hpke
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .encoding import lp, lp_join, require_id, require_profile
from .errors import ShareFormatError, UnwrapError


IMAGE_KEY_BYTES = 32
NONCE_BYTES = 16
WRAP_BYTES = 80  # X25519 enc (32) + wrapped key (32) + AEAD tag (16)
SUBKEY_BYTES = 32
SUBKEY_LABELS = ("round/0/sign-pre", "round/0/perm", "round/0/sign-post", "mac", "private-metadata-aead")
SUITE = hpke.Suite(hpke.KEM.X25519, hpke.KDF.HKDF_SHA256, hpke.AEAD.CHACHA20_POLY1305)


class SecretKey:
    """Mutable 32-byte secret with explicit zeroization; redacted everywhere."""

    __slots__ = ("_buf", "_wiped")

    def __init__(self, material: bytes | bytearray):
        if not isinstance(material, (bytes, bytearray)) or len(material) != IMAGE_KEY_BYTES:
            raise ShareFormatError("secret keys are exactly 32 bytes")
        self._buf = bytearray(material)
        self._wiped = False

    def raw(self) -> bytes:
        """Short-lived bytes copy for library calls (the copy itself cannot be wiped)."""
        if self._wiped:
            raise ShareFormatError("secret key already wiped")
        return bytes(self._buf)

    @property
    def wiped(self) -> bool:
        return self._wiped

    def buffer_is_zero(self) -> bool:
        return not any(self._buf)

    def wipe(self) -> None:
        for index in range(len(self._buf)):
            self._buf[index] = 0
        self._wiped = True

    def __enter__(self) -> "SecretKey":
        return self

    def __exit__(self, *exc: object) -> None:
        self.wipe()

    def __repr__(self) -> str:
        return "SecretKey(<redacted>)"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("SecretKey must not be pickled")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SecretKey):
            return NotImplemented
        return hmac.compare_digest(bytes(self._buf), bytes(other._buf))

    __hash__ = None  # type: ignore[assignment]


def new_image_key() -> SecretKey:
    """Fresh 256-bit image key from the OS CSPRNG (production path)."""
    return SecretKey(secrets.token_bytes(IMAGE_KEY_BYTES))


def new_nonce() -> bytes:
    return secrets.token_bytes(NONCE_BYTES)


def build_info(profile_id: str, nonce: bytes, inst_id_b: str, key_id_b: str,
               inst_id_a: str, signing_key_id_a: str) -> bytes:
    """§8.4.1 (4): HPKE info binds profile, object nonce, recipient and sender."""
    require_profile(profile_id)
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
        raise ShareFormatError("nonce_B must be 16 bytes")
    ids = [require_id(v, w) for v, w in ((inst_id_b, "inst_id_B"), (key_id_b, "key_id_B"),
                                          (inst_id_a, "inst_id_A"), (signing_key_id_a, "signing_key_id_A"))]
    return lp_join(profile_id.encode("ascii"), nonce, *(i.encode("ascii") for i in ids))


def wrap_image_key(image_key: SecretKey, recipient: X25519PublicKey, info: bytes) -> bytes:
    if not isinstance(recipient, X25519PublicKey):
        raise ShareFormatError("recipient key must be an X25519 public key")
    wrap = SUITE.encrypt(image_key.raw(), recipient, info)
    if len(wrap) != WRAP_BYTES:  # pragma: no cover - fixed by the suite
        raise ShareFormatError("unexpected HPKE output length")
    return wrap


def unwrap_image_key(wrap: bytes, recipient: X25519PrivateKey, info: bytes) -> SecretKey:
    """Internal helper (returns the key to the service flow, never to callers of the service API)."""
    if not isinstance(wrap, bytes) or len(wrap) != WRAP_BYTES:
        raise UnwrapError("malformed image-key wrap")
    if not isinstance(recipient, X25519PrivateKey):
        raise ShareFormatError("recipient key must be an X25519 private key")
    try:
        material = SUITE.decrypt(wrap, recipient, info)
    except InvalidTag:
        raise UnwrapError("image key could not be unwrapped") from None
    if len(material) != IMAGE_KEY_BYTES:
        raise UnwrapError("unwrapped key has the wrong length")
    return SecretKey(material)


def derive_subkeys(image_key: SecretKey, nonce: bytes, profile_id: str) -> dict[str, SecretKey]:
    """§8.4.1 (5): HKDF-SHA256(ikm=k_img_B, salt=nonce_B, info=LP(profile_id)||LP(label))."""
    require_profile(profile_id)
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
        raise ShareFormatError("nonce_B must be 16 bytes")
    ikm = image_key.raw()
    out: dict[str, SecretKey] = {}
    for label in SUBKEY_LABELS:
        info = lp(profile_id.encode("ascii")) + lp(label.encode("ascii"))
        out[label] = SecretKey(HKDF(algorithm=hashes.SHA256(), length=SUBKEY_BYTES, salt=nonce, info=info).derive(ikm))
    return out


def wipe_all(*keys: SecretKey | dict[str, SecretKey] | None) -> None:
    for item in keys:
        if item is None:
            continue
        for key in (item.values() if isinstance(item, dict) else (item,)):
            key.wipe()


class TestFixtureStream:
    """PUBLIC, PREDICTABLE byte stream for reproducible research runs only (§8.8).

    SHA-256 in counter mode over a labelled seed. Anything derived from it is a
    public test fixture: results never claim security against an attacker who
    knows the label.
    """

    __test__ = False  # not a pytest class
    PREFIX = b"soteria-share/test-fixture/v2/"

    def __init__(self, label: str):
        if not isinstance(label, str) or not label:
            raise ShareFormatError("fixture label must be a non-empty string")
        self._seed = self.PREFIX + label.encode("utf-8")
        self._counter = 0

    def read(self, size: int) -> bytes:
        out = bytearray()
        while len(out) < size:
            out += hashlib.sha256(self._seed + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
        return bytes(out[:size])

    def image_key(self) -> SecretKey:
        return SecretKey(self.read(IMAGE_KEY_BYTES))

    def nonce(self) -> bytes:
        return self.read(NONCE_BYTES)


def fixture_x25519(label: str) -> X25519PrivateKey:
    """Deterministic, PUBLIC test recipient key pair (never for real use)."""
    return X25519PrivateKey.from_private_bytes(TestFixtureStream("x25519/" + label).read(32))
