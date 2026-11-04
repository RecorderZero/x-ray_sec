"""v2 outer container, private-range PNG checks, metadata AEAD, MAC and signature.

Byte layout is §8.4.1 exactly::

    container = PREAMBLE || LP(H) || LP(meta_ct) || LP(PNG) || tag(32) || sig(64)

Parsing checks every length against its cap and rejects trailing bytes before
any field is decoded; PNG structure (signature, chunk order/types, CRCs, IHDR
geometry, bounded inflate size) is validated before the image is decoded.
"""

from __future__ import annotations

import hmac
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import hmac as crypto_hmac
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from scripts.m1_storage import decode_private_range_png, encode_private_range_png

from .encoding import (
    DIGEST_BYTES, MAX_H_BYTES, MAX_PNG_BYTES, META_CT_BYTES, PREAMBLE, PREAMBLE_MAGIC, SIG_BYTES, TAG_BYTES,
    Reader, decode_id, decode_profile, lp, lp_join, require_id, require_profile,
)
from .errors import MacError, MetadataError, ShareFormatError, SignatureError
from .keys import NONCE_BYTES, WRAP_BYTES, SecretKey


MAC_DOMAIN = b"soteria-share/v2/mac"
SIG_DOMAIN = b"soteria-share/v2/sig"
META_NONCE = bytes(12)
META_PT_BYTES = 16
IMAGE_SIDE = 256
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_RAW_BYTES = IMAGE_SIDE * (1 + IMAGE_SIDE)  # filter byte + 256 grayscale samples per row
MAX_CONTAINER_BYTES = len(PREAMBLE) + 4 + MAX_H_BYTES + 4 + META_CT_BYTES + 4 + MAX_PNG_BYTES + TAG_BYTES + SIG_BYTES


# --------------------------------------------------------------------------- header
@dataclass(frozen=True)
class Header:
    profile_id: str
    manifest_digest: bytes
    inst_id_a: str
    signing_key_id_a: str
    inst_id_b: str
    key_id_b: str
    nonce: bytes
    wrap: bytes

    def encode(self) -> bytes:
        require_profile(self.profile_id)
        for value, what in ((self.inst_id_a, "inst_id_A"), (self.signing_key_id_a, "signing_key_id_A"),
                            (self.inst_id_b, "inst_id_B"), (self.key_id_b, "key_id_B")):
            require_id(value, what)
        if len(self.manifest_digest) != DIGEST_BYTES or len(self.nonce) != NONCE_BYTES or len(self.wrap) != WRAP_BYTES:
            raise ShareFormatError("header binary field has the wrong length")
        encoded = lp_join(self.profile_id.encode("ascii"), self.manifest_digest,
                          self.inst_id_a.encode("ascii"), self.signing_key_id_a.encode("ascii"),
                          self.inst_id_b.encode("ascii"), self.key_id_b.encode("ascii"), self.nonce, self.wrap)
        if len(encoded) > MAX_H_BYTES:  # pragma: no cover - bounded by the field limits
            raise ShareFormatError("header too large")
        return encoded

    @classmethod
    def decode(cls, data: bytes) -> "Header":
        reader = Reader(data)
        header = cls(
            profile_id=decode_profile(reader.lp(64)),
            manifest_digest=reader.lp(DIGEST_BYTES, exact=DIGEST_BYTES),
            inst_id_a=decode_id(reader.lp(64), "inst_id_A"),
            signing_key_id_a=decode_id(reader.lp(64), "signing_key_id_A"),
            inst_id_b=decode_id(reader.lp(64), "inst_id_B"),
            key_id_b=decode_id(reader.lp(64), "key_id_B"),
            nonce=reader.lp(NONCE_BYTES, exact=NONCE_BYTES),
            wrap=reader.lp(WRAP_BYTES, exact=WRAP_BYTES),
        )
        reader.done()
        return header


# --------------------------------------------------------------------------- metadata
def _check_meta_values(norm: float, lo: np.float32, hi: np.float32, error: type[Exception]) -> None:
    if not np.isfinite(norm) or norm <= 0:
        raise error("norm must be positive and finite")
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
        raise error("lo/hi must be finite with hi >= lo")


def encode_meta_pt(norm: float, lo: float, hi: float) -> bytes:
    lo32, hi32 = np.float32(lo), np.float32(hi)
    _check_meta_values(float(norm), lo32, hi32, ShareFormatError)
    return struct.pack("<d", float(norm)) + lo32.astype("<f4").tobytes() + hi32.astype("<f4").tobytes()


def decode_meta_pt(pt: bytes) -> tuple[float, np.float32, np.float32]:
    if len(pt) != META_PT_BYTES:
        raise MetadataError("metadata plaintext has the wrong length")
    (norm,) = struct.unpack("<d", pt[:8])
    lo = np.frombuffer(pt[8:12], dtype="<f4")[0]
    hi = np.frombuffer(pt[12:16], dtype="<f4")[0]
    _check_meta_values(norm, lo, hi, MetadataError)
    return float(norm), np.float32(lo), np.float32(hi)


def _meta_aad(h_bytes: bytes) -> bytes:
    return PREAMBLE + lp(h_bytes)


def seal_metadata(meta_key: SecretKey, h_bytes: bytes, norm: float, lo: float, hi: float) -> bytes:
    """meta_key is single-use (fresh per image key), so the fixed zero nonce is safe (§8.4.1 (6))."""
    ct = ChaCha20Poly1305(meta_key.raw()).encrypt(META_NONCE, encode_meta_pt(norm, lo, hi), _meta_aad(h_bytes))
    assert len(ct) == META_CT_BYTES
    return ct


def open_metadata(meta_key: SecretKey, h_bytes: bytes, meta_ct: bytes) -> tuple[float, np.float32, np.float32]:
    if len(meta_ct) != META_CT_BYTES:
        raise MetadataError("metadata ciphertext has the wrong length")
    try:
        pt = ChaCha20Poly1305(meta_key.raw()).decrypt(META_NONCE, meta_ct, _meta_aad(h_bytes))
    except InvalidTag:
        raise MetadataError("metadata authentication failed") from None
    return decode_meta_pt(pt)


# --------------------------------------------------------------------------- PNG (§8.4.1 (8))
def validate_png(png: bytes) -> None:
    """Structure, CRC, geometry and bounded-inflate checks; raises ShareFormatError."""
    if len(png) > MAX_PNG_BYTES or not png.startswith(PNG_SIGNATURE):
        raise ShareFormatError("not a size-bounded PNG")
    pos, chunks, idat = len(PNG_SIGNATURE), [], bytearray()
    while pos < len(png):
        if pos + 8 > len(png):
            raise ShareFormatError("truncated PNG chunk header")
        (length,) = struct.unpack(">I", png[pos:pos + 4])
        ctype = png[pos + 4:pos + 8]
        end = pos + 12 + length
        if length > MAX_PNG_BYTES or end > len(png):
            raise ShareFormatError("PNG chunk runs past the payload")
        data = png[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", png[pos + 8 + length:end])
        if zlib.crc32(ctype + data) & 0xFFFFFFFF != crc:
            raise ShareFormatError("PNG chunk CRC mismatch")
        chunks.append(ctype)
        if ctype == b"IHDR":
            if len(chunks) != 1 or length != 13:
                raise ShareFormatError("IHDR must be the first 13-byte chunk")
            width, height, depth, color, comp, filt, interlace = struct.unpack(">IIBBBBB", data)
            if (width, height, depth, color, comp, filt, interlace) != (IMAGE_SIDE, IMAGE_SIDE, 8, 0, 0, 0, 0):
                raise ShareFormatError("PNG must be 256x256, 8-bit grayscale, non-interlaced")
        elif ctype == b"IDAT":
            if chunks[-2:-1] not in ([b"IHDR"], [b"IDAT"]):
                raise ShareFormatError("IDAT chunks must be contiguous after IHDR")
            idat += data
        elif ctype == b"IEND":
            if length != 0 or end != len(png):
                raise ShareFormatError("IEND must be empty and final")
        else:
            raise ShareFormatError(f"PNG chunk {ctype!r} is not allowed")
        pos = end
    if not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND" or b"IDAT" not in chunks:
        raise ShareFormatError("PNG must be IHDR, IDAT+, IEND")
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(bytes(idat), PNG_RAW_BYTES + 1)
    except zlib.error as exc:
        raise ShareFormatError("PNG image data does not inflate") from exc
    if len(raw) != PNG_RAW_BYTES or inflater.unconsumed_tail or not inflater.eof:
        raise ShareFormatError("PNG inflates to the wrong size")


def encode_anon_png(image: np.ndarray) -> tuple[bytes, np.float32, np.float32]:
    """Private-range quantization (no lo/hi chunks); returns PNG bytes and exact float32 lo/hi."""
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (IMAGE_SIDE, IMAGE_SIDE):
        raise ShareFormatError("anonymous image must be 256x256")
    png, record = encode_private_range_png(array)
    validate_png(png)
    return png, np.float32(array.min()), np.float32(array.max())


def _f32_bits_be(value: np.float32) -> str:
    return struct.pack(">f", float(value)).hex()


def decode_anon_png(png: bytes, lo: np.float32, hi: np.float32) -> np.ndarray:
    validate_png(png)
    return decode_private_range_png(png, _f32_bits_be(lo), _f32_bits_be(hi))


# --------------------------------------------------------------------------- MAC / signature
def mac_message(h_bytes: bytes, meta_ct: bytes, png: bytes) -> bytes:
    return lp(MAC_DOMAIN) + PREAMBLE + lp_join(h_bytes, meta_ct, png)


def compute_tag(mac_key: SecretKey, h_bytes: bytes, meta_ct: bytes, png: bytes) -> bytes:
    signer = crypto_hmac.HMAC(mac_key.raw(), hashes.SHA256())
    signer.update(mac_message(h_bytes, meta_ct, png))
    return signer.finalize()


def verify_tag(mac_key: SecretKey, h_bytes: bytes, meta_ct: bytes, png: bytes, tag: bytes) -> None:
    if not hmac.compare_digest(compute_tag(mac_key, h_bytes, meta_ct, png), tag):
        raise MacError("container MAC mismatch")


def sig_message(h_bytes: bytes, meta_ct: bytes, png: bytes, tag: bytes) -> bytes:
    return lp(SIG_DOMAIN) + PREAMBLE + lp_join(h_bytes, meta_ct, png) + tag


def sign(signing_key: Ed25519PrivateKey, h_bytes: bytes, meta_ct: bytes, png: bytes, tag: bytes) -> bytes:
    return signing_key.sign(sig_message(h_bytes, meta_ct, png, tag))


def verify_signature(public_key: Ed25519PublicKey, h_bytes: bytes, meta_ct: bytes, png: bytes, tag: bytes, sig: bytes) -> None:
    try:
        public_key.verify(sig, sig_message(h_bytes, meta_ct, png, tag))
    except InvalidSignature:
        raise SignatureError("sender signature did not verify") from None


# --------------------------------------------------------------------------- framing
@dataclass(frozen=True)
class Parsed:
    h_bytes: bytes
    header: Header
    meta_ct: bytes
    png: bytes
    tag: bytes
    sig: bytes


def assemble(h_bytes: bytes, meta_ct: bytes, png: bytes, tag: bytes, sig: bytes) -> bytes:
    if len(tag) != TAG_BYTES or len(sig) != SIG_BYTES or len(meta_ct) != META_CT_BYTES:
        raise ShareFormatError("tag/sig/meta_ct have fixed lengths")
    return PREAMBLE + lp_join(h_bytes, meta_ct, png) + tag + sig


def parse(container: bytes) -> Parsed:
    """Bounded framing + schema parse; no PNG decode, no crypto."""
    if not isinstance(container, (bytes, bytearray)) or len(container) > MAX_CONTAINER_BYTES:
        raise ShareFormatError("container missing or over the size limit")
    reader = Reader(container)
    magic = reader.fixed(len(PREAMBLE_MAGIC))
    if magic != PREAMBLE_MAGIC:
        raise ShareFormatError("not a SOTERIA share container")
    (version,) = struct.unpack(">H", reader.fixed(2))
    if version != 2:
        raise ShareFormatError(f"unsupported container version {version}")
    h_bytes = reader.lp(MAX_H_BYTES)
    meta_ct = reader.lp(META_CT_BYTES, exact=META_CT_BYTES)
    png = reader.lp(MAX_PNG_BYTES)
    tag = reader.fixed(TAG_BYTES)
    sig = reader.fixed(SIG_BYTES)
    reader.done()
    return Parsed(h_bytes, Header.decode(h_bytes), meta_ct, png, tag, sig)
