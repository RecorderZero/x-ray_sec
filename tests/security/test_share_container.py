"""S4: v2 container (P2, V4 container layers, V9, V11, F4 fixed test vectors)."""

import hashlib
import struct
import zlib

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from scripts.m1_storage import encode_range_preserving_png
from src.share.container import (
    META_NONCE, Header, _meta_aad, assemble, compute_tag, decode_anon_png, encode_anon_png, open_metadata,
    parse, seal_metadata, sign, validate_png, verify_signature, verify_tag,
)
from src.share.encoding import PREAMBLE, PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE, lp, lp_join
from src.share.errors import MacError, MetadataError, ShareFormatError, SignatureError
from src.share.keys import TestFixtureStream, derive_subkeys


def _signer(label: str = "A") -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(TestFixtureStream("ed25519/" + label).read(32))


def _header(**override) -> Header:
    stream = TestFixtureStream("container/header")
    fields = dict(profile_id=PROFILE, manifest_digest=hashlib.sha256(b"manifest").digest(),
                  inst_id_a="hospital-A", signing_key_id_a="A-ed25519-2026a",
                  inst_id_b="hospital-B", key_id_b="B-x25519-2026a",
                  nonce=stream.read(16), wrap=stream.read(80))
    fields.update(override)
    return Header(**fields)


def _subkeys():
    stream = TestFixtureStream("container/keys")
    return derive_subkeys(stream.image_key(), stream.nonce(), PROFILE)


def _image() -> np.ndarray:
    rng = np.random.default_rng(1911)
    return (np.linspace(-0.8, 0.9, 65536, dtype=np.float32).reshape(256, 256)
            + rng.normal(0, 0.05, (256, 256)).astype(np.float32))


def _build(png: bytes | None = None, norm: float = 123.4):
    sk = _subkeys()
    h = _header().encode()
    if png is None:
        png, lo, hi = encode_anon_png(_image())
    else:
        lo, hi = np.float32(-1), np.float32(1)
    meta = seal_metadata(sk["private-metadata-aead"], h, norm, lo, hi)
    tag = compute_tag(sk["mac"], h, meta, png)
    sig = sign(_signer(), h, meta, png, tag)
    return assemble(h, meta, png, tag, sig), sk, (h, meta, png, tag, sig)


# ------------------------------------------------------------------ F4 fixed vector (normal encoding)
def test_fixed_vector_header_meta_tag_sig() -> None:
    sk = _subkeys()
    h = _header().encode()
    assert h[:4] == struct.pack(">I", len(PROFILE)) and h[4:4 + len(PROFILE)] == PROFILE.encode()
    meta = seal_metadata(sk["private-metadata-aead"], h, 248.261, -1.25, 0.75)
    dummy_png = b"\x89PNG-fixed-vector"
    tag = compute_tag(sk["mac"], h, meta, dummy_png)
    sig = sign(_signer(), h, meta, dummy_png, tag)
    digest = hashlib.sha256(assemble(h, meta, dummy_png, tag, sig)).hexdigest()
    assert len(h) == 242 and len(meta) == 32  # 8 LP prefixes (32) + 33+32+10+15+10+14+16+80
    assert digest == FROZEN_CONTAINER_SHA256, digest


# Self-generated known answer (frozen 2026-11-04 at S4) guarding against silent format drift;
# test_header_bytes_match_spec_by_hand checks the layout independently of the encoder.
FROZEN_CONTAINER_SHA256 = "ef697b0e83eb66bf53078514ce522d2e96e82866fffaf8746819330294c948a1"


def test_header_bytes_match_spec_by_hand() -> None:
    hd = _header()
    fields = [PROFILE.encode(), hd.manifest_digest, b"hospital-A", b"A-ed25519-2026a",
              b"hospital-B", b"B-x25519-2026a", hd.nonce, hd.wrap]
    manual = b"".join(len(f).to_bytes(4, "big") + f for f in fields)
    assert hd.encode() == manual
    assert PREAMBLE == b"SOTERIA\x00\x00\x02"


# ------------------------------------------------------------------ P2 round trip
def test_p2_container_bytes_round_trip() -> None:
    blob, _, parts = _build()
    parsed = parse(blob)
    assert (parsed.h_bytes, parsed.meta_ct, parsed.png, parsed.tag, parsed.sig) == parts
    assert parsed.header == _header()
    assert assemble(parsed.h_bytes, parsed.meta_ct, parsed.png, parsed.tag, parsed.sig) == blob


# ------------------------------------------------------------------ V11 / F4 negative framing vectors
def _bad_framings(blob: bytes) -> dict[str, bytes]:
    h_len = struct.unpack(">I", blob[10:14])[0]
    return {
        "empty": b"",
        "truncated_preamble": blob[:5],
        "truncated_mid": blob[: len(blob) // 2],
        "truncated_sig": blob[:-1],
        "trailing_bytes": blob + b"\x00",
        "bad_magic": b"SOTERIX\x00" + blob[8:],
        "unknown_version": blob[:8] + struct.pack(">H", 3) + blob[10:],
        "h_length_overflow": blob[:10] + struct.pack(">I", 0xFFFFFFF0) + blob[14:],
        "h_over_cap": blob[:10] + struct.pack(">I", 4097) + blob[14:],
        "meta_wrong_len": blob[:14 + h_len] + struct.pack(">I", 31) + blob[18 + h_len:],
        "over_total_cap": blob + bytes(2_000_000),
    }


@pytest.mark.parametrize("case", list(_bad_framings(b"\x00" * 64)))
def test_parse_rejects_bad_framing(case: str) -> None:
    blob, _, _ = _build()
    with pytest.raises(ShareFormatError):
        parse(_bad_framings(blob)[case])


@pytest.mark.parametrize("override", [
    dict(profile_id="soteria-share/other/v2"), dict(inst_id_a="bad id"), dict(nonce=bytes(15)),
    dict(wrap=bytes(79)), dict(manifest_digest=bytes(31)),
])
def test_header_encode_rejects_illegal_fields(override) -> None:
    with pytest.raises(ShareFormatError):
        _header(**override).encode()


def test_header_decode_rejects_extra_or_reordered_fields() -> None:
    h = _header().encode()
    with pytest.raises(ShareFormatError):
        Header.decode(h + lp(b"patient00241"))   # extra field (e.g. a patient id)
    reader_fields = []
    from src.share.encoding import Reader
    r = Reader(h)
    for _ in range(8):
        reader_fields.append(r.lp(4096))
    swapped = lp_join(reader_fields[1], reader_fields[0], *reader_fields[2:])
    with pytest.raises(ShareFormatError):
        Header.decode(swapped)


# ------------------------------------------------------------------ PNG checks (§8.4.1 (8), V9)
def _png(width=256, height=256, depth=8, color=0, interlace=0, raw=None, extra_chunks=(), idat_split=1) -> bytes:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
    if raw is None:
        raw = b"".join(b"\x00" + bytes(range(256)) for _ in range(256))
    comp = zlib.compress(raw, 9)
    step = max(1, len(comp) // idat_split)
    idats = b"".join(chunk(b"IDAT", comp[i:i + step]) for i in range(0, len(comp), step))
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, interlace))
    return b"\x89PNG\r\n\x1a\n" + ihdr + b"".join(chunk(t, d) for t, d in extra_chunks) + idats + chunk(b"IEND", b"")


def test_handmade_valid_png_passes_and_split_idat_ok() -> None:
    validate_png(_png())
    validate_png(_png(idat_split=4))


@pytest.mark.parametrize("bad", [
    "text_chunk_v1_public_range", "tEXt", "size128", "rgb", "depth16", "interlaced",
    "bad_crc", "after_iend", "no_idat", "inflate_too_big", "inflate_too_small", "not_png",
])
def test_png_validator_rejects(bad: str) -> None:
    image = _image()
    cases = {
        "text_chunk_v1_public_range": lambda: encode_range_preserving_png(image)[0],
        "tEXt": lambda: _png(extra_chunks=[(b"tEXt", b"k\x00v")]),
        "size128": lambda: _png(width=128, height=128, raw=b"".join(b"\x00" + bytes(128) for _ in range(128))),
        "rgb": lambda: _png(color=2),
        "depth16": lambda: _png(depth=16),
        "interlaced": lambda: _png(interlace=1),
        "bad_crc": lambda: (lambda p: p[:-5] + bytes([p[-5] ^ 1]) + p[-4:])(_png()),
        "after_iend": lambda: _png() + b"junk",
        "no_idat": lambda: b"\x89PNG\r\n\x1a\n" + _png()[8:33] + _png()[-12:],
        "inflate_too_big": lambda: _png(raw=bytes(10_000_000)),
        "inflate_too_small": lambda: _png(raw=bytes(100)),
        "not_png": lambda: b"GIF89a" + bytes(100),
    }
    with pytest.raises(ShareFormatError):
        validate_png(cases[bad]())


def test_anon_png_round_trip_and_public_view_has_no_range() -> None:
    image = _image()
    png, lo, hi = encode_anon_png(image)
    assert b"tEXt" not in png and b"x-range" not in png and b"iTXt" not in png and b"zTXt" not in png
    back = decode_anon_png(png, lo, hi)
    assert back.dtype == np.float32 and back.shape == (256, 256)
    assert float(np.abs(back - image).max()) <= float(hi - lo) / 510 + 1e-6


def test_constant_image_contract() -> None:
    png, lo, hi = encode_anon_png(np.full((256, 256), 0.25, dtype=np.float32))
    assert lo == hi
    assert np.all(decode_anon_png(png, lo, hi) == np.float32(0.25))


# ------------------------------------------------------------------ metadata (§8.4.1 (6))
@pytest.mark.parametrize("norm,lo,hi", [(0.0, 0, 1), (-1.0, 0, 1), (float("nan"), 0, 1), (float("inf"), 0, 1),
                                        (1.0, float("nan"), 1), (1.0, 0, float("inf")), (1.0, 1.0, 0.5)])
def test_seal_metadata_rejects_illegal_values(norm, lo, hi) -> None:
    with pytest.raises(ShareFormatError):
        seal_metadata(_subkeys()["private-metadata-aead"], _header().encode(), norm, lo, hi)


def test_metadata_round_trip_and_aad_binding() -> None:
    key = _subkeys()["private-metadata-aead"]
    h = _header().encode()
    ct = seal_metadata(key, h, 248.261, -1.5, 2.5)
    norm, lo, hi = open_metadata(key, h, ct)
    assert norm == 248.261 and lo == np.float32(-1.5) and hi == np.float32(2.5)
    with pytest.raises(MetadataError):
        open_metadata(key, _header(key_id_b="B-x25519-other").encode(), ct)   # header (AAD) changed
    with pytest.raises(MetadataError):
        open_metadata(key, h, ct[:-1] + bytes([ct[-1] ^ 1]))                  # ciphertext tampered


def test_open_metadata_rejects_authenticated_but_illegal_plaintext() -> None:
    key = _subkeys()["private-metadata-aead"]
    h = _header().encode()
    illegal = struct.pack("<d", 1.0) + np.float32(2).astype("<f4").tobytes() + np.float32(1).astype("<f4").tobytes()
    ct = ChaCha20Poly1305(key.raw()).encrypt(META_NONCE, illegal, _meta_aad(h))   # hi < lo
    with pytest.raises(MetadataError):
        open_metadata(key, h, ct)


# ------------------------------------------------------------------ V4 container layers (F2)
def test_v4_signature_layer_catches_any_signed_byte_flip() -> None:
    blob, _, (h, meta, png, tag, sig) = _build()
    pub = _signer().public_key()
    verify_signature(pub, h, meta, png, tag, sig)
    signed_region = len(blob) - 64
    for pos in np.linspace(0, signed_region - 1, 25).astype(int):
        flipped = blob[:pos] + bytes([blob[pos] ^ 0x01]) + blob[pos + 1:]
        try:
            p = parse(flipped)
        except ShareFormatError:
            continue  # framing/schema caught it first
        with pytest.raises(SignatureError):
            verify_signature(pub, p.h_bytes, p.meta_ct, p.png, p.tag, p.sig)
    with pytest.raises(SignatureError):
        verify_signature(_signer("not-A").public_key(), h, meta, png, tag, sig)


def test_v4_mac_layer_reached_after_controlled_resign() -> None:
    _, sk, (h, meta, png, tag, _) = _build()
    png2 = encode_anon_png(np.flipud(_image()).copy())[0]   # scaling alone would quantize to the same PNG
    sig2 = sign(_signer(), h, meta, png2, tag)                       # test signer re-signs, tag left stale
    verify_signature(_signer().public_key(), h, meta, png2, tag, sig2)
    assert png2 != png
    with pytest.raises(MacError):
        verify_tag(sk["mac"], h, meta, png2, tag)


def test_v4_metadata_aead_layer_reached_after_controlled_mac_and_resign() -> None:
    _, sk, (h, meta, png, _, _) = _build()
    meta2 = meta[:3] + bytes([meta[3] ^ 0x80]) + meta[4:]
    tag2 = compute_tag(sk["mac"], h, meta2, png)                     # fixture mac_key recomputes the tag
    sig2 = sign(_signer(), h, meta2, png, tag2)
    verify_signature(_signer().public_key(), h, meta2, png, tag2, sig2)
    verify_tag(sk["mac"], h, meta2, png, tag2)
    with pytest.raises(MetadataError):
        open_metadata(sk["private-metadata-aead"], h, meta2)
