"""Spec-level reference for P3 (AF-054): computes everything from x0 independently.

Nothing here reads values the service wrote, except where a function says so
explicitly (``service_q`` reads the public PNG; ``service_metadata`` opens the
service's meta_ct with an independently derived key so it can be COMPARED with the
reference's own norm/lo/hi). Own HKDF, float64 numpy FWHT, rint quantization.
Identity stub pipelines are assumed (z = x0, anon = protected latent).
"""

from __future__ import annotations

import io
import struct

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from PIL import Image

from src.crypto.prng import inverse_permutation, permutation_from_key, sign_vector_from_key
from src.share.container import parse
from src.share.encoding import PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE
from src.share.keys import TestFixtureStream


FIXED_NORM = 248.261
PREAMBLE = b"SOTERIA\x00\x00\x02"


def lp(b: bytes) -> bytes:
    return len(b).to_bytes(4, "big") + b


def subkey(k: bytes, nonce: bytes, label: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=nonce, info=lp(PROFILE.encode()) + lp(label.encode())).derive(k)


def fwht(v: np.ndarray) -> np.ndarray:
    out = v.astype(np.float64).copy()
    h = 1
    while h < out.size:
        out = out.reshape(-1, 2, h)
        a, b = out[:, 0, :].copy(), out[:, 1, :].copy()
        out[:, 0, :], out[:, 1, :] = a + b, a - b
        out = out.reshape(-1)
        h *= 2
    return out / np.sqrt(out.size)


def streams(label: str):
    stream = TestFixtureStream(label)
    k, nonce = stream.read(32), stream.read(16)
    pre = sign_vector_from_key(subkey(k, nonce, "round/0/sign-pre"), 65536).astype(np.float64)
    post = sign_vector_from_key(subkey(k, nonce, "round/0/sign-post"), 65536).astype(np.float64)
    perm = permutation_from_key(subkey(k, nonce, "round/0/perm"), 65536)
    return k, nonce, pre, perm, post


def quantize(anon: np.ndarray) -> tuple[np.ndarray, np.float32, np.float32]:
    """§8.4.1 (8) in float32: q = clip(rint((x-lo)/(hi-lo)*255), 0, 255); constant image -> zeros."""
    anon = np.asarray(anon, dtype=np.float32)
    lo, hi = np.float32(anon.min()), np.float32(anon.max())
    span = np.float32(hi - lo)
    if span > 0:
        q = np.clip(np.rint((anon - lo) / span * np.float32(255)), 0, 255).astype(np.uint8)
    else:
        q = np.zeros(anon.shape, dtype=np.uint8)
    return q, lo, hi


def encode(x0: np.ndarray, label: str) -> dict:
    """Independent sender: expected norm, lo, hi and q, all from x0."""
    _, _, pre, perm, post = streams(label)
    z = np.asarray(x0, dtype=np.float32).reshape(-1).astype(np.float64)
    norm = float(np.linalg.norm(z))
    c = (fwht(z * (FIXED_NORM / norm) * pre)[perm] * post).astype(np.float32).reshape(256, 256)
    q, lo, hi = quantize(c)
    return {"norm": norm, "lo": lo, "hi": hi, "q": q}


def decode(q: np.ndarray, norm: float, lo: np.float32, hi: np.float32, label: str) -> np.ndarray:
    """Independent receiver half: dequantize -> DPHD^-1 -> restore norm."""
    _, _, pre, perm, post = streams(label)
    deq = (q.astype(np.float32) / np.float32(255) * np.float32(hi - lo) + lo).reshape(-1).astype(np.float64)
    u = fwht((deq * post)[inverse_permutation(perm)]) * pre
    return (u * (norm / FIXED_NORM)).astype(np.float32).reshape(256, 256)


def service_q(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(parse(blob).png)) as im:
        return np.asarray(im, dtype=np.uint8)


def service_metadata(blob: bytes, label: str) -> tuple[float, np.float32, np.float32]:
    """Open the service's meta_ct with an independently derived key (for comparison only)."""
    k, nonce, *_ = streams(label)
    p = parse(blob)
    pt = ChaCha20Poly1305(subkey(k, nonce, "private-metadata-aead")).decrypt(bytes(12), p.meta_ct, PREAMBLE + lp(p.h_bytes))
    norm = struct.unpack("<d", pt[:8])[0]
    return norm, np.frombuffer(pt[8:12], "<f4")[0], np.frombuffer(pt[12:16], "<f4")[0]


# Fixtures that were looked at before the encode-half criterion was written (2026-11-04).
THRESHOLD_SETTING_SEEDS = (1911, 7, 42, 3, 5, 11, 13, 17)
# Held-out fixtures used only to validate that post-hoc criterion (AF-054).
HELD_OUT_SEEDS = tuple(range(1000, 1032))
