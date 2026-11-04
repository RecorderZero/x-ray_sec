"""Signed model/sampler manifest (§8.4 manifest, opinion N3).

The container carries only ``manifest_digest``; the receiver must obtain the
full manifest, check ``sha256(payload) == manifest_digest``, verify A's Ed25519
signature, and confirm every field equals the profile's expectation before any
model is loaded. No local defaults are used.

    payload = LP("soteria-share/manifest/v1") || LP(profile_id) || LP(checkpoint_sha256[32])
              || LP(sampler) || u32 steps || u32 n || u32 timesteps[n] || LP(backward)
              || LP(generation) || u32 image_size || LP(x0_preprocess)
    blob    = LP(payload) || sig(64)     sig = Ed25519_A(LP(SIG_DOMAIN) || payload)
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .encoding import PROFILE_M1_PRIVATE_RANGE_V2, Reader, decode_profile, lp
from .errors import ManifestError, ShareFormatError


PAYLOAD_DOMAIN = b"soteria-share/manifest/v1"
SIG_DOMAIN = b"soteria-share/v2/manifest-sig"
MAX_MANIFEST_BYTES = 64 * 1024
CHECKPOINT_SHA256 = bytes.fromhex("3f61b80a2b2df4f5f1780fe9a7ce639f6a94ddb5057e5f9e88dc300de4786b14")


def deployment_timesteps(n: int = 100, top: int = 499) -> tuple[int, ...]:
    """Same boundary schedule as scripts/respaced_ddim_sweep.schedule(n) (deployment n=100)."""
    return tuple(int(t) for t in np.unique(np.rint(np.linspace(0, top, n + 1)).astype(int)))


@dataclass(frozen=True)
class Manifest:
    profile_id: str
    checkpoint_sha256: bytes
    sampler: str
    steps: int
    timesteps: tuple[int, ...]
    backward: str
    generation: str
    image_size: int
    x0_preprocess: str

    def payload(self) -> bytes:
        if len(self.checkpoint_sha256) != 32:
            raise ShareFormatError("checkpoint digest must be 32 bytes")
        return (lp(PAYLOAD_DOMAIN) + lp(self.profile_id.encode("ascii")) + lp(self.checkpoint_sha256)
                + lp(self.sampler.encode("ascii")) + struct.pack(">II", self.steps, len(self.timesteps))
                + struct.pack(f">{len(self.timesteps)}I", *self.timesteps)
                + lp(self.backward.encode("ascii")) + lp(self.generation.encode("ascii"))
                + struct.pack(">I", self.image_size) + lp(self.x0_preprocess.encode("ascii")))

    def digest(self) -> bytes:
        return hashlib.sha256(self.payload()).digest()


# The only manifest accepted for the only allowed profile (§8.4, rq4_formal_dphd operating point).
PROFILE_MANIFESTS = {
    PROFILE_M1_PRIVATE_RANGE_V2: Manifest(
        profile_id=PROFILE_M1_PRIVATE_RANGE_V2,
        checkpoint_sha256=CHECKPOINT_SHA256,
        sampler="DPM-Solver++(2M)",
        steps=100,
        timesteps=deployment_timesteps(100),
        backward="null",
        generation="unconditional",
        image_size=256,
        x0_preprocess="equalizeHist|resize256-INTER_AREA|jpeg-q100-roundtrip|normalize",
    ),
}


def sign_manifest(manifest: Manifest, signer: Ed25519PrivateKey) -> bytes:
    payload = manifest.payload()
    return lp(payload) + signer.sign(lp(SIG_DOMAIN) + payload)


def _parse(payload: bytes) -> Manifest:
    reader = Reader(payload)
    if reader.lp(64) != PAYLOAD_DOMAIN:
        raise ShareFormatError("not a manifest payload")
    profile_id = decode_profile(reader.lp(64))
    checkpoint = reader.lp(32, exact=32)
    sampler = reader.lp(64).decode("ascii")
    steps, count = struct.unpack(">II", reader.fixed(8))
    if count > 10_000:
        raise ShareFormatError("too many timesteps")
    timesteps = struct.unpack(f">{count}I", reader.fixed(4 * count))
    backward = reader.lp(64).decode("ascii")
    generation = reader.lp(64).decode("ascii")
    (image_size,) = struct.unpack(">I", reader.fixed(4))
    x0_preprocess = reader.lp(256).decode("ascii")
    reader.done()
    return Manifest(profile_id, checkpoint, sampler, steps, tuple(timesteps), backward, generation, image_size, x0_preprocess)


def verify_manifest(blob: bytes, expected_digest: bytes, signer: Ed25519PublicKey, profile_id: str) -> Manifest:
    """Digest -> signature -> parse -> exact equality with the profile's manifest."""
    try:
        if not isinstance(blob, (bytes, bytearray)) or len(blob) > MAX_MANIFEST_BYTES:
            raise ShareFormatError("manifest missing or too large")
        reader = Reader(blob)
        payload = reader.lp(MAX_MANIFEST_BYTES)
        sig = reader.fixed(64)
        reader.done()
    except ShareFormatError as exc:
        raise ManifestError("malformed manifest blob") from exc
    if hashlib.sha256(payload).digest() != expected_digest:
        raise ManifestError("manifest digest does not match the container")
    try:
        signer.verify(sig, lp(SIG_DOMAIN) + payload)
    except InvalidSignature:
        raise ManifestError("manifest signature does not verify") from None
    try:
        manifest = _parse(payload)
    except (ShareFormatError, UnicodeDecodeError, struct.error) as exc:
        raise ManifestError("manifest payload is malformed") from exc
    expected = PROFILE_MANIFESTS.get(profile_id)
    if expected is None or manifest != expected:
        raise ManifestError("manifest is outside the allowed profile")
    return manifest
