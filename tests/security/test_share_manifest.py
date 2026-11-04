"""S6: signed manifest (digest + signature + exact profile match before any inference)."""

import dataclasses
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.respaced_ddim_sweep import schedule
from src.share.encoding import PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE
from src.share.errors import ManifestError
from src.share.keys import TestFixtureStream
from src.share.manifest import (
    CHECKPOINT_SHA256, PROFILE_MANIFESTS, deployment_timesteps, sign_manifest, verify_manifest,
)


MANIFEST = PROFILE_MANIFESTS[PROFILE]


def _signer(label: str = "A") -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(TestFixtureStream("ed25519/" + label).read(32))


def test_profile_manifest_matches_formal_operating_point() -> None:
    assert deployment_timesteps(100) == tuple(int(t) for t in schedule(100))
    assert len(MANIFEST.timesteps) == 101 and MANIFEST.timesteps[0] == 0 and MANIFEST.timesteps[-1] == 499
    assert CHECKPOINT_SHA256.hex() == "3f61b80a2b2df4f5f1780fe9a7ce639f6a94ddb5057e5f9e88dc300de4786b14"


def test_valid_manifest_verifies() -> None:
    blob = sign_manifest(MANIFEST, _signer())
    assert verify_manifest(blob, MANIFEST.digest(), _signer().public_key(), PROFILE) == MANIFEST


@pytest.mark.parametrize("change", [
    dict(steps=50), dict(sampler="DDIM"), dict(backward="y=0"), dict(timesteps=deployment_timesteps(50)),
    dict(checkpoint_sha256=hashlib.sha256(b"other").digest()), dict(generation="conditional"), dict(image_size=512),
])
def test_signed_but_off_profile_manifest_is_rejected(change) -> None:
    other = dataclasses.replace(MANIFEST, **change)
    blob = sign_manifest(other, _signer())
    with pytest.raises(ManifestError):
        verify_manifest(blob, other.digest(), _signer().public_key(), PROFILE)


def test_digest_mismatch_wrong_signer_and_tamper() -> None:
    blob = sign_manifest(MANIFEST, _signer())
    with pytest.raises(ManifestError):
        verify_manifest(blob, bytes(32), _signer().public_key(), PROFILE)
    with pytest.raises(ManifestError):
        verify_manifest(blob, MANIFEST.digest(), _signer("not-A").public_key(), PROFILE)
    tampered = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(ManifestError):
        verify_manifest(tampered, MANIFEST.digest(), _signer().public_key(), PROFILE)
    with pytest.raises(ManifestError):
        verify_manifest(blob + b"\x00", MANIFEST.digest(), _signer().public_key(), PROFILE)
    with pytest.raises(ManifestError):
        verify_manifest(b"", MANIFEST.digest(), _signer().public_key(), PROFILE)
