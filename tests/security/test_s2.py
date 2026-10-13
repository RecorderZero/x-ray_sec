"""D6.7 authenticated S2 integration and ordering tests."""

import json

import pytest
import torch

from src.crypto.errors import AuthenticationError
from src.crypto.errors import NonceReuseError
from src.crypto.nonce import NonceRegistry
from src.crypto.s2 import open_anonymous_image, open_latent, seal_anonymous_image, seal_latent
from src.crypto.sot import transform


KEY = bytes(range(32))
NONCE = bytes(range(16))
AAD = {
    "study_instance_uid": "1.2.3",
    "sop_instance_uid": "1.2.3.4",
    "model_sha256": "ab" * 32,
    "sampler_version": "legacy-ddim-anonymization/v1",
}


@pytest.mark.parametrize("rounds", [1, 2, 4])
@pytest.mark.parametrize("use_permutation", [False, True])
def test_m2_authenticated_roundtrip(tmp_path, rounds: int, use_permutation: bool) -> None:
    original = torch.randn(1, 8, 8, generator=torch.Generator().manual_seed(1911))
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        sealed = seal_latent(
            original, KEY, NONCE, rounds=rounds, use_permutation=use_permutation,
            aad=AAD, registry=registry,
        )
    restored = open_latent(sealed, KEY)
    assert restored.shape == original.shape and restored.dtype == original.dtype
    assert torch.max(torch.abs(restored - original)).item() <= 2e-6


def test_m1_authenticates_before_decode_inversion_and_inverse(tmp_path) -> None:
    latent = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(1911))
    protected = transform(latent, KEY, NONCE, rounds=2, use_permutation=True)
    image = protected.clone()  # Identity image/inversion fixture isolates control flow.
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        sealed = seal_anonymous_image(
            image, KEY, NONCE, rounds=2, use_permutation=True,
            latent_shape=tuple(latent.shape), latent_dtype="float32", aad=AAD,
            registry=registry,
        )
    calls = []
    restored = open_anonymous_image(sealed, KEY, lambda value: calls.append(value) or value)
    assert len(calls) == 1
    # M1 is quantized, so only a bounded approximate round-trip is expected.
    assert torch.max(torch.abs(restored - latent)).item() < 0.03

    document = json.loads(sealed)
    document["tag"] = "00" * 32
    tampered = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    calls.clear()
    with pytest.raises(AuthenticationError, match="authentication failed"):
        open_anonymous_image(tampered, KEY, lambda value: calls.append(value) or value)
    assert calls == []


def test_wrong_key_and_nonce_fail_closed(tmp_path) -> None:
    original = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        sealed = seal_latent(
            original, KEY, NONCE, rounds=2, use_permutation=True, aad=AAD,
            registry=registry,
        )
    with pytest.raises(AuthenticationError, match="authentication failed"):
        open_latent(sealed, bytes(reversed(range(32))))
    document = json.loads(sealed)
    document["nonce"] = (b"x" * 16).hex()
    tampered = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(AuthenticationError, match="authentication failed"):
        open_latent(tampered, KEY)


def test_sealing_rejects_nonce_reuse_across_m1_and_m2(tmp_path) -> None:
    original = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        seal_latent(
            original, KEY, NONCE, rounds=1, use_permutation=False, aad=AAD,
            registry=registry,
        )
        with pytest.raises(NonceReuseError, match="nonce reuse rejected"):
            seal_anonymous_image(
                original.reshape(1, 1, 8, 8), KEY, NONCE, rounds=1,
                use_permutation=False, latent_shape=tuple(original.shape),
                latent_dtype="float32", aad=AAD, registry=registry,
            )
