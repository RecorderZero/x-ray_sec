"""AF-025 fixed-norm S2 v2 and private recovery metadata."""

import base64
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.m1_storage import HI_KEY, LO_KEY, PRIVATE_RANGE_PNG
from src.crypto.container import deserialize_container
from src.crypto.errors import AuthenticationError
from src.crypto.nonce import NonceRegistry
from src.crypto.s2_fixed_norm import (
    FIXED_NORM,
    normalize_to_fixed_norm,
    open_anonymous_image,
    open_latent,
    public_anonymous_image,
    seal_anonymous_image,
    seal_latent,
)
from src.crypto.sot import transform


KEY = bytes(range(32))
AAD = {
    "study_instance_uid": "research:fixed-norm",
    "sop_instance_uid": "research:fixed-norm:0",
    "model_sha256": "ab" * 32,
    "sampler_version": "legacy-ddim-anonymization/v1",
}


def test_m2_fixed_norm_is_public_but_original_norm_is_private(tmp_path) -> None:
    original = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(1911)) * 3
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        sealed = seal_latent(
            original, KEY, bytes(range(16)), rounds=1, use_permutation=False,
            aad=AAD, registry=registry,
        )
    container = deserialize_container(sealed)
    protected = np.frombuffer(container.ciphertext, dtype="<f4")
    assert np.linalg.norm(protected.astype(np.float64)) == pytest.approx(float(FIXED_NORM), abs=2e-5)
    serialized_text = sealed.decode("utf-8")
    assert "original_norm_float64_bits" not in serialized_text
    restored = open_latent(sealed, KEY)
    assert torch.max(torch.abs(restored - original)).item() <= 3e-6


def test_m1_hides_range_and_restores_original_norm_after_authentication(tmp_path) -> None:
    original = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(1911))
    normalized, original_norm = normalize_to_fixed_norm(original)
    nonce = bytes(reversed(range(16)))
    protected = transform(normalized, KEY, nonce, rounds=1, use_permutation=False)
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        sealed = seal_anonymous_image(
            protected, original_norm, KEY, nonce, rounds=1, use_permutation=False,
            latent_shape=tuple(original.shape), latent_dtype="float32", aad=AAD,
            registry=registry,
        )
    container = deserialize_container(sealed)
    assert container.storage["storage_protocol"] == PRIVATE_RANGE_PNG
    assert container.storage["lo_float32_bits"] == container.storage["hi_float32_bits"] == ""
    with Image.open(io.BytesIO(container.ciphertext)) as image:
        assert LO_KEY not in image.text and HI_KEY not in image.text
    public = public_anonymous_image(sealed)
    assert float(public.min()) >= 0 and float(public.max()) <= 1
    restored = open_anonymous_image(sealed, KEY, lambda image: image)
    assert torch.max(torch.abs(restored - original)).item() < 0.04


def test_private_metadata_tamper_fails_before_inverter(tmp_path) -> None:
    original = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(1911))
    normalized, original_norm = normalize_to_fixed_norm(original)
    nonce = bytes(range(16))
    protected = transform(normalized, KEY, nonce, rounds=1, use_permutation=False)
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        sealed = seal_anonymous_image(
            protected, original_norm, KEY, nonce, rounds=1, use_permutation=False,
            latent_shape=tuple(original.shape), latent_dtype="float32", aad=AAD,
            registry=registry,
        )
    document = json.loads(sealed)
    raw = bytearray(base64.b64decode(document["parameters"]["encrypted_private_metadata"]))
    raw[0] ^= 1
    document["parameters"]["encrypted_private_metadata"] = base64.b64encode(raw).decode()
    tampered = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    calls = []
    with pytest.raises(AuthenticationError, match="authentication failed"):
        open_anonymous_image(tampered, KEY, lambda image: calls.append(image) or image)
    assert calls == []
