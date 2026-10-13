"""D5.5 canonical container and M1 metadata consistency."""

import json

import pytest

from src.crypto.container import (
    VERSION,
    AuthenticatedContainer,
    deserialize_container,
    validate_m1_png,
)
from src.crypto.errors import CryptoConfigurationError
from src.crypto.mac import seal_container

from .conftest import TEST_KEY


def test_canonical_serialize_parse_roundtrip(unsigned_m1: AuthenticatedContainer) -> None:
    serialized = seal_container(TEST_KEY, unsigned_m1, preseal_validator=validate_m1_png)
    parsed = deserialize_container(serialized)
    assert parsed.version == VERSION
    assert parsed.nonce == unsigned_m1.nonce
    assert parsed.ciphertext == unsigned_m1.ciphertext
    assert parsed.mac_bytes() == unsigned_m1.mac_bytes()
    assert parsed.serialize() == serialized


def test_unknown_version_missing_extra_duplicate_and_float_fail_closed(unsigned_m1) -> None:
    signed = seal_container(TEST_KEY, unsigned_m1, preseal_validator=validate_m1_png)
    document = json.loads(signed)
    for mutation in (
        lambda d: d.update(version="cxr-auth-container/v999"),
        lambda d: d.pop("aad"),
        lambda d: d.update(extra="forbidden"),
        lambda d: d["parameters"].update(weight=0.5),
    ):
        changed = json.loads(signed)
        mutation(changed)
        with pytest.raises(CryptoConfigurationError):
            deserialize_container(json.dumps(changed).encode())
    duplicate = signed.rstrip()[:-1] + b',"version":"cxr-auth-container/v1"}'
    with pytest.raises(CryptoConfigurationError, match="duplicate"):
        deserialize_container(duplicate)


def test_lo_hi_are_exact_bits_and_png_mismatch_is_rejected_before_seal(unsigned_m1) -> None:
    assert len(unsigned_m1.storage["lo_float32_bits"]) == 8
    assert len(unsigned_m1.storage["hi_float32_bits"]) == 8
    changed = AuthenticatedContainer(
        nonce=unsigned_m1.nonce, parameters=unsigned_m1.parameters, aad=unsigned_m1.aad,
        storage={**unsigned_m1.storage, "lo_float32_bits": "00000000"},
        ciphertext=unsigned_m1.ciphertext,
    )
    with pytest.raises(CryptoConfigurationError, match="differ"):
        seal_container(TEST_KEY, changed, preseal_validator=validate_m1_png)
