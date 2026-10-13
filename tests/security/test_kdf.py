"""D5.1/D5.2 key source, password mode, and RFC 5869 vectors."""

import pytest

from src.crypto.errors import CryptoConfigurationError
from src.crypto.kdf import (
    MAC_LABEL,
    ScryptParameters,
    derive_subkey,
    generate_master_key,
    hkdf_sha256,
    master_key_from_password,
    round_label,
)


def test_random_master_key_is_256_bits_and_not_reused() -> None:
    keys = {generate_master_key() for _ in range(32)}
    assert len(keys) == 32
    assert all(isinstance(key, bytes) and len(key) == 32 for key in keys)


def test_password_mode_is_deterministic_only_with_same_salt_and_parameters() -> None:
    salt = bytes(range(16))
    first = master_key_from_password("correct horse battery staple", salt=salt)
    second = master_key_from_password("correct horse battery staple", salt=salt)
    changed = master_key_from_password("correct horse battery staple", salt=bytes(reversed(range(16))))
    assert first.master_key == second.master_key
    assert first.master_key != changed.master_key
    assert "correct horse" not in repr(first) and first.master_key.hex() not in repr(first)
    assert first.parameters.manifest()["algorithm"] == "scrypt"


@pytest.mark.parametrize("password", ["", b"not text", None])
def test_password_mode_rejects_invalid_password(password) -> None:
    with pytest.raises(CryptoConfigurationError):
        master_key_from_password(password)  # type: ignore[arg-type]


def test_scrypt_rejects_weak_or_malformed_parameters() -> None:
    with pytest.raises(CryptoConfigurationError):
        master_key_from_password("pw", parameters=ScryptParameters(n=1024))
    with pytest.raises(CryptoConfigurationError):
        master_key_from_password("pw", salt=b"short")


def test_rfc5869_sha256_case_1() -> None:
    output = hkdf_sha256(
        bytes.fromhex("0b" * 22),
        salt=bytes.fromhex("000102030405060708090a0b0c"),
        info=bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"),
        length=42,
    )
    assert output.hex() == (
        "3cb25f25faacd57a90434f64d0362f2a"
        "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865"
    )


def test_domain_labels_and_full_inputs_are_separated() -> None:
    master = bytes(range(32))
    nonce = bytes(range(16))
    labels = [round_label(r, purpose) for r in range(4) for purpose in
              ("sign-pre", "perm", "sign-post")] + [MAC_LABEL]
    assert len(labels) == len(set(labels))
    outputs = [derive_subkey(master, nonce, label) for label in labels]
    assert all(len(output) == 32 for output in outputs)
    assert len(outputs) == len(set(outputs))
    assert derive_subkey(master, nonce, labels[0]) == outputs[0]
    assert derive_subkey(master, bytes(reversed(range(16))), labels[0]) != outputs[0]
    assert derive_subkey(bytes(reversed(range(32))), nonce, labels[0]) != outputs[0]


def test_key_and_nonce_lengths_fail_closed() -> None:
    with pytest.raises(CryptoConfigurationError):
        derive_subkey(b"short", bytes(16), MAC_LABEL)
    with pytest.raises(CryptoConfigurationError):
        derive_subkey(bytes(32), b"short", MAC_LABEL)
