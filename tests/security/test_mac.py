"""D5.6 HMAC tamper matrix and verify-before-open gate."""

import base64
import hashlib
import json

import pytest

from src.crypto.container import validate_m1_png
from src.crypto.errors import AuthenticationError
from src.crypto.mac import open_authenticated, seal_container

from .conftest import TEST_KEY, mutate_json


def test_unmodified_container_opens_once(unsigned_m1) -> None:
    sealed = seal_container(TEST_KEY, unsigned_m1, preseal_validator=validate_m1_png)
    calls = []
    result = open_authenticated(sealed, TEST_KEY, lambda container: calls.append(container) or validate_m1_png(container))
    assert result == unsigned_m1.ciphertext
    assert len(calls) == 1


def _tamper_cases(sealed: bytes):
    return {
        "tag": mutate_json(sealed, lambda d: d.update(tag="00" * 32)),
        "nonce": mutate_json(sealed, lambda d: d.update(nonce="ff" * 16)),
        "version": mutate_json(sealed, lambda d: d.update(version="cxr-auth-container/v2")),
        "parameters": mutate_json(sealed, lambda d: d["parameters"].update(rounds=3)),
        "aad": mutate_json(sealed, lambda d: d["aad"].update(sop_instance_uid="9.9.9")),
        "storage_protocol": mutate_json(sealed, lambda d: d["storage"].update(storage_protocol="other")),
        "quantization": mutate_json(sealed, lambda d: d["storage"].update(quantization="floor")),
        "record_lo_only": mutate_json(sealed, lambda d: d["storage"].update(lo_float32_bits="00000000")),
        "record_hi_only": mutate_json(sealed, lambda d: d["storage"].update(hi_float32_bits="3f800000")),
        "payload_hash": mutate_json(sealed, lambda d: d["storage"].update(payload_sha256="00" * 32)),
        "payload_byte": mutate_json(sealed, _change_payload_and_hash),
    }


def _change_payload_and_hash(document) -> None:
    payload = bytearray(base64.b64decode(document["ciphertext"]))
    payload[-12] ^= 1
    changed = bytes(payload)
    document["ciphertext"] = base64.b64encode(changed).decode()
    document["storage"]["payload_sha256"] = hashlib.sha256(changed).hexdigest()


def test_every_tamper_and_wrong_key_reject_before_opener(unsigned_m1) -> None:
    sealed = seal_container(TEST_KEY, unsigned_m1, preseal_validator=validate_m1_png)
    cases = _tamper_cases(sealed)
    cases["wrong_key"] = sealed
    for name, payload in cases.items():
        calls = []
        key = bytes(reversed(range(32))) if name == "wrong_key" else TEST_KEY
        with pytest.raises(AuthenticationError, match="authentication failed"):
            open_authenticated(payload, key, lambda container: calls.append(container))
        assert calls == [], name


def test_all_external_failures_use_same_error(unsigned_m1) -> None:
    sealed = seal_container(TEST_KEY, unsigned_m1, preseal_validator=validate_m1_png)
    messages = set()
    for payload in (b"not json", sealed[:-10], _tamper_cases(sealed)["record_lo_only"]):
        with pytest.raises(AuthenticationError) as caught:
            open_authenticated(payload, TEST_KEY, lambda _container: None)
        messages.add(str(caught.value))
    assert messages == {"authentication failed"}
