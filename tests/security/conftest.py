"""Shared D5 authenticated-container fixtures."""

import json

import numpy as np
import pytest

from scripts.m1_storage import QUANTIZATION, encode_range_preserving_png
from src.crypto.container import AuthenticatedContainer, make_storage_record


TEST_KEY = bytes(range(32))


@pytest.fixture
def unsigned_m1() -> AuthenticatedContainer:
    image = np.linspace(-0.4, 0.9, 64, dtype=np.float32).reshape(8, 8)
    payload, record = encode_range_preserving_png(image)
    return AuthenticatedContainer(
        nonce=bytes(range(16)),
        parameters={"scheme": "S2", "rounds": 2, "permutation": True},
        aad={
            "study_instance_uid": "1.2.3",
            "sop_instance_uid": "1.2.3.4",
            "shape": [1, 256, 256],
            "dtype": "float32",
            "model_sha256": "ab" * 32,
            "sampler_version": "legacy-ddim-anonymization/v1",
        },
        storage=make_storage_record(
            mode="M1", storage_protocol=record["storage_protocol"],
            quantization=QUANTIZATION,
            lo_float32_bits=record["lo_float32_bits"],
            hi_float32_bits=record["hi_float32_bits"], payload=payload,
        ),
        ciphertext=payload,
    )


def mutate_json(serialized: bytes, mutator) -> bytes:
    document = json.loads(serialized)
    mutator(document)
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
