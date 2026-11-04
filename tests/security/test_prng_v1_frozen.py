"""S1 / V1: v1 PRNG and SOT outputs stay bit-identical across the prng.py refactor.

Vectors were frozen from the pre-refactor code by scripts/freeze_prng_v1_vectors.py
(see the file's ``generated_from`` block). Test key bytes(range(32)) is public.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.crypto.errors import CryptoConfigurationError
from src.crypto.kdf import derive_subkey, round_label
from src.crypto.prng import permutation, permutation_from_key, sign_vector, sign_vector_from_key
from src.crypto.sot import transform


VECTORS = json.loads((Path(__file__).parent / "vectors/prng_v1_frozen.json").read_text(encoding="utf-8"))
TEST_KEY = bytes(range(32))
NONCES = {name: bytes.fromhex(value) for name, value in VECTORS["nonces"].items()}


def _sha(array: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(array.astype(dtype)).tobytes()).hexdigest()


def test_vectors_were_frozen_from_clean_crypto_files() -> None:
    assert VECTORS["generated_from"]["crypto_files_dirty"] is False
    assert len(VECTORS["sign_vectors"]) == 36 and len(VECTORS["permutations"]) == 18


@pytest.mark.parametrize("vec", VECTORS["sign_vectors"], ids=lambda v: f"{v['nonce']}-r{v['round']}-{v['purpose']}-{v['size']}")
def test_v1_sign_vectors_bit_identical(vec: dict) -> None:
    out = sign_vector(TEST_KEY, NONCES[vec["nonce"]], vec["round"], vec["purpose"], vec["size"])
    assert _sha(out, "<i1") == vec["sha256_int8"]


@pytest.mark.parametrize("vec", VECTORS["permutations"], ids=lambda v: f"{v['nonce']}-r{v['round']}-{v['size']}")
def test_v1_permutations_bit_identical(vec: dict) -> None:
    out = permutation(TEST_KEY, NONCES[vec["nonce"]], vec["round"], vec["size"])
    assert _sha(out, "<i8") == vec["sha256_int64"]


@pytest.mark.parametrize("vec", VECTORS["sot_transform"], ids=lambda v: f"R{v['rounds']}-P{int(v['permutation'])}")
def test_v1_sot_transform_bit_identical(vec: dict) -> None:
    x = torch.from_numpy(np.random.default_rng(1911).standard_normal(65_536).astype(np.float32)).reshape(1, 1, 256, 256)
    out = transform(x, TEST_KEY, NONCES["ascending"], rounds=vec["rounds"], use_permutation=vec["permutation"])
    assert _sha(out.numpy().reshape(-1), "<f4") == vec["sha256_float32"]


def test_from_key_matches_v1_with_v1_derived_subkey() -> None:
    nonce = NONCES["sha256_label"]
    for purpose in ("sign-pre", "sign-post"):
        key = derive_subkey(TEST_KEY, nonce, round_label(1, purpose))
        assert np.array_equal(sign_vector_from_key(key, 4096), sign_vector(TEST_KEY, nonce, 1, purpose, 4096))
    key = derive_subkey(TEST_KEY, nonce, round_label(1, "perm"))
    assert np.array_equal(permutation_from_key(key, 4096), permutation(TEST_KEY, nonce, 1, 4096))


@pytest.mark.parametrize("bad", [b"", bytes(31), bytes(33), "k" * 32])
def test_from_key_rejects_non_256_bit_keys(bad) -> None:
    with pytest.raises(CryptoConfigurationError):
        sign_vector_from_key(bad, 16)
    with pytest.raises(CryptoConfigurationError):
        permutation_from_key(bad, 16)


# bool sizes (True == 1) are accepted by the pre-refactor v1 validation; D11 keeps
# v1 behavior unchanged, so they are intentionally not listed here.
@pytest.mark.parametrize("size", [0, -1, 1.5])
def test_from_key_rejects_bad_sizes(size) -> None:
    key = bytes(32)
    with pytest.raises(CryptoConfigurationError):
        sign_vector_from_key(key, size)
    with pytest.raises(CryptoConfigurationError):
        permutation_from_key(key, size)
