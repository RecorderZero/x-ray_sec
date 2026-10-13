"""D5.3 ChaCha20 sign/permutation construction and AF-030 guards."""

import ast
from pathlib import Path

import numpy as np

from src.crypto.prng import _uniform_below, inverse_permutation, permutation, sign_vector


KEY = bytes(range(32))
NONCE = bytes(range(16))


class FakeSource:
    def __init__(self, values: list[int]):
        self.values = iter(values)
        self.calls = 0

    def read(self, size: int) -> bytes:
        assert size == 4
        self.calls += 1
        return next(self.values).to_bytes(4, "little")


def test_frozen_chacha_vectors() -> None:
    assert sign_vector(KEY, NONCE, 0, "sign-pre", 32).tolist() == [
        -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1,
        1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1,
    ]
    assert permutation(KEY, NONCE, 0, 16).tolist() == [
        4, 5, 0, 6, 12, 9, 2, 14, 3, 13, 1, 7, 10, 15, 11, 8,
    ]


def test_rejection_sampling_rejects_incomplete_tail() -> None:
    source = FakeSource([2**32 - 1, 7])
    assert _uniform_below(source, 3) == 1
    assert source.calls == 2


def test_permutation_is_complete_reproducible_and_invertible() -> None:
    first = permutation(KEY, NONCE, 2, 4096)
    second = permutation(KEY, NONCE, 2, 4096)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(4096))
    assert np.array_equal(first[inverse_permutation(first)], np.arange(4096))
    assert not np.array_equal(first, permutation(KEY, bytes(reversed(range(16))), 2, 4096))


def test_signs_are_reproducible_balanced_and_nonce_separated() -> None:
    first = sign_vector(KEY, NONCE, 1, "sign-post", 65_536)
    assert np.array_equal(first, sign_vector(KEY, NONCE, 1, "sign-post", 65_536))
    assert set(np.unique(first)) == {-1, 1}
    assert abs(float(first.mean())) < 0.02
    assert not np.array_equal(first, sign_vector(KEY, bytes(reversed(range(16))), 1, "sign-post", 65_536))


def test_image_nonce_near_chacha_counter_limit_cannot_overflow_stream() -> None:
    # Before AF-037, these first four bytes became ChaCha20's initial counter
    # and a full d=65,536 permutation deterministically exhausted the stream.
    near_wrap_nonce = b"\xff\xff\xff\xff" + bytes(range(12))
    signs = sign_vector(KEY, near_wrap_nonce, 0, "sign-pre", 65_536)
    indices = permutation(KEY, near_wrap_nonce, 0, 65_536)
    assert signs.shape == (65_536,) and set(np.unique(signs)) == {-1, 1}
    assert np.array_equal(np.sort(indices), np.arange(65_536))
    assert np.array_equal(indices[inverse_permutation(indices)], np.arange(65_536))


def test_crypto_key_path_contains_no_noncrypto_seed_calls() -> None:
    banned = {"manual_seed", "seed", "default_rng", "RandomState"}
    for path in Path("src/crypto").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = {
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not (calls & banned), (path, calls & banned)
