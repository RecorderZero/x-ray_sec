"""D6.1/D6.2 correctness and AF-030 transform integration tests."""

import pytest
import torch

from src.crypto.errors import CryptoConfigurationError
from src.crypto.prng import permutation, sign_vector
from src.crypto.sot import fwht, inverse_transform, transform, transform_vectors


KEY = bytes(range(32))
OTHER_KEY = bytes(reversed(range(32)))
NONCE = bytes(range(16))
OTHER_NONCE = bytes(reversed(range(16)))


@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
@pytest.mark.parametrize("use_permutation", [False, True])
def test_float64_inverse_d65536(rounds: int, use_permutation: bool) -> None:
    generator = torch.Generator().manual_seed(1911)
    original = torch.randn(1, 256, 256, dtype=torch.float64, generator=generator)
    protected = transform(original, KEY, NONCE, rounds=rounds, use_permutation=use_permutation)
    restored = inverse_transform(
        protected, KEY, NONCE, rounds=rounds, use_permutation=use_permutation,
    )
    assert protected.shape == original.shape
    assert protected.dtype == original.dtype
    assert torch.max(torch.abs(restored - original)).item() <= 1e-12


def test_explicit_q_is_orthogonal_and_matches_round_composition() -> None:
    dimension = 1024
    identity = torch.eye(dimension, dtype=torch.float64)
    # Each row is a basis vector; transpose so columns form the conventional Q.
    q1 = transform_vectors(identity, KEY, NONCE, rounds=1, use_permutation=True).T
    q_total = transform_vectors(identity, KEY, NONCE, rounds=2, use_permutation=True).T
    hadamard = fwht(identity).T
    pre = torch.from_numpy(sign_vector(KEY, NONCE, 1, "sign-pre", dimension)).double()
    post = torch.from_numpy(sign_vector(KEY, NONCE, 1, "sign-post", dimension)).double()
    indices = torch.from_numpy(permutation(KEY, NONCE, 1, dimension))
    permutation_matrix = identity[indices]
    q_round_two = torch.diag(post) @ permutation_matrix @ hadamard @ torch.diag(pre)
    assert torch.max(torch.abs(q_total - q_round_two @ q1)).item() <= 1e-12
    assert torch.max(torch.abs(q_total.T @ q_total - identity)).item() <= 1e-12
    assert abs(torch.linalg.matrix_norm(q_total, ord=2).item() - 1.0) <= 1e-12


def test_wrong_key_and_nonce_are_not_valid_inverse_controls() -> None:
    generator = torch.Generator().manual_seed(1911)
    original = torch.randn(65536, dtype=torch.float64, generator=generator)
    protected = transform(original, KEY, NONCE, rounds=2)
    wrong_key = inverse_transform(protected, OTHER_KEY, NONCE, rounds=2)
    wrong_nonce = inverse_transform(protected, KEY, OTHER_NONCE, rounds=2)
    for candidate in (wrong_key, wrong_nonce):
        cosine = torch.nn.functional.cosine_similarity(original, candidate, dim=0).item()
        assert abs(cosine) < 0.03
        assert not torch.allclose(candidate, original)


def test_fwht_contract_and_involution() -> None:
    values = torch.arange(16, dtype=torch.float64).reshape(2, 8)
    assert torch.max(torch.abs(fwht(fwht(values)) - values)).item() <= 1e-12
    with pytest.raises(CryptoConfigurationError):
        fwht(torch.ones(6, dtype=torch.float64))
    with pytest.raises(CryptoConfigurationError):
        fwht(torch.ones(8, dtype=torch.int64))
    with pytest.raises(CryptoConfigurationError):
        fwht(torch.tensor([float("nan"), 1.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_inverse() -> None:
    original = torch.arange(1024, dtype=torch.float64, device="cuda")
    protected = transform(original, KEY, NONCE, rounds=2)
    restored = inverse_transform(protected, KEY, NONCE, rounds=2)
    assert protected.is_cuda and restored.is_cuda
    assert torch.max(torch.abs(restored - original)).item() <= 1e-12
