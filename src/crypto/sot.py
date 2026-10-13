"""Key-derived structured orthogonal transform used by S2/S2a.

For one round this module implements exactly

    z_next = D_post P (H / sqrt(d)) D_pre z

where all signs and the optional permutation come directly from the D5
HKDF/ChaCha20 pipeline.  No integer seed or framework PRNG participates in
key generation.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .errors import CryptoConfigurationError
from .kdf import require_master_key, require_nonce
from .prng import inverse_permutation, permutation, sign_vector


def _require_power_of_two(size: int) -> None:
    if size < 1 or size & (size - 1):
        raise CryptoConfigurationError("transform dimension must be a positive power of two")


def fwht(values: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over the final dimension.

    The input is never modified.  CPU and CUDA tensors are supported; integer,
    complex, sparse, and non-finite inputs are rejected to keep the stored
    transform contract unambiguous.
    """
    if not isinstance(values, torch.Tensor) or values.layout != torch.strided:
        raise CryptoConfigurationError("FWHT input must be a dense torch tensor")
    if not values.is_floating_point():
        raise CryptoConfigurationError("FWHT input must use a real floating dtype")
    if values.ndim < 1:
        raise CryptoConfigurationError("FWHT input must have at least one dimension")
    size = values.shape[-1]
    _require_power_of_two(size)
    if not bool(torch.isfinite(values).all()):
        raise CryptoConfigurationError("FWHT input must be finite")

    output = values.clone()
    width = 1
    while width < size:
        blocks = output.reshape(*output.shape[:-1], -1, 2 * width)
        left = blocks[..., :width].clone()
        right = blocks[..., width:].clone()
        blocks[..., :width] = left + right
        blocks[..., width:] = left - right
        width *= 2
    return output / math.sqrt(size)


def _flat(values: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if not isinstance(values, torch.Tensor) or values.ndim < 1:
        raise CryptoConfigurationError("latent must be a torch tensor with at least one dimension")
    if not values.is_floating_point() or values.layout != torch.strided:
        raise CryptoConfigurationError("latent must be a dense real floating tensor")
    if not bool(torch.isfinite(values).all()):
        raise CryptoConfigurationError("latent must be finite")
    _require_power_of_two(values.numel())
    return values.reshape(1, -1), values.shape


def _round_material(
    master_key: bytes,
    nonce: bytes,
    round_index: int,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_permutation: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    pre = torch.as_tensor(
        sign_vector(master_key, nonce, round_index, "sign-pre", size),
        device=device, dtype=dtype,
    )
    post = torch.as_tensor(
        sign_vector(master_key, nonce, round_index, "sign-post", size),
        device=device, dtype=dtype,
    )
    indices = None
    if use_permutation:
        indices = torch.as_tensor(
            permutation(master_key, nonce, round_index, size),
            device=device, dtype=torch.long,
        )
    return pre, indices, post


def _validate_parameters(master_key: bytes, nonce: bytes, rounds: int, use_permutation: bool) -> None:
    require_master_key(master_key)
    require_nonce(nonce)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise CryptoConfigurationError("rounds must be a positive integer")
    if not isinstance(use_permutation, bool):
        raise CryptoConfigurationError("use_permutation must be bool")


def transform(
    latent: torch.Tensor,
    master_key: bytes,
    nonce: bytes,
    *,
    rounds: int,
    use_permutation: bool = True,
) -> torch.Tensor:
    """Apply ``Q_R ... Q_1`` to one latent tensor, preserving shape/dtype/device."""
    _validate_parameters(master_key, nonce, rounds, use_permutation)
    output, shape = _flat(latent)
    size = output.shape[-1]
    for round_index in range(rounds):
        pre, indices, post = _round_material(
            master_key, nonce, round_index, size,
            device=output.device, dtype=output.dtype,
            use_permutation=use_permutation,
        )
        output = fwht(output * pre)
        if indices is not None:
            output = output.index_select(-1, indices)
        output = output * post
    return output.reshape(shape)


def transform_vectors(
    vectors: torch.Tensor,
    master_key: bytes,
    nonce: bytes,
    *,
    rounds: int,
    use_permutation: bool = True,
) -> torch.Tensor:
    """Apply one shared transform to vectors stored along the final dimension.

    This batched form exists for explicit-matrix diagnostics.  Deployment code
    should call :func:`transform` once per image because every image requires a
    unique nonce.
    """
    _validate_parameters(master_key, nonce, rounds, use_permutation)
    if not isinstance(vectors, torch.Tensor) or vectors.ndim < 1:
        raise CryptoConfigurationError("vectors must be a torch tensor")
    if not vectors.is_floating_point() or vectors.layout != torch.strided:
        raise CryptoConfigurationError("vectors must be dense real floating tensors")
    if not bool(torch.isfinite(vectors).all()):
        raise CryptoConfigurationError("vectors must be finite")
    size = vectors.shape[-1]
    _require_power_of_two(size)
    output = vectors
    for round_index in range(rounds):
        pre, indices, post = _round_material(
            master_key, nonce, round_index, size,
            device=output.device, dtype=output.dtype,
            use_permutation=use_permutation,
        )
        output = fwht(output * pre)
        if indices is not None:
            output = output.index_select(-1, indices)
        output = output * post
    return output


def inverse_transform(
    protected: torch.Tensor,
    master_key: bytes,
    nonce: bytes,
    *,
    rounds: int,
    use_permutation: bool = True,
) -> torch.Tensor:
    """Apply the exact reverse round order and inverse operations."""
    _validate_parameters(master_key, nonce, rounds, use_permutation)
    output, shape = _flat(protected)
    size = output.shape[-1]
    for round_index in reversed(range(rounds)):
        pre, indices, post = _round_material(
            master_key, nonce, round_index, size,
            device=output.device, dtype=output.dtype,
            use_permutation=use_permutation,
        )
        output = output * post
        if indices is not None:
            inverse = torch.as_tensor(
                inverse_permutation(indices.detach().cpu().numpy().astype(np.int64, copy=False)),
                device=output.device, dtype=torch.long,
            )
            output = output.index_select(-1, inverse)
        output = fwht(output) * pre
    return output.reshape(shape)
