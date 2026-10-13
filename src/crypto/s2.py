"""Authenticated S2 wrappers around the structured orthogonal transform.

S2a is the transform alone (:mod:`src.crypto.sot`).  S2 adds the D5
authenticated container.  All open paths enter inversion/decode logic only
through ``open_authenticated`` so tag verification necessarily happens first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from scripts.m1_storage import (
    QUANTIZATION,
    decode_range_preserving_png,
    encode_range_preserving_png,
)

from .container import AuthenticatedContainer, make_storage_record, validate_m1_png
from .errors import CryptoConfigurationError
from .mac import open_authenticated, seal_container
from .sot import inverse_transform, transform


TRANSFORM_NAME = "hadamard-rademacher/v1"
LATENT_STORAGE = "latent-payload/v1"
SUPPORTED_DTYPES = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8")}


def transform_parameters(rounds: int, use_permutation: bool) -> dict[str, Any]:
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise CryptoConfigurationError("rounds must be a positive integer")
    if not isinstance(use_permutation, bool):
        raise CryptoConfigurationError("use_permutation must be bool")
    return {
        "scheme": "S2", "transform": TRANSFORM_NAME,
        "rounds": rounds, "permutation": use_permutation,
    }


def _aad(aad: Mapping[str, Any], latent: torch.Tensor) -> dict[str, Any]:
    output = dict(aad)
    output["shape"] = list(latent.shape)
    output["dtype"] = str(latent.detach().cpu().numpy().dtype)
    return output


def seal_latent(
    latent: torch.Tensor,
    master_key: bytes,
    nonce: bytes,
    *,
    rounds: int,
    use_permutation: bool,
    aad: Mapping[str, Any],
) -> bytes:
    """Transform and authenticate an exact M2 latent payload."""
    protected = transform(
        latent, master_key, nonce, rounds=rounds, use_permutation=use_permutation,
    )
    array = protected.detach().cpu().contiguous().numpy()
    if str(array.dtype) not in SUPPORTED_DTYPES:
        raise CryptoConfigurationError("S2 payload dtype must be float32 or float64")
    payload = array.astype(SUPPORTED_DTYPES[str(array.dtype)], copy=False).tobytes(order="C")
    unsigned = AuthenticatedContainer(
        nonce=nonce,
        parameters=transform_parameters(rounds, use_permutation),
        aad=_aad(aad, latent),
        storage=make_storage_record(
            mode="M2", storage_protocol=LATENT_STORAGE, quantization="none",
            lo_float32_bits="", hi_float32_bits="", payload=payload,
        ),
        ciphertext=payload,
    )
    return seal_container(master_key, unsigned)


def open_latent(serialized: bytes, master_key: bytes, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Authenticate, parse, and inverse an M2 payload in that order."""
    def opener(container: AuthenticatedContainer) -> torch.Tensor:
        if container.storage != make_storage_record(
            mode="M2", storage_protocol=LATENT_STORAGE, quantization="none",
            lo_float32_bits="", hi_float32_bits="", payload=container.ciphertext,
        ):
            raise CryptoConfigurationError("invalid S2 M2 storage record")
        parameters = dict(container.parameters)
        expected = transform_parameters(parameters.get("rounds"), parameters.get("permutation"))
        if parameters != expected:
            raise CryptoConfigurationError("invalid S2 parameters")
        dtype_name = container.aad["dtype"]
        if dtype_name not in SUPPORTED_DTYPES:
            raise CryptoConfigurationError("unsupported authenticated latent dtype")
        shape = tuple(container.aad["shape"])
        expected_bytes = int(np.prod(shape)) * SUPPORTED_DTYPES[dtype_name].itemsize
        if len(container.ciphertext) != expected_bytes:
            raise CryptoConfigurationError("latent payload length disagrees with authenticated shape")
        array = np.frombuffer(container.ciphertext, dtype=SUPPORTED_DTYPES[dtype_name]).reshape(shape).copy()
        protected = torch.from_numpy(array).to(device)
        return inverse_transform(
            protected, master_key, container.nonce,
            rounds=parameters["rounds"], use_permutation=parameters["permutation"],
        )

    return open_authenticated(serialized, master_key, opener)

def seal_anonymous_image(
    anonymous_image: torch.Tensor,
    master_key: bytes,
    nonce: bytes,
    *,
    rounds: int,
    use_permutation: bool,
    latent_shape: tuple[int, ...],
    latent_dtype: str,
    aad: Mapping[str, Any],
) -> bytes:
    """Encode one anonymous image as authenticated range-preserving M1 PNG."""
    if anonymous_image.shape != (1, 1, anonymous_image.shape[-2], anonymous_image.shape[-1]):
        raise CryptoConfigurationError("anonymous image must have shape (1,1,H,W)")
    payload, png = encode_range_preserving_png(anonymous_image[0, 0].detach().cpu().numpy())
    full_aad = dict(aad)
    full_aad.update(shape=list(latent_shape), dtype=latent_dtype)
    unsigned = AuthenticatedContainer(
        nonce=nonce,
        parameters=transform_parameters(rounds, use_permutation),
        aad=full_aad,
        storage=make_storage_record(
            mode="M1", storage_protocol=png["storage_protocol"], quantization=QUANTIZATION,
            lo_float32_bits=png["lo_float32_bits"], hi_float32_bits=png["hi_float32_bits"],
            payload=payload,
        ),
        ciphertext=payload,
    )
    return seal_container(master_key, unsigned, preseal_validator=validate_m1_png)


def open_anonymous_image(
    serialized: bytes,
    master_key: bytes,
    inverter: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Verify tag, decode PNG, run inversion, then inverse the SOT."""
    def opener(container: AuthenticatedContainer) -> torch.Tensor:
        parameters = dict(container.parameters)
        expected = transform_parameters(parameters.get("rounds"), parameters.get("permutation"))
        if parameters != expected:
            raise CryptoConfigurationError("invalid S2 parameters")
        image = torch.from_numpy(decode_range_preserving_png(container.ciphertext).copy())[None, None].to(device)
        protected = inverter(image)
        if tuple(protected.shape) != tuple(container.aad["shape"]):
            raise CryptoConfigurationError("inverter output shape disagrees with authenticated latent shape")
        return inverse_transform(
            protected, master_key, container.nonce,
            rounds=parameters["rounds"], use_permutation=parameters["permutation"],
        )

    return open_authenticated(serialized, master_key, opener)
