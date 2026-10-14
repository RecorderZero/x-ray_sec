"""AF-025 S2 v2: fixed-norm transform with encrypted recovery metadata.

The public transform input always has the frozen norm ``FIXED_NORM``.  The
original norm, and M1 lo/hi when applicable, are encrypted with a dedicated
ChaCha20-Poly1305 subkey.  The existing outer HMAC still authenticates the
complete container before any metadata decryption, image decoding, inversion,
or inverse transform is attempted.
"""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from scripts.m1_storage import (
    PRIVATE_RANGE_PNG,
    QUANTIZATION,
    decode_private_png_attacker_view,
    decode_private_range_png,
    encode_private_range_png,
)

from .container import (
    AuthenticatedContainer,
    deserialize_container,
    make_storage_record,
    validate_m1_png,
)
from .errors import CryptoConfigurationError
from .kdf import PRIVATE_METADATA_LABEL, derive_subkey
from .mac import open_authenticated, seal_container
from .nonce import NonceRegistry
from .sot import inverse_transform, transform


SCHEME = "S2-fixed-norm/v2"
TRANSFORM_NAME = "hadamard-rademacher-fixed-norm/v2"
LATENT_STORAGE = "latent-payload-fixed-norm/v2"
FIXED_NORM = np.float64("248.261")
FIXED_NORM_FLOAT64_BITS = struct.pack(">d", float(FIXED_NORM)).hex()
SUPPORTED_DTYPES = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8")}
AEAD_NONCE = bytes(12)  # Safe because the HKDF subkey is unique per image nonce.


def _f64_bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _f64_from_bits(text: str) -> float:
    if not isinstance(text, str) or len(text) != 16:
        raise CryptoConfigurationError("invalid float64 bit string")
    try:
        value = struct.unpack(">d", bytes.fromhex(text))[0]
    except (ValueError, struct.error) as error:
        raise CryptoConfigurationError("invalid float64 bit string") from error
    if not np.isfinite(value) or value <= 0:
        raise CryptoConfigurationError("original norm must be positive and finite")
    return float(value)


def _base_parameters(rounds: int, use_permutation: bool) -> dict[str, Any]:
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise CryptoConfigurationError("rounds must be a positive integer")
    if not isinstance(use_permutation, bool):
        raise CryptoConfigurationError("use_permutation must be bool")
    return {
        "scheme": SCHEME,
        "transform": TRANSFORM_NAME,
        "rounds": rounds,
        "permutation": use_permutation,
        "fixed_norm_float64_bits": FIXED_NORM_FLOAT64_BITS,
        "private_metadata_aead": "chacha20-poly1305/hkdf-per-image-key/v1",
    }


def _metadata_aad(nonce: bytes, mode: str, base: Mapping[str, Any]) -> bytes:
    document = {"nonce": nonce.hex(), "mode": mode, "parameters": dict(base)}
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _encrypt_metadata(
    master_key: bytes, nonce: bytes, mode: str, base: Mapping[str, Any], metadata: Mapping[str, str],
) -> str:
    plaintext = json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    key = derive_subkey(master_key, nonce, PRIVATE_METADATA_LABEL)
    ciphertext = ChaCha20Poly1305(key).encrypt(AEAD_NONCE, plaintext, _metadata_aad(nonce, mode, base))
    return base64.b64encode(ciphertext).decode("ascii")


def _decrypt_metadata(
    master_key: bytes, container: AuthenticatedContainer, mode: str, base: Mapping[str, Any],
) -> dict[str, str]:
    encoded = container.parameters.get("encrypted_private_metadata")
    if not isinstance(encoded, str):
        raise CryptoConfigurationError("missing encrypted private metadata")
    try:
        ciphertext = base64.b64decode(encoded, validate=True)
        key = derive_subkey(master_key, container.nonce, PRIVATE_METADATA_LABEL)
        plaintext = ChaCha20Poly1305(key).decrypt(
            AEAD_NONCE, ciphertext, _metadata_aad(container.nonce, mode, base),
        )
        document = json.loads(plaintext.decode("utf-8"))
    except Exception as error:
        raise CryptoConfigurationError("invalid encrypted private metadata") from error
    if not isinstance(document, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in document.items()):
        raise CryptoConfigurationError("private metadata schema mismatch")
    return document


def _parameters(
    master_key: bytes, nonce: bytes, mode: str, rounds: int, use_permutation: bool,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    base = _base_parameters(rounds, use_permutation)
    return {**base, "encrypted_private_metadata": _encrypt_metadata(master_key, nonce, mode, base, metadata)}


def _private_metadata(container: AuthenticatedContainer, master_key: bytes, mode: str) -> tuple[dict[str, Any], dict[str, str]]:
    parameters = dict(container.parameters)
    base = _base_parameters(parameters.get("rounds"), parameters.get("permutation"))
    if set(parameters) != set(base) | {"encrypted_private_metadata"} or any(
        parameters.get(key) != value for key, value in base.items()
    ):
        raise CryptoConfigurationError("invalid fixed-norm S2 parameters")
    return base, _decrypt_metadata(master_key, container, mode, base)


def normalize_to_fixed_norm(latent: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return a dtype-preserving tensor with the public frozen L2 norm."""
    if not torch.is_floating_point(latent) or not bool(torch.isfinite(latent).all()):
        raise CryptoConfigurationError("latent must be finite floating point")
    original_norm = float(torch.linalg.vector_norm(latent.detach().double()).item())
    if not np.isfinite(original_norm) or original_norm <= 0:
        raise CryptoConfigurationError("latent norm must be positive and finite")
    normalized = latent * latent.new_tensor(float(FIXED_NORM / original_norm))
    return normalized, original_norm


def restore_original_norm(latent: torch.Tensor, original_norm: float) -> torch.Tensor:
    return latent * latent.new_tensor(float(original_norm / FIXED_NORM))


def _aad(aad: Mapping[str, Any], latent: torch.Tensor) -> dict[str, Any]:
    output = dict(aad)
    output["shape"] = list(latent.shape)
    output["dtype"] = str(latent.detach().cpu().numpy().dtype)
    return output


def seal_latent(
    latent: torch.Tensor, master_key: bytes, nonce: bytes, *, rounds: int,
    use_permutation: bool, aad: Mapping[str, Any], registry: NonceRegistry,
) -> bytes:
    if not isinstance(registry, NonceRegistry):
        raise CryptoConfigurationError("S2 sealing requires a persistent nonce registry")
    registry.register(master_key, nonce)
    normalized, original_norm = normalize_to_fixed_norm(latent)
    protected = transform(normalized, master_key, nonce, rounds=rounds, use_permutation=use_permutation)
    array = protected.detach().cpu().contiguous().numpy()
    if str(array.dtype) not in SUPPORTED_DTYPES:
        raise CryptoConfigurationError("S2 payload dtype must be float32 or float64")
    payload = array.astype(SUPPORTED_DTYPES[str(array.dtype)], copy=False).tobytes(order="C")
    parameters = _parameters(
        master_key, nonce, "M2", rounds, use_permutation,
        {"original_norm_float64_bits": _f64_bits(original_norm)},
    )
    unsigned = AuthenticatedContainer(
        nonce=nonce, parameters=parameters, aad=_aad(aad, latent),
        storage=make_storage_record(
            mode="M2", storage_protocol=LATENT_STORAGE, quantization="none",
            lo_float32_bits="", hi_float32_bits="", payload=payload,
        ), ciphertext=payload,
    )
    return seal_container(master_key, unsigned)


def open_latent(serialized: bytes, master_key: bytes, *, device: torch.device | str = "cpu") -> torch.Tensor:
    def opener(container: AuthenticatedContainer) -> torch.Tensor:
        base, metadata = _private_metadata(container, master_key, "M2")
        if set(metadata) != {"original_norm_float64_bits"}:
            raise CryptoConfigurationError("M2 private metadata schema mismatch")
        if container.storage != make_storage_record(
            mode="M2", storage_protocol=LATENT_STORAGE, quantization="none",
            lo_float32_bits="", hi_float32_bits="", payload=container.ciphertext,
        ):
            raise CryptoConfigurationError("invalid fixed-norm M2 storage record")
        dtype_name = container.aad["dtype"]
        if dtype_name not in SUPPORTED_DTYPES:
            raise CryptoConfigurationError("unsupported authenticated latent dtype")
        shape = tuple(container.aad["shape"])
        if len(container.ciphertext) != int(np.prod(shape)) * SUPPORTED_DTYPES[dtype_name].itemsize:
            raise CryptoConfigurationError("latent payload length disagrees with authenticated shape")
        array = np.frombuffer(container.ciphertext, dtype=SUPPORTED_DTYPES[dtype_name]).reshape(shape).copy()
        normalized = inverse_transform(
            torch.from_numpy(array).to(device), master_key, container.nonce,
            rounds=base["rounds"], use_permutation=base["permutation"],
        )
        return restore_original_norm(normalized, _f64_from_bits(metadata["original_norm_float64_bits"]))
    return open_authenticated(serialized, master_key, opener)


def seal_anonymous_image(
    anonymous_image: torch.Tensor, original_norm: float, master_key: bytes, nonce: bytes, *,
    rounds: int, use_permutation: bool, latent_shape: tuple[int, ...], latent_dtype: str,
    aad: Mapping[str, Any], registry: NonceRegistry,
) -> bytes:
    if anonymous_image.ndim != 4 or anonymous_image.shape[:2] != (1, 1):
        raise CryptoConfigurationError("anonymous image must have shape (1,1,H,W)")
    if not isinstance(registry, NonceRegistry):
        raise CryptoConfigurationError("S2 sealing requires a persistent nonce registry")
    _f64_from_bits(_f64_bits(original_norm))
    registry.register(master_key, nonce)
    payload, png = encode_private_range_png(anonymous_image[0, 0].detach().cpu().numpy())
    metadata = {
        "original_norm_float64_bits": _f64_bits(original_norm),
        "lo_float32_bits": png["lo_float32_bits"],
        "hi_float32_bits": png["hi_float32_bits"],
    }
    full_aad = dict(aad)
    full_aad.update(shape=list(latent_shape), dtype=latent_dtype)
    unsigned = AuthenticatedContainer(
        nonce=nonce,
        parameters=_parameters(master_key, nonce, "M1", rounds, use_permutation, metadata),
        aad=full_aad,
        storage=make_storage_record(
            mode="M1", storage_protocol=PRIVATE_RANGE_PNG, quantization=QUANTIZATION,
            lo_float32_bits="", hi_float32_bits="", payload=payload,
        ), ciphertext=payload,
    )
    return seal_container(master_key, unsigned, preseal_validator=validate_m1_png)


def public_anonymous_image(serialized: bytes, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Extract the unauthenticated attacker-visible q/255 image, without lo/hi."""
    container = deserialize_container(serialized)
    if container.storage["mode"] != "M1" or container.storage["storage_protocol"] != PRIVATE_RANGE_PNG:
        raise CryptoConfigurationError("container is not private-range M1")
    image = decode_private_png_attacker_view(container.ciphertext).copy()
    return torch.from_numpy(image)[None, None].to(device)


def open_anonymous_image(
    serialized: bytes, master_key: bytes, inverter: Callable[[torch.Tensor], torch.Tensor], *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    def opener(container: AuthenticatedContainer) -> torch.Tensor:
        base, metadata = _private_metadata(container, master_key, "M1")
        if set(metadata) != {"original_norm_float64_bits", "lo_float32_bits", "hi_float32_bits"}:
            raise CryptoConfigurationError("M1 private metadata schema mismatch")
        if container.storage["storage_protocol"] != PRIVATE_RANGE_PNG:
            raise CryptoConfigurationError("invalid fixed-norm M1 storage record")
        image = decode_private_range_png(
            container.ciphertext, metadata["lo_float32_bits"], metadata["hi_float32_bits"],
        ).copy()
        protected = inverter(torch.from_numpy(image)[None, None].to(device))
        if tuple(protected.shape) != tuple(container.aad["shape"]):
            raise CryptoConfigurationError("inverter output shape disagrees with authenticated latent shape")
        normalized = inverse_transform(
            protected, master_key, container.nonce,
            rounds=base["rounds"], use_permutation=base["permutation"],
        )
        return restore_original_norm(normalized, _f64_from_bits(metadata["original_norm_float64_bits"]))
    return open_authenticated(serialized, master_key, opener)
