"""D5.6 HMAC-SHA256 and verify-before-open control flow."""

from __future__ import annotations

from typing import Callable, TypeVar

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, hmac

from .container import AuthenticatedContainer, deserialize_container, validate_m1_png
from .errors import AuthenticationError
from .kdf import MAC_LABEL, derive_subkey, require_master_key


T = TypeVar("T")


def compute_tag(master_key: bytes, container: AuthenticatedContainer) -> bytes:
    require_master_key(master_key)
    key = derive_subkey(master_key, container.nonce, MAC_LABEL)
    signer = hmac.HMAC(key, hashes.SHA256())
    signer.update(container.mac_bytes())
    return signer.finalize()


def seal_container(
    master_key: bytes,
    unsigned: AuthenticatedContainer,
    *,
    preseal_validator: Callable[[AuthenticatedContainer], object] | None = None,
) -> bytes:
    unsigned.validate(require_tag=False)
    if unsigned.storage["mode"] == "M1":
        validate_m1_png(unsigned)
    if preseal_validator is not None:
        preseal_validator(unsigned)
    return unsigned.with_tag(compute_tag(master_key, unsigned)).serialize()


def _verify(master_key: bytes, container: AuthenticatedContainer) -> None:
    key = derive_subkey(master_key, container.nonce, MAC_LABEL)
    verifier = hmac.HMAC(key, hashes.SHA256())
    verifier.update(container.mac_bytes())
    try:
        verifier.verify(container.tag)
    except InvalidSignature as error:
        raise AuthenticationError("authentication failed") from error


def open_authenticated(
    serialized: bytes,
    master_key: bytes,
    opener: Callable[[AuthenticatedContainer], T],
) -> T:
    """Parse enough to authenticate, then and only then invoke plaintext work."""
    try:
        require_master_key(master_key)
        container = deserialize_container(serialized)
        _verify(master_key, container)
    except AuthenticationError:
        raise
    except Exception as error:
        raise AuthenticationError("authentication failed") from error
    try:
        if container.storage["mode"] == "M1":
            validate_m1_png(container)
        return opener(container)
    except Exception as error:
        # Auth succeeded, but malformed authenticated contents still fail closed.
        raise AuthenticationError("authentication failed") from error
