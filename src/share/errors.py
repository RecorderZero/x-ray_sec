"""Layer-specific failures for the sharing prototype.

Every verification failure subclasses the v1 :class:`AuthenticationError`, so a
service boundary can catch one type and report nothing more specific, while
tests can assert which layer fired (opinion F2 / N10).
"""

from __future__ import annotations

from src.crypto.errors import AuthenticationError, CryptoConfigurationError


class ShareFormatError(CryptoConfigurationError):
    """Framing, schema, size limit, or canonical-encoding violation."""


class RegistryError(AuthenticationError):
    """Registry signature, freshness, rollback, or key-status check failed."""


class SignatureError(AuthenticationError):
    """The sender's Ed25519 signature did not verify."""


class UnwrapError(AuthenticationError):
    """HPKE could not unwrap the image key (wrong key, wrong info, malformed wrap)."""


class MacError(AuthenticationError):
    """The container HMAC did not verify under the unwrapped image key."""


class MetadataError(AuthenticationError):
    """Private metadata failed AEAD authentication or its value checks."""


class ManifestError(AuthenticationError):
    """The model/sampler manifest is missing, mismatched, unsigned, or outside the profile."""
