"""Public error types. Authentication failures intentionally share one type."""


class CryptoConfigurationError(ValueError):
    """A local key, nonce, version, or schema configuration is invalid."""


class NonceReuseError(RuntimeError):
    """The same nonce was registered twice in one master-key scope."""


class AuthenticationError(RuntimeError):
    """Generic external error; does not reveal which authenticated field failed."""
