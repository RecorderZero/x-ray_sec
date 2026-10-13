"""D5.4 persistent nonce registry."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.crypto.errors import CryptoConfigurationError, NonceReuseError
from src.crypto.kdf import generate_master_key
from src.crypto.nonce import NonceRegistry


def test_explicit_reuse_is_rejected_and_scope_is_per_key(tmp_path: Path) -> None:
    path = tmp_path / "nonces.sqlite3"
    key_a, key_b = generate_master_key(), generate_master_key()
    nonce = bytes(range(16))
    with NonceRegistry(path) as registry:
        registry.register(key_a, nonce)
        with pytest.raises(NonceReuseError):
            registry.register(key_a, nonce)
        registry.register(key_b, nonce)
        assert registry.count() == 2
    with NonceRegistry(path) as reopened:
        with pytest.raises(NonceReuseError):
            reopened.register(key_a, nonce)


def test_issue_is_unique_and_atomic_across_threads(tmp_path: Path) -> None:
    key = generate_master_key()
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(lambda _: registry.issue(key), range(1000)))
        assert len(values) == len(set(values)) == registry.count() == 1000


def test_nonce_length_fails_closed(tmp_path: Path) -> None:
    with NonceRegistry(tmp_path / "nonces.sqlite3") as registry:
        with pytest.raises(CryptoConfigurationError):
            registry.register(generate_master_key(), b"short")
