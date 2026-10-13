"""D5.4 persistent, atomic nonce-reuse registry scoped by master key."""

from __future__ import annotations

import hmac
import secrets
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from .errors import NonceReuseError
from .kdf import NONCE_BYTES, require_master_key, require_nonce


SCOPE_LABEL = b"cxr-transform/v1/nonce-registry-scope"


def key_scope(master_key: bytes) -> bytes:
    """Internal irreversible namespace; raw master keys never enter the registry."""
    require_master_key(master_key)
    return hmac.digest(master_key, SCOPE_LABEL, "sha256")


class NonceRegistry:
    """SQLite UNIQUE(scope,nonce) provides cross-process check-and-insert."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._lock = threading.Lock()
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS nonces (scope BLOB NOT NULL, nonce BLOB NOT NULL, "
            "PRIMARY KEY(scope, nonce)) WITHOUT ROWID"
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "NonceRegistry":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def register(self, master_key: bytes, nonce: bytes) -> None:
        require_nonce(nonce)
        scope = key_scope(master_key)
        try:
            with self._lock, self._connection:
                self._connection.execute("INSERT INTO nonces(scope, nonce) VALUES (?, ?)", (scope, nonce))
        except sqlite3.IntegrityError as error:
            raise NonceReuseError("nonce reuse rejected") from error

    def register_many(self, master_key: bytes, nonces: Iterable[bytes]) -> int:
        materialized = list(nonces)
        for nonce in materialized:
            require_nonce(nonce)
        scope = key_scope(master_key)
        try:
            with self._lock, self._connection:
                self._connection.executemany(
                    "INSERT INTO nonces(scope, nonce) VALUES (?, ?)",
                    ((scope, nonce) for nonce in materialized),
                )
        except sqlite3.IntegrityError as error:
            raise NonceReuseError("nonce reuse rejected") from error
        return len(materialized)

    def issue(self, master_key: bytes) -> bytes:
        nonce = secrets.token_bytes(NONCE_BYTES)
        self.register(master_key, nonce)
        return nonce

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM nonces").fetchone()[0])
