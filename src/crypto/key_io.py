"""D5.7 strict raw key files; no pickle-capable loader is used."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import CryptoConfigurationError
from .kdf import require_master_key


def write_master_key(path: Path, master_key: bytes) -> None:
    key = require_master_key(master_key)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def read_master_key(path: Path) -> bytes:
    payload = Path(path).read_bytes()
    if len(payload) != 32:
        raise CryptoConfigurationError("key file must contain exactly 32 raw bytes")
    return require_master_key(payload)
