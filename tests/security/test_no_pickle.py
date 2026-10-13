"""D5.7 raw key format and static no-pickle guard."""

import ast
import stat
from pathlib import Path

import pytest

from src.crypto.errors import CryptoConfigurationError
from src.crypto.kdf import generate_master_key
from src.crypto.key_io import read_master_key, write_master_key


def test_raw_key_roundtrip_permissions_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "master.key"
    key = generate_master_key()
    write_master_key(path, key)
    assert read_master_key(path) == key
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        write_master_key(path, generate_master_key())


def test_key_reader_rejects_trailing_or_short_data(tmp_path: Path) -> None:
    path = tmp_path / "master.key"
    for payload in (bytes(31), bytes(33), b"80" * 32):
        path.write_bytes(payload)
        with pytest.raises(CryptoConfigurationError):
            read_master_key(path)


def test_src_crypto_has_no_pickle_capable_key_loader() -> None:
    forbidden_calls = {("pickle", "load"), ("torch", "load"), ("numpy", "load"), ("np", "load")}
    for path in Path("src/crypto").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    aliases[name.asname or name.name] = name.name
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module = aliases.get(node.func.value.id, node.func.value.id)
                found.add((module, node.func.attr))
        assert not (found & forbidden_calls), (path, found & forbidden_calls)
