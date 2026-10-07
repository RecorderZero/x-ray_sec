from __future__ import annotations

import types

import pytest

from scripts.env_guard import enforce_active_prefix


def test_env_guard_accepts_module_inside_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(__file__="/opt/env/lib/python3.10/site-packages/pkg.py")
    monkeypatch.setattr("scripts.env_guard.sys.prefix", "/opt/env")
    monkeypatch.setattr("scripts.env_guard.site.ENABLE_USER_SITE", False)
    enforce_active_prefix({"pkg": module})


def test_env_guard_rejects_module_outside_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(__file__="/home/user/.local/lib/python3.10/site-packages/pkg.py")
    monkeypatch.setattr("scripts.env_guard.sys.prefix", "/opt/env")
    monkeypatch.setattr("scripts.env_guard.site.ENABLE_USER_SITE", False)
    with pytest.raises(SystemExit, match="packages loaded outside sys.prefix"):
        enforce_active_prefix({"pkg": module})


def test_env_guard_rejects_enabled_user_site(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(__file__="/opt/env/lib/python3.10/site-packages/pkg.py")
    monkeypatch.setattr("scripts.env_guard.sys.prefix", "/opt/env")
    monkeypatch.setattr("scripts.env_guard.site.ENABLE_USER_SITE", True)
    with pytest.raises(SystemExit, match="user-site is enabled"):
        enforce_active_prefix({"pkg": module})
