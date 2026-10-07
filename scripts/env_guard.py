#!/usr/bin/env python3
"""Fail closed when research scripts import packages outside the active env."""

from __future__ import annotations

import site
import sys
from pathlib import Path
from types import ModuleType
from typing import Mapping


def enforce_active_prefix(package_modules: Mapping[str, ModuleType]) -> None:
    """Reject modules loaded outside ``sys.prefix`` or an enabled user site."""
    prefix = Path(sys.prefix).resolve()
    outside: dict[str, str] = {}
    for name, module in package_modules.items():
        module_file = getattr(module, "__file__", None)
        if not module_file:
            outside[name] = "<missing __file__>"
            continue
        resolved = Path(module_file).resolve()
        if prefix not in resolved.parents:
            outside[name] = str(resolved)
    if outside:
        raise SystemExit(f"packages loaded outside sys.prefix: {outside}")
    if site.ENABLE_USER_SITE:
        raise SystemExit(
            "user-site is enabled; run through scripts/run_cfg_ddim.sh "
            "or set PYTHONNOUSERSITE=1"
        )
