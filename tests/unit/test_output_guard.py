"""AF-029: tracked outputs are not overwritten without --force."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.output_guard import add_force_flag, refuse_overwrite


def test_existing_output_is_refused_without_force(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        refuse_overwrite([target], force=False)
    assert target.read_text(encoding="utf-8") == "{}"


def test_force_or_missing_output_is_allowed(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    refuse_overwrite([target], force=False)  # missing: allowed
    target.write_text("{}", encoding="utf-8")
    refuse_overwrite([target], force=True)  # explicit force: allowed


def test_force_flag_defaults_to_false() -> None:
    parser = argparse.ArgumentParser()
    add_force_flag(parser)
    assert parser.parse_args([]).force is False
    assert parser.parse_args(["--force"]).force is True
