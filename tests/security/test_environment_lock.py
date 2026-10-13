"""AF-038: D5 dependency pins and the Week 3 freeze must stay aligned."""

from pathlib import Path


def test_d5_dependency_pins_are_present_in_week3_freeze() -> None:
    requirements = {
        line.strip()
        for line in Path("requirements-d5.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    freeze = set(
        Path("artifacts/environment_freeze_w3.txt").read_text(encoding="utf-8").splitlines()
    )
    assert requirements == {"cryptography==50.0.2", "cffi==2.1.1", "pycparser==3.11"}
    assert requirements <= freeze
