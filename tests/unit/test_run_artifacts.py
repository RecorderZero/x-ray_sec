"""Shared run-directory writer: standalone validation and start-time git state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_artifacts
from scripts.artifact_schema import validate_run


def test_standalone_run_is_valid_and_records_start_commit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EXPERIMENT_RUN_DIR", raising=False)
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> str:
        calls.append(args)
        return "start" if args[0] == "rev-parse" else ""

    monkeypatch.setattr(run_artifacts, "git_value", fake_git)
    context = run_artifacts.open_run("UNIT", {"a": 1}, tmp_path)
    # A commit made while the run is in progress must not be recorded.
    monkeypatch.setattr(run_artifacts, "git_value", lambda *args: "later" if args[0] == "rev-parse" else "")
    rows = [{"sample_id": "x", "metric": 1.0}, {"sample_id": "y", "metric": 2.0}]
    manifest_fields = {
        "dataset_split_hash": "s", "checkpoint_hash": "c", "seed_list": [1911, 1911],
        "steps": 1, "scheme": "P", "rounds": 0, "nonce_mode": "none", "container_version": "none",
    }
    run_dir = run_artifacts.finalize_run(
        context, Path(__file__), rows, ["metric"], manifest_fields, "out\n", ""
    )
    assert validate_run(run_dir)["valid"] is True
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"] == "start"
    assert manifest["git_dirty"] is False
    # AF-023(5): untracked files must not make every run look dirty.
    assert ("status", "--porcelain=v1", "--untracked-files=no") in calls
    modules = manifest["imported_project_modules"]
    assert modules["scripts/run_artifacts.py"] == run_artifacts.sha256_file(
        run_artifacts.PROJECT_ROOT / "scripts/run_artifacts.py"
    )
    assert all(not name.startswith(("/", "..")) for name in modules)


def test_managed_run_dir_is_adopted(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "UNIT" / "UNIT_20260101T000000Z_deadbeef"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("EXPERIMENT_RUN_DIR", str(run_dir))
    context = run_artifacts.open_run("UNIT", {"a": 1}, tmp_path)
    assert context.managed and context.run_dir == run_dir
    assert (run_dir / "config.json").is_file()


def test_source_edited_during_run_fails_finalize(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EXPERIMENT_RUN_DIR", raising=False)
    monkeypatch.setattr(run_artifacts, "git_value", lambda *args: "")
    snapshots = iter([{"scripts/example.py": "a" * 64}, {"scripts/example.py": "b" * 64}])
    monkeypatch.setattr(run_artifacts, "imported_project_modules", lambda: next(snapshots))
    context = run_artifacts.open_run("UNIT", {"a": 1}, tmp_path)
    rows = [{"sample_id": "x", "metric": 1.0}]
    with pytest.raises(RuntimeError, match="changed during the run"):
        run_artifacts.finalize_run(context, Path(__file__), rows, ["metric"], {}, "", "")
