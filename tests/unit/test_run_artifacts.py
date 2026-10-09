"""Shared run-directory writer: standalone validation and start-time git state."""

from __future__ import annotations

from pathlib import Path

from scripts import run_artifacts
from scripts.artifact_schema import validate_run


def test_standalone_run_is_valid_and_records_start_commit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EXPERIMENT_RUN_DIR", raising=False)
    monkeypatch.setattr(run_artifacts, "git_value", lambda *args: "start" if args[0] == "rev-parse" else "")
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
    manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert '"git_commit": "start"' in manifest
    assert '"git_dirty": false' in manifest


def test_managed_run_dir_is_adopted(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "UNIT" / "UNIT_20260101T000000Z_deadbeef"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("EXPERIMENT_RUN_DIR", str(run_dir))
    context = run_artifacts.open_run("UNIT", {"a": 1}, tmp_path)
    assert context.managed and context.run_dir == run_dir
    assert (run_dir / "config.json").is_file()
