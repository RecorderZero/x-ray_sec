#!/usr/bin/env python3
"""Create and validate the run artifacts required by WORKFLOW section 3.2."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


TERMINAL_STATES = {"succeeded", "failed", "interrupted"}
REQUIRED_MANIFEST = {
    "schema_version", "task_id", "run_id", "config_hash", "git_commit",
    "dataset_split_hash", "checkpoint_hash", "expected_samples", "command",
    "started_at", "finished_at", "exit_code", "script_sha256", "git_dirty",
    "sample_ids", "python", "pytorch", "cuda", "gpu", "dtype", "seed_list",
    "steps", "packages", "scheme", "rounds", "nonce_mode",
    "container_version", "numeric_fields", "failure_reason", "skip_reason",
}
REQUIRED_FILES = (
    "manifest.json", "config.json", "per_sample.csv", "summary.json",
    "stdout.log", "stderr.log", "exit_code",
)
REQUIRED_PACKAGES = {"numpy", "scipy", "scikit-image", "pillow", "opencv", "torch"}


def write_synthetic_run(run_dir: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a tiny deterministic run fixture for validator tests."""
    materialized = list(rows)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 2,
        "task_id": "E1.5",
        "run_id": "E1.5_synthetic_test",
        "config_hash": "0" * 64,
        "git_commit": "0" * 40,
        "dataset_split_hash": "1" * 64,
        "checkpoint_hash": "2" * 64,
        "expected_samples": len(materialized),
        "command": "synthetic",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "exit_code": 0,
        "script_sha256": "3" * 64,
        "git_dirty": False,
        "sample_ids": [row["sample_id"] for row in materialized],
        "python": "test", "pytorch": "test", "cuda": "none", "gpu": "none",
        "dtype": "float32", "seed_list": [1911], "steps": 1,
        "packages": {name: "test" for name in sorted(REQUIRED_PACKAGES)},
        "scheme": "synthetic", "rounds": 0, "nonce_mode": "none",
        "container_version": "none", "numeric_fields": ["metric"],
        "failure_reason": "", "skip_reason": "",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "stdout.log").write_text("synthetic run\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    (run_dir / "exit_code").write_text("0\n", encoding="utf-8")
    with (run_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "metric"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
    mean_metric = sum(float(row["metric"]) for row in materialized) / len(materialized)
    (run_dir / "summary.json").write_text(
        json.dumps({
            "schema_version": 2,
            "sample_count": len(materialized),
            "numeric_means": {"metric": mean_metric},
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_run(run_dir: Path) -> dict[str, Any]:
    """Validate presence, uniqueness, finiteness, and declared summary means."""
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (run_dir / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return {"valid": False, "errors": errors}

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_MANIFEST - set(manifest))
    if missing:
        errors.append(f"manifest missing keys: {missing}")

    try:
        recorded_exit_code = int((run_dir / "exit_code").read_text(encoding="utf-8").strip())
    except ValueError:
        errors.append("exit_code file is not an integer")
    else:
        if recorded_exit_code != manifest.get("exit_code"):
            errors.append("exit_code file differs from manifest exit_code")

    packages = manifest.get("packages", {})
    missing_packages = sorted(REQUIRED_PACKAGES - set(packages))
    if missing_packages:
        errors.append(f"manifest packages missing: {missing_packages}")

    with (run_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    ids = [row.get("sample_id", "") for row in rows]
    if not ids or any(not sample_id for sample_id in ids):
        errors.append("empty or missing sample_id")
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample_id")
    if len(rows) != manifest.get("expected_samples"):
        errors.append("row count differs from manifest expected_samples")
    if ids != manifest.get("sample_ids"):
        errors.append("sample IDs differ from manifest sample_ids")
    if len(rows) != summary.get("sample_count"):
        errors.append("row count differs from summary sample_count")

    numeric_fields = manifest.get("numeric_fields")
    if not isinstance(numeric_fields, list) or not numeric_fields:
        errors.append("numeric_fields must be a non-empty list")
        numeric_fields = []
    for field in numeric_fields:
        if field not in (fieldnames or []):
            errors.append(f"numeric field absent from per_sample.csv header: {field}")
            continue
        values: list[float] = []
        for row in rows:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"missing or non-numeric {field}")
                continue
            if not math.isfinite(value):
                errors.append(f"{field} contains NaN/Inf")
            values.append(value)
        if values:
            observed = sum(values) / len(values)
            expected = summary.get("numeric_means", {}).get(field, math.nan)
            if not math.isclose(observed, float(expected), rel_tol=1e-9, abs_tol=1e-12):
                errors.append(f"summary {field} mean is inconsistent with per_sample.csv")
    return {"valid": not errors, "errors": errors, "sample_count": len(rows)}
