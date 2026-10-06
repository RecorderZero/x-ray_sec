#!/usr/bin/env python3
"""Create and validate the minimum run artifacts required by the loop spec."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


TERMINAL_STATES = {"succeeded", "failed", "interrupted"}
REQUIRED_MANIFEST = {
    "schema_version",
    "task_id",
    "run_id",
    "config_hash",
    "git_commit",
    "dataset_split_hash",
    "checkpoint_hash",
    "expected_samples",
}


def write_synthetic_run(run_dir: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a tiny deterministic run fixture for validator tests/smoke tests."""
    materialized = list(rows)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "task_id": "E1.5",
        "run_id": "E1.5_synthetic_test",
        "config_hash": "0" * 64,
        "git_commit": "0" * 40,
        "dataset_split_hash": "1" * 64,
        "checkpoint_hash": "2" * 64,
        "expected_samples": len(materialized),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "metric"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
    mean_metric = sum(float(row["metric"]) for row in materialized) / len(materialized)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"schema_version": 1, "sample_count": len(materialized), "metric_mean": mean_metric},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_run(run_dir: Path) -> dict[str, Any]:
    """Validate presence, uniqueness, finiteness, and summary consistency."""
    errors: list[str] = []
    for name in ("manifest.json", "per_sample.csv", "summary.json"):
        if not (run_dir / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return {"valid": False, "errors": errors}

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_MANIFEST - set(manifest))
    if missing:
        errors.append(f"manifest missing keys: {missing}")

    with (run_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("sample_id", "") for row in rows]
    if not ids or any(not sample_id for sample_id in ids):
        errors.append("empty or missing sample_id")
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample_id")
    if len(rows) != manifest.get("expected_samples"):
        errors.append("row count differs from manifest expected_samples")
    if len(rows) != summary.get("sample_count"):
        errors.append("row count differs from summary sample_count")

    metric_values = []
    for row in rows:
        try:
            value = float(row["metric"])
        except (KeyError, TypeError, ValueError):
            errors.append("missing or non-numeric metric")
            continue
        if not math.isfinite(value):
            errors.append("metric contains NaN/Inf")
        metric_values.append(value)
    if metric_values:
        observed_mean = sum(metric_values) / len(metric_values)
        if not math.isclose(observed_mean, float(summary.get("metric_mean", math.nan)), rel_tol=1e-12, abs_tol=1e-12):
            errors.append("summary metric_mean is inconsistent with per_sample.csv")
    return {"valid": not errors, "errors": errors, "sample_count": len(rows)}
