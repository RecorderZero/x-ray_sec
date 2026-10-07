import json
from pathlib import Path

from scripts.artifact_schema import validate_run, write_synthetic_run


def make_run(path: Path) -> None:
    write_synthetic_run(
        path,
        [{"sample_id": "a", "metric": 1.0}, {"sample_id": "b", "metric": 3.0}],
    )


def test_synthetic_run_is_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    assert validate_run(run_dir) == {"valid": True, "errors": [], "sample_count": 2}


def test_duplicate_and_nonfinite_rows_fail(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_synthetic_run(
        run_dir,
        [{"sample_id": "a", "metric": 1.0}, {"sample_id": "a", "metric": float("nan")}],
    )
    result = validate_run(run_dir)
    assert result["valid"] is False
    assert "duplicate sample_id" in result["errors"]
    assert "metric contains NaN/Inf" in result["errors"]


def test_missing_manifest_field_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["script_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_run(run_dir)
    assert result["valid"] is False
    assert "manifest missing keys: ['script_sha256']" in result["errors"]


def test_tampered_summary_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["numeric_means"]["metric"] = 99
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = validate_run(run_dir)
    assert result["valid"] is False
    assert "summary metric mean is inconsistent with per_sample.csv" in result["errors"]
