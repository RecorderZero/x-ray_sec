from pathlib import Path

from scripts.artifact_schema import validate_run, write_synthetic_run


def test_synthetic_run_is_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_synthetic_run(
        run_dir,
        [{"sample_id": "a", "metric": 1.0}, {"sample_id": "b", "metric": 3.0}],
    )
    result = validate_run(run_dir)
    assert result == {"valid": True, "errors": [], "sample_count": 2}


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
