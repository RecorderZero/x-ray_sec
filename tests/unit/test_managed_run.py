from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/managed_run.py"


def execute(
    tmp_path: Path, child_code: str, *runner_flags: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--task-id",
            "CONTROL",
            "--artifacts-root",
            str(tmp_path),
            "--heartbeat-seconds",
            "0.01",
            *runner_flags,
            "--",
            sys.executable,
            "-c",
            child_code,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    run_dirs = list((tmp_path / "CONTROL").glob("CONTROL_*"))
    assert len(run_dirs) == 1
    return result, run_dirs[0]


def test_managed_run_captures_real_output_and_success(tmp_path: Path) -> None:
    result, run_dir = execute(tmp_path, "print('actual stdout')")
    assert result.returncode == 0
    assert (run_dir / "stdout.log").read_text(encoding="utf-8") == "actual stdout\n"
    assert (run_dir / "stderr.log").read_text(encoding="utf-8") == ""
    assert (run_dir / "exit_code").read_text(encoding="utf-8").strip() == "0"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "succeeded"
    assert status["exit_code"] == 0
    for name in ("runner_config.json", "command.txt", "pid", "heartbeat", "runner_manifest.json"):
        assert (run_dir / name).is_file()


def test_managed_run_preserves_traceback_and_failure_code(tmp_path: Path) -> None:
    result, run_dir = execute(tmp_path, "raise RuntimeError('intentional control failure')")
    assert result.returncode != 0
    stderr = (run_dir / "stderr.log").read_text(encoding="utf-8")
    assert "Traceback" in stderr
    assert "intentional control failure" in stderr
    exit_code = int((run_dir / "exit_code").read_text(encoding="utf-8"))
    assert exit_code != 0
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["exit_code"] == exit_code
    assert "intentional control failure" in status["last_error"]


SYNTHETIC_CHILD = """
import os, sys
from pathlib import Path
sys.path.insert(0, {root!r})
from scripts.artifact_schema import write_synthetic_run
write_synthetic_run(
    Path(os.environ["EXPERIMENT_RUN_DIR"]),
    [{{"sample_id": "a", "metric": 1.0}}, {{"sample_id": "b", "metric": {second}}}],
    managed=True,
)
print("child wrote artifacts")
"""


def test_validated_child_artifacts_succeed(tmp_path: Path) -> None:
    child = SYNTHETIC_CHILD.format(root=str(ROOT), second="3.0")
    result, run_dir = execute(tmp_path, child, "--validate-artifacts")
    assert result.returncode == 0, result.stderr
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "succeeded"
    assert status["artifact_validation"]["valid"] is True
    assert (run_dir / "stdout.log").read_text(encoding="utf-8") == "child wrote artifacts\n"


def test_invalid_child_artifacts_mark_run_failed(tmp_path: Path) -> None:
    child = SYNTHETIC_CHILD.format(root=str(ROOT), second="float('nan')")
    result, run_dir = execute(tmp_path, child, "--validate-artifacts")
    assert result.returncode == 3
    assert (run_dir / "exit_code").read_text(encoding="utf-8").strip() == "0"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert "metric contains NaN/Inf" in status["last_error"]


def test_missing_child_artifacts_mark_run_failed(tmp_path: Path) -> None:
    result, run_dir = execute(tmp_path, "print('no artifacts')", "--validate-artifacts")
    assert result.returncode == 3
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert "missing manifest.json" in status["last_error"]
