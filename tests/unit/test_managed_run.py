from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/managed_run.py"


def execute(tmp_path: Path, child_code: str) -> tuple[subprocess.CompletedProcess[str], Path]:
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
    for name in ("config.json", "command.txt", "pid", "heartbeat", "runner_manifest.json"):
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
