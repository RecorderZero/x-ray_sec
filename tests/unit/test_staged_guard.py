from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)

def initialize_repo(path: Path) -> Path:
    run("git", "init", "-q", cwd=path)
    guard = path / "check.sh"
    shutil.copy2(ROOT / "scripts/check_staged_files.sh", guard)
    return guard

def test_guard_rejects_checkpoint_suffix(tmp_path: Path) -> None:
    guard = initialize_repo(tmp_path)
    (tmp_path / "a.pt").write_bytes(b"not really a checkpoint")
    assert run("git", "add", "-f", "a.pt", cwd=tmp_path).returncode == 0
    result = run("bash", str(guard), cwd=tmp_path)
    assert result.returncode != 0
    assert "forbidden" in result.stderr

def test_guard_rejects_private_key_content(tmp_path: Path) -> None:
    guard = initialize_repo(tmp_path)
    (tmp_path / "innocent.txt").write_text(
        "-----BEGIN OPENSSH PRIVATE KEY-----\\nredacted\\n", encoding="utf-8"
    )
    assert run("git", "add", "innocent.txt", cwd=tmp_path).returncode == 0
    result = run("bash", str(guard), cwd=tmp_path)
    assert result.returncode != 0
    assert "private-key material" in result.stderr
