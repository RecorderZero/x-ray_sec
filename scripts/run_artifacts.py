#!/usr/bin/env python3
"""Shared WORKFLOW 3.2 run-directory writer for standalone and managed runs.

Standalone: the script creates ``artifacts/runs/<task>/<run_id>/``, writes its
captured stdout/stderr and exit code, and validates immediately.
Managed (``EXPERIMENT_RUN_DIR`` set by ``managed_run.py``): the script writes
config, manifest, per-sample rows and summary into the managed directory;
``managed_run.py`` owns the logs and exit code and validates after exit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.artifact_schema import validate_run
    from scripts.managed_run import MANAGED_RUN_ENV
except ModuleNotFoundError:
    from artifact_schema import validate_run
    from managed_run import MANAGED_RUN_ENV


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunContext:
    task_id: str
    run_id: str
    run_dir: Path
    config_hash: str
    managed: bool
    started_at: str
    git_commit: str
    git_dirty: bool
    # SHA-256 of the project modules already imported when the run started.
    source_snapshot: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def git_dirty() -> bool:
    """Uncommitted changes to tracked files; untracked files (e.g. AUDIT.md) are ignored."""
    return bool(git_value("status", "--porcelain=v1", "--untracked-files=no"))


def imported_project_modules() -> dict[str, str]:
    """SHA-256 of every imported Python module that lives inside the repository.

    Covers the entry script, shared helpers (e.g. ``scripts/e2_legacy_wrapper.py``)
    and the predecessor's ``past/SourceCode`` modules actually loaded by the run.
    """
    modules: dict[str, str] = {}
    for module in list(sys.modules.values()):
        location = getattr(module, "__file__", None)
        # Some extension modules report a bare relative name (e.g. torch's "_ops.py").
        if not location or not Path(location).is_absolute():
            continue
        resolved = Path(location).resolve()
        if resolved.suffix != ".py" or PROJECT_ROOT not in resolved.parents or not resolved.is_file():
            continue
        modules[str(resolved.relative_to(PROJECT_ROOT))] = sha256_file(resolved)
    return dict(sorted(modules.items()))


def open_run(
    task_id: str,
    config: dict[str, Any],
    artifacts_root: Path,
    started_at: str | None = None,
) -> RunContext:
    """Create (standalone) or adopt (managed) the run directory and write config.json."""
    config_bytes = (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    started_at = started_at or datetime.now(timezone.utc).isoformat()
    managed_dir = os.environ.get(MANAGED_RUN_ENV)
    if managed_dir:
        run_dir = Path(managed_dir)
        run_id = run_dir.name
        if not run_id.startswith(f"{task_id}_"):
            raise RuntimeError(f"managed run {run_id} does not belong to task {task_id}")
        if (run_dir / "config.json").exists():
            raise RuntimeError(f"refusing to overwrite config.json in {run_dir}")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{task_id}_{timestamp}_{config_hash[:8]}"
        run_dir = artifacts_root / task_id / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_bytes(config_bytes)
    # Git state is captured when the run starts, not when it finishes, so that
    # commits made while a long run is in progress are not misattributed.
    return RunContext(
        task_id, run_id, run_dir, config_hash, bool(managed_dir), started_at,
        git_value("rev-parse", "HEAD"), git_dirty(), imported_project_modules(),
    )


def package_versions() -> dict[str, str]:
    import cv2
    import numpy
    import PIL
    import scipy
    import skimage
    import torch

    return {
        "numpy": numpy.__version__, "opencv": cv2.__version__,
        "pillow": PIL.__version__, "scipy": scipy.__version__,
        "scikit-image": skimage.__version__, "torch": torch.__version__,
    }


def finalize_run(
    context: RunContext,
    script_path: Path,
    per_sample_rows: list[dict[str, Any]],
    numeric_fields: list[str],
    manifest_fields: dict[str, Any],
    stdout_text: str = "",
    stderr_text: str = "",
) -> Path:
    """Write per-sample rows, summary and manifest; validate standalone runs."""
    import numpy as np
    import torch

    run_dir = context.run_dir
    # A source file edited while the run was in progress would otherwise be
    # recorded with the hash of the edited file, not of the code that ran.
    current_modules = imported_project_modules()
    modified = sorted(
        path for path, digest in context.source_snapshot.items()
        if current_modules.get(path) != digest
    )
    if modified:
        raise RuntimeError(f"project source files changed during the run: {modified}")
    with (run_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_sample_rows)
    summary = {
        "schema_version": 2,
        "task_id": context.task_id,
        "run_id": context.run_id,
        "sample_count": len(per_sample_rows),
        "numeric_means": {
            field: float(np.mean([float(row[field]) for row in per_sample_rows]))
            for field in numeric_fields
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "task_id": context.task_id,
        "run_id": context.run_id,
        "config_hash": context.config_hash,
        "git_commit": context.git_commit,
        "git_dirty": context.git_dirty,
        "expected_samples": len(per_sample_rows),
        "sample_ids": [str(row["sample_id"]) for row in per_sample_rows],
        "command": " ".join(sys.argv),
        "started_at": context.started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": 0,
        "script_sha256": sha256_file(script_path),
        "imported_project_modules": current_modules,
        "source_snapshot_at_start": sorted(context.source_snapshot),
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "dtype": "float32",
        "packages": package_versions(),
        "numeric_fields": numeric_fields,
        "managed_run": context.managed,
        "failure_reason": "",
        "skip_reason": "",
        **manifest_fields,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if context.managed:
        print(f"wrote run artifacts to managed run {run_dir}; managed_run validates after exit")
        return run_dir
    (run_dir / "stdout.log").write_text(stdout_text, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr_text, encoding="utf-8")
    (run_dir / "exit_code").write_text("0\n", encoding="utf-8")
    validation = validate_run(run_dir)
    if not validation["valid"]:
        raise RuntimeError(f"invalid run artifacts: {validation['errors']}")
    print(f"validated run artifacts: {run_dir}")
    return run_dir
