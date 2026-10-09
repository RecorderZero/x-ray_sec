#!/usr/bin/env python3
"""Run a command with durable logs, status, heartbeat, and exclusive locks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.artifact_schema import validate_run
except ModuleNotFoundError:
    from artifact_schema import validate_run

MANAGED_RUN_ENV = "EXPERIMENT_RUN_DIR"
VALIDATION_FAILURE_EXIT_CODE = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"active run lock exists: {path}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"holder_pid={os.getpid()}\n")
    handle.flush()
    return handle


def tail_text(path: Path, lines: int = 20) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--validate-artifacts",
        action="store_true",
        help="after a zero exit, require the child's manifest/per-sample/summary "
        "in the run directory and validate them before marking success",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("a command is required after --")
    if args.heartbeat_seconds <= 0:
        raise SystemExit("--heartbeat-seconds must be positive")

    config = {
        "schema_version": 1,
        "task_id": args.task_id,
        "command": command,
        "cwd": str(args.cwd.resolve()),
        "gpu_id": args.gpu_id,
        "heartbeat_seconds": args.heartbeat_seconds,
        "validate_artifacts": args.validate_artifacts,
        "resumable": False,
    }
    config_bytes = (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{args.task_id}_{timestamp}_{config_hash[:8]}"
    lock_root = args.artifacts_root / ".locks"
    task_lock = acquire_lock(lock_root / f"task_{args.task_id}_{config_hash}.lock")
    gpu_lock = acquire_lock(lock_root / f"gpu_{args.gpu_id}.lock")
    # Keep both handles referenced until main exits so advisory locks remain held.
    _held_locks = (task_lock, gpu_lock)

    run_dir = args.artifacts_root / args.task_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # config.json is reserved for the child's resolved experiment config.
    (run_dir / "runner_config.json").write_bytes(config_bytes)
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    os.environ[MANAGED_RUN_ENV] = str(run_dir.resolve())
    started_at = utc_now()
    status = {
        "schema_version": 1,
        "task_id": args.task_id,
        "run_id": run_id,
        "state": "queued",
        "host": os.uname().nodename,
        "gpu": args.gpu_id,
        "start_time": started_at,
        "update_time": started_at,
        "end_time": None,
        "pid": None,
        "progress": {"completed": 0, "total": None},
        "last_checkpoint": None,
        "exit_code": None,
        "last_error": "",
        "resumable": False,
    }
    atomic_json(run_dir / "status.json", status)

    process = None
    interrupted = False

    def stop_child(signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True
        if process is not None and process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGINT, stop_child)
    signal.signal(signal.SIGTERM, stop_child)
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout_handle, (
        run_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
            env=os.environ.copy(),
        )
        (run_dir / "pid").write_text(f"{process.pid}\n", encoding="utf-8")
        status.update(state="running", pid=process.pid, update_time=utc_now())
        atomic_json(run_dir / "status.json", status)
        last_heartbeat = 0.0
        while process.poll() is None:
            now = time.monotonic()
            if now - last_heartbeat >= args.heartbeat_seconds:
                heartbeat = utc_now()
                (run_dir / "heartbeat").write_text(heartbeat + "\n", encoding="utf-8")
                status["update_time"] = heartbeat
                atomic_json(run_dir / "status.json", status)
                last_heartbeat = now
            time.sleep(min(1.0, args.heartbeat_seconds))
        return_code = int(process.returncode)

    ended_at = utc_now()
    (run_dir / "exit_code").write_text(f"{return_code}\n", encoding="utf-8")
    final_state = "interrupted" if interrupted else ("succeeded" if return_code == 0 else "failed")
    last_error = tail_text(run_dir / "stderr.log") if return_code else ""
    validation = None
    if args.validate_artifacts and final_state == "succeeded":
        # Loop spec 4.4: verify sample counts, IDs, finiteness and summary
        # consistency before the run may be recorded as succeeded.
        validation = validate_run(run_dir)
        if not validation["valid"]:
            final_state = "failed"
            last_error = "artifact validation failed: " + "; ".join(validation["errors"])
    status.update(
        state=final_state,
        update_time=ended_at,
        end_time=ended_at,
        exit_code=return_code,
        last_error=last_error,
        artifact_validation=validation,
    )
    atomic_json(run_dir / "status.json", status)
    runner_manifest = {
        **config,
        "run_id": run_id,
        "config_hash": config_hash,
        "started_at": started_at,
        "finished_at": ended_at,
        "exit_code": return_code,
        "state": final_state,
        "artifact_validation": validation,
    }
    atomic_json(run_dir / "runner_manifest.json", runner_manifest)
    print(run_dir)
    if return_code == 0 and final_state == "failed":
        raise SystemExit(VALIDATION_FAILURE_EXIT_CODE)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
