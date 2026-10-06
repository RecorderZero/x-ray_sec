#!/usr/bin/env python3
"""Create the reproducible E1.1 environment and checkpoint inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import types
from pathlib import Path

import torch
import cv2
import PIL
import pytest


MODEL_CONFIG = {
    "image_size": 256,
    "in_channels": 1,
    "num_channels": 128,
    "num_classes": 2,
    "class_cond": True,
    "num_res_blocks": 2,
    "num_heads": 1,
    "learn_sigma": True,
    "use_scale_shift_norm": False,
    "attention_resolutions": "16",
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "rescale_learned_sigmas": False,
    "rescale_timesteps": False,
    "timestep_respacing": "ddim1000",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument(
        "--environment-output",
        type=Path,
        default=Path("artifacts/environment_baseline.txt"),
    )
    parser.add_argument(
        "--inventory-output", type=Path, default=Path("model_inventory.csv")
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(source_root))

    # Legacy modules create Visdom clients at import time. Inventory is headless.
    visdom_module = types.ModuleType("visdom")
    visdom_module.Visdom = lambda *args, **kwargs: types.SimpleNamespace()
    sys.modules["visdom"] = visdom_module

    from guided_diffusion.script_util import (  # pylint: disable=import-error
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    config = model_and_diffusion_defaults()
    config.update(MODEL_CONFIG)
    model, diffusion = create_model_and_diffusion(**config)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    incompat = model.load_state_dict(state_dict, strict=False)
    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(state_dict, strict=True)

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory if cuda_available else 0
    )
    environment = {
        "schema_version": 1,
        "timestamp_timezone": "Asia/Taipei",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch": torch.__version__,
        "pillow": PIL.__version__,
        "pytest": pytest.__version__,
        "opencv": cv2.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_memory,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain=v1"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "strict_load": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "diffusion_timesteps": int(diffusion.num_timesteps),
        "model_config": MODEL_CONFIG,
    }
    args.environment_output.parent.mkdir(parents=True, exist_ok=True)
    args.environment_output.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
    with args.inventory_output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "schema_version",
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "strict_load",
            "missing_key_count",
            "unexpected_key_count",
            "parameter_count",
            "diffusion_timesteps",
            "model_config_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "schema_version": 1,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": environment["checkpoint_sha256"],
                "checkpoint_bytes": environment["checkpoint_bytes"],
                "strict_load": True,
                "missing_key_count": len(missing),
                "unexpected_key_count": len(unexpected),
                "parameter_count": environment["model_parameter_count"],
                "diffusion_timesteps": environment["diffusion_timesteps"],
                "model_config_json": json.dumps(MODEL_CONFIG, sort_keys=True),
            }
        )


if __name__ == "__main__":
    main()
