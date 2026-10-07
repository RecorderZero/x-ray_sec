#!/usr/bin/env python3
"""E2.2 wrapper and regression check for the legacy S0/S1 pipelines.

The wrapper deliberately calls the predecessor's sampler rather than copying its
DDIM schedule.  The command-line verification compares that call with an
independently spelled-out direct call using the same public legacy parameters.
No raw key material is written to the result file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, git_value, preprocess, sha256_file


LEGACY_PASSWORD = "Ki@13579"
LEGACY_KEY_SEED = 42
KEY_SHAPE = (1, 256, 256)


def _load_crypto(source_root: Path):
    source = str(source_root.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from guided_diffusion.anonymization import (
        DiffusionAnonymizer,
        RademacherKey,
        SignedPermutationKey,
    )

    return DiffusionAnonymizer, RademacherKey, SignedPermutationKey


def build_legacy_components(source_root: Path, scheme: str, device: torch.device):
    """Construct the exact full-mask key objects used by the predecessor."""
    DiffusionAnonymizer, RademacherKey, SignedPermutationKey = _load_crypto(source_root)
    if scheme not in {"S0", "S1"}:
        raise ValueError(f"unknown legacy scheme: {scheme}")
    key_class = RademacherKey if scheme == "S0" else SignedPermutationKey
    # Both arguments were present in the predecessor shell command.  In the
    # legacy constructors password takes precedence over seed for key material.
    key = key_class(
        shape=KEY_SHAPE,
        seed=LEGACY_KEY_SEED,
        password=LEGACY_PASSWORD,
    ).to(device)
    anonymizer = DiffusionAnonymizer(
        latent_shape=KEY_SHAPE,
        mask_type="full",
        margin=0,
        use_permutation=scheme == "S1",
    ).to(device)
    return anonymizer, key


@torch.inference_mode()
def legacy_anonymize(
    diffusion,
    model_fn,
    x0: torch.Tensor,
    source_root: Path,
    scheme: str,
    noise_level: int = 500,
    guidance_scale: float = 0.0,
):
    """Call the predecessor's complete anonymization sampler unchanged."""
    anonymizer, key = build_legacy_components(source_root, scheme, x0.device)
    kwargs = {
        "y": torch.zeros(x0.shape[0], device=x0.device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }
    return diffusion.ddim_sample_loop_anonymization(
        model_fn,
        tuple(x0.shape),
        img=x0,
        anonymizer=anonymizer,
        key=key,
        org=x0,
        mode="anonymize",
        clip_denoised=True,
        model_kwargs=kwargs,
        device=x0.device,
        noise_level=noise_level,
        progress=False,
        guidance_scale=guidance_scale,
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.detach().float() - b.detach().float())).cpu())


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.inference_mode()
def verify_scheme(
    diffusion,
    model_fn,
    x0: torch.Tensor,
    source_root: Path,
    scheme: str,
    noise_level: int,
    guidance_scale: float,
) -> dict[str, Any]:
    """Compare wrapper output with an explicit direct predecessor call."""
    observed_output, observed_latent, observed_input = legacy_anonymize(
        diffusion, model_fn, x0, source_root, scheme, noise_level, guidance_scale
    )

    anonymizer, key = build_legacy_components(source_root, scheme, x0.device)
    kwargs = {
        "y": torch.zeros(x0.shape[0], device=x0.device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }
    expected_output, expected_latent, expected_input = (
        diffusion.ddim_sample_loop_anonymization(
            model_fn,
            tuple(x0.shape),
            img=x0,
            anonymizer=anonymizer,
            key=key,
            org=x0,
            mode="anonymize",
            clip_denoised=True,
            model_kwargs=kwargs,
            device=x0.device,
            noise_level=noise_level,
            progress=False,
            guidance_scale=guidance_scale,
        )
    )

    z_anon = anonymizer.anonymize_latent(observed_latent, key)
    z_roundtrip = anonymizer.deanonymize_latent(z_anon, key)
    result = {
        "scheme": scheme,
        "wrapper_vs_direct_output_max_abs": _max_abs(observed_output, expected_output),
        "wrapper_vs_direct_latent_max_abs": _max_abs(observed_latent, expected_latent),
        "wrapper_vs_direct_input_max_abs": _max_abs(observed_input, expected_input),
        "transform_roundtrip_max_abs": _max_abs(observed_latent, z_roundtrip),
        "output_finite": bool(torch.isfinite(observed_output).all()),
        "latent_finite": bool(torch.isfinite(observed_latent).all()),
        "output_sha256": _tensor_sha256(observed_output),
        "latent_sha256": _tensor_sha256(observed_latent),
    }
    numeric = [value for value in result.values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError(f"non-finite E2.2 metric for {scheme}")
    if any(result[name] != 0.0 for name in (
        "wrapper_vs_direct_output_max_abs",
        "wrapper_vs_direct_latent_max_abs",
        "wrapper_vs_direct_input_max_abs",
        "transform_roundtrip_max_abs",
    )):
        raise RuntimeError(f"legacy regression mismatch for {scheme}: {result}")
    if not result["output_finite"] or not result["latent_finite"]:
        raise RuntimeError(f"NaN/Inf in legacy regression for {scheme}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("results/E2.2.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for E2.2 full-pipeline verification")

    import csv

    with args.split.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[args.sample_index]
    x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).cuda()
    _, diffusion, model_fn = create_runtime(
        args.source_root, args.checkpoint, torch.device("cuda:0")
    )
    results = [
        verify_scheme(
            diffusion,
            model_fn,
            x0,
            args.source_root,
            scheme,
            args.noise_level,
            args.guidance_scale,
        )
        for scheme in ("S0", "S1")
    ]
    payload = {
        "schema_version": 1,
        "task_id": "E2.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": row["sample_id"],
        "split_sha256": row["split_sha256"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "script_sha256": sha256_file(Path(__file__)),
        "source_sampler_sha256": sha256_file(args.source_root / "guided_diffusion" / "gaussian_diffusion.py"),
        "command": " ".join(sys.argv),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain=v1")),
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "float32",
        "legacy_key_seed": LEGACY_KEY_SEED,
        "noise_level": args.noise_level,
        "guidance_scale": args.guidance_scale,
        "legacy_parameter_profile": "public predecessor password plus seed 42; password precedence",
        "raw_key_material_recorded": False,
        "schemes": results,
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
