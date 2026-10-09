#!/usr/bin/env python3
"""E2.2 wrapper and regression check for the legacy S0/S1 pipelines.

The wrapper deliberately calls the predecessor's sampler rather than copying its
DDIM schedule.  The command-line verification compares that call with an
independently spelled-out direct call using the same public legacy parameters.
No raw key material is written to the result file.

Generation uses ``guidance_scale=-1`` exactly as the predecessor hard-codes it
(``cfg_image_sample_anonymization.py`` anonymize branch and the de-anonymize
step of ``ddim_sample_loop_anomaly_detection_with_deanonymization``): a single
model call conditioned on y=0 (healthy class) without CFG mixing.  This is not
unconditional generation; only the inversion uses ``null=True``.
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
LEGACY_GUIDANCE_SCALE = -1
LEGACY_PIPELINE = "legacy_anonymization:ddim_sample_loop_anonymization"
GENERATION_CONDITIONING = "y=0 healthy-class conditional, single model call, no CFG mixing (guidance_scale=-1)"


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


def legacy_model_kwargs(batch_size: int, device: torch.device) -> dict[str, Any]:
    """Model kwargs the predecessor passes to its anonymization sampler."""
    return {
        "y": torch.zeros(batch_size, device=device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }


@torch.inference_mode()
def _legacy_sampler(
    diffusion,
    model_fn,
    image: torch.Tensor,
    source_root: Path,
    scheme: str,
    mode: str,
    noise_level: int,
):
    anonymizer, key = build_legacy_components(source_root, scheme, image.device)
    return diffusion.ddim_sample_loop_anonymization(
        model_fn,
        tuple(image.shape),
        img=image,
        anonymizer=anonymizer,
        key=key,
        org=image,
        mode=mode,
        clip_denoised=True,
        model_kwargs=legacy_model_kwargs(image.shape[0], image.device),
        device=image.device,
        noise_level=noise_level,
        progress=False,
        guidance_scale=LEGACY_GUIDANCE_SCALE,
    )


def legacy_anonymize(
    diffusion,
    model_fn,
    x0: torch.Tensor,
    source_root: Path,
    scheme: str,
    noise_level: int = 500,
):
    """Call the predecessor's complete anonymization sampler unchanged.

    Returns ``(anonymous_image, x_T, input_image)`` as the legacy sampler does.
    """
    return _legacy_sampler(
        diffusion, model_fn, x0, source_root, scheme, "anonymize", noise_level
    )


def legacy_deanonymize(
    diffusion,
    model_fn,
    x_anonymous: torch.Tensor,
    source_root: Path,
    scheme: str,
    noise_level: int = 500,
):
    """Recover an image with the predecessor's de-anonymization step.

    This is step 1 of ``ddim_sample_loop_anomaly_detection_with_deanonymization``
    (inversion of the anonymous image, inverse key, generation).  The later
    lesion-localization step is out of scope and is not reproduced.
    Returns ``(recovered_image, inverted_anonymous_latent, input_image)``.
    """
    return _legacy_sampler(
        diffusion, model_fn, x_anonymous, source_root, scheme, "deanonymize", noise_level
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.detach().float() - b.detach().float())).cpu())


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _direct_call(
    diffusion, model_fn, image, source_root, scheme, mode, noise_level, guidance_scale
):
    """Independently spelled-out predecessor call used as the regression oracle."""
    anonymizer, key = build_legacy_components(source_root, scheme, image.device)
    kwargs = {
        "y": torch.zeros(image.shape[0], device=image.device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }
    return diffusion.ddim_sample_loop_anonymization(
        model_fn,
        tuple(image.shape),
        img=image,
        anonymizer=anonymizer,
        key=key,
        org=image,
        mode=mode,
        clip_denoised=True,
        model_kwargs=kwargs,
        device=image.device,
        noise_level=noise_level,
        progress=False,
        guidance_scale=guidance_scale,
    )


@torch.inference_mode()
def verify_scheme(
    diffusion,
    model_fn,
    x0: torch.Tensor,
    source_root: Path,
    scheme: str,
    noise_level: int,
) -> dict[str, Any]:
    """Compare wrapper output with explicit direct predecessor calls.

    Besides the guidance=-1 oracle, the anonymization is also compared with a
    direct call at guidance 0 (AF-021): mathematically eps=(1+0)eps_cond-0*eps_uncond
    equals eps_cond, so both must be bit-exact.
    """
    observed_output, observed_latent, observed_input = legacy_anonymize(
        diffusion, model_fn, x0, source_root, scheme, noise_level
    )
    expected_output, expected_latent, expected_input = _direct_call(
        diffusion, model_fn, x0, source_root, scheme, "anonymize",
        noise_level, LEGACY_GUIDANCE_SCALE,
    )
    guidance0_output, guidance0_latent, _ = _direct_call(
        diffusion, model_fn, x0, source_root, scheme, "anonymize", noise_level, 0.0
    )
    recovered, recovered_latent, _ = legacy_deanonymize(
        diffusion, model_fn, observed_output, source_root, scheme, noise_level
    )
    expected_recovered, expected_recovered_latent, _ = _direct_call(
        diffusion, model_fn, observed_output, source_root, scheme, "deanonymize",
        noise_level, LEGACY_GUIDANCE_SCALE,
    )

    anonymizer, key = build_legacy_components(source_root, scheme, x0.device)
    z_anon = anonymizer.anonymize_latent(observed_latent, key)
    z_roundtrip = anonymizer.deanonymize_latent(z_anon, key)
    result = {
        "scheme": scheme,
        "wrapper_vs_direct_output_max_abs": _max_abs(observed_output, expected_output),
        "wrapper_vs_direct_latent_max_abs": _max_abs(observed_latent, expected_latent),
        "wrapper_vs_direct_input_max_abs": _max_abs(observed_input, expected_input),
        "wrapper_vs_guidance0_output_max_abs": _max_abs(observed_output, guidance0_output),
        "wrapper_vs_guidance0_latent_max_abs": _max_abs(observed_latent, guidance0_latent),
        "deanonymize_wrapper_vs_direct_output_max_abs": _max_abs(recovered, expected_recovered),
        "deanonymize_wrapper_vs_direct_latent_max_abs": _max_abs(
            recovered_latent, expected_recovered_latent
        ),
        "transform_roundtrip_max_abs": _max_abs(observed_latent, z_roundtrip),
        "output_finite": bool(torch.isfinite(observed_output).all()),
        "latent_finite": bool(torch.isfinite(observed_latent).all()),
        "recovered_finite": bool(torch.isfinite(recovered).all()),
        "output_sha256": _tensor_sha256(observed_output),
        "latent_sha256": _tensor_sha256(observed_latent),
        "recovered_sha256": _tensor_sha256(recovered),
    }
    numeric = [value for value in result.values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError(f"non-finite E2.2 metric for {scheme}")
    exact_fields = [name for name in result if name.endswith("_max_abs")]
    if any(result[name] != 0.0 for name in exact_fields):
        raise RuntimeError(f"legacy regression mismatch for {scheme}: {result}")
    if not all(result[name] for name in ("output_finite", "latent_finite", "recovered_finite")):
        raise RuntimeError(f"NaN/Inf in legacy regression for {scheme}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--noise-level", type=int, default=500)
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
        )
        for scheme in ("S0", "S1")
    ]
    payload = {
        "schema_version": 2,
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
        "guidance_scale": LEGACY_GUIDANCE_SCALE,
        "generation_conditioning": GENERATION_CONDITIONING,
        "inversion_conditioning": "null=True (unconditional)",
        "pipeline": LEGACY_PIPELINE,
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
