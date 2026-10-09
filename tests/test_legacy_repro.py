"""E2.2 regression tests against the predecessor's S0/S1 implementation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_legacy_wrapper import (
    LEGACY_GUIDANCE_SCALE,
    build_legacy_components,
    legacy_anonymize,
    legacy_deanonymize,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "past/SourceCode"
CHECKPOINT = SOURCE / "results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt"
GPU_NOISE_LEVEL = 16
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not CHECKPOINT.is_file(),
    reason="requires CUDA and legacy checkpoint",
)


@pytest.fixture(scope="module")
def runtime():
    _, diffusion, model_fn = create_runtime(SOURCE, CHECKPOINT, torch.device("cuda:0"))
    row = next(csv.DictReader((ROOT / "splits/dev_v1.1.csv").open(encoding="utf-8")))
    x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).cuda()
    return diffusion, model_fn, x0


def direct_legacy_call(diffusion, model_fn, image, scheme, mode, guidance_scale):
    anonymizer, key = build_legacy_components(SOURCE, scheme, image.device)
    kwargs = {
        "y": torch.zeros(1, device=image.device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }
    with torch.inference_mode():
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
            noise_level=GPU_NOISE_LEVEL,
            progress=False,
            guidance_scale=guidance_scale,
        )


def assert_all_equal(observed, expected) -> None:
    assert all(torch.equal(actual, oracle) for actual, oracle in zip(observed, expected))


def test_wrapper_uses_predecessor_hardcoded_guidance() -> None:
    assert LEGACY_GUIDANCE_SCALE == -1


@pytest.mark.parametrize("scheme", ["S0", "S1"])
def test_legacy_transform_is_exactly_invertible(scheme: str) -> None:
    anonymizer, key = build_legacy_components(SOURCE, scheme, torch.device("cpu"))
    z = torch.linspace(-2.0, 2.0, 256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256)
    encrypted = anonymizer.anonymize_latent(z, key)
    recovered = anonymizer.deanonymize_latent(encrypted, key)
    assert torch.equal(recovered, z)


@requires_gpu
@pytest.mark.parametrize("scheme", ["S0", "S1"])
def test_wrapper_is_bit_exact_with_legacy_anonymization_sampler(runtime, scheme: str) -> None:
    diffusion, model_fn, x0 = runtime
    observed = legacy_anonymize(diffusion, model_fn, x0, SOURCE, scheme, GPU_NOISE_LEVEL)
    expected = direct_legacy_call(diffusion, model_fn, x0, scheme, "anonymize", -1)
    assert_all_equal(observed, expected)


@requires_gpu
@pytest.mark.parametrize("scheme", ["S0", "S1"])
def test_guidance_minus_one_equals_guidance_zero(runtime, scheme: str) -> None:
    """AF-021: -1 (single conditional call) and 0 (CFG with w=0) are bit-exact."""
    diffusion, model_fn, x0 = runtime
    observed = legacy_anonymize(diffusion, model_fn, x0, SOURCE, scheme, GPU_NOISE_LEVEL)
    guidance_zero = direct_legacy_call(diffusion, model_fn, x0, scheme, "anonymize", 0.0)
    assert_all_equal(observed, guidance_zero)


@requires_gpu
@pytest.mark.parametrize("scheme", ["S0", "S1"])
def test_deanonymize_wrapper_is_bit_exact_with_legacy_sampler(runtime, scheme: str) -> None:
    diffusion, model_fn, x0 = runtime
    anonymous, _, _ = legacy_anonymize(diffusion, model_fn, x0, SOURCE, scheme, GPU_NOISE_LEVEL)
    observed = legacy_deanonymize(
        diffusion, model_fn, anonymous, SOURCE, scheme, GPU_NOISE_LEVEL
    )
    expected = direct_legacy_call(diffusion, model_fn, anonymous, scheme, "deanonymize", -1)
    assert_all_equal(observed, expected)
