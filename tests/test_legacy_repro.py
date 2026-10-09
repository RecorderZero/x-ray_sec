"""E2.2 regression tests against the predecessor's S0/S1 implementation.

P (AF-017) is the identity-key control on the same anonymization pipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_legacy_wrapper import (
    KEY_SHAPE,
    LEGACY_GUIDANCE_SCALE,
    apply_legacy_key,
    build_legacy_components,
    legacy_anonymize,
    legacy_deanonymize,
    legacy_generate,
    legacy_invert,
    legacy_model_kwargs,
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


@pytest.mark.parametrize("scheme", ["P", "S0", "S1"])
def test_legacy_transform_is_exactly_invertible(scheme: str) -> None:
    anonymizer, key = build_legacy_components(SOURCE, scheme, torch.device("cpu"))
    z = torch.linspace(-2.0, 2.0, 256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256)
    encrypted = anonymizer.anonymize_latent(z, key)
    recovered = anonymizer.deanonymize_latent(encrypted, key)
    assert torch.equal(recovered, z)


@requires_gpu
@pytest.mark.parametrize("scheme", ["P", "S0", "S1"])
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
@pytest.mark.parametrize("scheme", ["P", "S0", "S1"])
def test_deanonymize_wrapper_is_bit_exact_with_legacy_sampler(runtime, scheme: str) -> None:
    diffusion, model_fn, x0 = runtime
    anonymous, _, _ = legacy_anonymize(diffusion, model_fn, x0, SOURCE, scheme, GPU_NOISE_LEVEL)
    observed = legacy_deanonymize(
        diffusion, model_fn, anonymous, SOURCE, scheme, GPU_NOISE_LEVEL
    )
    expected = direct_legacy_call(diffusion, model_fn, anonymous, scheme, "deanonymize", -1)
    assert_all_equal(observed, expected)


def test_identity_key_leaves_latent_unchanged_in_legacy_expression() -> None:
    """AF-017: the sampler's inline full-mask Rademacher step is the identity for P."""
    import sys

    sys.path.insert(0, str(SOURCE))
    from guided_diffusion.anonymization import AnonymizationMask

    anonymizer, key = build_legacy_components(SOURCE, "P", torch.device("cpu"))
    assert torch.equal(key.key, torch.ones(KEY_SHAPE))
    z = torch.randn((1, *KEY_SHAPE), generator=torch.Generator().manual_seed(1911))
    mask = AnonymizationMask(shape=KEY_SHAPE, mask_type=anonymizer.mask_type, margin=anonymizer.margin).mask
    processed = mask * (key.key * z) + (1 - mask) * z
    assert torch.equal(processed, z)
    assert torch.equal(anonymizer.anonymize_latent(z, key), z)


@requires_gpu
def test_p_identity_key_matches_keyless_legacy_passes(runtime) -> None:
    """AF-017: P output equals legacy forward then backward with no key at all."""
    diffusion, model_fn, x0 = runtime
    output, latent, _ = legacy_anonymize(diffusion, model_fn, x0, SOURCE, "P", GPU_NOISE_LEVEL)
    kwargs = legacy_model_kwargs(1, x0.device)
    with torch.inference_mode():
        keyless_latent, _ = diffusion.ddim_anonymization_forward(
            model_fn, x0, GPU_NOISE_LEVEL, clip_denoised=True, model_kwargs=kwargs
        )
        keyless_output = diffusion.ddim_anonymization_backward(
            model_fn, keyless_latent, GPU_NOISE_LEVEL, clip_denoised=True,
            model_kwargs=kwargs, guidance_scale=LEGACY_GUIDANCE_SCALE,
        )
    assert torch.equal(latent, keyless_latent)
    assert torch.equal(output, keyless_output)


@requires_gpu
@pytest.mark.parametrize("scheme", ["P", "S0", "S1"])
def test_cached_latent_path_matches_full_legacy_sampler(runtime, scheme: str) -> None:
    """E2.3: invert -> apply_legacy_key -> generate equals the full legacy calls."""
    diffusion, model_fn, x0 = runtime
    anonymous, latent, _ = legacy_anonymize(diffusion, model_fn, x0, SOURCE, scheme, GPU_NOISE_LEVEL)
    cached = legacy_invert(diffusion, model_fn, x0, GPU_NOISE_LEVEL)
    assert torch.equal(cached, latent)
    z_anonymous = apply_legacy_key(cached, SOURCE, scheme, "anonymize")
    assert torch.equal(legacy_generate(diffusion, model_fn, z_anonymous, GPU_NOISE_LEVEL), anonymous)

    recovered, reinverted, _ = legacy_deanonymize(
        diffusion, model_fn, anonymous, SOURCE, scheme, GPU_NOISE_LEVEL
    )
    reinverted_cached = legacy_invert(diffusion, model_fn, anonymous, GPU_NOISE_LEVEL)
    assert torch.equal(reinverted_cached, reinverted)
    z_recovered = apply_legacy_key(reinverted_cached, SOURCE, scheme, "deanonymize")
    assert torch.equal(legacy_generate(diffusion, model_fn, z_recovered, GPU_NOISE_LEVEL), recovered)


@pytest.mark.parametrize("scheme", ["P", "S0", "S1"])
def test_cached_key_application_round_trips_exactly(scheme: str) -> None:
    z = torch.randn((2, *KEY_SHAPE), generator=torch.Generator().manual_seed(1911))
    encrypted = apply_legacy_key(z, SOURCE, scheme, "anonymize")
    assert torch.equal(apply_legacy_key(encrypted, SOURCE, scheme, "deanonymize"), z)
    if scheme != "P":
        assert not torch.equal(encrypted, z)
