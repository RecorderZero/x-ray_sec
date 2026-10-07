"""E2.2 regression tests against the predecessor's S0/S1 implementation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_legacy_wrapper import (
    build_legacy_components,
    legacy_anonymize,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "past/SourceCode"
CHECKPOINT = SOURCE / "results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt"


@pytest.mark.parametrize("scheme", ["S0", "S1"])
def test_legacy_transform_is_exactly_invertible(scheme: str) -> None:
    anonymizer, key = build_legacy_components(SOURCE, scheme, torch.device("cpu"))
    z = torch.linspace(-2.0, 2.0, 256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256)
    encrypted = anonymizer.anonymize_latent(z, key)
    recovered = anonymizer.deanonymize_latent(encrypted, key)
    assert torch.equal(recovered, z)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not CHECKPOINT.is_file(),
    reason="requires CUDA and legacy checkpoint",
)
@pytest.mark.parametrize("scheme", ["S0", "S1"])
def test_wrapper_is_bit_exact_with_legacy_anonymization_sampler(scheme: str) -> None:
    _, diffusion, model_fn = create_runtime(SOURCE, CHECKPOINT, torch.device("cuda:0"))
    row = next(csv.DictReader((ROOT / "splits/dev_v1.1.csv").open(encoding="utf-8")))
    x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).cuda()

    observed = legacy_anonymize(
        diffusion, model_fn, x0, SOURCE, scheme, noise_level=16, guidance_scale=0.0
    )
    anonymizer, key = build_legacy_components(SOURCE, scheme, x0.device)
    kwargs = {
        "y": torch.zeros(1, device=x0.device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }
    expected = diffusion.ddim_sample_loop_anonymization(
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
        noise_level=16,
        progress=False,
        guidance_scale=0.0,
    )
    assert all(torch.equal(actual, oracle) for actual, oracle in zip(observed, expected))
