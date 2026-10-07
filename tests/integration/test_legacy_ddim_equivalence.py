from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from scripts.e1_ddim_runner import create_runtime, invert_reconstruct, preprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "past/SourceCode"
CHECKPOINT = SOURCE / "results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt"


@pytest.mark.skipif(not torch.cuda.is_available() or not CHECKPOINT.is_file(), reason="requires CUDA and legacy checkpoint")
def test_wrapper_is_bit_exact_with_legacy_progressive_sampler() -> None:
    _, diffusion, model_fn = create_runtime(SOURCE, CHECKPOINT, torch.device("cuda:0"))
    row = next(csv.DictReader((ROOT / "splits/dev_v1.1.csv").open(encoding="utf-8")))
    x0 = preprocess(ROOT / row["local_path"])[0].unsqueeze(0).cuda()
    seed, time, guidance = 1911, 500, 0.0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = {"y": torch.zeros(1, device=x0.device, dtype=torch.int64), "p_uncond": -1, "clf_free": True}
    t0 = torch.zeros(1, device=x0.device, dtype=torch.long)
    x_noisy = diffusion.q_sample(x_start=x0, t=t0, noise=torch.randn_like(x0))
    outputs = [out["sample"] for out in diffusion.ddim_sample_loop_known_progressive(
        model_fn, tuple(x0.shape), time=time, noise=x_noisy, clip_denoised=True,
        model_kwargs=kwargs, device=x0.device, guidance_scale=guidance,
    )]
    expected_z, expected_x = outputs[time - 2], outputs[-1]
    observed_z, observed_x = invert_reconstruct(diffusion, model_fn, x0, time, guidance, seed)
    assert torch.equal(observed_z, expected_z)
    assert torch.equal(observed_x, expected_x)
