"""CPU-only branch-contract tests; no checkpoints or image datasets required."""
import pytest
import torch

from scripts.localization_sampler_smoke import (
    canonical_cfg_backward, cfg_data_prediction, gs0_sanity, map_agreement,
    map_statistics, parallel_branches, recovered_latent,
)
from scripts.respaced_ddim_sweep import Respaced
from scripts.exact_inv_sweep import DPMSolverpp
from scripts.layerA_dpm_ablation import ddim_cfg_backward, dpm_cfg_backward


class Sampler:
    def __init__(self):
        self.inversions = 0
        self.backward_inputs = []

    def forward(self, x, boundaries):
        self.inversions += 1
        return x + 10

    def backward(self, z, boundaries):
        self.backward_inputs.append(z.clone())
        # Deliberate mutation tests isolation of sibling branch inputs.
        return z.sub_(10)


def test_parallel_branches_share_latent_without_reinversion():
    s = Sampler()
    x = torch.arange(4.0).reshape(1, 1, 2, 2)
    z = recovered_latent(s, x, [0, 1], "original", "test")
    seen = []

    def cfg(value, gs):
        seen.append(value.clone())
        return value.sub_(10 + gs)

    rec, healthy, maps = parallel_branches(s, cfg, z, [0, 1], [0.0, 2.0])
    assert s.inversions == 1
    assert torch.equal(z, x + 10)
    assert all(torch.equal(v, z) for v in seen + s.backward_inputs)
    assert torch.equal(rec, x)
    assert torch.equal(maps[2.0], torch.full_like(x, 2))
    assert gs0_sanity(rec, healthy[0.0]) == 0


def test_gs0_failure_is_not_silently_reported_as_success():
    with pytest.raises(ValueError, match="equivalence"):
        gs0_sanity(torch.zeros(2), torch.ones(2))


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        parallel_branches(Sampler(), lambda z, gs: z, torch.tensor([float("nan")]), [0, 1], [0])


def test_map_stats_and_undefined_correlation():
    assert map_statistics(torch.zeros(4))["mean"] == 0
    assert map_agreement(torch.zeros(4), torch.zeros(4))["pearson"] is None
    result = map_agreement(torch.arange(4.0), torch.arange(4.0))
    assert result["mae"] == 0
    assert result["pearson"] == pytest.approx(1)


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError, match="unknown scheme"):
        recovered_latent(Sampler(), torch.zeros(4), [0, 1], "typo", "test")


@pytest.mark.parametrize("use_dpm", [False, True])
def test_actual_sampler_gs0_conditional_identity_with_mock_model(use_dpm):
    class Diffusion:
        alphas_cumprod = [0.99, 0.85, 0.6, 0.3]

        def p_mean_variance(self, model, x, t, clip_denoised, model_kwargs):
            # Null and healthy predictions deliberately differ.
            shift = 0.04 if model_kwargs.get("null", False) else 0.02
            return {"pred_xstart": (x * 0.9 + shift).clamp(-1, 1)}

        def _predict_eps_from_xstart(self, x, t, px):
            a = torch.tensor(self.alphas_cumprod, dtype=x.dtype)[t].reshape(-1, 1, 1, 1)
            return (x - a.sqrt() * px) / (1-a).sqrt()

    rs = Respaced(Diffusion(), None, torch.device("cpu"))
    sampler = DPMSolverpp(rs) if use_dpm else rs
    boundaries = [0, 1, 2, 3]
    cfg = lambda z, gs: canonical_cfg_backward(sampler, z, boundaries, gs)
    latent = torch.linspace(0.1, 0.7, 4).reshape(1, 1, 2, 2)
    rec, healthy, maps = parallel_branches(sampler, cfg, latent, boundaries, [0.0, 2.0])
    assert gs0_sanity(rec, healthy[0.0]) == 0
    assert float(maps[2.0].max()) > 1e-3


def test_direct_x0_guidance_preserves_clip_before_mix_and_skips_null_at_gs0():
    class Diffusion:
        alphas_cumprod = [0.99, 0.2]

        def __init__(self):
            self.calls = []

        def p_mean_variance(self, model, x, t, clip_denoised, model_kwargs):
            null = model_kwargs.get("null", False)
            self.calls.append(null)
            return {"pred_xstart": torch.full_like(x, -1 if null else 1)}

    diffusion = Diffusion()
    rs = Respaced(diffusion, None, torch.device("cpu"))
    x = torch.zeros(1, 1, 2, 2)
    pred, _ = cfg_data_prediction(rs, x, 1, 0)
    assert diffusion.calls == [False]
    assert torch.equal(pred, torch.ones_like(x))
    pred, _ = cfg_data_prediction(rs, x, 1, 2)
    assert torch.equal(pred, torch.full_like(x, 5))  # no post-mix clipping
