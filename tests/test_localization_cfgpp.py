"""CPU-only equation tests; no checkpoint, image dataset or CUDA required."""
import math
from types import SimpleNamespace

import pytest
import torch

from scripts.localization_cfgpp import ddim_cfgpp_backward, dpm_cfgpp_backward


class FakeDiffusion:
    def __init__(self, equal=False, nonfinite=False):
        self.equal, self.nonfinite = equal, nonfinite

    def p_mean_variance(self, model, x, t, clip_denoised, model_kwargs):
        assert clip_denoised
        shift = 0 if self.equal or model_kwargs.get("null", False) else 0.13
        result = 0.19 * x + 0.02 * t.to(x.dtype).reshape(-1, 1) + shift
        if self.nonfinite:
            result = result * float("nan")
        return {"pred_xstart": result}


class FakeDPM:
    def __init__(self, equal=False, nonfinite=False):
        self.rs = SimpleNamespace(
            abar=torch.tensor([0.99, 0.9, 0.7, 0.4, 0.15], dtype=torch.float64),
            d=FakeDiffusion(equal, nonfinite), m=None,
            _base_kwargs=lambda batch: {"y": torch.zeros(batch, dtype=torch.long), "clf_free": True})

    def _coef(self, s, t, dtype):
        ac = self.rs.abar
        logsnr = 0.5 * (ac.log() - (1 - ac).log())
        h = float(logsnr[t] - logsnr[s])
        c1 = ((1 - ac[t]) / (1 - ac[s])).sqrt().to(dtype)
        return c1, -float(ac[t].sqrt()) * math.expm1(-h), h


Z = torch.tensor([[0.3, -0.2], [0.6, 0.1]], dtype=torch.float64)
B = [0, 1, 3, 4]


def unconditional_oracle(dpm, z):
    x, prev, hprev = z.clone(), None, None
    for s, t in zip(reversed(B[1:]), reversed(B[:-1])):
        c1, c2, h = dpm._coef(s, t, x.dtype)
        xu = 0.19 * x + 0.02 * s
        d = xu if prev is None else xu + (xu - prev) / (2 * hprev / h)
        x = c1 * x + c2 * d
        prev, hprev = xu, h
    return x


def test_lambda_zero_matches_unconditional_multistep():
    dpm = FakeDPM()
    torch.testing.assert_close(dpm_cfgpp_backward(dpm, Z, B, 0), unconditional_oracle(dpm, Z))


@pytest.mark.parametrize("lam", [0, 0.25, 0.5, 1])
def test_equal_branches_guidance_invariant(lam):
    dpm = FakeDPM(equal=True)
    torch.testing.assert_close(dpm_cfgpp_backward(dpm, Z, B, lam), unconditional_oracle(dpm, Z))


@pytest.mark.parametrize("lam", [0, 0.6, 1])
def test_first_step_matches_ddim_closed_form(lam):
    dpm = FakeDPM()
    ac = dpm.rs.abar
    xu = 0.19 * Z + 0.02 * 4
    xc = xu + 0.13
    eps_u = (Z - ac[4].sqrt() * xu) / (1 - ac[4]).sqrt()
    expected = ac[1].sqrt() * (xu + lam * (xc - xu)) + (1 - ac[1]).sqrt() * eps_u
    torch.testing.assert_close(dpm_cfgpp_backward(dpm, Z, [1, 4], lam), expected)
    torch.testing.assert_close(ddim_cfgpp_backward(dpm.rs, Z, [1, 4], lam), expected)


@pytest.mark.parametrize("lam", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_lambda(lam):
    with pytest.raises(ValueError):
        dpm_cfgpp_backward(FakeDPM(), Z, B, lam)


@pytest.mark.parametrize("boundaries", [[], [1], [1, 1], [2, 1], [-1, 2], [0, 5], [0, 2.1], [False, 2]])
def test_invalid_boundaries(boundaries):
    with pytest.raises(ValueError):
        dpm_cfgpp_backward(FakeDPM(), Z, boundaries, 0.5)


def test_nonfinite_input_and_prediction():
    with pytest.raises(ValueError):
        dpm_cfgpp_backward(FakeDPM(), Z * float("nan"), B, 0.5)
    with pytest.raises(FloatingPointError):
        dpm_cfgpp_backward(FakeDPM(nonfinite=True), Z, B, 0.5)


@pytest.mark.parametrize("bad_alpha", [0, 1, float("nan"), float("inf")])
def test_invalid_schedule(bad_alpha):
    dpm = FakeDPM()
    dpm.rs.abar[1] = bad_alpha
    with pytest.raises(ValueError):
        dpm_cfgpp_backward(dpm, Z, B, 0.5)


def test_inputs_unchanged_and_output_finite():
    original = Z.clone()
    output = dpm_cfgpp_backward(FakeDPM(), Z, B, 0.7)
    assert bool(torch.isfinite(output).all())
    torch.testing.assert_close(Z, original)
