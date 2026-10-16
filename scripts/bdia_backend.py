#!/usr/bin/env python3
"""Implementer BDIA (gamma=1) sampler backend for the Week-7 appendix 2x2 ablation.

Reproduces the gamma=1 bidirectional-integration update (Zhang et al., ECCV 2024)
validated by the auditor reference ``audit/bdia_vs_ddim_s2fix.py`` (AUD-20261015-02).
One model evaluation per step yields both the DDIM generation step ``G`` (-> t-1)
and the DDIM reverse step ``R`` (-> t+1); the leapfrog recurrence

    generate:  z_{i-1} = z_{i+1} + G(z_i) - R(z_i)
    invert:    z_{i+1} = z_{i-1} + R(z_i) - G(z_i)

is algebraically reversible step-for-step, so ``invert(generate(z))`` recovers the
input far better than DDIM at identical NFE (the boundary step is pure DDIM and is
the only non-exact part). Conditioning kwargs are supplied by the caller so the
2x2 runner can set asymmetric (invert null / generate conditional) or symmetric
(both null) per cell.

``BDIABackend`` mirrors ``scripts.q8_pilot_runner.DiffusionBackend`` (.generate /
.invert) and additionally exposes an NFE counter to confirm cost parity with DDIM.

CPU self-test (no model, no GPU): ``scripts/run_cfg_ddim.sh python -m scripts.bdia_backend``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def _extract(arr, t: torch.Tensor, shape) -> torch.Tensor:
    """Gather ``arr[t]`` and broadcast to ``shape`` (matches guided_diffusion)."""
    res = torch.from_numpy(np.asarray(arr, dtype=np.float64)).to(t.device)[t].float()
    while res.dim() < len(shape):
        res = res[..., None]
    return res.expand(shape)


@dataclass
class NFE:
    """Number of function (model) evaluations, for DDIM/BDIA cost-parity checks."""

    count: int = 0

    def bump(self) -> None:
        self.count += 1


def _gr(diffusion, model, x: torch.Tensor, t: torch.Tensor, kw: dict, nfe: NFE | None = None):
    """One model evaluation -> (G generation step -> t-1, R reverse step -> t+1)."""
    out = diffusion.p_mean_variance(model, x, t, clip_denoised=True, model_kwargs=kw, clf_free=None)
    if nfe is not None:
        nfe.bump()
    x0 = out["pred_xstart"]
    eps = diffusion._predict_eps_from_xstart(x, t, x0)
    ap = _extract(diffusion.alphas_cumprod_prev, t, x.shape)
    an = _extract(diffusion.alphas_cumprod_next, t, x.shape)
    g = x0 * torch.sqrt(ap) + torch.sqrt(1 - ap) * eps
    r = x0 * torch.sqrt(an) + torch.sqrt(1 - an) * eps
    return g, r


def bdia_generate(diffusion, model, z_T: torch.Tensor, kw: dict, steps: int,
                  device: torch.device, nfe: NFE | None = None) -> torch.Tensor:
    """BDIA generation x_T -> x_0 (leapfrog; boundary step pure DDIM)."""
    bs = z_T.shape[0]
    prev, curr = None, z_T
    for i in list(range(steps - 1))[::-1]:
        t = torch.tensor([i] * bs, device=device)
        g, r = _gr(diffusion, model, curr, t, kw, nfe)
        nxt = g if prev is None else prev + g - r
        prev, curr = curr, nxt
    return curr


def bdia_invert(diffusion, model, x0: torch.Tensor, kw: dict, steps: int,
                device: torch.device, nfe: NFE | None = None) -> torch.Tensor:
    """BDIA inversion x_0 -> x_T (leapfrog; boundary step pure DDIM)."""
    bs = x0.shape[0]
    prev, curr = None, x0
    for i in list(range(steps - 1)):
        t = torch.tensor([i] * bs, device=device)
        g, r = _gr(diffusion, model, curr, t, kw, nfe)
        nxt = r if prev is None else prev + r - g
        prev, curr = curr, nxt
    return curr


@dataclass
class BDIABackend:
    """Drop-in for q8_pilot_runner.DiffusionBackend using the BDIA sampler.

    ``invert_kwargs`` / ``generate_kwargs`` carry the per-cell conditioning:
    asymmetric -> invert null, generate conditional; symmetric -> both null.
    """

    diffusion: object
    model_fn: object
    noise_level: int
    device: torch.device
    invert_kwargs: dict
    generate_kwargs: dict
    nfe: NFE = field(default_factory=NFE)

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return bdia_generate(self.diffusion, self.model_fn, latent, dict(self.generate_kwargs),
                             self.noise_level, self.device, self.nfe)

    def invert(self, image: torch.Tensor) -> torch.Tensor:
        return bdia_invert(self.diffusion, self.model_fn, image, dict(self.invert_kwargs),
                           self.noise_level, self.device, self.nfe)


# --------------------------------------------------------------------------- self-test
# CPU self-test goal: prove this module REPRODUCES the auditor's validated BDIA
# reference (audit/bdia_vs_ddim_s2fix.py, AUD-20261015-02) bit-for-bit, and that
# NFE == DDIM (steps-1 model evals per pass). The *quantitative* "BDIA roundtrip <
# DDIM" advantage is a stability property of the real fine-step model and is
# validated on-GPU (auditor diagnostic + the appendix smoke), not on a toy.
class _ToyDiffusion:
    """Minimal stand-in exposing the three members ``_gr`` uses. Denoiser is a
    bounded deterministic map so a short trajectory stays finite. No model/GPU."""

    def __init__(self, steps: int):
        acp = np.linspace(0.9999, 0.3, steps)  # decreasing alphas_cumprod
        self.alphas_cumprod = acp
        self.alphas_cumprod_prev = np.append(1.0, acp[:-1])
        self.alphas_cumprod_next = np.append(acp[1:], acp[-1] * 0.9)

    def p_mean_variance(self, model, x, t, clip_denoised, model_kwargs, clf_free):
        shift = 0.0 if model_kwargs.get("null") else 0.05
        return {"pred_xstart": 0.5 * torch.tanh(x + shift)}

    def _predict_eps_from_xstart(self, x, t, x0):
        acp = _extract(self.alphas_cumprod, t, x.shape)
        return (x - torch.sqrt(acp) * x0) / torch.sqrt(1 - acp)


def _selftest() -> None:
    torch.manual_seed(1911)
    steps, device = 10, torch.device("cpu")
    d = _ToyDiffusion(steps)
    kw = {"y": torch.zeros(1, dtype=torch.int64), "p_uncond": -1, "clf_free": True, "null": True}
    z = torch.randn(1, 1, 4, 4)

    nfe_g, nfe_i = NFE(), NFE()
    anon = bdia_generate(d, None, z, kw, steps, device, nfe_g)
    back = bdia_invert(d, None, anon, kw, steps, device, nfe_i)
    assert torch.isfinite(anon).all() and torch.isfinite(back).all()
    assert nfe_g.count == steps - 1 == nfe_i.count, (nfe_g.count, nfe_i.count, steps - 1)

    # Bit-fidelity to the auditor reference (skip gracefully if audit/ absent).
    # Importing the reference pulls in the predecessor training stack, which spams
    # visdom connection errors on stderr; silence them so the result stays legible.
    ref_ok = "unavailable"
    try:
        import contextlib
        import importlib
        import io as _io
        _sink = _io.StringIO()
        with contextlib.redirect_stderr(_sink), contextlib.redirect_stdout(_sink):
            ref = importlib.import_module("audit.bdia_vs_ddim_s2fix")
        ref.T = steps  # the reference reads a module-global horizon
        ref_anon = ref.bdia_generate(d, None, z, dict(kw), device)
        ref_back = ref.bdia_invert(d, None, ref_anon, dict(kw), device)
        dg = float((anon - ref_anon).abs().max())
        di = float((back - ref_back).abs().max())
        assert dg < 1e-5 and di < 1e-5, f"diverges from reference: gen={dg:.2e} inv={di:.2e}"
        ref_ok = f"match (gen Δ={dg:.1e}, inv Δ={di:.1e})"
    except ModuleNotFoundError:
        pass

    print(f"[bdia_backend selftest] steps={steps} NFE/pass={nfe_g.count} (==DDIM steps-1) "
          f"| finite OK | auditor-reference: {ref_ok}")


if __name__ == "__main__":
    _selftest()
