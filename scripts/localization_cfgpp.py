"""Experimental pseudo-healthy branch: CFG++ paper-equation variant.

Reference: Chung et al., CFG++, ICLR 2025; arXiv:2406.08070v2,
Algorithm 2 and Appendix A Eq. (26):
https://arxiv.org/html/2406.08070v2#A1

This uses unconditional-current minus unconditional-previous in the 2M
correction. The upstream repository's latent_diffusion.py (main, inspected
2026-10-27, lines 796--799) instead uses guided-current minus unconditional-
previous. These are different variants; this module follows the PAPER.
https://github.com/CFGpp-diffusion/CFGpp/blob/main/latent_diffusion.py

Only the healthy branch changes: encryption, inversion and recovery are untouched.
lambda is the paper interpolation scale [0, 1], NOT legacy gs (omega=1+gs).
As in layerA_dpm_ablation, each branch x0 is clipped before interpolation.
This is an adaptation to the existing pixel-space checkpoint, not a claim of
improved medical localization or a full reproduction of Stable Diffusion results.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch

VARIANT = "cfgpp-paper-v2-eq26-vp-clipped-branch-x0"


def _validate(rs, z, boundaries, guidance_lambda):
    lam = float(guidance_lambda)
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("guidance_lambda must be finite and in [0, 1]")
    b = list(boundaries)
    if len(b) < 2 or any(isinstance(t, bool) or not isinstance(t, Integral) for t in b):
        raise ValueError("boundaries must contain at least two integer timesteps")
    if any(t < 0 or t >= len(rs.abar) for t in b) or any(s >= t for s, t in zip(b, b[1:])):
        raise ValueError("boundaries must be strictly increasing and in schedule range")
    a = rs.abar[b]
    if not bool(torch.isfinite(a).all()) or not bool(((a > 0) & (a < 1)).all()):
        raise ValueError("selected cumulative alphas must be finite and strictly in (0, 1)")
    if not bool((a[:-1] > a[1:]).all()):
        raise ValueError("cumulative alphas must strictly decrease along boundaries")
    if not z.is_floating_point() or z.ndim < 1 or not bool(torch.isfinite(z).all()):
        raise ValueError("latent must be a finite floating-point batch tensor")
    return b, lam


def _branch_x0(rs, x, timestep):
    t = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
    cond = dict(rs._base_kwargs(x.shape[0]))
    cond.pop("null", None)
    uncond = dict(cond, null=True)
    xc = rs.d.p_mean_variance(rs.m, x, t, clip_denoised=True, model_kwargs=cond)["pred_xstart"]
    xu = rs.d.p_mean_variance(rs.m, x, t, clip_denoised=True, model_kwargs=uncond)["pred_xstart"]
    if not bool(torch.isfinite(xc).all()) or not bool(torch.isfinite(xu).all()):
        raise FloatingPointError("non-finite conditional/unconditional x0 prediction")
    return xc, xu


@torch.inference_mode()
def ddim_cfgpp_backward(rs, z, boundaries, guidance_lambda):
    """Algorithm 2: guided x0 with unconditional renoising; same rs API."""
    b, lam = _validate(rs, z, boundaries, guidance_lambda)
    x = z.clone()
    for s, t in zip(reversed(b[1:]), reversed(b[:-1])):
        xc, xu = _branch_x0(rs, x, s)
        a_s = rs.abar[s].to(x.dtype).sqrt()
        a_t = rs.abar[t].to(x.dtype).sqrt()
        sigma_s = (1 - rs.abar[s]).to(x.dtype).sqrt()
        sigma_t = (1 - rs.abar[t]).to(x.dtype).sqrt()
        x = a_t * (xu + lam * (xc - xu)) + sigma_t * (x - a_s * xu) / sigma_s
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("non-finite CFG++ DDIM state")
    return x


@torch.inference_mode()
def dpm_cfgpp_backward(dpm, z, boundaries, guidance_lambda):
    """Eq.26 in VP coordinates; unconditional history, first-order first step.

    With y=x/alpha and VE sigma=sigma_VP/alpha, q=exp(-h), the
    update is x_next=c1*x + alpha_next*(x0_lambda-q*x0_u
                  +(1-q)/(2*r)*(x0_u-x0_u_previous)).
    The history term is omitted at the first step. No terminal extrapolation
    to alpha=1 is added: boundaries match the existing deployed solver.
    """
    rs = dpm.rs
    b, lam = _validate(rs, z, boundaries, guidance_lambda)
    x = z.clone()
    previous_u, previous_h = None, None
    for s, t in zip(reversed(b[1:]), reversed(b[:-1])):
        c1, _, h = dpm._coef(s, t, x.dtype)
        if not math.isfinite(h) or h <= 0:
            raise ValueError("DPM++ denoising log-SNR step must be finite and positive")
        xc, xu = _branch_x0(rs, x, s)
        q = math.exp(-h)
        update = xu + lam * (xc - xu) - q * xu
        if previous_u is not None:
            r = previous_h / h
            update = update + (-math.expm1(-h) / (2 * r)) * (xu - previous_u)
        x = c1 * x + rs.abar[t].to(x.dtype).sqrt() * update
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("non-finite CFG++ DPM++ state")
        previous_u, previous_h = xu, h
    return x
