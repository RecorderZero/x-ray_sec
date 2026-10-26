#!/usr/bin/env python3
"""B2: BELM (Bidirectional Explicit Linear Multi-step) sampler -- exact inversion.

Implements the O-BELM update (Zhang et al., NeurIPS 2024, arXiv:2410.07273),
Eqs. (18) forward / (19) exact inverse, over the same pixel-space CFG-DDIM model
and respaced timestep grid as the aligned DDIM (no change to past/SourceCode/).

Notation (VP diffusion): for timestep t, alpha=sqrt(abar[t]), sigma=sqrt(1-abar),
sigbar=sigma/alpha; grid t_0=0 < t_1 < ... < t_M=TOP; h_i = sigbar[t_i]-sigbar[t_{i-1}].
Two-step method: first point bootstrapped by one aligned DDIM step.

  eps_i = eps_theta(x_i, t_i)
  decode (x_{i-1} from x_{i+1},x_i):  Eq.(18)
  encode (x_{i+1} from x_{i-1},x_i):  Eq.(19), the exact inverse

This module is a SMOKE validation harness first: key-free round-trip vs the
aligned DDIM at matched NFE.  Numbers are not trusted until the round-trip is
sane (finite, and at least as good as DDIM at full density).

  scripts/run_cfg_ddim.sh python -m scripts.belm_sampler
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.respaced_ddim_sweep import Respaced, schedule, psnr, SEED, TOP

CKPT = Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt")
INDICES = (0, 1)
SWEEP_N = (499, 200, 100, 50, 25)


def schedule_sigbar(n: int, belm, top: int = TOP) -> np.ndarray:
    """n+1 ascending timesteps roughly uniform in sigbar=sigma/alpha (BELM-friendly)."""
    sb = belm.sigbar[: top + 1].detach().cpu().numpy()
    targets = np.linspace(sb[0], sb[top], n + 1)
    grid = np.array([int(np.abs(sb - t).argmin()) for t in targets])
    return np.unique(grid)


class BELM:
    def __init__(self, diffusion, model_fn, device, clip: bool = True):
        self.d, self.m, self.dev, self.clip = diffusion, model_fn, device, clip
        ac = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float64, device=device)
        self.alpha = ac.sqrt()                    # alpha[t] = sqrt(abar[t])
        self.sigbar = ((1 - ac) / ac).sqrt()      # sigbar[t] = sigma/alpha

    def _kw(self, b, null):
        kw = {"y": torch.zeros(b, device=self.dev, dtype=torch.int64), "p_uncond": -1, "clf_free": True}
        if null:
            kw["null"] = True
        return kw

    @torch.inference_mode()
    def _eps_and_x0(self, x, t_int, mk):
        t = torch.full((x.shape[0],), int(t_int), device=self.dev, dtype=torch.long)
        px = self.d.p_mean_variance(self.m, x, t, clip_denoised=self.clip, model_kwargs=mk)["pred_xstart"]
        eps = self.d._predict_eps_from_xstart(x, t, px)
        return eps, px

    def _ddim_jump(self, x, t_from, t_to, mk):
        """One aligned DDIM step: eval at t_from, land at abar[t_to] (bootstrap)."""
        eps, px = self._eps_and_x0(x, t_from, mk)
        a = self.alpha[int(t_to)].to(x.dtype) ** 2  # abar[t_to]
        return px * torch.sqrt(a) + torch.sqrt(1 - a) * eps

    def _coeffs(self, grid, i):
        """h_i, h_{i+1} and the alpha lookups around index i (float64 scalars)."""
        a = self.alpha
        sb = self.sigbar
        tm1, ti, tp1 = int(grid[i - 1]), int(grid[i]), int(grid[i + 1])
        hi = (sb[ti] - sb[tm1]).item()
        hip1 = (sb[tp1] - sb[ti]).item()
        return hi, hip1, a[tm1].item(), a[ti].item(), a[tp1].item()

    @torch.inference_mode()
    def encode(self, x0, grid):
        """Image -> noise (inversion), Eq.(19).  grid ascending [0..TOP]."""
        mk = self._kw(x0.shape[0], null=True)
        M = len(grid) - 1
        x = [None] * (M + 1)
        x[0] = x0
        x[1] = self._ddim_jump(x0, grid[0], grid[1], mk)          # bootstrap
        for i in range(1, M):
            hi, hip1, am1, ai, ap1 = self._coeffs(grid, i)
            eps, _ = self._eps_and_x0(x[i], grid[i], mk)
            c2 = (hip1 ** 2 / hi ** 2) * (ap1 / am1)
            c1 = ((hi ** 2 - hip1 ** 2) / hi ** 2) * (ap1 / ai)
            ce = (hip1 * (hi + hip1) / hi) * ap1
            x[i + 1] = c2 * x[i - 1] + c1 * x[i] + ce * eps
        return x[M]

    @torch.inference_mode()
    def decode(self, z, grid):
        """Noise -> image (generation), Eq.(18).  grid ascending [0..TOP]."""
        mk = self._kw(z.shape[0], null=False)
        M = len(grid) - 1
        x = [None] * (M + 1)
        x[M] = z
        x[M - 1] = self._ddim_jump(z, grid[M], grid[M - 1], mk)   # bootstrap
        for i in range(M - 1, 0, -1):
            hi, hip1, am1, ai, ap1 = self._coeffs(grid, i)
            eps, _ = self._eps_and_x0(x[i], grid[i], mk)
            c2 = (hi ** 2 / hip1 ** 2) * (am1 / ap1)
            c1 = ((hip1 ** 2 - hi ** 2) / hip1 ** 2) * (am1 / ai)
            ce = -(hi * (hi + hip1) / hip1) * am1
            x[i - 1] = c2 * x[i + 1] + c1 * x[i] + ce * eps
        return x[0]


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), CKPT, device)
    belm_clip = BELM(diffusion, model_fn, device, clip=True)      # current (t-grid, clip)
    belm_noclip = BELM(diffusion, model_fn, device, clip=False)   # candidate (sigbar-grid, no clip)
    rs = Respaced(diffusion, model_fn, device)

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    imgs = [preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device) for i in INDICES]

    def rt(sampler, x0, g):
        z = sampler.encode(x0, g)
        xr = sampler.decode(z, g)
        ok = bool(torch.isfinite(z).all() and torch.isfinite(xr).all())
        return psnr(xr, x0), ok

    print(f"{'N':>5} {'DDIM(t)':>9} {'BELM(t,clip)':>13} {'BELM(sb,noclip)':>16}")
    out = []
    for n in SWEEP_N:
        gt = schedule(n)
        gsb = schedule_sigbar(n, belm_noclip)
        ddim, belm_tc, belm_sb, fin = [], [], [], True
        for x0 in imgs:
            ddim.append(psnr(rs.backward(rs.forward(x0, gt), gt), x0))
            p1, _ = rt(belm_clip, x0, gt); belm_tc.append(p1)
            p2, ok = rt(belm_noclip, x0, gsb); belm_sb.append(p2); fin = fin and ok
        md, m1, m2 = float(np.mean(ddim)), float(np.mean(belm_tc)), float(np.mean(belm_sb))
        out.append({"N": n, "ddim_t": round(md, 2), "belm_t_clip": round(m1, 2),
                    "belm_sigbar_noclip": round(m2, 2), "sigbar_grid_pts": int(len(gsb)), "finite": fin})
        print(f"{n:>5} {md:>9.2f} {m1:>13.2f} {m2:>16.2f}", flush=True)

    outdir = Path("paper_assets/belm_smoke"); outdir.mkdir(parents=True, exist_ok=True)
    import json
    (outdir / "smoke.json").write_text(json.dumps({"seed": SEED, "indices": list(INDICES), "rows": out},
                                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {outdir}/smoke.json")


if __name__ == "__main__":
    main()
