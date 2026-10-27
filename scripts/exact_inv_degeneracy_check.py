#!/usr/bin/env python3
"""Sanity: is DPM-Solver++'s very high round-trip PSNR a real win or a degenerate
latent?  A sampler can fake reconstruction fidelity if its encode(x0)=z keeps
image structure (z not ~N(0,1)): then decode trivially returns x0, but z cannot
be encrypted and produces no valid pseudo-healthy image.

For each sampler at N=100 we report:
  * z statistics: mean, std, |max|   (a proper latent is ~0 mean, ~1 std)
  * anon divergence: PSNR(decode(sign-flip(z)), x0) -- MUST be LOW (encryption
    bites: the anon image is far from x0), else the latent carries no usable
    structure to scramble.
  * recon: PSNR(decode(z), x0) -- the headline number, for reference.

  scripts/run_cfg_ddim.sh python -m scripts.exact_inv_degeneracy_check
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.respaced_ddim_sweep import Respaced, schedule, psnr, SEED
from scripts.exact_inv_sweep import BDIARespaced, DPMSolverpp, CKPT

N = 100
INDICES = (0, 1)


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), CKPT, device)
    rs = Respaced(diffusion, model_fn, device)
    samplers = {"DDIM": rs, "BDIA": BDIARespaced(rs), "DPM++": DPMSolverpp(rs)}

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    imgs = [preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device) for i in INDICES]
    b = schedule(N)

    out = {}
    for name, s in samplers.items():
        zs, recon, anon = [], [], []
        for x0 in imgs:
            z = s.forward(x0, b)
            # sign-flip "encryption" (S0 Rademacher), bit-exact & norm-preserving
            flip = (torch.randint(0, 2, z.shape, device=device, generator=torch.Generator(device).manual_seed(7)) * 2 - 1).to(z.dtype)
            zc = z * flip
            recon.append(psnr(s.backward(z, b), x0))
            anon.append(psnr(s.backward(zc, b), x0))
            zs.append(z)
        zc_all = torch.cat(zs)
        out[name] = {
            "z_mean": round(float(zc_all.mean()), 4),
            "z_std": round(float(zc_all.std()), 4),
            "z_absmax": round(float(zc_all.abs().max()), 3),
            "recon_psnr_db": round(sum(recon) / len(recon), 2),
            "anon_vs_x0_psnr_db": round(sum(anon) / len(anon), 2),
        }
        print(f"{name:>6}  z: mean={out[name]['z_mean']:+.3f} std={out[name]['z_std']:.3f} "
              f"absmax={out[name]['z_absmax']:.2f} | recon={out[name]['recon_psnr_db']:.2f} dB "
              f"| anon(sign-flip) vs x0={out[name]['anon_vs_x0_psnr_db']:.2f} dB", flush=True)

    Path("paper_assets/exact_inv_sweep").mkdir(parents=True, exist_ok=True)
    Path("paper_assets/exact_inv_sweep/degeneracy_check.json").write_text(
        json.dumps({"seed": SEED, "N": N, "indices": list(INDICES), "samplers": out},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nInterpretation: a REAL win has z_std~1, z_mean~0, AND anon(sign-flip) "
          "far below recon (encryption moves the image). A DEGENERATE latent shows "
          "z_std<<1 or anon~recon (scrambling the latent barely changes the output).")


if __name__ == "__main__":
    main()
