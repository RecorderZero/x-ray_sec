#!/usr/bin/env python3
"""Layer-A equivalence ablation under the new sampler (s11.2 item 2).

The 4-pass pipeline bifurcates the LAST stage: from the decrypted/recovered latent
x_rec_T it generates BOTH the recovery image (unconditional) and the pseudo-healthy
image (CFG), instead of re-inverting the recovery image into a fresh latent first.
The predecessor ablation proved this redundancy for the legacy DDIM sampler
(A~=B, cos 0.9999); here we re-verify it for the DEPLOYED sampler DPM-Solver++(2M)
at N=100, and for aligned respaced DDIM at N=100 as a reference.

Per image, per sampler, per guidance gs:
  recovery_B = backward(x_rec_T)                       (direct; the shipped path)
  recovery_A = backward(forward(recovery_B))           (re-invert then regenerate)
  healthy_B  = cfg_backward(x_rec_T, gs)               (direct CFG)
  healthy_A  = cfg_backward(forward(recovery_B), gs)   (re-invert then CFG)
We report psnr/ssim/cosine(A,B) for both the recovery and the pseudo-healthy, so a
high agreement means the re-inversion pass is redundant under this sampler too.
CFG mixes eps = (1+gs)*eps_cond - gs*eps_uncond (cond = y=0 healthy class, uncond
= null), the standard classifier-free guidance, 2 model evals/step at gs!=... .

  smoke: scripts/run_cfg_ddim.sh python -m scripts.layerA_dpm_ablation --smoke
  full : scripts/run_cfg_ddim.sh python -m scripts.layerA_dpm_ablation
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED
from scripts.exact_inv_sweep import DPMSolverpp, CKPT

FULL_INDICES = (0, 1, 10, 11)
SMOKE_INDICES = (0, 1)
N = 100
GUIDANCE = (0.0, 2.0)


def _cfg_eps(rs: Respaced, x, t_int, gs):
    """CFG noise prediction: (1+gs) eps_cond - gs eps_uncond (cond=y0, uncond=null)."""
    t = torch.full((x.shape[0],), int(t_int), device=rs.dev, dtype=torch.long)
    mk_c = rs._base_kwargs(x.shape[0])                       # conditional (no null)
    mk_u = rs._base_kwargs(x.shape[0]); mk_u["null"] = True  # unconditional
    px_c = rs.d.p_mean_variance(rs.m, x, t, clip_denoised=True, model_kwargs=mk_c)["pred_xstart"]
    px_u = rs.d.p_mean_variance(rs.m, x, t, clip_denoised=True, model_kwargs=mk_u)["pred_xstart"]
    eps_c = rs.d._predict_eps_from_xstart(x, t, px_c)
    eps_u = rs.d._predict_eps_from_xstart(x, t, px_u)
    return (1.0 + gs) * eps_c - gs * eps_u


def _x0_from_eps(rs, x, t_int, eps):
    a = rs.abar[int(t_int)].to(x.dtype)
    return (x - torch.sqrt(1 - a) * eps) / torch.sqrt(a)


@torch.inference_mode()
def ddim_cfg_backward(rs: Respaced, z, b, gs):
    x = z
    for k in range(len(b) - 1, 0, -1):
        eps = _cfg_eps(rs, x, b[k], gs)
        px = _x0_from_eps(rs, x, b[k], eps)
        a = rs.abar[int(b[k - 1])].to(x.dtype)
        x = px * torch.sqrt(a) + torch.sqrt(1 - a) * eps
    return x


@torch.inference_mode()
def dpm_cfg_backward(dpm: DPMSolverpp, z, b, gs):
    """DPM-Solver++(2M) generation with CFG data-prediction (x0 from CFG eps)."""
    rs = dpm.rs
    order = [(int(b[i]), int(b[i - 1])) for i in range(len(b) - 1, 0, -1)]
    x = z
    x0_prev, h_prev = None, None
    for k, (s, t) in enumerate(order):
        c1, c2, h = dpm._coef(s, t, x.dtype)
        eps = _cfg_eps(rs, x, s, gs)
        x0_s = _x0_from_eps(rs, x, s, eps)
        if k == 0:
            D = x0_s
        else:
            r = h_prev / h
            D = (1 + 1 / (2 * r)) * x0_s - (1 / (2 * r)) * x0_prev
        x = c1 * x + c2 * D
        x0_prev, h_prev = x0_s, h
    return x


def _ab(a, b):
    m = image_pair_metrics("", a, b)
    return {"psnr_db": round(float(m["_psnr_db"]), 3), "ssim": round(float(m["_ssim"]), 4),
            "cosine": round(float(m["_cosine"]), 6)}


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/layerA_dpm"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    indices = list(SMOKE_INDICES if args.smoke else FULL_INDICES)

    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), CKPT, device)
    rs = Respaced(diffusion, model_fn, device)
    dpm = DPMSolverpp(rs)
    b = schedule(N)
    samplers = {
        "aligned-DDIM": (rs.forward, rs.backward, lambda z, gs: ddim_cfg_backward(rs, z, b, gs)),
        "DPM++": (dpm.forward, dpm.backward, lambda z, gs: dpm_cfg_backward(dpm, z, b, gs)),
    }

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    items = [(rows[i]["sample_id"], preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device))
             for i in indices]

    records = []
    for name, (fwd, bwd, cfg_bwd) in samplers.items():
        for sid, x0 in items:
            x_rec_T = fwd(x0, b)
            recovery_B = bwd(x_rec_T, b)
            z_reinv = fwd(recovery_B, b)
            recovery_A = bwd(z_reinv, b)
            rec = {"sampler": name, "N": N, "sample_id": sid,
                   "recovery_AB": _ab(recovery_A, recovery_B)}
            for gs in GUIDANCE:
                healthy_B = cfg_bwd(x_rec_T, gs)
                healthy_A = cfg_bwd(z_reinv, gs)
                rec[f"healthy_gs{gs:g}_AB"] = _ab(healthy_A, healthy_B)
                rec[f"healthy_gs{gs:g}_finite"] = bool(torch.isfinite(healthy_A).all() and torch.isfinite(healthy_B).all())
            records.append(rec)
            print(f"{name:>12} {sid:<16} recovery cos={rec['recovery_AB']['cosine']:.5f} "
                  f"psnr={rec['recovery_AB']['psnr_db']:.1f} | "
                  + " ".join(f"gs{gs:g} cos={rec[f'healthy_gs{gs:g}_AB']['cosine']:.5f} "
                             f"psnr={rec[f'healthy_gs{gs:g}_AB']['psnr_db']:.1f}" for gs in GUIDANCE), flush=True)

    # aggregate per sampler
    agg = {}
    for name in samplers:
        rr = [r for r in records if r["sampler"] == name]
        a = {"recovery_mean_cos": round(float(np.mean([r["recovery_AB"]["cosine"] for r in rr])), 6),
             "recovery_mean_psnr": round(float(np.mean([r["recovery_AB"]["psnr_db"] for r in rr])), 3)}
        for gs in GUIDANCE:
            a[f"healthy_gs{gs:g}_mean_cos"] = round(float(np.mean([r[f"healthy_gs{gs:g}_AB"]["cosine"] for r in rr])), 6)
            a[f"healthy_gs{gs:g}_mean_psnr"] = round(float(np.mean([r[f"healthy_gs{gs:g}_AB"]["psnr_db"] for r in rr])), 3)
        agg[name] = a

    name_out = "smoke.json" if args.smoke else "ablation.json"
    (args.out_dir / name_out).write_text(json.dumps(
        {"seed": SEED, "smoke": args.smoke, "N": N, "guidance": list(GUIDANCE), "indices": indices,
         "variants": {"A": "re-invert recovery then (CFG) generate", "B": "generate directly from recovered latent"},
         "aggregate": agg, "per_sample": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== layer-A equivalence (A re-invert vs B direct; mean cos / psnr dB) ===")
    for name in samplers:
        a = agg[name]
        print(f"{name:>12}  recovery cos={a['recovery_mean_cos']:.5f} psnr={a['recovery_mean_psnr']:.1f}  | "
              + "  ".join(f"healthy gs{gs:g} cos={a[f'healthy_gs{gs:g}_mean_cos']:.5f} psnr={a[f'healthy_gs{gs:g}_mean_psnr']:.1f}"
                          for gs in GUIDANCE))
    print(f"\nwrote {args.out_dir}/{name_out}")


if __name__ == "__main__":
    main()
