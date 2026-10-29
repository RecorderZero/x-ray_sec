#!/usr/bin/env python3
"""SMOKE (1 image): dump our deployment pipeline's x0 / anon / recovered float
arrays for two presentation figures (value-range histogram + fidelity side-by-side).

Deployment operating point reused from rq4_formal_redo: DPM-Solver++(2M) N=100,
null backward, S2fn cipher.  Also runs the P (identity) path so the histogram can
contrast a plaintext-range anon against the encrypted anon.  Raw float `anon` is
captured BEFORE the PNG hand-off so the true value range is visible.

  scripts/run_cfg_ddim.sh python -m scripts.smoke_presentation_panels
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED
from scripts.exact_inv_sweep import DPMSolverpp, CKPT
from scripts.rq4_formal_redo import pipeline


def _hw(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().reshape(256, 256)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    ap.add_argument("--index", type=int, default=0, help="which image in the split")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True, help="output .npz")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    row = list(csv.DictReader(args.split.open(encoding="utf-8")))[args.index]
    sid = row["sample_id"]
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), CKPT, device)
    rs = Respaced(diffusion, model_fn, device, backward_null=True)
    dpm = DPMSolverpp(rs)
    b = schedule(args.steps)

    x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
    out = {"x0": _hw(x0), "sample_id": np.array(sid), "steps": np.array(args.steps)}
    for scheme in ("P", "S2fn"):
        x_rec, anon = pipeline(dpm, x0, sid, scheme, b)
        out[f"anon_{scheme}"] = _hw(anon)
        out[f"rec_{scheme}"] = _hw(x_rec)
        out[f"psnr_{scheme}"] = np.array(float(image_pair_metrics("", x_rec, x0)["_psnr_db"]))
        a = _hw(anon)
        print(f"{scheme}: recon PSNR={float(out[f'psnr_{scheme}']):.2f} dB | "
              f"anon float range=[{a.min():.3f},{a.max():.3f}] "
              f"frac outside [0,1]={(np.mean((a < 0) | (a > 1))):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
