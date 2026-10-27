#!/usr/bin/env python3
"""Deployment-realistic step sweep with DPM-Solver++ vs DDIM.

Same full M1 pipeline as `scripts.deploy_s2_sweep` (aligned decode + real
range-preserving PNG store/read-back + S2 encryption), but run with each sampler
so we see whether DPM-Solver++'s key-free round-trip advantage survives the
deployment path (encryption + lossy-ish PNG handoff) at matched NFE:

  z = Enc_sampler.forward(x0)
  anon = Enc_sampler.backward(E(z))
  anon_png = range_preserving_PNG(anon)          # real store + read-back
  zhat' = Dec_sampler.forward(anon_png)
  x_rec = Dec_sampler.backward(D(zhat'))

Samplers (identical .forward/.backward interface): DDIM (respaced), DPM++ (2M).
Schemes: P (identity baseline, double round-trip) and S2fn (shipped fixed-norm).
NFE == len(boundaries)-1 per pass for both samplers, so pipeline_seconds compares
apples-to-apples.  Reports recon PSNR vs x0 (the recovery) and single round-trip.

  smoke: scripts/run_cfg_ddim.sh python -m scripts.deploy_dpm_sweep --smoke
  full : scripts/run_cfg_ddim.sh python -m scripts.deploy_dpm_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.m1_storage import range_preserving_png_handoff
from scripts.respaced_ddim_sweep import Respaced, schedule, psnr, SEED, TOP
from scripts.exact_inv_sweep import DPMSolverpp, CKPT
from scripts.deploy_s2_sweep import encrypt, decrypt, SCHEMES as ALL_SCHEMES  # noqa: F401

FULL_INDICES = (0, 1, 10, 11)
SMOKE_INDICES = (0, 1)
SWEEP_N = (499, 200, 100, 50, 25)
SCHEMES = ("P", "S2fn")


@torch.inference_mode()
def pipeline(sampler, x0, sid, scheme, b):
    z = sampler.forward(x0, b)
    zc, orig = encrypt(z, scheme, sid)
    anon = sampler.backward(zc, b)
    anon_png, _ = range_preserving_png_handoff(anon)
    zc2 = sampler.forward(anon_png, b)
    x_rec = sampler.backward(decrypt(zc2, scheme, sid, orig), b)
    return x_rec, anon


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/deploy_dpm_sweep"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    indices = list(SMOKE_INDICES if args.smoke else FULL_INDICES)

    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), CKPT, device)
    rs = Respaced(diffusion, model_fn, device)
    samplers = {"DDIM": rs, "DPM++": DPMSolverpp(rs)}

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    items = [(rows[i]["sample_id"], preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device))
             for i in indices]

    # crypto round-trip sanity on each sampler's latent: D(E(z)) ~= z
    for sname, s in samplers.items():
        z0 = s.forward(items[0][1], schedule(TOP + 1))
        for sch in SCHEMES:
            zc, orig = encrypt(z0, sch, items[0][0])
            rt = float((decrypt(zc, sch, items[0][0], orig) - z0).abs().max())
            print(f"[crypto-sanity] {sname}/{sch}: max|D(E(z))-z| = {rt:.3e}", flush=True)

    rs.backward(rs.forward(items[0][1], schedule(10)), schedule(10))  # warm-up

    records = []
    for n in SWEEP_N:
        b = schedule(n)
        for sid, x0 in items:
            for sname, s in samplers.items():
                for sch in SCHEMES:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    x_rec, anon = pipeline(s, x0, sid, sch, b)
                    torch.cuda.synchronize()
                    records.append({
                        "N": n, "sampler": sname, "scheme": sch, "sample_id": sid,
                        "recon_psnr_vs_x0_db": psnr(x_rec, x0),
                        "single_roundtrip_psnr_vs_x0_db": psnr(anon, x0),
                        "finite": bool(torch.isfinite(x_rec).all()),
                        "pipeline_seconds": round(time.perf_counter() - t0, 3),
                    })
        done = [r for r in records if r["N"] == n]
        msg = []
        for sname in samplers:
            for sch in SCHEMES:
                v = np.mean([r["recon_psnr_vs_x0_db"] for r in done if r["sampler"] == sname and r["scheme"] == sch])
                msg.append(f"{sname}/{sch}={v:.2f}")
        print(f"N={n}: " + "  ".join(msg), flush=True)

    summary = []
    for n in SWEEP_N:
        for sname in samplers:
            for sch in SCHEMES:
                rr = [r for r in records if r["N"] == n and r["sampler"] == sname and r["scheme"] == sch]
                summary.append({
                    "N": n, "sampler": sname, "scheme": sch,
                    "mean_recon_psnr_db": round(float(np.mean([r["recon_psnr_vs_x0_db"] for r in rr])), 2),
                    "mean_single_roundtrip_db": round(float(np.mean([r["single_roundtrip_psnr_vs_x0_db"] for r in rr])), 2),
                    "mean_pipeline_seconds": round(float(np.mean([r["pipeline_seconds"] for r in rr])), 2),
                    "all_finite": all(r["finite"] for r in rr),
                })

    name = "smoke.json" if args.smoke else "sweep.json"
    (args.out_dir / name).write_text(json.dumps(
        {"seed": SEED, "smoke": args.smoke, "indices": indices, "samplers": list(samplers),
         "schemes": list(SCHEMES), "storage": "range_preserving_png/v1",
         "summary": summary, "per_sample": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        with (args.out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["N", "sampler", "scheme", "mean_recon_psnr_db", "mean_single_roundtrip_db", "mean_pipeline_seconds"])
            for s in summary:
                w.writerow([s["N"], s["sampler"], s["scheme"], s["mean_recon_psnr_db"],
                            s["mean_single_roundtrip_db"], s["mean_pipeline_seconds"]])

    print(f"\n=== deployment recon PSNR vs x0 (range-preserving PNG; mean {len(items)} imgs) ===")
    print(f"{'N':>5} {'DDIM/P':>8} {'DPM/P':>8} {'DDIM/S2fn':>10} {'DPM/S2fn':>9}  {'DPM-DDIM(S2fn)':>14} {'s/4pass':>8}")
    for n in SWEEP_N:
        v = {(s["sampler"], s["scheme"]): s for s in summary if s["N"] == n}
        dgain = v[("DPM++", "S2fn")]["mean_recon_psnr_db"] - v[("DDIM", "S2fn")]["mean_recon_psnr_db"]
        print(f"{n:>5} {v[('DDIM','P')]['mean_recon_psnr_db']:>8.2f} {v[('DPM++','P')]['mean_recon_psnr_db']:>8.2f} "
              f"{v[('DDIM','S2fn')]['mean_recon_psnr_db']:>10.2f} {v[('DPM++','S2fn')]['mean_recon_psnr_db']:>9.2f}  "
              f"{dgain:>+14.2f} {v[('DPM++','S2fn')]['mean_pipeline_seconds']:>8.2f}")
    print(f"\nwrote {args.out_dir}/{name}")


if __name__ == "__main__":
    main()
