#!/usr/bin/env python3
"""Deployment-realistic step sweep: aligned decode + range-preserving PNG, S2.

Full M1 pipeline per image/scheme (aligned respaced DDIM decode throughout):

  z = F(x0)
  anon = B(E(z))
  anon_png = range_preserving_PNG(anon)        # real store + read-back (AF-024)
  zhat' = F(anon_png)
  x_rec = B(D(zhat'))

Schemes:
  P    : E = D = identity (baseline, double round-trip)
  S2a  : SOT orthogonal transform (norm-preserving)
  S2fn : S2 fixed-norm v2 fidelity -- normalize_to_fixed_norm -> transform ...
         inverse_transform -> restore_original_norm.  The AEAD container only
         protects the norm/lo-hi from tampering (security), not fidelity, so we
         carry the exact original norm directly (= what AEAD delivers byte-exact).

Records recon PSNR vs x0 (double round-trip, the recovery) AND anon PSNR vs x0
(single round-trip; for P this is the ~46 dB figure, shown for comparison).

  scripts/run_cfg_ddim.sh python -m scripts.deploy_s2_sweep
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.m1_storage import range_preserving_png_handoff
from scripts.respaced_ddim_sweep import Respaced, schedule, psnr, SEED, TOP
from src.crypto.sot import transform, inverse_transform
from src.crypto.s2_fixed_norm import normalize_to_fixed_norm, restore_original_norm

CKPT = Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt")
INDICES = (0, 1, 10, 11)
SWEEP_N = (499, 400, 200, 100, 50, 25)
SCHEMES = ("P", "S2a", "S2fn")
S2_KEY = hashlib.sha256(b"cxr-s2/D6/public-reproducibility-fixture/v1").digest()
S2_ROUNDS = 2
S2_PERM = True


def nonce(sid: str) -> bytes:
    return hashlib.sha256(f"cxr-s2/deploy-sweep/{sid}".encode()).digest()[:16]


def encrypt(z, scheme, sid):
    if scheme == "P":
        return z, None
    if scheme == "S2a":
        return transform(z, S2_KEY, nonce(sid), rounds=S2_ROUNDS, use_permutation=S2_PERM), None
    normalized, orig = normalize_to_fixed_norm(z)
    return transform(normalized, S2_KEY, nonce(sid), rounds=S2_ROUNDS, use_permutation=S2_PERM), orig


def decrypt(zc, scheme, sid, orig):
    if scheme == "P":
        return zc
    if scheme == "S2a":
        return inverse_transform(zc, S2_KEY, nonce(sid), rounds=S2_ROUNDS, use_permutation=S2_PERM)
    inv = inverse_transform(zc, S2_KEY, nonce(sid), rounds=S2_ROUNDS, use_permutation=S2_PERM)
    return restore_original_norm(inv, orig)


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    out_dir = Path("paper_assets/deploy_s2_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), CKPT, device)
    rs = Respaced(diffusion, model_fn, device)

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    items = [(rows[i]["sample_id"], preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device))
             for i in INDICES]

    # crypto round-trip sanity (latent only): D(E(z)) ~= z
    z0 = rs.forward(items[0][1], schedule(TOP + 1))
    for sch in SCHEMES:
        zc, orig = encrypt(z0, sch, items[0][0])
        rt = float((decrypt(zc, sch, items[0][0], orig) - z0).abs().max())
        print(f"[crypto-sanity] {sch}: max|D(E(z))-z| = {rt:.3e}", flush=True)

    rs.backward(rs.forward(items[0][1], schedule(10)), schedule(10))  # warm-up

    records = []
    for n in SWEEP_N:
        b = schedule(n)
        for sid, x0 in items:
            for sch in SCHEMES:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                z = rs.forward(x0, b)
                zc, orig = encrypt(z, sch, sid)
                anon = rs.backward(zc, b)
                anon_png, _ = range_preserving_png_handoff(anon)
                zc2 = rs.forward(anon_png, b)
                x_rec = rs.backward(decrypt(zc2, sch, sid, orig), b)
                torch.cuda.synchronize()
                records.append({
                    "N": n, "scheme": sch, "sample_id": sid,
                    "recon_psnr_vs_x0_db": psnr(x_rec, x0),
                    "anon_png_vs_anon_psnr_db": psnr(anon_png, anon),
                    "single_roundtrip_psnr_vs_x0_db": psnr(anon, x0),
                    "pipeline_seconds": round(time.perf_counter() - t0, 3),
                })
        done = [r for r in records if r["N"] == n]
        print(f"N={n}: " + "  ".join(f"{s}={np.mean([r['recon_psnr_vs_x0_db'] for r in done if r['scheme']==s]):.2f}"
                                     for s in SCHEMES), flush=True)

    summary = []
    for n in SWEEP_N:
        pmean = float(np.mean([r["recon_psnr_vs_x0_db"] for r in records if r["N"] == n and r["scheme"] == "P"]))
        for sch in SCHEMES:
            rr = [r for r in records if r["N"] == n and r["scheme"] == sch]
            m = float(np.mean([r["recon_psnr_vs_x0_db"] for r in rr]))
            summary.append({"N": n, "scheme": sch, "mean_recon_psnr_db": round(m, 2),
                            "gap_vs_P_db": round(m - pmean, 3),
                            "mean_single_roundtrip_db": round(float(np.mean([r["single_roundtrip_psnr_vs_x0_db"] for r in rr])), 2),
                            "mean_pipeline_seconds": round(float(np.mean([r["pipeline_seconds"] for r in rr])), 2)})

    (out_dir / "sweep.json").write_text(json.dumps(
        {"seed": SEED, "indices": list(INDICES), "schemes": list(SCHEMES), "storage": "range_preserving_png/v1",
         "s2_rounds": S2_ROUNDS, "s2_use_permutation": S2_PERM, "summary": summary, "per_sample": records},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "scheme", "mean_recon_psnr_db", "gap_vs_P_db", "mean_single_roundtrip_db", "mean_pipeline_seconds"])
        for s in summary:
            w.writerow([s["N"], s["scheme"], s["mean_recon_psnr_db"], s["gap_vs_P_db"],
                        s["mean_single_roundtrip_db"], s["mean_pipeline_seconds"]])

    print(f"\n=== deployment (range-preserving PNG) recon PSNR vs x0 (mean {len(items)} imgs) ===")
    print(f"{'N':>5} {'P':>8} {'S2a':>8} {'S2fn':>8}  {'S2fn-P':>8} {'s/4pass':>8}")
    for n in SWEEP_N:
        v = {s["scheme"]: s for s in summary if s["N"] == n}
        print(f"{n:>5} {v['P']['mean_recon_psnr_db']:>8.2f} {v['S2a']['mean_recon_psnr_db']:>8.2f} "
              f"{v['S2fn']['mean_recon_psnr_db']:>8.2f}  {v['S2fn']['gap_vs_P_db']:>+8.2f} "
              f"{v['P']['mean_pipeline_seconds']:>8.2f}")
    print(f"\nwrote {out_dir}/sweep.json + .csv")


if __name__ == "__main__":
    main()
