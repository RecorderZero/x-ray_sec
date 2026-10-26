#!/usr/bin/env python3
"""B1 on the ENCRYPTED pipeline, using the aligned respaced DDIM decode.

The key-free sweep (respaced_ddim_sweep.py) was a best case: encryption scatters
the invert/regen error into generator-sensitive directions, so the real question
is how P / S0 / S1 / S2 recover x0 through the full encrypt->generate->invert->
decrypt->generate chain, and whether S2 keeps up with P.

Uses the aligned forward/backward (the ~46 dB decode) consistently; float
hand-off (no PNG) to isolate the sampler+cipher from storage.  Four passes per
image per scheme:  z=F(x0); anon=B(E(z)); zhat'=F(anon); x_rec=B(D(zhat')).

  P   : E = D = identity
  S0  : E/D = legacy Rademacher sign flip (apply_legacy_key, self-inverse)
  S1  : E/D = legacy signed permutation (apply / apply_inverse)
  S2a : E/D = SOT orthogonal structured transform (src.crypto.sot), norm-
        preserving; the fixed-norm S2 v2 adds AEAD norm restore (separate).

  scripts/run_cfg_ddim.sh python -m scripts.aligned_encrypted_sweep
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
from scripts.e2_legacy_wrapper import apply_legacy_key
from scripts.respaced_ddim_sweep import Respaced, schedule, psnr, SEED, TOP
from src.crypto.sot import transform, inverse_transform

CKPT = Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt")
SOURCE_ROOT = Path("past/SourceCode")
INDICES = (0, 1, 10, 11)
SWEEP_N = (499, 100, 50)
SCHEMES = ("P", "S0", "S1", "S2a")
S2_KEY = hashlib.sha256(b"cxr-s2/D6/public-reproducibility-fixture/v1").digest()
S2_ROUNDS = 2
S2_PERM = True


def s2_nonce(sample_id: str) -> bytes:
    return hashlib.sha256(f"cxr-s2/B1-aligned-sweep/{sample_id}".encode()).digest()[:16]


def encrypt(z, scheme, sid):
    if scheme == "P":
        return z
    if scheme in ("S0", "S1"):
        return apply_legacy_key(z, SOURCE_ROOT, scheme, "anonymize")
    return transform(z, S2_KEY, s2_nonce(sid), rounds=S2_ROUNDS, use_permutation=S2_PERM)


def decrypt(zc, scheme, sid):
    if scheme == "P":
        return zc
    if scheme in ("S0", "S1"):
        return apply_legacy_key(zc, SOURCE_ROOT, scheme, "deanonymize")
    return inverse_transform(zc, S2_KEY, s2_nonce(sid), rounds=S2_ROUNDS, use_permutation=S2_PERM)


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    out_dir = Path("paper_assets/aligned_encrypted_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, diffusion, model_fn = create_runtime(SOURCE_ROOT, CKPT, device)
    rs = Respaced(diffusion, model_fn, device)

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    items = [(rows[i]["sample_id"], preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device))
             for i in INDICES]

    # crypto round-trip sanity (latent, no generator): D(E(z)) must ~= z
    z0 = rs.forward(items[0][1], schedule(TOP + 1))
    for sch in SCHEMES:
        rt = float((decrypt(encrypt(z0, sch, items[0][0]), sch, items[0][0]) - z0).abs().max())
        print(f"[crypto-sanity] {sch}: max|D(E(z))-z| = {rt:.3e}", flush=True)

    rs.backward(rs.forward(items[0][1], schedule(10)), schedule(10))  # warm-up

    records = []
    for n in SWEEP_N:
        b = schedule(n)
        for sid, x0 in items:
            for sch in SCHEMES:
                t0 = time.perf_counter(); torch.cuda.synchronize()
                z = rs.forward(x0, b)
                anon = rs.backward(encrypt(z, sch, sid), b)
                zc = rs.forward(anon, b)
                x_rec = rs.backward(decrypt(zc, sch, sid), b)
                torch.cuda.synchronize()
                records.append({
                    "N": n, "scheme": sch, "sample_id": sid,
                    "recon_psnr_vs_x0_db": psnr(x_rec, x0),
                    "anon_psnr_vs_x0_db": psnr(anon, x0),
                    "pipeline_seconds": round(time.perf_counter() - t0, 3),
                })
        print(json.dumps([r for r in records if r["N"] == n], ensure_ascii=False)[:400] + " ...", flush=True)

    # summary: mean recon PSNR per (N, scheme) and gap vs P
    summary = []
    for n in SWEEP_N:
        pmean = float(np.mean([r["recon_psnr_vs_x0_db"] for r in records if r["N"] == n and r["scheme"] == "P"]))
        for sch in SCHEMES:
            rr = [r for r in records if r["N"] == n and r["scheme"] == sch]
            m = float(np.mean([r["recon_psnr_vs_x0_db"] for r in rr]))
            summary.append({"N": n, "scheme": sch,
                            "mean_recon_psnr_db": round(m, 2),
                            "gap_vs_P_db": round(m - pmean, 3),
                            "mean_pipeline_seconds": round(float(np.mean([r["pipeline_seconds"] for r in rr])), 2)})

    (out_dir / "sweep.json").write_text(json.dumps(
        {"seed": SEED, "indices": list(INDICES), "schemes": list(SCHEMES),
         "s2_rounds": S2_ROUNDS, "s2_use_permutation": S2_PERM, "handoff": "float",
         "summary": summary, "per_sample": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "scheme", "mean_recon_psnr_db", "gap_vs_P_db", "mean_pipeline_seconds"])
        for s in summary:
            w.writerow([s["N"], s["scheme"], s["mean_recon_psnr_db"], s["gap_vs_P_db"], s["mean_pipeline_seconds"]])

    print(f"\n=== aligned encrypted end-to-end recon PSNR vs x0 (mean over {len(items)} imgs, float handoff) ===")
    print(f"{'N':>5} {'P':>8} {'S0':>8} {'S1':>8} {'S2a':>8}   (gap vs P)")
    for n in SWEEP_N:
        vals = {s["scheme"]: s["mean_recon_psnr_db"] for s in summary if s["N"] == n}
        gaps = {s["scheme"]: s["gap_vs_P_db"] for s in summary if s["N"] == n}
        print(f"{n:>5} {vals['P']:>8.2f} {vals['S0']:>8.2f} {vals['S1']:>8.2f} {vals['S2a']:>8.2f}   "
              f"S0 {gaps['S0']:+.2f}  S1 {gaps['S1']:+.2f}  S2a {gaps['S2a']:+.2f}")
    print(f"\nwrote {out_dir}/sweep.json + .csv")


if __name__ == "__main__":
    main()
