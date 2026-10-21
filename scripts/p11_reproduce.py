#!/usr/bin/env python3
"""P11.2 key-result reproduction: re-derive the frozen E2.3 latent cache (N=10 slice).

The E2.3 DDIM-null latent cache is the deterministic input every Q8_FORMAL main
table derives from. This re-runs the x0 -> z_T inversion for the first N rows of
the frozen split on a fresh GPU process and checks the latents against the frozen
cache within tolerance (DDIM eta=0 is deterministic, so drift is only GPU float
non-determinism). Reproducing this reproduces the foundation of the main tables.

  scripts/run_cfg_ddim.sh python -m scripts.p11_reproduce --limit 10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, load_split, preprocess
from scripts.e2_legacy_wrapper import legacy_invert

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a"
CKPT = ROOT / "past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt"
NOISE_LEVEL = 500
# tolerance: deterministic DDIM, so expect tiny drift from GPU float non-determinism
COS_MIN = 0.9999
MAXABS_MAX = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--split", type=Path, default=ROOT / "splits/security_v1.csv")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "paper_assets/reproduction")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    args.out.mkdir(parents=True, exist_ok=True)

    frozen = np.load(CACHE / "latents.npy")  # (N,1,H,W)
    idx = {r["sample_id"]: int(r["cache_index"]) for r in
           csv.DictReader((CACHE / "per_sample.csv").open(newline="", encoding="utf-8"))}
    rows = load_split(args.split)[: args.limit]

    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(ROOT / "past/SourceCode", CKPT, device)

    records, cos_all, maxabs_all, bitexact = [], [], [], 0
    for r in rows:
        x0 = preprocess(Path(r["local_path"]))[0].unsqueeze(0).to(device)
        z_new = legacy_invert(diffusion, model_fn, x0, NOISE_LEVEL).detach().float().cpu().numpy()
        z_ref = frozen[idx[r["sample_id"]]].reshape(z_new.shape)
        maxabs = float(np.abs(z_new - z_ref).max())
        a, b = z_new.ravel(), z_ref.ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        bx = maxabs == 0.0
        bitexact += int(bx)
        cos_all.append(cos); maxabs_all.append(maxabs)
        records.append({"sample_id": r["sample_id"], "cosine": round(cos, 8),
                        "max_abs": round(maxabs, 8), "bit_exact": bx,
                        "within_tol": cos >= COS_MIN and maxabs <= MAXABS_MAX})
        print(f"{r['sample_id']}  cos={cos:.7f} max_abs={maxabs:.2e} bit_exact={bx}", flush=True)

    passed = all(x["within_tol"] for x in records)
    summary = {
        "task_id": "P11.2-reproduction", "target": "E2.3 latent cache (Q8_FORMAL input)",
        "cache_run": CACHE.name, "split": args.split.name, "n": len(records),
        "tolerance": {"cosine_min": COS_MIN, "max_abs_max": MAXABS_MAX},
        "result": {
            "all_within_tolerance": passed,
            "bit_exact_count": bitexact,
            "cosine_min": round(min(cos_all), 8), "cosine_mean": round(float(np.mean(cos_all)), 8),
            "max_abs_max": round(max(maxabs_all), 8), "max_abs_mean": round(float(np.mean(maxabs_all)), 8),
        },
        "note": ("Deterministic DDIM eta=0 inversion re-run on a fresh GPU process vs the "
                 "frozen E2.3 cache that Q8_FORMAL consumed. Within-tolerance reproduction of "
                 "this cache reproduces the deterministic input of every main table; residual "
                 "drift (if any) is GPU float non-determinism, not a pipeline change."),
        "per_sample": records,
    }
    (args.out / "reproduction_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.out / "reproduction_report.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["sample_id", "cosine", "max_abs", "bit_exact", "within_tol"],
                           lineterminator="\n")
        w.writeheader(); w.writerows(records)
    print(f"\nreproduction {'PASS' if passed else 'FAIL'}: bit_exact {bitexact}/{len(records)}, "
          f"cos_min {min(cos_all):.6f}, max_abs_max {max(maxabs_all):.2e}")
    print(f"wrote {args.out}/reproduction_report.{{json,csv}}")


if __name__ == "__main__":
    main()
