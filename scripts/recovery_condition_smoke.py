#!/usr/bin/env python3
"""Smoke: should the recovery backward be unconditional (null) or y=0 conditional?

The inversion is unconditional (`forward` sets null=True), but the recovery
backward currently conditions on the healthy class y=0 (`_base_kwargs`, no null)
-- an asymmetry inherited from the predecessor.  Two hypotheses motivate testing
the matched (null) recovery:

  (1) Fidelity: DDIM inversion computes z under the *null* model, so retracing
      the ODE under the *null* model (not y=0) should reconstruct x0 more
      faithfully (matched forward/backward trajectory).
  (2) Localization: the heatmap is |recovery - healthy(CFG toward y=0)|.  A y=0
      recovery is already biased toward healthy, shrinking the lesion signal; a
      null recovery stays neutral to class, so the difference vs the guided
      healthy should be larger and more lesion-specific.

Holding the inversion at null, we compare recovery under {null, y0} for both
samplers at N=100 on dev images, reporting image-space fidelity (PSNR/SSIM/cos
vs x0) AND the gs=2 difference-map magnitude under each recovery condition.
null recovery with null=False reproduces the shipped `sampler.backward` exactly.

  scripts/run_cfg_ddim.sh python -m scripts.recovery_condition_smoke
  scripts/run_cfg_ddim.sh python -m scripts.recovery_condition_smoke --smoke
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
from scripts.exact_inv_sweep import CKPT, DPMSolverpp
from scripts.localization_sampler_smoke import canonical_cfg_backward
from scripts.respaced_ddim_sweep import Respaced, SEED, schedule

FULL_INDICES = (0, 1, 10, 11)
SMOKE_INDICES = (0, 1)
N = 100
GS = 2.0  # pseudo-healthy guidance for the localization-signal probe


@torch.inference_mode()
def recovery_backward(sampler, z, b, null: bool):
    """sampler.backward, but with the recovery conditioning made explicit.

    null=False is bit-identical to the shipped sampler.backward (y=0 conditional);
    null=True matches the unconditional inversion.
    """
    rs = sampler.rs if isinstance(sampler, DPMSolverpp) else sampler
    mk = rs._base_kwargs(z.shape[0])
    if null:
        mk["null"] = True
    if isinstance(sampler, DPMSolverpp):
        order = [(int(b[i]), int(b[i - 1])) for i in range(len(b) - 1, 0, -1)]
        return sampler._run(z, order, mk)
    x = z
    for k in range(len(b) - 1, 0, -1):
        x = rs._step(x, b[k], b[k - 1], mk)
    return x


def _fidelity(x_rec, x0):
    m = image_pair_metrics("", x0, x_rec)
    return {"psnr_db": round(float(m["_psnr_db"]), 3),
            "ssim": round(float(m["_ssim"]), 4),
            "cosine": round(float(m["_cosine"]), 6)}


def _map_stats(m):
    a = m.detach().double().flatten()
    return {"mean": round(float(a.mean()), 6), "max": round(float(a.max()), 6),
            "fraction_gt_001": round(float((a > 0.01).double().mean()), 6)}


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/recovery_condition"))
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
    samplers = {"aligned-DDIM": rs, "DPM++2M": dpm}

    rows = list(csv.DictReader(args.split.open(encoding="utf-8")))
    items = [(rows[i]["sample_id"], preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device))
             for i in indices]

    rs.backward(rs.forward(items[0][1], schedule(10)), schedule(10))  # warm-up

    records = []
    for name, sampler in samplers.items():
        for sid, x0 in items:
            z = sampler.forward(x0, b)                       # unconditional inversion
            healthy = canonical_cfg_backward(sampler, z.clone(), b, GS)
            rec = {}
            for cond in ("null", "y0"):
                x_rec = recovery_backward(sampler, z.clone(), b, null=(cond == "null"))
                heat = (x_rec - healthy).abs()
                rec[cond] = {"fidelity": _fidelity(x_rec, x0), "gs2_map": _map_stats(heat)}
            row = {"sampler": name, "sample_id": sid, **rec}
            records.append(row)
            print(f"{name:>12} {sid:<16} "
                  f"null psnr={rec['null']['fidelity']['psnr_db']:.2f} cos={rec['null']['fidelity']['cosine']:.5f} "
                  f"map_mean={rec['null']['gs2_map']['mean']:.5f} | "
                  f"y0 psnr={rec['y0']['fidelity']['psnr_db']:.2f} cos={rec['y0']['fidelity']['cosine']:.5f} "
                  f"map_mean={rec['y0']['gs2_map']['mean']:.5f}", flush=True)

    agg = {}
    for name in samplers:
        rr = [r for r in records if r["sampler"] == name]
        a = {}
        for cond in ("null", "y0"):
            a[cond] = {
                "mean_psnr_db": round(float(np.mean([r[cond]["fidelity"]["psnr_db"] for r in rr])), 3),
                "mean_ssim": round(float(np.mean([r[cond]["fidelity"]["ssim"] for r in rr])), 4),
                "mean_cosine": round(float(np.mean([r[cond]["fidelity"]["cosine"] for r in rr])), 6),
                "mean_gs2_map_mean": round(float(np.mean([r[cond]["gs2_map"]["mean"] for r in rr])), 6),
                "mean_gs2_map_frac_gt_001": round(float(np.mean([r[cond]["gs2_map"]["fraction_gt_001"] for r in rr])), 6),
            }
        a["psnr_gain_null_minus_y0_db"] = round(a["null"]["mean_psnr_db"] - a["y0"]["mean_psnr_db"], 3)
        a["map_mean_ratio_null_over_y0"] = round(a["null"]["mean_gs2_map_mean"] / a["y0"]["mean_gs2_map_mean"], 3) \
            if a["y0"]["mean_gs2_map_mean"] > 0 else None
        agg[name] = a

    name_out = "smoke.json" if args.smoke else "recovery_condition.json"
    (args.out_dir / name_out).write_text(json.dumps(
        {"seed": SEED, "smoke": args.smoke, "N": N, "gs": GS, "indices": indices,
         "inversion": "null=True (unconditional)",
         "recovery_conditions": {"null": "null=True matches inversion",
                                 "y0": "y=0 conditional == shipped sampler.backward"},
         "aggregate": agg, "per_sample": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== recovery condition: null (matched) vs y=0 (shipped); mean over images ===")
    for name in samplers:
        a = agg[name]
        print(f"{name:>12}  null: psnr={a['null']['mean_psnr_db']:.2f} ssim={a['null']['mean_ssim']:.4f} "
              f"map_mean={a['null']['mean_gs2_map_mean']:.5f}  |  "
              f"y0: psnr={a['y0']['mean_psnr_db']:.2f} ssim={a['y0']['mean_ssim']:.4f} "
              f"map_mean={a['y0']['mean_gs2_map_mean']:.5f}  |  "
              f"null-y0 psnr {a['psnr_gain_null_minus_y0_db']:+.2f} dB, "
              f"map ratio {a['map_mean_ratio_null_over_y0']}")
    print(f"\nwrote {args.out_dir}/{name_out}")


if __name__ == "__main__":
    main()
