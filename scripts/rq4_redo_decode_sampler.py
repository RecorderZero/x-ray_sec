#!/usr/bin/env python3
"""RQ4 re-report, step 1: decode-convention x sampler decomposition (AF-044).

The frozen Q8_FORMAL fidelity used the predecessor's *legacy* backward (a decode
grid off-by-several vs the textbook-aligned grid; forward is bit-identical, so the
whole gap is in the backward -- see scripts/backward_offset_sweep.py).  Before any
RQ4 number is re-reported on DPM-Solver++, we must separate two gains that would
otherwise be conflated:

  (1) decode-convention gain : legacy-DDIM  ->  aligned-DDIM   (same sampler, N=499)
  (2) sampler gain           : aligned-DDIM ->  aligned-DPM++  (same convention, each N)

Holding storage (range-preserving PNG) and scheme fixed, we run the full 4-pass
deployment pipeline with each backward and report the full fidelity metric set
(PSNR/SSIM/cosine/L-inf/RMSE/MAE) of the recovery vs x0, plus the single
round-trip (anon vs x0).  This is the aligned-decode re-baseline the auditor asked
for; the formal N=200 re-report builds on it.

  smoke: scripts/run_cfg_ddim.sh python -m scripts.rq4_redo_decode_sampler --smoke
  full : scripts/run_cfg_ddim.sh python -m scripts.rq4_redo_decode_sampler
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_legacy_wrapper import legacy_invert, legacy_generate
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.m1_storage import range_preserving_png_handoff
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED, TOP
from scripts.exact_inv_sweep import DPMSolverpp, CKPT
from scripts.deploy_s2_sweep import encrypt, decrypt

FULL_INDICES = (0, 1, 10, 11)
SMOKE_INDICES = (0, 1)
SWEEP_N = (499, 200, 100)
SCHEMES = ("P", "S2fn")
METRICS = ("psnr_db", "ssim", "cosine", "linf", "rmse", "mae")


class LegacyBackend:
    """Frozen convention: legacy_invert / legacy_generate (Delta t=1, T=500).

    Dense only -- ignores the respaced boundary grid, so it is run at N=499 to
    represent the Q8_FORMAL decode convention on the same deployment pipeline.
    """

    def __init__(self, diffusion, model_fn):
        self.d, self.m = diffusion, model_fn

    def forward(self, x0, b):
        return legacy_invert(self.d, self.m, x0, TOP + 1)

    def backward(self, z, b):
        return legacy_generate(self.d, self.m, z, TOP + 1)


@torch.inference_mode()
def pipeline(sampler, x0, sid, scheme, b):
    z = sampler.forward(x0, b)
    zc, orig = encrypt(z, scheme, sid)
    anon = sampler.backward(zc, b)
    anon_png, _ = range_preserving_png_handoff(anon)
    zc2 = sampler.forward(anon_png, b)
    x_rec = sampler.backward(decrypt(zc2, scheme, sid, orig), b)
    return x_rec, anon


def _mdict(x_rec, x0):
    m = image_pair_metrics("", x_rec, x0)
    return {k: round(float(m[f"_{k}"]), 4) for k in METRICS}


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/rq4_redo"))
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
    legacy = LegacyBackend(diffusion, model_fn)
    # (sampler, label, Ns): legacy only dense; aligned samplers across the sweep.
    plan = [(legacy, "legacy-DDIM", (499,)), (rs, "aligned-DDIM", SWEEP_N), (dpm, "aligned-DPM++", SWEEP_N)]

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    items = [(rows[i]["sample_id"], preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device))
             for i in indices]

    rs.backward(rs.forward(items[0][1], schedule(10)), schedule(10))  # warm-up
    records = []
    for sampler, label, Ns in plan:
        for n in Ns:
            b = schedule(n)
            for sch in SCHEMES:
                per = []
                for sid, x0 in items:
                    x_rec, anon = pipeline(sampler, x0, sid, sch, b)
                    md = _mdict(x_rec, x0)
                    md["single_roundtrip_psnr_db"] = round(float(image_pair_metrics("", anon, x0)["_psnr_db"]), 4)
                    md.update({"sampler": label, "N": n, "scheme": sch, "sample_id": sid,
                               "finite": bool(torch.isfinite(x_rec).all())})
                    per.append(md); records.append(md)
                agg = {k: round(float(np.mean([r[k] for r in per])), 3) for k in (*METRICS, "single_roundtrip_psnr_db")}
                print(f"{label:>14} N={n:>4} {sch:>5}  " +
                      "  ".join(f"{k}={agg[k]}" for k in ("psnr_db", "ssim", "cosine", "linf")), flush=True)

    # decomposition summary (scheme-wise, at N=499 for the convention delta)
    def mean_psnr(label, n, sch):
        v = [r["psnr_db"] for r in records if r["sampler"] == label and r["N"] == n and r["scheme"] == sch]
        return round(float(np.mean(v)), 3) if v else None

    decomp = {}
    for sch in SCHEMES:
        leg = mean_psnr("legacy-DDIM", 499, sch)
        ddim499 = mean_psnr("aligned-DDIM", 499, sch)
        decomp[sch] = {
            "legacy_ddim_N499": leg, "aligned_ddim_N499": ddim499,
            "decode_convention_gain_db": round(ddim499 - leg, 3),
            "sampler_gain_aligned_N100_db": round(mean_psnr("aligned-DPM++", 100, sch) - mean_psnr("aligned-DDIM", 100, sch), 3),
            "dpmpp_N100": mean_psnr("aligned-DPM++", 100, sch),
            "dpmpp_N200": mean_psnr("aligned-DPM++", 200, sch),
        }

    name = "smoke.json" if args.smoke else "decompose.json"
    (args.out_dir / name).write_text(json.dumps(
        {"seed": SEED, "smoke": args.smoke, "indices": indices, "storage": "range_preserving_png/v1",
         "metrics": list(METRICS), "decomposition": decomp, "per_sample": records},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== decode-convention vs sampler decomposition (PSNR dB, recon vs x0) ===")
    for sch in SCHEMES:
        d = decomp[sch]
        print(f"[{sch}] legacy-DDIM(499)={d['legacy_ddim_N499']}  aligned-DDIM(499)={d['aligned_ddim_N499']}  "
              f"=> decode-convention +{d['decode_convention_gain_db']}  | "
              f"aligned-DPM++(100)={d['dpmpp_N100']} (sampler gain @N=100 +{d['sampler_gain_aligned_N100_db']})")
    print(f"\nwrote {args.out_dir}/{name}")


if __name__ == "__main__":
    main()
