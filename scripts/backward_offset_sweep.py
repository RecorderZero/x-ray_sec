#!/usr/bin/env python3
"""Dose-response probe: how much of the aligned-vs-legacy decode gap is a
uniform timestep offset in the backward pass?

Holds the latent z fixed (forward is bit-identical to legacy, verified = 0) and
decodes it with the aligned backward but pretending z sits `delta` timesteps
BELOW its true level 499.  delta=0 is aligned.

ACTUAL RESULT (not the original one-step hypothesis): delta0/1/2/3 =
47.4/47.0/44.7/42.2 dB and legacy backward = 42.8 dB, i.e. legacy sits near
delta2-3, NOT at a clean delta=1.  So the gap is DOMINATED by a decode-grid
misalignment in the legacy backward, but it is NOT a single-step offset -- the
exact mechanism in `ddim_anonymization_backward` remains [推定] (unlocated line).
This script documents that dose-response; it does not claim a one-liner root cause.

  scripts/run_cfg_ddim.sh python -m scripts.backward_offset_sweep
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_legacy_wrapper import legacy_invert, legacy_generate
from scripts.respaced_ddim_sweep import Respaced, schedule, psnr, SEED, TOP

INDICES = (0, 1)
DELTAS = (0, 1, 2, 3)
CKPT = Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt")


@torch.inference_mode()
def backward_offset(rs: Respaced, z, delta: int):
    """Aligned DDIM decode, but pretend z is `delta` steps below level TOP."""
    mk = rs._base_kwargs(z.shape[0])
    x = z
    for L in range(TOP - delta, 0, -1):   # eval at L (matched), jump to abar[L-1]
        x = rs._step(x, L, L - 1, mk)
    return x


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    source_root = Path("past/SourceCode")
    _, diffusion, model_fn = create_runtime(source_root, CKPT, device)
    rs = Respaced(diffusion, model_fn, device)
    b = schedule(TOP + 1)  # [0..499]

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    records = []
    for idx in INDICES:
        x0 = preprocess(Path(rows[idx]["local_path"]))[0].unsqueeze(0).to(device)
        z = rs.forward(x0, b)
        fwd_max_abs = float((z - legacy_invert(diffusion, model_fn, x0, 500)).abs().max())
        rec = {"sample_id": rows[idx]["sample_id"], "forward_z_max_abs_mine_vs_legacy": fwd_max_abs,
               "legacy_backward_psnr": psnr(legacy_generate(diffusion, model_fn, z, 500), x0)}
        for d in DELTAS:
            rec[f"delta{d}_psnr"] = psnr(backward_offset(rs, z, d), x0)
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    keys = ["forward_z_max_abs_mine_vs_legacy", "legacy_backward_psnr"] + [f"delta{d}_psnr" for d in DELTAS]
    mean = {k: float(np.mean([r[k] for r in records])) for k in keys}
    out = Path("paper_assets/respaced_ddim_sweep/backward_offset_sweep.json")
    out.write_text(json.dumps({"seed": SEED, "indices": list(INDICES), "deltas": list(DELTAS),
                               "mean": mean, "per_sample": records}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("\n=== mean over %d images (full density) ===" % len(records))
    print(f"  forward z max|mine-legacy| = {mean['forward_z_max_abs_mine_vs_legacy']:.3e}")
    for d in DELTAS:
        legacy = mean["legacy_backward_psnr"]
        tag = "  (~ legacy)" if abs(mean[f"delta{d}_psnr"] - legacy) < 1.0 else ""
        print(f"  delta={d} decode PSNR vs x0 = {mean[f'delta{d}_psnr']:.2f} dB{tag}")
    print(f"  legacy backward PSNR       = {mean['legacy_backward_psnr']:.2f} dB (reference)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
