#!/usr/bin/env python3
"""Controlled check: is the 46 vs 42 dB gap the legacy backward off-by-one?

Isolates the cause of respaced DDIM's higher full-density round-trip PSNR by
holding the latent fixed and swapping only the backward pass:

  0. z_mine = respaced_forward(x0)  vs  z_legacy = legacy_invert(x0)
     -> max|z_mine - z_legacy| should be ~0 (forward identical => not the cause).
  On that SAME z, N=499:
  A. aligned   : my backward, eval timestep == state's noise level (499..1)
  B. legacy    : the predecessor's ddim_anonymization_backward (legacy_generate)
  C. shifted   : my backward but eval one timestep BELOW the state (mimic legacy)

ACTUAL RESULT (not the original hypothesis): forward z max|mine-legacy| = 0
(so the gap is entirely in the backward pass), A aligned = 47.4 dB,
B legacy = 42.8 dB, but C "one-below" COLLAPSED to 6.4 dB -- i.e. a persistent
one-step-below shift is NOT what legacy does.  Conclusion: the gap lives in the
backward decode grid, but the exact legacy mechanism is NOT a clean one-step
offset and remains [推定].  See backward_offset_sweep.py for the dose-response.

  scripts/run_cfg_ddim.sh python -m scripts.backward_offbyone_check
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

INDICES = (0, 1)  # two images is enough to localize a 4 dB systematic gap
CKPT = Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt")


@torch.inference_mode()
def backward_shifted(rs: Respaced, z, boundaries):
    """Reintroduce the legacy off-by-one: eval one timestep below the state."""
    mk = rs._base_kwargs(z.shape[0])
    x = z
    for k in range(len(boundaries) - 1, 0, -1):
        t_eval = max(int(boundaries[k]) - 1, 0)   # <-- one below the true level
        x = rs._step(x, t_eval, int(boundaries[k - 1]), mk)
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
    b = schedule(TOP + 1)  # full density: boundaries [0..499]

    rows = list(csv.DictReader(Path("splits/dev_v1.1.csv").open(encoding="utf-8")))
    records = []
    for idx in INDICES:
        x0 = preprocess(Path(rows[idx]["local_path"]))[0].unsqueeze(0).to(device)
        z_mine = rs.forward(x0, b)
        z_legacy = legacy_invert(diffusion, model_fn, x0, 500)
        fwd_max_abs = float((z_mine - z_legacy).abs().max())

        xA = rs.backward(z_mine, b)                       # aligned (mine)
        xB = legacy_generate(diffusion, model_fn, z_mine, 500)  # legacy backward, same z
        xC = backward_shifted(rs, z_mine, b)              # mine + reintroduced shift
        rec = {
            "sample_id": rows[idx]["sample_id"],
            "forward_z_max_abs_mine_vs_legacy": fwd_max_abs,
            "A_aligned_psnr": psnr(xA, x0),
            "B_legacy_psnr": psnr(xB, x0),
            "C_shifted_psnr": psnr(xC, x0),
        }
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    mean = {k: float(np.mean([r[k] for r in records]))
            for k in ("forward_z_max_abs_mine_vs_legacy", "A_aligned_psnr", "B_legacy_psnr", "C_shifted_psnr")}
    out = Path("paper_assets/respaced_ddim_sweep/backward_offbyone_check.json")
    out.write_text(json.dumps({"seed": SEED, "indices": list(INDICES),
                               "mean": mean, "per_sample": records}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("\n=== mean over %d images (full density N=499) ===" % len(records))
    print(f"  forward z max|mine-legacy|      = {mean['forward_z_max_abs_mine_vs_legacy']:.3e}  (0 => forward identical)")
    print(f"  A aligned backward (mine)  PSNR = {mean['A_aligned_psnr']:.2f} dB")
    print(f"  B legacy backward          PSNR = {mean['B_legacy_psnr']:.2f} dB")
    print(f"  C shifted backward (mine)  PSNR = {mean['C_shifted_psnr']:.2f} dB")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
