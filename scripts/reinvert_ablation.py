#!/usr/bin/env python3
"""Ablation: redundant re-inversion in the predecessor's healthy-generation step.

Question (user, 2026-10-21): in the deanonymize/anomaly flow the predecessor
feeds the recovered IMAGE x_rec into ddim_sample_loop_known, which re-inverts it
to a latent and then CFG-denoises. But we already hold the decrypted/inverted
latent x_rec_T. Does denoising straight from x_rec_T (skipping the re-inversion)
hurt quality, and is it faster?

Two variants, same CFG denoise, per image, per guidance scale:
  A (predecessor)  x_rec -> ddim_sample_loop_known  (re-invert x_rec -> z_T' -> CFG denoise)
  B (proposed)     x_rec_T -> ddim_anonymization_backward(guidance=gs)   (CFG denoise directly)

Here x_rec_T is produced by legacy_invert(x0) and x_rec by legacy_generate(x_rec_T),
exactly the predecessor's own building blocks (read-only import of past/SourceCode;
nothing there is modified). We compare healthy_A vs healthy_B (psnr/ssim/cosine),
their anomaly maps |x_rec - healthy|, and the wall-clock of the healthy step only
(x_rec is needed either way as the reconstruction / heatmap reference).

  scripts/run_cfg_ddim.sh python -m scripts.reinvert_ablation \
    --split splits/security_v1.csv --limit 10 --guidance 0.0 2.0
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.e1_ddim_runner import create_runtime, load_split, preprocess
from scripts.e2_legacy_wrapper import legacy_invert, legacy_generate, legacy_model_kwargs

SEED = 1911
NOISE_LEVEL = 500


def to_u8(t: torch.Tensor) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    a = (a + 1.0) / 2.0  # [-1,1] -> [0,1]
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return 99.0
    return 10.0 * np.log10((2.0 ** 2) / mse)  # data range 2 ([-1,1])


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().double(); y = b.flatten().double()
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    return float(((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten(), b.flatten()
    return float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-12))


@torch.inference_mode()
def healthy_variant_A(diffusion, model_fn, x_rec, gs, device):
    """Predecessor: re-invert x_rec then CFG denoise (ddim_sample_loop_known)."""
    sample, _x_noisy, _org = diffusion.ddim_sample_loop_known(
        model_fn, tuple(x_rec.shape), img=(x_rec,), org=x_rec,
        clip_denoised=True, model_kwargs=legacy_model_kwargs(x_rec.shape[0], device),
        device=device, noise_level=NOISE_LEVEL, guidance_scale=gs,
    )
    return sample


@torch.inference_mode()
def healthy_variant_B(diffusion, model_fn, x_rec_T, gs, device):
    """Proposed: CFG denoise straight from the decrypted latent."""
    return diffusion.ddim_anonymization_backward(
        model_fn, x_rec_T, NOISE_LEVEL, clip_denoised=True,
        model_kwargs=legacy_model_kwargs(x_rec_T.shape[0], device), guidance_scale=gs,
    )


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--guidance", type=float, nargs="+", default=[0.0, 2.0])
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/reinvert_ablation"))
    ap.add_argument("--fig-gs", type=float, default=2.0, help="which gs to render in the PNG grid")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_split(args.split)[: args.limit]
    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)

    records = []
    fig_rows = []
    for i, row in enumerate(rows):
        x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)  # (1,1,H,W)
        # shared, needed either way: decrypted/inverted latent + reconstruction image
        x_rec_T = legacy_invert(diffusion, model_fn, x0, NOISE_LEVEL)
        x_rec = legacy_generate(diffusion, model_fn, x_rec_T, NOISE_LEVEL)
        print(f"[{i}] {row['sample_id']}  recon psnr(x0)={psnr(x0, x_rec):.2f}", flush=True)

        per_img = {"sample_id": row["sample_id"], "recon_psnr_x0": round(psnr(x0, x_rec), 3)}
        fig_entry = {"x0": x0, "x_rec": x_rec}
        for gs in args.guidance:
            hB, tB = timed(lambda: healthy_variant_B(diffusion, model_fn, x_rec_T, gs, device), device)
            hA, tA = timed(lambda: healthy_variant_A(diffusion, model_fn, x_rec, gs, device), device)
            tag = f"gs{gs:g}"
            per_img.update({
                f"{tag}_psnr_AB": round(psnr(hA, hB), 3),
                f"{tag}_ssim_AB": round(ssim(hA, hB), 4),
                f"{tag}_cos_AB": round(cosine(hA, hB), 6),
                f"{tag}_heatmapA_mean": round(torch.mean((x_rec - hA).abs()).item(), 5),
                f"{tag}_heatmapB_mean": round(torch.mean((x_rec - hB).abs()).item(), 5),
                f"{tag}_time_A_s": round(tA, 3),
                f"{tag}_time_B_s": round(tB, 3),
                f"{tag}_speedup_A_over_B": round(tA / tB, 3),
            })
            print(f"    gs={gs:g}  psnr(A,B)={per_img[f'{tag}_psnr_AB']:.2f} "
                  f"cos={per_img[f'{tag}_cos_AB']:.5f}  tA={tA:.2f}s tB={tB:.2f}s "
                  f"({per_img[f'{tag}_speedup_A_over_B']:.2f}x)", flush=True)
            if abs(gs - args.fig_gs) < 1e-9:
                fig_entry[f"hA"] = hA
                fig_entry[f"hB"] = hB
        records.append(per_img)
        fig_rows.append(fig_entry)

    # --- figure grid: x0 | x_rec | healthy_A | healthy_B | 10x|A-B| ---
    cols = []
    for e in fig_rows:
        diff = (e["hA"] - e["hB"]).abs() * 10.0
        strip = [to_u8(e["x0"][0, 0]), to_u8(e["x_rec"][0, 0]),
                 to_u8(e["hA"][0, 0]), to_u8(e["hB"][0, 0]),
                 np.clip((diff[0, 0].detach().cpu().numpy()) / 2.0 * 255.0, 0, 255).astype(np.uint8)]
        cols.append(np.hstack([np.pad(s, 2, constant_values=255) for s in strip]))
    grid = np.vstack(cols)
    fig_path = args.out_dir / f"grid_gs{args.fig_gs:g}.png"
    Image.fromarray(grid, mode="L").save(fig_path)

    csv_path = args.out_dir / "reinvert_ablation.csv"
    fields = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(records)

    # aggregate
    agg = {}
    for gs in args.guidance:
        tag = f"gs{gs:g}"
        agg[tag] = {
            "mean_psnr_AB": round(float(np.mean([r[f"{tag}_psnr_AB"] for r in records])), 3),
            "mean_cos_AB": round(float(np.mean([r[f"{tag}_cos_AB"] for r in records])), 6),
            "mean_time_A_s": round(float(np.mean([r[f"{tag}_time_A_s"] for r in records])), 3),
            "mean_time_B_s": round(float(np.mean([r[f"{tag}_time_B_s"] for r in records])), 3),
            "mean_speedup_A_over_B": round(float(np.mean([r[f"{tag}_speedup_A_over_B"] for r in records])), 3),
        }
    summary = {
        "task_id": "reinvert-ablation", "seed": SEED, "n": len(records),
        "split": args.split.name, "noise_level": NOISE_LEVEL, "guidance": args.guidance,
        "variants": {"A": "predecessor: re-invert x_rec then CFG denoise (ddim_sample_loop_known)",
                     "B": "proposed: CFG denoise straight from decrypted x_rec_T"},
        "aggregate": agg, "per_image": records,
        "note": ("Timing is the healthy step only; x_rec is needed either way. A high "
                 "psnr/cosine(A,B) means the re-inversion is a redundant round-trip: B "
                 "reaches the same healthy image while skipping one inversion pass."),
    }
    (args.out_dir / "reinvert_ablation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {fig_path}\nwrote {csv_path}\nagg={json.dumps(agg)}")


if __name__ == "__main__":
    main()
