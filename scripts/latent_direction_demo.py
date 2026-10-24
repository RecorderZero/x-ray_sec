#!/usr/bin/env python3
"""Demo: a latent error's DIRECTION, not its size, decides image reconstruction.

Illustrates the 重要.md §「cos 其實不代表還原程度的好壞」 point with a figure the
user can read directly. For each dev image we take z = invert(x0), base = G(z)
(the P output), then add four errors to z and regenerate, reporting latent
cos/RMSE and image PSNR vs base:

  1. white_small   white Gaussian, RMSE ~0.135 (≈ S0/S1 decryption-error size)
  2. S0_actual     S0's real decrypted latent after the 8-bit PNG hand-off
  3. white_large   white Gaussian, RMSE ~1.12  (≈ P re-inversion-error size)
  4. P_actual      P's real re-inversion latent after the PNG hand-off

Punch line (reproduces appendix_audit/.../w01_fig03): white_large and P_actual
have the SAME latent RMSE (~1.1) yet opposite image PSNR -- white_large collapses
(~6 dB), P_actual is near-perfect (~44 dB) -- because P's error lies in the
generator-insensitive null directions while random / key-scrambled errors do not.

This is an implementer-side independent reproduction of the auditor diagnostic
`audit/e2_latent_direction_sensitivity.py` (same legacy flow, guidance -1, T=500),
adding a side-by-side IMAGE grid so the direction effect is visible, not just tabulated.

  scripts/run_cfg_ddim.sh python -m scripts.latent_direction_demo
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "past/SourceCode")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_legacy_wrapper import legacy_invert, legacy_generate

SEED = 1911
T = 500
INDICES = (0, 1, 10, 11)  # dev_v1.1 rows used by the auditor diagnostic


def legacy_png(x: torch.Tensor) -> torch.Tensor:
    """Predecessor 8-bit PNG hand-off: min-max to [0,1], quantize to 8-bit, back."""
    lo, hi = x.min(), x.max()
    return ((x - lo) / (hi - lo) * 255).to(torch.uint8).float() / 255.0


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    # auditor convention: compare on [0,1] (matches legacy_png range)
    a = a.double().clamp(0, 1)
    b = b.double().clamp(0, 1)
    mse = float((a - b).pow(2).mean())
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def lat(a: torch.Tensor, b: torch.Tensor) -> dict:
    x, y = a.double().flatten(), b.double().flatten()
    return {"cos": float(x @ y / (x.norm() * y.norm())),
            "rmse": float((x - y).pow(2).mean().sqrt())}


def disp(t: torch.Tensor) -> np.ndarray:
    """Per-image min-max to 8-bit purely for display (quantitative claim is the PSNR)."""
    a = t.detach().float().cpu().numpy()
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo + 1e-8)
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt"))
    ap.add_argument("--indices", type=int, nargs="+", default=list(INDICES))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/latent_direction_demo"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from guided_diffusion.anonymization import RademacherKey
    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    key = RademacherKey(shape=(1, 256, 256), seed=42, password="Ki@13579").key.to(device)

    fwd = lambda x: legacy_invert(diffusion, model_fn, x, T)          # noqa: E731  x0 -> z_T
    gen = lambda z: legacy_generate(diffusion, model_fn, z, T)        # noqa: E731  z_T -> image

    rows = list(csv.DictReader(args.split.open(encoding="utf-8")))
    # fixed column order for the table + grid
    COND = ["white_small", "S0_actual", "white_large", "P_actual"]
    LABEL = {"white_small": "+white (RMSE~0.14)", "S0_actual": "+S0 decrypt err",
             "white_large": "+white (RMSE~1.1)", "P_actual": "+P re-invert err"}
    records, fig_rows = [], []

    for i, ridx in enumerate(args.indices):
        r = rows[ridx]
        x0 = preprocess(Path(r["local_path"]))[0].unsqueeze(0).to(device)
        z = fwd(x0)
        base = gen(z)
        if i == 0:
            print(f"[range] x0 [{x0.min():.3f},{x0.max():.3f}]  "
                  f"base [{base.min():.3f},{base.max():.3f}]", flush=True)

        g = torch.Generator(device=device).manual_seed(SEED + i)
        white = torch.randn(z.shape, generator=g, device=device)
        perturbed = {
            "white_small": z + white * 0.135,
            "white_large": z + white * 1.12,
            "P_actual": fwd(legacy_png(base)),            # real P re-inversion latent
            "S0_actual": fwd(legacy_png(gen(z * key))) * key,  # real S0 decrypted latent
        }

        rec = {"sample_id": r["sample_id"]}
        imgs = {"x0": x0, "base": base}
        for name in COND:
            zz = perturbed[name]
            img = gen(zz)
            m = lat(z, zz)
            rec[name] = {**m, "psnr_vs_base": psnr(base, img)}
            imgs[name] = img
        records.append(rec)
        fig_rows.append(imgs)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    # ---- summary (mean over images) ----
    summary = {c: {k: float(np.mean([x[c][k] for x in records]))
                   for k in ("cos", "rmse", "psnr_vs_base")} for c in COND}
    (args.out_dir / "latent_direction_demo.json").write_text(
        json.dumps({"summary": summary, "per_sample": records,
                    "indices": args.indices, "seed": SEED, "T": T}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with (args.out_dir / "latent_direction_demo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "latent_cos", "latent_rmse", "image_psnr_vs_base_dB"])
        for c in COND:
            w.writerow([c, round(summary[c]["cos"], 4), round(summary[c]["rmse"], 4),
                        round(summary[c]["psnr_vs_base"], 2)])

    # ---- image grid: rows = images; cols = x0 | G(z) | 4 conditions ----
    cols = ["x0", "base"] + COND
    head = {"x0": "original x0", "base": "G(z) = P output"}
    head.update(LABEL)
    tile, pad, top = 196, 4, 34
    H = top + len(fig_rows) * (tile + pad) + pad
    W = pad + len(cols) * (tile + pad)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for ci, c in enumerate(cols):
        x = pad + ci * (tile + pad)
        draw.text((x + 2, 4), head[c], fill="black")
        for ri, imgs in enumerate(fig_rows):
            y = top + ri * (tile + pad)
            arr = disp(imgs[c][0, 0])
            canvas.paste(Image.fromarray(arr, mode="L").convert("RGB").resize((tile, tile)), (x, y))
            if c in COND:
                p = records[ri][c]["psnr_vs_base"]
                draw.text((x + 3, y + 3), f"{p:.1f} dB", fill="yellow")
                draw.text((x + 3, y + tile - 14), f"cos {records[ri][c]['cos']:.2f}", fill="yellow")
    grid_path = args.out_dir / "grid.png"
    canvas.save(grid_path)

    print("\n=== summary (mean over %d images) ===" % len(records))
    for c in COND:
        s = summary[c]
        print(f"  {c:12s}  latent cos={s['cos']:.3f} rmse={s['rmse']:.3f}  "
              f"image PSNR vs G(z)={s['psnr_vs_base']:.1f} dB")
    print(f"\nwrote {grid_path} + .json/.csv")


if __name__ == "__main__":
    main()
