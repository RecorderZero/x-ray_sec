#!/usr/bin/env python3
"""Visual grid for the deployment S2 sweep: recovered image across step counts.

Shows x0 and the range-preserving-PNG recovery for P and S2fn (fixed-norm) at
N in {499,200,100,50,25} on two dev images, so the step-count collapse and the
S2-vs-P difference are visible at a glance.  Same aligned decode + pipeline as
`scripts/deploy_s2_sweep.py`.

  scripts/run_cfg_ddim.sh python -m scripts.deploy_s2_grid
"""
from __future__ import annotations

from pathlib import Path
import csv

import torch
from PIL import Image, ImageDraw

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.m1_storage import range_preserving_png_handoff
from scripts.respaced_ddim_sweep import Respaced, schedule, psnr, disp, SEED, TOP
from scripts.deploy_s2_sweep import encrypt, decrypt

CKPT = Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt")
IMG_IDX = (0, 10)
GRID_N = (499, 200, 100, 50, 25)
SCHEMES = ("P", "S2fn")


@torch.inference_mode()
def recover(rs, x0, sid, scheme, n):
    b = schedule(n)
    z = rs.forward(x0, b)
    zc, orig = encrypt(z, scheme, sid)
    anon = rs.backward(zc, b)
    anon_png, _ = range_preserving_png_handoff(anon)
    zc2 = rs.forward(anon_png, b)
    return rs.backward(decrypt(zc2, scheme, sid, orig), b)


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
    imgs = [(rows[i]["sample_id"], preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device))
            for i in IMG_IDX]

    row_keys = [(sid, x0, sch) for sid, x0 in imgs for sch in SCHEMES]
    cols = ["x0"] + [f"N={n}" for n in GRID_N]
    tile, pad, top = 196, 4, 32
    H = top + len(row_keys) * (tile + pad) + pad
    W = pad + len(cols) * (tile + pad)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for ci, c in enumerate(cols):
        draw.text((pad + ci * (tile + pad) + 2, 4), c, fill="black")

    for ri, (sid, x0, sch) in enumerate(row_keys):
        y = top + ri * (tile + pad)
        # x0 column with row label
        draw.text((pad + 2, y + 2), f"{sid.split('_')[-1]}·{sch}", fill="black")
        canvas.paste(Image.fromarray(disp(x0[0, 0]), mode="L").convert("RGB").resize((tile, tile)), (pad, y))
        for ci, n in enumerate(GRID_N, start=1):
            x = pad + ci * (tile + pad)
            rec = recover(rs, x0, sid, sch, n)
            p = psnr(rec, x0)
            canvas.paste(Image.fromarray(disp(rec[0, 0]), mode="L").convert("RGB").resize((tile, tile)), (x, y))
            draw.text((x + 3, y + 3), f"{p:.1f} dB", fill="yellow")
        print(f"row {sid}·{sch} done", flush=True)

    path = out_dir / "grid.png"
    canvas.save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
