#!/usr/bin/env python3
"""Lesion-region recovery fidelity: is the effusion REGION preserved by the
key-controlled 4-pass recovery as well as the whole image?

For CheXlocalize val patients WITH a Pleural Effusion GT mask: run the deployment
pipeline (invert -> encrypt S2fn -> anon -> range-PNG -> invert -> decrypt ->
recover) and compute PSNR(recovered, x0) GLOBAL vs INSIDE-mask vs OUTSIDE-mask.

Supports the claim "recovery is near-lossless in the lesion region too" (so a
key-holder's recovered image preserves lesion detail for downstream training).
This is NOT the counterfactual localization task (that is chance-level); this is
regional reconstruction fidelity.

  PYTHONPATH=/data2/paper/.cache/localization_deps scripts/run_cfg_ddim.sh \
    python -m scripts.lesion_region_fidelity --n 24
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED
from scripts.exact_inv_sweep import DPMSolverpp, CKPT
from scripts.rq4_formal_redo import pipeline  # 4-pass, uses deploy_s2_sweep S2_ROUNDS/S2_PERM (DPHD)

GT = Path("dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json")
IMG_ROOT = Path("dataset/chexlocalize/CheXpert/val")


def decode_mask(entry) -> np.ndarray:
    from pycocotools import mask as coco
    obj = dict(entry)
    if isinstance(obj["counts"], str):
        obj["counts"] = obj["counts"].encode("ascii")
    return coco.decode(obj)                       # (H, W) uint8 {0,1}


def img_path(gt_key: str) -> Path:
    # patient64622_study1_view1_frontal -> val/patient64622/study1/view1_frontal.jpg
    pid, study, rest = gt_key.split("_", 2)
    return IMG_ROOT / pid / study / f"{rest}.jpg"


def psnr(a: np.ndarray, b: np.ndarray, m: np.ndarray | None = None) -> float:
    if m is not None:
        a, b = a[m], b[m]
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 0 else 10 * np.log10(1.0 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("paper_assets/tables/table_lesion_region_fidelity.json"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    dev = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), CKPT, dev)
    rs = Respaced(diffusion, model_fn, dev, backward_null=True)
    dpm = DPMSolverpp(rs)
    b = schedule(args.steps)

    gt = json.loads(GT.read_text())
    keys = sorted(k for k, v in gt.items()
                  if isinstance(v, dict) and v.get("Pleural Effusion", {}).get("counts"))

    rows = []
    for gk in keys:
        if len(rows) >= args.n:
            break
        p = img_path(gk)
        if not p.exists():
            continue
        # decode + align the mask FIRST (cheap); skip empty masks (effusion-negative
        # entries carry an all-zero RLE) before the expensive 4-pass pipeline.
        full = decode_mask(gt[gk]["Pleural Effusion"]).astype(np.uint8)   # (H,W)
        m = cv2.resize(full, (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
        if m.sum() < 16:
            continue
        x0 = preprocess(p)[0].unsqueeze(0).to(dev)          # [1,1,256,256] in [0,1], equalized
        x_rec, _ = pipeline(dpm, x0, gk, "S2fn", b)
        a = x0[0, 0].float().cpu().numpy()
        r = x_rec[0, 0].float().cpu().numpy()
        rows.append({"gt_key": gk, "mask_frac": round(float(m.mean()), 4),
                     "psnr_global": round(psnr(r, a), 3),
                     "psnr_in_mask": round(psnr(r, a, m), 3),
                     "psnr_out_mask": round(psnr(r, a, ~m), 3)})
        print(f"{gk:42s} mask%={rows[-1]['mask_frac']*100:4.1f}  "
              f"global={rows[-1]['psnr_global']:.2f}  in={rows[-1]['psnr_in_mask']:.2f}  "
              f"out={rows[-1]['psnr_out_mask']:.2f}", flush=True)

    g = np.array([x["psnr_global"] for x in rows])
    im = np.array([x["psnr_in_mask"] for x in rows])
    om = np.array([x["psnr_out_mask"] for x in rows])
    summary = {"schema_version": 1, "task_id": "lesion-region-fidelity", "seed": SEED,
               "pathology": "Pleural Effusion", "n": len(rows), "steps": args.steps,
               "scheme": "S2fn (deployment D.P.H.D)",
               "psnr_global_mean": round(float(g.mean()), 3),
               "psnr_in_mask_mean": round(float(im.mean()), 3),
               "psnr_out_mask_mean": round(float(om.mean()), 3),
               "in_minus_global_mean_db": round(float((im - g).mean()), 3),
               "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nn={len(rows)}  PSNR global={summary['psnr_global_mean']:.2f}  "
          f"in-mask={summary['psnr_in_mask_mean']:.2f}  out-mask={summary['psnr_out_mask_mean']:.2f}  "
          f"(in-global {summary['in_minus_global_mean_db']:+.2f} dB)\nwrote {args.out}")


if __name__ == "__main__":
    main()
